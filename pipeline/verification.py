"""Identity and background similarity verification.

Uses:
  - InsightFace  → face embedding cosine similarity
  - CLIP         → semantic background similarity
  - SSIM         → pixel-level background fidelity
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Face verification (InsightFace)
# ---------------------------------------------------------------------------

def load_face_analyzer(
    model_name: str = "buffalo_l",
    providers: Optional[list] = None,
):
    """Load the InsightFace FaceAnalysis app.

    Args:
        model_name: InsightFace model pack (e.g. "buffalo_l", "antelopev2").
        providers: ONNX execution providers. Defaults to CUDA then CPU.

    Returns:
        Prepared FaceAnalysis instance.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ImportError(
            "insightface is not installed. Run: pip install insightface"
        ) from e

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    logger.info(f"Loading InsightFace model '{model_name}' …")
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace ready.")
    return app


def get_face_embedding(
    analyzer,
    image: Image.Image,
) -> Optional[np.ndarray]:
    """Extract a normalised face embedding from the largest face in `image`.

    Returns:
        1-D float32 embedding, or None if no face is detected.
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = analyzer.get(img_bgr)
    if not faces:
        logger.warning("No face detected in image.")
        return None
    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return largest.normed_embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# CLIP-based background similarity
# ---------------------------------------------------------------------------

def load_clip_model(
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
):
    """Load a CLIP model and its processor.

    Returns:
        Tuple of (CLIPModel, CLIPProcessor).
    """
    from transformers import CLIPModel, CLIPProcessor

    logger.info(f"Loading CLIP model '{model_id}' …")
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.to(device)
    model.eval()
    logger.info("CLIP model ready.")
    return model, processor


def get_clip_embedding(
    model,
    processor,
    image: Image.Image,
    device: str = "cpu",
) -> np.ndarray:
    """Return a normalised CLIP image embedding."""
    import torch

    # Pass only pixel_values to avoid unexpected kwargs that can cause
    # get_image_features to return a BaseModelOutputWithPooling instead of a tensor.
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        out = model.get_image_features(pixel_values=pixel_values)
    # Newer transformers versions may wrap the result in a dataclass
    features: torch.Tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Structural Similarity Index between two PIL images (grayscale)."""
    from skimage.metrics import structural_similarity

    arr1 = np.array(img1.convert("L"))
    arr2 = np.array(img2.convert("L").resize(img1.size, Image.LANCZOS))
    score, _ = structural_similarity(arr1, arr2, full=True)
    return float(score)


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_similarity(
    original: Image.Image,
    edited: Image.Image,
    mask: np.ndarray,
    face_analyzer,
    clip_model=None,
    clip_processor=None,
    clip_device: str = "cpu",
) -> Dict[str, Optional[float]]:
    """Measure how faithfully the edit preserved identity and background.

    Args:
        original: Unmodified source image.
        edited: Composited result to evaluate.
        mask: Soft mask used for the edit (0=background, 255=edited region).
        face_analyzer: InsightFace FaceAnalysis instance.
        clip_model: Optional CLIPModel; if None, CLIP scores are skipped.
        clip_processor: CLIPProcessor paired with clip_model.
        clip_device: Device for CLIP inference.

    Returns:
        Dict with keys:
            "face_similarity"           – cosine sim of face embeddings (or None)
            "background_ssim"           – SSIM over the background region
            "background_clip_similarity"– CLIP cosine sim of background crops (or None)
            "edit_region_ssim"          – SSIM inside the mask (lower = more changed)
    """
    results: Dict[str, Optional[float]] = {}

    # --- Face similarity ---
    orig_emb = get_face_embedding(face_analyzer, original)
    edit_emb = get_face_embedding(face_analyzer, edited)
    if orig_emb is not None and edit_emb is not None:
        results["face_similarity"] = cosine_similarity(orig_emb, edit_emb)
        logger.info(f"Face similarity: {results['face_similarity']:.4f}")
    else:
        results["face_similarity"] = None
        logger.info("Face similarity: N/A (face not detected in one or both images)")

    # --- Region extraction ---
    orig_arr = np.array(original.convert("RGB"))
    edit_arr = np.array(edited.convert("RGB").resize(original.size, Image.LANCZOS))

    edit_alpha = mask.astype(np.float32) / 255.0  # 1.0 inside edited region
    inv_alpha = 1.0 - edit_alpha                  # 1.0 in background
    if edit_alpha.ndim == 2:
        edit_alpha = edit_alpha[:, :, np.newaxis]
        inv_alpha = inv_alpha[:, :, np.newaxis]

    # Zero-out the opposite region for each crop so SSIM is region-focused
    orig_bg = (orig_arr * inv_alpha).astype(np.uint8)
    edit_bg = (edit_arr * inv_alpha).astype(np.uint8)
    orig_fg = (orig_arr * edit_alpha).astype(np.uint8)
    edit_fg = (edit_arr * edit_alpha).astype(np.uint8)

    # --- Edit-region SSIM (lower = more changed — what we WANT for edits) ---
    results["edit_region_ssim"] = compute_ssim(
        Image.fromarray(orig_fg), Image.fromarray(edit_fg)
    )
    logger.info(f"Edit-region SSIM: {results['edit_region_ssim']:.4f} (lower = more changed)")

    # --- Background SSIM ---
    results["background_ssim"] = compute_ssim(
        Image.fromarray(orig_bg), Image.fromarray(edit_bg)
    )
    logger.info(f"Background SSIM: {results['background_ssim']:.4f}")

    # --- Background CLIP similarity (optional) ---
    if clip_model is not None and clip_processor is not None:
        orig_clip = get_clip_embedding(clip_model, clip_processor, Image.fromarray(orig_bg), clip_device)
        edit_clip = get_clip_embedding(clip_model, clip_processor, Image.fromarray(edit_bg), clip_device)
        results["background_clip_similarity"] = float(np.dot(orig_clip, edit_clip))
        logger.info(
            f"Background CLIP similarity: {results['background_clip_similarity']:.4f}"
        )
    else:
        results["background_clip_similarity"] = None

    return results
