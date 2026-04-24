"""SAM-based segmentation: mask generation and post-processing."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import setup_logger

logger = setup_logger(__name__)


def load_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_h",
    device: str = "cuda",
):
    """Load a SAM model from a local checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint (e.g. sam_vit_h_4b8939.pth).
        model_type: One of "vit_h", "vit_l", "vit_b".
        device: "cuda", "mps", or "cpu".

    Returns:
        Loaded SAM model placed on `device`.
    """
    try:
        from segment_anything import sam_model_registry
    except ImportError as e:
        raise ImportError(
            "segment-anything is not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    logger.info(f"Loading SAM model ({model_type}) from {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    logger.info("SAM model loaded.")
    return sam


def generate_mask(
    image: Image.Image,
    sam_model,
    point_coords: Optional[List[Tuple[int, int]]] = None,
    point_labels: Optional[List[int]] = None,
    box: Optional[List[int]] = None,
    auto_select: bool = False,
    auto_index: int = 0,
) -> np.ndarray:
    """Generate a binary mask (uint8, values 0 or 255) using SAM.

    Provide one of:
      - point_coords + optional point_labels  → prompt-based prediction
      - box                                   → bounding-box prediction
      - auto_select=True                      → automatic mask (picks auto_index-th largest)

    Returns:
        np.ndarray of shape (H, W) with values in {0, 255}.
    """
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor

    img_array = np.array(image)

    if auto_select:
        logger.info("Running automatic mask generation …")
        mask_gen = SamAutomaticMaskGenerator(
            sam_model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )
        masks = mask_gen.generate(img_array)
        if not masks:
            raise RuntimeError("SAM produced no masks for this image.")
        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
        chosen = masks_sorted[auto_index]["segmentation"]
        logger.info(
            f"Auto-selected mask #{auto_index} (area={masks_sorted[auto_index]['area']})"
        )
        return chosen.astype(np.uint8) * 255

    # Prompt-based
    predictor = SamPredictor(sam_model)
    predictor.set_image(img_array)

    pts = np.array(point_coords) if point_coords is not None else None
    lbls = (
        np.array(point_labels)
        if point_labels is not None
        else (np.ones(len(point_coords), dtype=int) if point_coords is not None else None)
    )
    bx = np.array(box) if box is not None else None

    if pts is None and bx is None:
        raise ValueError(
            "Provide point_coords, box, or set auto_select=True."
        )

    logger.info(
        f"Running SAM prediction (points={pts is not None}, box={bx is not None}) …"
    )
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbls,
        box=bx,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    logger.info(
        f"Best mask score: {scores[best_idx]:.4f} (index {best_idx})"
    )
    return masks[best_idx].astype(np.uint8) * 255


def load_grounding_dino(
    model_id: str = "IDEA-Research/grounding-dino-base",
    device: str = "cpu",
):
    """Load Grounding DINO processor and model from HuggingFace.

    Downloads automatically on first use (~900 MB for base variant).
    Returns:
        (processor, model) tuple — pass both to generate_grounded_mask().
    """
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as e:
        raise ImportError("transformers>=4.38.0 is required for Grounded-SAM.") from e

    logger.info(f"Loading Grounding DINO ({model_id}) …")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    logger.info("Grounding DINO loaded.")
    return processor, model


def generate_grounded_mask(
    image: Image.Image,
    gdino_processor,
    gdino_model,
    sam_model,
    text_query: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    device: str = "cpu",
) -> np.ndarray:
    """Find all regions matching text_query via Grounding DINO, then mask with SAM.

    Runs Grounding DINO to detect all instances of text_query, feeds each
    bounding box into SAM, and returns the union of all resulting masks.

    Args:
        image: Input PIL image.
        gdino_processor: Processor returned by load_grounding_dino().
        gdino_model: Model returned by load_grounding_dino().
        sam_model: SAM model returned by load_sam_model().
        text_query: Region to find, e.g. "hair", "eyes", "shirt".
        box_threshold: Grounding DINO confidence threshold for boxes.
        text_threshold: Grounding DINO confidence threshold for text tokens.
        device: Device the Grounding DINO model lives on.

    Returns:
        np.ndarray of shape (H, W) with values in {0, 255} — union of all masks.

    Raises:
        RuntimeError: If no regions are found for text_query.
    """
    import torch
    from segment_anything import SamPredictor

    query = text_query.strip()
    if not query.endswith("."):
        query += "."

    img_array = np.array(image)
    H, W = img_array.shape[:2]

    inputs = gdino_processor(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(H, W)],
    )
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]

    if len(boxes) == 0:
        raise RuntimeError(
            f"Grounding DINO found no regions matching '{text_query}'. "
            "Try lowering --grounded-box-threshold or rephrasing the query."
        )

    logger.info(
        f"Grounding DINO: {len(boxes)} region(s) for '{text_query}' "
        f"(scores: {[f'{s:.2f}' for s in scores.tolist()]})"
    )

    predictor = SamPredictor(sam_model)
    predictor.set_image(img_array)

    union_mask = np.zeros((H, W), dtype=np.uint8)
    for box in boxes.cpu().numpy():
        masks, mask_scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
        best = masks[int(np.argmax(mask_scores))]
        union_mask = np.maximum(union_mask, best.astype(np.uint8) * 255)

    logger.info(f"Grounded mask: {int(union_mask.sum() / 255)} pixels selected across {len(boxes)} region(s).")
    return union_mask


def process_mask(
    mask: np.ndarray,
    dilation_kernel_size: int = 15,
    dilation_iterations: int = 2,
    blur_kernel_size: int = 21,
) -> np.ndarray:
    """Dilate and Gaussian-blur a binary mask to soften edges.

    Args:
        mask: uint8 mask with values in {0, 255}.
        dilation_kernel_size: Size of the elliptic structuring element.
        dilation_iterations: Number of dilation passes.
        blur_kernel_size: Kernel size for Gaussian blur (must be odd).

    Returns:
        Processed mask (uint8, values 0-255).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_kernel_size, dilation_kernel_size),
    )
    dilated = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1  # must be odd
    blurred = cv2.GaussianBlur(dilated, (blur_kernel_size, blur_kernel_size), 0)

    logger.info(
        f"Mask processed — dilation={dilation_kernel_size}x{dilation_kernel_size}x{dilation_iterations}, "
        f"blur={blur_kernel_size}x{blur_kernel_size}"
    )
    return blurred
