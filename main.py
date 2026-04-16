"""
sd-image-editor — end-to-end controlled image editing pipeline.

Usage (single function call):

    from main import run_pipeline

    result = run_pipeline(
        image_path="input/portrait.jpg",
        prompt="make hair curly",
        point_coords=[(320, 80)],   # click on the hair region
        output_path="output/result.png",
    )
    print(result["scores"])

CLI usage:

    python main.py --image input/portrait.jpg --prompt "make hair curly" \\
        --point 320 80 --output output/result.png
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline.compositing import blend_images
from pipeline.controlnet import apply_controlnet, extract_canny_edges, load_controlnet_pipeline
from pipeline.diffusion import load_inpaint_pipeline, run_inpainting
from pipeline.segmentation import generate_mask, load_sam_model, process_mask
from pipeline.utils import get_device, load_image, save_image, setup_logger
from pipeline.verification import (
    evaluate_similarity,
    load_clip_model,
    load_face_analyzer,
)

logger = setup_logger("pipeline", level=logging.INFO)


# ---------------------------------------------------------------------------
# Model registry — loaded lazily and cached between retry attempts
# ---------------------------------------------------------------------------

class _ModelCache:
    sam = None
    inpaint_pipe = None
    controlnet_pipe = None
    face_analyzer = None
    clip_model = None
    clip_processor = None


def _ensure_models(
    sam_checkpoint: str,
    sam_model_type: str,
    use_controlnet: bool,
    device: str,
    inpaint_model_id: str,
    controlnet_model_id: str,
    base_model_id: str,
    face_providers: List[str],
    clip_model_id: str,
    load_clip: bool,
) -> None:
    """Lazy-load every model once and cache in _ModelCache."""
    if _ModelCache.sam is None:
        _ModelCache.sam = load_sam_model(sam_checkpoint, sam_model_type, device)

    if use_controlnet:
        if _ModelCache.controlnet_pipe is None:
            _ModelCache.controlnet_pipe = load_controlnet_pipeline(
                controlnet_model_id, base_model_id, device
            )
    else:
        if _ModelCache.inpaint_pipe is None:
            _ModelCache.inpaint_pipe = load_inpaint_pipeline(inpaint_model_id, device)

    if _ModelCache.face_analyzer is None:
        _ModelCache.face_analyzer = load_face_analyzer(providers=face_providers)

    if load_clip and _ModelCache.clip_model is None:
        _ModelCache.clip_model, _ModelCache.clip_processor = load_clip_model(clip_model_id)


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def retry_logic(
    original: Image.Image,
    mask_processed: np.ndarray,
    prompt: str,
    use_controlnet: bool,
    canny_image: Optional[Image.Image],
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    controlnet_conditioning_scale: float,
    negative_prompt: str,
    face_threshold: float,
    bg_ssim_threshold: float,
    min_edit_threshold: float,
    max_attempts: int,
    load_clip: bool,
    clip_device: str,
) -> Tuple[Optional[Image.Image], Dict, List[Dict]]:
    """Attempt inpainting up to `max_attempts` times, returning the first result
    that passes face-similarity, background-SSIM, AND minimum edit checks.

    Adaptive escalation: if the edit region is too similar to the original
    (edit_region_ssim > 1 - min_edit_threshold), strength is increased by 0.1
    and ControlNet scale is reduced by 0.1 on the next attempt to give the
    model more creative freedom.

    Returns:
        (accepted_image_or_None, last_scores, all_attempt_logs)
    """
    attempt_logs: List[Dict] = []
    best_result: Optional[Image.Image] = None
    best_scores: Dict = {}

    current_strength = strength
    current_cn_scale = controlnet_conditioning_scale

    for attempt in range(1, max_attempts + 1):
        seed = attempt * 137  # deterministic but different each attempt
        logger.info(
            f"--- Attempt {attempt}/{max_attempts} "
            f"(seed={seed}, strength={current_strength:.2f}, cn_scale={current_cn_scale:.2f}) ---"
        )

        # --- Generate ---
        if use_controlnet and _ModelCache.controlnet_pipe is not None:
            generated = apply_controlnet(
                pipe=_ModelCache.controlnet_pipe,
                image=original,
                mask=mask_processed,
                canny_image=canny_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=current_strength,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=current_cn_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        else:
            generated = run_inpainting(
                pipe=_ModelCache.inpaint_pipe,
                image=original,
                mask=mask_processed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=current_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

        # --- Composite ---
        blended = blend_images(original, generated, mask_processed)

        # --- Evaluate ---
        scores = evaluate_similarity(
            original=original,
            edited=blended,
            mask=mask_processed,
            face_analyzer=_ModelCache.face_analyzer,
            clip_model=_ModelCache.clip_model if load_clip else None,
            clip_processor=_ModelCache.clip_processor if load_clip else None,
            clip_device=clip_device,
        )

        # --- Decision ---
        face_sim = scores.get("face_similarity")
        bg_ssim = scores.get("background_ssim", 1.0)
        edit_ssim = scores.get("edit_region_ssim", 0.0)

        face_ok = face_sim is None or face_sim >= face_threshold
        bg_ok = bg_ssim >= bg_ssim_threshold
        # edit_ok: the masked region must differ from the original by at least min_edit_threshold
        edit_ok = edit_ssim <= (1.0 - min_edit_threshold)

        log_entry = {
            "attempt": attempt,
            "seed": seed,
            "strength": current_strength,
            "cn_scale": current_cn_scale,
            "scores": scores,
            "accepted": face_ok and bg_ok and edit_ok,
        }
        attempt_logs.append(log_entry)

        if face_ok and bg_ok and edit_ok:
            logger.info(
                f"Attempt {attempt} ACCEPTED — face_sim={face_sim}, "
                f"bg_ssim={bg_ssim:.4f}, edit_ssim={edit_ssim:.4f}"
            )
            return blended, scores, attempt_logs

        reasons = []
        if not face_ok:
            reasons.append(f"face_sim={face_sim:.4f} < {face_threshold}")
        if not bg_ok:
            reasons.append(f"bg_ssim={bg_ssim:.4f} < {bg_ssim_threshold}")
        if not edit_ok:
            reasons.append(
                f"edit_ssim={edit_ssim:.4f} too high (edit barely happened) — escalating"
            )
            # Adaptive escalation: more noise, looser ControlNet
            current_strength = min(0.75, current_strength + 0.1)
            current_cn_scale = max(0.2, current_cn_scale - 0.1)
        logger.warning(f"Attempt {attempt} REJECTED — {'; '.join(reasons)}")

        best_result = blended
        best_scores = scores

    logger.error(
        f"All {max_attempts} attempts failed. Returning best result from last attempt."
    )
    return best_result, best_scores, attempt_logs


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    prompt: str,
    # --- Output ---
    output_path: str = "output/result.png",
    log_path: str = "output/run_log.json",
    # --- SAM ---
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth",
    sam_model_type: str = "vit_h",
    # --- Mask selection (one of: points, box, or auto) ---
    point_coords: Optional[List[Tuple[int, int]]] = None,
    point_labels: Optional[List[int]] = None,
    box: Optional[List[int]] = None,
    auto_select: bool = False,
    auto_mask_index: int = 0,
    # --- Mask post-processing ---
    dilation_kernel_size: int = 15,
    dilation_iterations: int = 2,
    blur_kernel_size: int = 21,
    # --- Diffusion ---
    inpaint_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    use_controlnet: bool = True,
    controlnet_model_id: str = "diffusers/controlnet-canny-sdxl-1.0",
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    strength: float = 0.6,
    guidance_scale: float = 9.0,
    num_inference_steps: int = 30,
    controlnet_conditioning_scale: float = 0.5,
    canny_low: int = 100,
    canny_high: int = 200,
    negative_prompt: str = (
        "blurry, distorted, low quality, artifacts, bad anatomy, "
        "extra limbs, cloned face, disfigured, deformed"
    ),
    # --- Verification ---
    face_similarity_threshold: float = 0.85,
    background_ssim_threshold: float = 0.80,
    min_edit_threshold: float = 0.05,   # edit-region SSIM must drop by at least this much
    face_providers: Optional[List[str]] = None,
    load_clip: bool = True,
    clip_model_id: str = "openai/clip-vit-base-patch32",
    clip_device: str = "cpu",
    # --- Retry ---
    max_attempts: int = 3,
    # --- Device ---
    device: Optional[str] = None,
) -> Dict:
    """Run the full controlled-inpainting pipeline.

    Args:
        image_path: Path to the input image.
        prompt: Text description of the desired edit (e.g. "make hair curly").
        output_path: Where to save the final result.
        log_path: Where to save the JSON run log.
        sam_checkpoint: Local path to SAM checkpoint (.pth).
        sam_model_type: SAM architecture ("vit_h", "vit_l", "vit_b").
        point_coords: [(x, y), …] SAM prompt points.
        point_labels: 1=foreground, 0=background per point (default all 1).
        box: [x0, y0, x1, y1] bounding box prompt for SAM.
        auto_select: Use automatic mask generation (no prompt needed).
        auto_mask_index: Which auto mask to use (0 = largest area).
        dilation_kernel_size: Kernel size for mask dilation.
        dilation_iterations: Number of dilation passes.
        blur_kernel_size: Gaussian blur kernel (softens mask edges).
        inpaint_model_id: HF model ID for SDXL inpainting.
        use_controlnet: Enable ControlNet for structure preservation.
        controlnet_model_id: HF model ID for canny ControlNet.
        base_model_id: HF model ID for SDXL base (used with ControlNet).
        strength: Denoising strength (clamped to 0.2–0.5).
        guidance_scale: CFG scale.
        num_inference_steps: Diffusion steps.
        controlnet_conditioning_scale: ControlNet influence weight.
        canny_low: Canny lower threshold.
        canny_high: Canny upper threshold.
        negative_prompt: Negative conditioning text.
        face_similarity_threshold: Minimum acceptable face cosine similarity.
        background_ssim_threshold: Minimum acceptable background SSIM.
        face_providers: ONNX providers for InsightFace.
        load_clip: Whether to compute CLIP background similarity.
        clip_model_id: HF model ID for CLIP.
        clip_device: Device for CLIP inference.
        max_attempts: Maximum inpainting retries.
        device: Compute device. Auto-detected if None.

    Returns:
        Dict with keys:
            "output_path"   – str path to saved result
            "scores"        – final similarity scores
            "attempt_logs"  – per-attempt score history
            "accepted"      – bool, True if thresholds were met
    """
    t0 = time.time()
    device = device or get_device()
    logger.info(f"Pipeline starting — device={device}, prompt='{prompt}'")

    if face_providers is None:
        face_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

    # 1. Load image
    logger.info(f"[1/7] Loading image: {image_path}")
    original = load_image(image_path)
    logger.info(f"      Image size: {original.size}")

    # 2. Load models (lazy, cached)
    logger.info("[2/7] Loading models …")
    _ensure_models(
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        use_controlnet=use_controlnet,
        device=device,
        inpaint_model_id=inpaint_model_id,
        controlnet_model_id=controlnet_model_id,
        base_model_id=base_model_id,
        face_providers=face_providers,
        clip_model_id=clip_model_id,
        load_clip=load_clip,
    )

    # 3. Generate mask
    logger.info("[3/7] Generating SAM mask …")
    raw_mask = generate_mask(
        image=original,
        sam_model=_ModelCache.sam,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        auto_select=auto_select,
        auto_index=auto_mask_index,
    )

    # 4. Process mask
    logger.info("[4/7] Processing mask (dilation + blur) …")
    mask_processed = process_mask(
        mask=raw_mask,
        dilation_kernel_size=dilation_kernel_size,
        dilation_iterations=dilation_iterations,
        blur_kernel_size=blur_kernel_size,
    )

    # 5. ControlNet: extract canny edges
    canny_image: Optional[Image.Image] = None
    if use_controlnet:
        logger.info("[5/7] Extracting Canny edges for ControlNet …")
        canny_image = extract_canny_edges(original, canny_low, canny_high)
    else:
        logger.info("[5/7] ControlNet disabled, skipping Canny extraction.")

    # 6. Generate + verify with retry
    logger.info(f"[6/7] Running inpainting with up to {max_attempts} attempts …")
    result_image, final_scores, attempt_logs = retry_logic(
        original=original,
        mask_processed=mask_processed,
        prompt=prompt,
        use_controlnet=use_controlnet,
        canny_image=canny_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        negative_prompt=negative_prompt,
        face_threshold=face_similarity_threshold,
        bg_ssim_threshold=background_ssim_threshold,
        min_edit_threshold=min_edit_threshold,
        max_attempts=max_attempts,
        load_clip=load_clip,
        clip_device=clip_device,
    )

    # 7. Save output
    logger.info(f"[7/7] Saving result to {output_path} …")
    if result_image is not None:
        save_image(result_image, output_path)
    else:
        logger.error("No result image produced (all attempts rejected).")

    accepted = any(log["accepted"] for log in attempt_logs)

    run_summary = {
        "image_path": image_path,
        "prompt": prompt,
        "output_path": output_path,
        "device": device,
        "use_controlnet": use_controlnet,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "face_similarity_threshold": face_similarity_threshold,
        "background_ssim_threshold": background_ssim_threshold,
        "min_edit_threshold": min_edit_threshold,
        "scores": final_scores,
        "attempt_logs": attempt_logs,
        "accepted": accepted,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # Save JSON log
    log_out = Path(log_path)
    log_out.parent.mkdir(parents=True, exist_ok=True)
    with open(log_out, "w") as f:
        json.dump(run_summary, f, indent=2)
    logger.info(f"Run log saved to {log_path}")

    logger.info(
        f"Pipeline complete in {run_summary['elapsed_seconds']}s — "
        f"accepted={accepted}, scores={final_scores}"
    )
    return run_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Controlled image editing via SAM + SDXL inpainting."
    )
    p.add_argument("--image", required=True, help="Path to input image.")
    p.add_argument("--prompt", required=True, help='Edit description, e.g. "make hair curly".')
    p.add_argument("--output", default="output/result.png", help="Output image path.")
    p.add_argument("--log", default="output/run_log.json", help="JSON log path.")

    # SAM
    p.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    p.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument(
        "--point", nargs=2, type=int, action="append", metavar=("X", "Y"),
        help="SAM point prompt (x y). Repeat for multiple points.",
    )
    p.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--auto-select", action="store_true")
    p.add_argument("--auto-mask-index", type=int, default=0)

    # Diffusion
    p.add_argument("--no-controlnet", action="store_true")
    p.add_argument("--strength", type=float, default=0.4)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=30)

    # Verification
    p.add_argument("--face-threshold", type=float, default=0.85)
    p.add_argument("--bg-ssim-threshold", type=float, default=0.80)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--no-clip", action="store_true")

    # Device
    p.add_argument("--device", default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    result = run_pipeline(
        image_path=args.image,
        prompt=args.prompt,
        output_path=args.output,
        log_path=args.log,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        point_coords=[tuple(p) for p in args.point] if args.point else None,
        box=args.box,
        auto_select=args.auto_select,
        auto_mask_index=args.auto_mask_index,
        use_controlnet=not args.no_controlnet,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        face_similarity_threshold=args.face_threshold,
        background_ssim_threshold=args.bg_ssim_threshold,
        max_attempts=args.max_attempts,
        load_clip=not args.no_clip,
        device=args.device,
    )

    print("\n=== Pipeline Result ===")
    print(f"Accepted : {result['accepted']}")
    print(f"Output   : {result['output_path']}")
    print(f"Elapsed  : {result['elapsed_seconds']}s")
    print(f"Scores   : {result['scores']}")
