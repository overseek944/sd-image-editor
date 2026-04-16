"""Alpha-composite the generated region back onto the original image,
plus crop-and-inpaint helpers to eliminate context anchoring."""

from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .utils import setup_logger

logger = setup_logger(__name__)


def blend_images(
    original: Image.Image,
    generated: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Alpha-composite `generated` (inside mask) onto `original` (outside mask).

    The soft edges produced by Gaussian-blurring the mask create a smooth
    transition so that the paste boundary is invisible.

    Args:
        original: The unmodified source image (RGB).
        generated: The inpainted image returned by the diffusion pipeline (RGB).
        mask: uint8 mask with soft edges (0=keep original, 255=use generated).

    Returns:
        Composited PIL image at the same resolution as `original`.
    """
    orig_arr = np.array(original.convert("RGB")).astype(np.float32)
    gen_arr = np.array(generated.convert("RGB").resize(original.size, Image.LANCZOS)).astype(
        np.float32
    )

    # Normalise mask to [0, 1], broadcast to (H, W, 1) for RGB blending
    alpha = mask.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]
    if alpha.shape[:2] != orig_arr.shape[:2]:
        from PIL import Image as _PIL
        alpha_pil = _PIL.fromarray((alpha[:, :, 0] * 255).astype(np.uint8)).resize(
            original.size, _PIL.LANCZOS
        )
        alpha = np.array(alpha_pil).astype(np.float32)[:, :, np.newaxis] / 255.0

    blended = orig_arr * (1.0 - alpha) + gen_arr * alpha
    result = Image.fromarray(blended.clip(0, 255).astype(np.uint8))

    logger.info("Images composited (original × (1−α) + generated × α).")
    return result


# ---------------------------------------------------------------------------
# Crop-and-inpaint helpers
# ---------------------------------------------------------------------------

def crop_for_inpaint(
    image: Image.Image,
    mask: np.ndarray,
    padding_fraction: float = 0.20,
    target_size: Tuple[int, int] = (1024, 1024),
) -> Tuple[Image.Image, np.ndarray, Dict]:
    """Crop the mask bounding box + proportional padding, then upscale to target_size.

    Passing a tight crop to the diffusion pipeline eliminates "context anchoring":
    the model no longer sees the original hair pixels in its attention field and is
    free to generate genuinely different content inside the mask.

    Args:
        image: Full-resolution original RGB image.
        mask: uint8 mask (0=background, 255=edit region) at the same resolution.
        padding_fraction: Fraction of bbox dimensions to add as padding on each side.
        target_size: (W, H) to upscale the crop to before passing to the pipeline.

    Returns:
        Tuple of:
          - crop_image  : PIL image of the cropped region, upscaled to target_size
          - crop_mask   : uint8 mask of the same crop, upscaled to target_size
          - crop_meta   : dict with "bbox" (x0,y0,x1,y1), "crop_size", "orig_size"
    """
    H, W = mask.shape[:2]

    # Find bounding box of non-zero mask pixels
    rows = np.where(np.any(mask > 127, axis=1))[0]
    cols = np.where(np.any(mask > 127, axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        # Mask is empty — fall back to full image
        logger.warning("crop_for_inpaint: mask is empty, using full image.")
        crop_meta = {
            "bbox": (0, 0, W, H),
            "crop_size": (W, H),
            "orig_size": image.size,
            "cropped": False,
        }
        return (
            image.resize(target_size, Image.LANCZOS),
            np.array(Image.fromarray(mask).resize(target_size, Image.NEAREST)),
            crop_meta,
        )

    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])

    # Add proportional padding, clamped to image bounds
    pad_y = max(1, int((y1 - y0) * padding_fraction))
    pad_x = max(1, int((x1 - x0) * padding_fraction))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(W, x1 + pad_x)
    y1 = min(H, y1 + pad_y)

    crop_w, crop_h = x1 - x0, y1 - y0

    # Crop and upscale
    crop_image = image.crop((x0, y0, x1, y1)).resize(target_size, Image.LANCZOS)
    crop_mask_pil = Image.fromarray(mask[y0:y1, x0:x1]).resize(target_size, Image.NEAREST)
    crop_mask = np.array(crop_mask_pil)

    crop_meta = {
        "bbox": (x0, y0, x1, y1),
        "crop_size": (crop_w, crop_h),
        "orig_size": image.size,
        "cropped": True,
    }

    logger.info(
        f"Crop-for-inpaint: bbox=({x0},{y0},{x1},{y1}), "
        f"crop={crop_w}×{crop_h} → upscaled to {target_size[0]}×{target_size[1]}"
    )
    return crop_image, crop_mask, crop_meta


def paste_inpaint_result(
    original: Image.Image,
    crop_result: Image.Image,
    crop_meta: Dict,
) -> Image.Image:
    """Scale the inpainted crop back to its original size and paste into the full image.

    Call this BEFORE blend_images so the soft-mask composite still handles edge blending.

    Args:
        original: Full-resolution original image (used as the base canvas).
        crop_result: Generated image at target_size (output of the pipeline).
        crop_meta: Dict returned by crop_for_inpaint (contains bbox and crop_size).

    Returns:
        Full-resolution PIL image with the inpainted crop pasted at the correct location.
    """
    if not crop_meta.get("cropped", True):
        # Was not cropped — resize directly back to full resolution
        return crop_result.resize(crop_meta["orig_size"], Image.LANCZOS)

    x0, y0, x1, y1 = crop_meta["bbox"]
    cw, ch = crop_meta["crop_size"]

    # Downscale crop result back to its original dimensions before pasting
    resized_crop = crop_result.resize((cw, ch), Image.LANCZOS)

    canvas = original.copy()
    canvas.paste(resized_crop, (x0, y0))

    logger.info(f"Pasted inpaint crop back at bbox=({x0},{y0},{x1},{y1}).")
    return canvas
