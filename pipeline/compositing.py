"""Alpha-composite the generated region back onto the original image."""

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
