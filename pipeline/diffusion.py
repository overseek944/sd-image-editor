"""Stable Diffusion XL inpainting pipeline wrapper."""

from typing import Optional

import numpy as np
import torch
from PIL import Image

from .utils import setup_logger

logger = setup_logger(__name__)

# Default SDXL inpainting model hosted on Hugging Face
_DEFAULT_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def load_inpaint_pipeline(
    model_id: str = _DEFAULT_INPAINT_MODEL,
    device: str = "cuda",
):
    """Load the SDXL inpainting pipeline.

    The pipeline is loaded in fp16 to reduce VRAM usage. CPU offload is
    enabled automatically when the device is "cuda".

    Args:
        model_id: Hugging Face model ID or local path.
        device: Target device.

    Returns:
        StableDiffusionXLInpaintPipeline ready for inference.
    """
    from diffusers import StableDiffusionXLInpaintPipeline

    logger.info(f"Loading SDXL inpainting pipeline from '{model_id}' …")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    logger.info("SDXL inpainting pipeline ready.")
    return pipe


def run_inpainting(
    pipe,
    image: Image.Image,
    mask: np.ndarray,
    prompt: str,
    negative_prompt: str = (
        "blurry, distorted, low quality, artifacts, bad anatomy, "
        "extra limbs, cloned face, disfigured, deformed"
    ),
    strength: float = 0.4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    target_size: Optional[tuple] = None,
) -> Image.Image:
    """Run SDXL inpainting.

    Args:
        pipe: Loaded inpainting pipeline.
        image: Original RGB image.
        mask: uint8 mask (0=preserve, 255=inpaint).
        prompt: Text description of the desired edit.
        negative_prompt: Things to avoid in the output.
        strength: Denoising strength (0.2–0.5 recommended).
        guidance_scale: CFG scale.
        num_inference_steps: Number of denoising steps.
        seed: RNG seed for reproducibility.
        target_size: (width, height) to resize inputs to. Defaults to image size.

    Returns:
        Generated PIL image at the same size as the input.
    """
    strength = max(0.2, min(0.75, strength))  # allow up to 0.75 for visible edits

    orig_size = image.size  # (W, H)
    if target_size is None:
        # Round to multiple of 8 for SDXL
        w = (orig_size[0] // 8) * 8
        h = (orig_size[1] // 8) * 8
        target_size = (w, h)

    resized_image = image.resize(target_size, Image.LANCZOS)
    mask_pil = Image.fromarray(mask).convert("L").resize(target_size, Image.NEAREST)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    logger.info(
        f"Running inpainting — strength={strength}, guidance={guidance_scale}, "
        f"steps={num_inference_steps}, seed={seed}"
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=resized_image,
        mask_image=mask_pil,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        width=target_size[0],
        height=target_size[1],
    ).images[0]

    # Restore to original resolution
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)

    return result
