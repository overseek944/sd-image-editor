"""ControlNet (canny) integration for structure-preserving inpainting."""

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .utils import setup_logger

logger = setup_logger(__name__)

_DEFAULT_CONTROLNET = "diffusers/controlnet-canny-sdxl-1.0"
_DEFAULT_BASE = "stabilityai/stable-diffusion-xl-base-1.0"


def extract_canny_edges(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Apply Canny edge detection and return a 3-channel PIL image.

    Args:
        image: Input RGB PIL image.
        low_threshold: Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.

    Returns:
        RGB PIL image with edges drawn in white on black.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    logger.info(
        f"Canny edges extracted (low={low_threshold}, high={high_threshold})"
    )
    return Image.fromarray(edges_rgb)


def load_controlnet_pipeline(
    controlnet_model_id: str = _DEFAULT_CONTROLNET,
    base_model_id: str = _DEFAULT_BASE,
    device: str = "cuda",
):
    """Load a ControlNet + SDXL inpainting pipeline.

    Args:
        controlnet_model_id: Canny ControlNet checkpoint on HF.
        base_model_id: SDXL base model checkpoint on HF.
        device: Target device.

    Returns:
        StableDiffusionXLControlNetInpaintPipeline.
    """
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    logger.info(f"Loading ControlNet from '{controlnet_model_id}' …")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id, torch_dtype=dtype
    )

    logger.info(f"Loading ControlNet+SDXL pipeline from '{base_model_id}' …")
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    logger.info("ControlNet pipeline ready.")
    return pipe


def apply_controlnet(
    pipe,
    image: Image.Image,
    mask: np.ndarray,
    canny_image: Image.Image,
    prompt: str,
    negative_prompt: str = (
        "blurry, distorted, low quality, artifacts, bad anatomy, "
        "extra limbs, cloned face, disfigured, deformed"
    ),
    strength: float = 0.4,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 0.8,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    target_size: Optional[tuple] = None,
) -> Image.Image:
    """Run structure-guided inpainting with ControlNet (canny).

    Args:
        pipe: Loaded ControlNet inpainting pipeline.
        image: Original RGB image.
        mask: uint8 mask (0=preserve, 255=inpaint).
        canny_image: Canny edge map as RGB PIL image.
        prompt: Target description.
        negative_prompt: Things to avoid.
        strength: Denoising strength (clamped to 0.2–0.5).
        guidance_scale: CFG scale.
        controlnet_conditioning_scale: How strongly ControlNet guides generation.
        num_inference_steps: Number of denoising steps.
        seed: RNG seed.
        target_size: (W, H) to resize inputs; defaults to image size.

    Returns:
        Generated PIL image at original resolution.
    """
    strength = max(0.2, min(0.5, strength))
    orig_size = image.size

    if target_size is None:
        w = (orig_size[0] // 8) * 8
        h = (orig_size[1] // 8) * 8
        target_size = (w, h)

    resized_image = image.resize(target_size, Image.LANCZOS)
    resized_canny = canny_image.resize(target_size, Image.LANCZOS)
    mask_pil = Image.fromarray(mask).convert("L").resize(target_size, Image.NEAREST)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    logger.info(
        f"Running ControlNet inpainting — strength={strength}, "
        f"guidance={guidance_scale}, cn_scale={controlnet_conditioning_scale}, "
        f"steps={num_inference_steps}, seed={seed}"
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=resized_image,
        mask_image=mask_pil,
        control_image=resized_canny,
        strength=strength,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        width=target_size[0],
        height=target_size[1],
    ).images[0]

    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)

    return result
