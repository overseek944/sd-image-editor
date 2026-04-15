"""Shared utilities: logging, image I/O, and device helpers."""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s — %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_image(path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(path).convert("RGB")
    return img


def save_image(image: Image.Image, path: str) -> None:
    """Save a PIL image, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def resize_to_multiple(image: Image.Image, multiple: int = 8) -> Image.Image:
    """Resize image so width and height are divisible by `multiple`."""
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
