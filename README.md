# sd-image-editor

A Python pipeline for controlled image editing using Stable Diffusion XL.  
Edit only a specific region of an image based on a text prompt while preserving identity, structure, and background.

---

## Architecture

```
Input image + Text prompt
        │
        ▼
┌─────────────────────┐
│   SAM Segmentation  │  → raw binary mask
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Mask Processing   │  → dilation + Gaussian blur → soft mask
└─────────────────────┘
        │              ┌──────────────────────────┐
        ├─────────────▶│ Canny Edge Extraction    │ (if ControlNet enabled)
        │              └──────────────────────────┘
        ▼                         │
┌─────────────────────┐           │
│  SDXL Inpainting    │◀──────────┘
│  (+ ControlNet opt.)│
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Image Compositing  │  original outside mask + generated inside mask
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Verification      │  InsightFace face similarity + CLIP/SSIM background
└─────────────────────┘
        │
    pass? ──No──▶ retry (max 3)
        │
       Yes
        ▼
   Saved result + JSON log
```

---

## Setup

> **Apple Silicon (M1/M2/M3) users:** PyTorch requires a native arm64 conda
> environment and Python ≤ 3.12. Follow the steps below exactly — do **not**
> use the conda `base` environment or Python 3.13.

### 1. Create a native arm64 conda environment

```bash
# Create a fresh arm64 environment with Python 3.11
CONDA_SUBDIR=osx-arm64 conda create -n sd-editor python=3.11 -y
conda activate sd-editor
conda config --env --set subdir osx-arm64   # pin this env to arm64 permanently
```

### 2. Install PyTorch (via conda)

```bash
conda install pytorch torchvision -c pytorch -y
```

Verify MPS (Apple GPU) is available:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# Expected: 2.x.x True
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Install SAM from source

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 6. Download the SAM checkpoint

```bash
mkdir -p checkpoints
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Available checkpoints:

| Model | Size | Flag |
|-------|------|------|
| `sam_vit_h_4b8939.pth` | 2.4 GB | `--sam-model-type vit_h` (default) |
| `sam_vit_l_0b3195.pth` | 1.2 GB | `--sam-model-type vit_l` |
| `sam_vit_b_01ec64.pth` | 375 MB | `--sam-model-type vit_b` |

> SDXL and ControlNet weights are downloaded automatically from Hugging Face on first run.

---

## Usage

### Python API (single function call)

```python
from main import run_pipeline

result = run_pipeline(
    image_path="input/portrait.jpg",
    prompt="make hair curly",
    point_coords=[(320, 80)],   # click on the region to edit
    output_path="output/result.png",
)

print(result["accepted"])  # True / False
print(result["scores"])    # {"face_similarity": 0.92, "background_ssim": 0.87, ...}
```

#### With bounding box selection

```python
result = run_pipeline(
    image_path="input/portrait.jpg",
    prompt="give a beard",
    box=[200, 300, 450, 500],   # [x0, y0, x1, y1]
    output_path="output/beard.png",
)
```

#### Automatic mask (no prompt needed)

```python
result = run_pipeline(
    image_path="input/object.jpg",
    prompt="make it red",
    auto_select=True,
    auto_mask_index=0,  # 0 = largest detected region
)
```

#### Without ControlNet (faster, less structure-preserving)

```python
result = run_pipeline(
    image_path="input/portrait.jpg",
    prompt="change background to forest",
    point_coords=[(512, 512)],
    use_controlnet=False,
    strength=0.5,
)
```

### CLI

```bash
# Point-based selection
python main.py \
    --image input/portrait.jpg \
    --prompt "make hair curly" \
    --point 320 80 \
    --output output/result.png

# Bounding box
python main.py \
    --image input/portrait.jpg \
    --prompt "give a beard" \
    --box 200 300 450 500

# Automatic mask, no ControlNet, faster
python main.py \
    --image input/object.jpg \
    --prompt "make it red" \
    --auto-select \
    --no-controlnet \
    --steps 20

# All flags
python main.py --help
```

---

## Module Reference

| Module | Key functions |
|--------|--------------|
| `pipeline/utils.py` | `load_image`, `save_image`, `get_device` |
| `pipeline/segmentation.py` | `load_sam_model`, `generate_mask`, `process_mask` |
| `pipeline/diffusion.py` | `load_inpaint_pipeline`, `run_inpainting` |
| `pipeline/controlnet.py` | `load_controlnet_pipeline`, `apply_controlnet`, `extract_canny_edges` |
| `pipeline/compositing.py` | `blend_images` |
| `pipeline/verification.py` | `load_face_analyzer`, `load_clip_model`, `evaluate_similarity` |
| `main.py` | `run_pipeline` (orchestrator), `retry_logic` |

---

## Configuration

Edit `config.yaml` for persistent defaults, or override any parameter in `run_pipeline()`.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strength` | `0.4` | Denoising strength (0.2–0.5 recommended) |
| `guidance_scale` | `7.5` | CFG scale |
| `face_similarity_threshold` | `0.85` | Reject if face cosine sim drops below this |
| `background_ssim_threshold` | `0.80` | Reject if background SSIM drops below this |
| `max_attempts` | `3` | Max retries before accepting best result |
| `use_controlnet` | `True` | Enable Canny ControlNet for structure preservation |
| `controlnet_conditioning_scale` | `0.8` | ControlNet influence weight |

---

## Output

Every run produces:
- `output/result.png` — the composited, verified result image
- `output/run_log.json` — scores and decisions for each attempt:

```json
{
  "accepted": true,
  "elapsed_seconds": 42.3,
  "scores": {
    "face_similarity": 0.921,
    "background_ssim": 0.874,
    "background_clip_similarity": 0.993
  },
  "attempt_logs": [
    {"attempt": 1, "seed": 137, "accepted": true, "scores": {...}}
  ]
}
```

---

## Hardware Requirements

| Config | VRAM |
|--------|------|
| SDXL inpainting (fp16) | ~10 GB |
| + ControlNet (fp16) | ~14 GB |
| CPU offload enabled | ~6 GB peak |

SAM `vit_b` reduces segmentation cost significantly on smaller GPUs.
