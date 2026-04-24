"""Gradio web UI for sd-image-editor.

Usage:
    pip install gradio
    python app.py
    # Open http://localhost:7860
"""

from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from main import run_pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("output")
RESULT_PATH = str(OUTPUT_DIR / "result.png")
LOG_PATH    = str(OUTPUT_DIR / "run_log.json")
INPUT_PATH  = str(OUTPUT_DIR / "ui_input.png")

OUTPUT_DIR.mkdir(exist_ok=True)

DOT_RADIUS = 8
DOT_COLOR  = (255, 50, 50)


# ── Helpers ───────────────────────────────────────────────────────────────────

def annotate_image(image: Image.Image, x: int, y: int) -> Image.Image:
    """Return a copy of image with a red dot at (x, y)."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    r = DOT_RADIUS
    # White border for visibility on any background
    draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
    draw.ellipse([x - r, y - r, x + r, y + r], fill=DOT_COLOR)
    return out


def format_scores(scores: dict, accepted: bool, elapsed: float) -> str:
    """Return a Markdown quality report."""
    status = "ACCEPTED" if accepted else "BEST RESULT (thresholds not fully met)"
    lines = [
        f"### {status}",
        f"**Time:** {elapsed:.1f}s",
    ]

    face = scores.get("face_similarity")
    if face is not None:
        tag = "OK" if face >= 0.85 else "LOW"
        lines.append(f"**Face similarity:** {face:.3f}  _{tag} (threshold 0.85)_")
    else:
        lines.append("**Face similarity:** N/A _(no face detected)_")

    bg = scores.get("background_ssim", 0.0)
    tag = "OK" if bg >= 0.80 else "LOW"
    lines.append(f"**Background SSIM:** {bg:.3f}  _{tag} (threshold 0.80)_")

    edit = scores.get("edit_region_ssim", 1.0)
    tag = "CHANGED" if edit <= 0.97 else "UNCHANGED"
    lines.append(f"**Edit-region SSIM:** {edit:.3f}  _{tag} (lower = more changed)_")

    clip = scores.get("background_clip_similarity")
    if clip is not None:
        lines.append(f"**Background CLIP:** {clip:.4f}")

    return "\n\n".join(lines)


# ── Pipeline runner (generator so Gradio shows progress) ─────────────────────

def run_edit(
    original_image,
    point_x,
    point_y,
    prompt,
    num_steps,
    strength,
    guidance_scale,
    controlnet_scale,
    max_attempts,
):
    """Generator that yields (result_image, scores_md, generate_btn) tuples."""

    errors = []
    if original_image is None:
        errors.append("Upload an image first.")
    if point_x is None or point_y is None:
        errors.append("Click the lower image to select the region you want to edit.")
    if not prompt or not prompt.strip():
        errors.append("Enter an edit prompt.")

    if errors:
        msg = "**Please fix the following:**\n" + "\n".join(f"- {e}" for e in errors)
        yield None, msg, gr.Button(value="Generate", interactive=True)
        return

    original_image.save(INPUT_PATH)

    yield (
        None,
        "Running pipeline… this typically takes **2–8 minutes** on MPS.",
        gr.Button(value="Running…", interactive=False),
    )

    try:
        result = run_pipeline(
            image_path=INPUT_PATH,
            prompt=prompt.strip(),
            point_coords=[(int(point_x), int(point_y))],
            output_path=RESULT_PATH,
            log_path=LOG_PATH,
            num_inference_steps=int(num_steps),
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_scale),
            max_attempts=int(max_attempts),
        )
    except Exception as exc:
        yield None, f"**Pipeline error:** {exc}", gr.Button(value="Generate", interactive=True)
        return

    result_pil = Image.open(result["output_path"]).convert("RGB") if result.get("output_path") else None
    scores_text = format_scores(result["scores"], result["accepted"], result["elapsed_seconds"])

    yield result_pil, scores_text, gr.Button(value="Generate", interactive=True)


# ── UI layout ─────────────────────────────────────────────────────────────────

def on_image_load(image):
    if image is None:
        return None, None, None, "_No image loaded._"
    return image.copy(), None, None, "_Click the image above to select a region._"


def on_click(original_image, evt: gr.SelectData):
    if original_image is None:
        return None, None, None, "_Upload an image first._"
    x, y = int(evt.index[0]), int(evt.index[1])
    return annotate_image(original_image, x, y), x, y, f"Selected point: ({x}, {y})"


with gr.Blocks(title="SD Image Editor") as demo:

    # Hidden state
    original_image_state = gr.State(None)
    point_x_state        = gr.State(None)
    point_y_state        = gr.State(None)

    gr.Markdown(
        "# SD Image Editor\n"
        "Upload a portrait, **click the region** you want to change, "
        "describe the edit, then hit **Generate**."
    )

    with gr.Row(equal_height=False):

        # ── Left column: image input ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=420):
            gr.Markdown("### 1 · Upload image")
            upload_image = gr.Image(
                type="pil",
                label="Upload",
                height=360,
                show_label=False,
            )
            gr.Markdown("### 2 · Click the region to edit")
            display_image = gr.Image(
                type="pil",
                label="Click to select region",
                interactive=False,
                height=360,
                buttons=[],
            )
            click_info = gr.Markdown("_No point selected yet._")

        # ── Right column: controls + output ───────────────────────────────────
        with gr.Column(scale=1, min_width=420):
            gr.Markdown("### 3 · Describe the edit")
            prompt_box = gr.Textbox(
                label="Edit prompt",
                placeholder='"make hair curly"  /  "add beard"  /  "change hair color to red"',
                lines=2,
            )

            with gr.Accordion("Advanced settings", open=False):
                steps_slider = gr.Slider(
                    minimum=10, maximum=50, value=20, step=1,
                    label="Inference steps  (more = slower + better quality)",
                )
                strength_slider = gr.Slider(
                    minimum=0.2, maximum=0.99, value=0.75, step=0.05,
                    label="Denoising strength  (higher = stronger edit)",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=20.0, value=9.0, step=0.5,
                    label="Guidance scale (CFG)",
                )
                cn_scale_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.4, step=0.05,
                    label="ControlNet scale  (lower = more creative)",
                )
                attempts_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Max retry attempts",
                )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

            gr.Markdown("### Result")
            result_image = gr.Image(
                type="pil",
                label="Result",
                interactive=False,
                height=360,
                show_label=False,
            )
            scores_md = gr.Markdown("_Run the pipeline to see quality scores._")

    # ── Event wiring ──────────────────────────────────────────────────────────

    # When a new image is uploaded (or changed), populate display and reset state
    for event in (upload_image.upload, upload_image.change):
        event(
            fn=on_image_load,
            inputs=[upload_image],
            outputs=[display_image, point_x_state, point_y_state, click_info],
        ).then(
            fn=lambda img: img,
            inputs=[upload_image],
            outputs=[original_image_state],
        )

    # When user clicks on the display image, mark the point
    display_image.select(
        fn=on_click,
        inputs=[original_image_state],
        outputs=[display_image, point_x_state, point_y_state, click_info],
    )

    # Generate button
    generate_btn.click(
        fn=run_edit,
        inputs=[
            original_image_state,
            point_x_state,
            point_y_state,
            prompt_box,
            steps_slider,
            strength_slider,
            guidance_slider,
            cn_scale_slider,
            attempts_slider,
        ],
        outputs=[result_image, scores_md, generate_btn],
        show_progress="full",
    )


# ── Launch ────────────────────────────────────────────────────────────────────

demo.queue(max_size=1, default_concurrency_limit=1)
demo.launch(server_name="0.0.0.0", server_port=7860, max_threads=2, theme=gr.themes.Soft())
