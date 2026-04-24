"""Gradio web UI for sd-image-editor.

Usage:
    python app.py
    # Open http://localhost:7860
"""

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from main import run_pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("output")
RESULT_PATH = str(OUTPUT_DIR / "result.png")
LOG_PATH    = str(OUTPUT_DIR / "run_log.json")
INPUT_PATH  = str(OUTPUT_DIR / "ui_input.png")

OUTPUT_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_background_and_mask(editor_value):
    """Return (background PIL image, constraint mask ndarray) from ImageEditor value."""
    if editor_value is None:
        return None, None
    background = editor_value.get("background")
    layers = editor_value.get("layers", [])
    if not layers:
        return background, None
    layer_rgba = np.array(layers[0].convert("RGBA"))
    alpha = layer_rgba[:, :, 3]
    constraint = (alpha > 10).astype(np.uint8) * 255
    if constraint.max() == 0:
        return background, None
    return background, constraint


def format_scores(scores: dict, accepted: bool, elapsed: float) -> str:
    status = "ACCEPTED" if accepted else "BEST RESULT (thresholds not fully met)"
    lines = [f"### {status}", f"**Time:** {elapsed:.1f}s"]

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


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_edit(editor_value, prompt, num_steps, strength, guidance_scale,
             controlnet_scale, max_attempts):
    """Generator yielding (result_image, scores_md, generate_btn) tuples."""

    errors = []
    background, constraint_mask = extract_background_and_mask(editor_value)

    if background is None:
        errors.append("Upload an image first.")
    if constraint_mask is None:
        errors.append("Paint the region you want to edit before generating.")
    if not prompt or not prompt.strip():
        errors.append("Enter an edit prompt.")

    if errors:
        msg = "**Please fix the following:**\n" + "\n".join(f"- {e}" for e in errors)
        yield None, msg, gr.Button(value="Generate", interactive=True)
        return

    background.save(INPUT_PATH)

    yield (
        None,
        "Running pipeline…",
        gr.Button(value="Running…", interactive=False),
    )

    try:
        result = run_pipeline(
            image_path=INPUT_PATH,
            prompt=prompt.strip(),
            constraint_mask=constraint_mask,
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

with gr.Blocks(title="SD Image Editor") as demo:

    gr.Markdown(
        "# SD Image Editor\n"
        "**1.** Upload a portrait — **2.** Paint over the region you want to change "
        "(use the brush tool) — **3.** Describe the edit — **4.** Hit **Generate**."
    )

    with gr.Row(equal_height=False):

        # ── Left column: image editor ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=460):
            gr.Markdown("### Upload & paint the region to edit")
            editor = gr.ImageEditor(
                type="pil",
                brush=gr.Brush(colors=["#ff3300"], default_size=25, color_mode="fixed"),
                label="Upload image, then paint the region",
                height=460,
                show_label=False,
            )
            gr.Markdown(
                "_Select the **brush** tool (pencil icon), then paint over the area "
                "you want to change. You can adjust brush size with the slider._"
            )

        # ── Right column: controls + output ──────────────────────────────────
        with gr.Column(scale=1, min_width=420):
            gr.Markdown("### Describe the edit")
            prompt_box = gr.Textbox(
                label="Edit prompt",
                placeholder='"make hair curly"  /  "change skin tone to darker"  /  "add beard"',
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

    generate_btn.click(
        fn=run_edit,
        inputs=[
            editor,
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
