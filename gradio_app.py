from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from enhancer import (
    DEFAULT_CONTROLNET_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    EnhanceConfig,
    SafetyCheckerTriggeredError,
    enhance_image,
    resolve_device,
    resolve_dtype,
)


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --canvas: #f5efe6;
    --canvas-deep: #eadfce;
    --ink: #221c16;
    --muted: #6a6054;
    --panel: rgba(255, 250, 243, 0.82);
    --panel-strong: rgba(255, 248, 238, 0.94);
    --line: rgba(76, 58, 37, 0.12);
    --line-strong: rgba(104, 74, 37, 0.22);
    --accent: #b06a28;
    --accent-deep: #8a511d;
    --accent-soft: #edd7bd;
    --shadow: 0 24px 80px rgba(49, 34, 18, 0.12);
    --shadow-soft: 0 14px 34px rgba(49, 34, 18, 0.08);
}

body, .gradio-container {
    font-family: 'Manrope', ui-sans-serif, sans-serif;
    color: var(--ink);
    background:
        radial-gradient(circle at top left, rgba(239, 217, 190, 0.75), transparent 30%),
        radial-gradient(circle at top right, rgba(236, 205, 162, 0.45), transparent 26%),
        linear-gradient(180deg, #fbf7f1 0%, var(--canvas) 52%, #efe5d7 100%);
}

.gradio-container {
    max-width: 1480px !important;
    padding: 24px 20px 60px !important;
}

.app-shell {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 34px;
    background: linear-gradient(180deg, rgba(255, 253, 249, 0.82), rgba(248, 241, 232, 0.88));
    box-shadow: var(--shadow);
    padding: 22px;
}

.app-shell::before,
.app-shell::after {
    content: "";
    position: absolute;
    border-radius: 999px;
    pointer-events: none;
    filter: blur(24px);
}

.app-shell::before {
    width: 320px;
    height: 320px;
    top: -120px;
    right: -80px;
    background: rgba(217, 162, 97, 0.18);
}

.app-shell::after {
    width: 240px;
    height: 240px;
    bottom: -100px;
    left: -90px;
    background: rgba(185, 127, 70, 0.12);
}

.hero-panel {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.95fr);
    gap: 18px;
    padding: 10px 6px 20px;
}

.hero-card,
.hero-note,
.section-card {
    border: 1px solid var(--line);
    border-radius: 28px;
    background: var(--panel);
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(18px);
}

.hero-card {
    padding: 28px 30px;
    background: linear-gradient(145deg, rgba(255, 249, 241, 0.96), rgba(246, 234, 217, 0.82));
}

.hero-note {
    padding: 24px;
    background: linear-gradient(160deg, rgba(255, 248, 238, 0.92), rgba(239, 226, 208, 0.75));
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.55);
    color: var(--accent-deep);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.eyebrow::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(180deg, #d9a05d, #9f6125);
    box-shadow: 0 0 0 6px rgba(176, 106, 40, 0.12);
}

.hero-title {
    margin: 18px 0 14px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(2.5rem, 5vw, 4.8rem);
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: #24160d;
}

.hero-copy {
    max-width: 720px;
    margin: 0;
    color: var(--muted);
    font-size: 1.05rem;
    line-height: 1.75;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 22px;
}

.metric-card {
    padding: 14px 16px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.58);
    border: 1px solid rgba(90, 67, 42, 0.1);
}

.metric-label {
    color: var(--muted);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 8px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.38rem;
    line-height: 1.05;
    color: #2a1a10;
}

.note-title {
    margin: 0 0 10px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.55rem;
    color: #26160c;
}

.note-copy,
.note-list {
    color: var(--muted);
    font-size: 0.96rem;
    line-height: 1.7;
}

.note-list {
    margin: 14px 0 0;
    padding-left: 18px;
}

.workspace-grid {
    position: relative;
    z-index: 1;
    gap: 18px;
}

.pane-stack {
    gap: 18px;
}

.section-card {
    padding: 18px;
    background: var(--panel-strong);
}

.section-card .gr-form,
.section-card .gr-group,
.section-card .gr-box,
.section-card .gr-block {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.section-heading {
    margin: 0 0 10px !important;
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.42rem !important;
    letter-spacing: -0.03em;
    color: #26170d !important;
}

.section-kicker {
    margin: 0 0 14px;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.7;
}

.lux-image,
.lux-image .image-container,
.lux-output,
.lux-output .image-container {
    border-radius: 24px !important;
    overflow: hidden !important;
}

.lux-image,
.lux-output {
    border: 1px solid rgba(88, 64, 38, 0.12);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(245, 238, 228, 0.9));
}

.lux-text textarea,
.lux-log textarea,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 18px !important;
    border: 1px solid rgba(95, 72, 46, 0.12) !important;
    background: rgba(255, 252, 247, 0.92) !important;
    color: var(--ink) !important;
}

.gradio-container label,
.gradio-container .block_label {
    font-weight: 700 !important;
    color: #433528 !important;
}

.gradio-container .wrap.svelte-1ipelgc,
.gradio-container .wrap.svelte-13io5gv {
    border-radius: 18px !important;
}

.tip-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}

.tip-card {
    padding: 16px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(255, 251, 246, 0.95), rgba(243, 233, 220, 0.82));
    border: 1px solid rgba(88, 64, 38, 0.1);
}

.tip-card h4 {
    margin: 0 0 8px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.1rem;
    color: #2a1a10;
}

.tip-card p {
    margin: 0;
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.65;
}

#enhance-button {
    min-height: 64px;
    border: 0 !important;
    border-radius: 20px !important;
    background: linear-gradient(135deg, #a85f1d, #cf8d45) !important;
    color: #fff8f0 !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 18px 36px rgba(168, 95, 29, 0.22);
}

#enhance-button:hover {
    filter: brightness(1.03);
    transform: translateY(-1px);
}

.micro-copy {
    margin: 10px 2px 0;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.7;
}

.status-banner {
    padding: 14px 16px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(252, 245, 232, 0.96), rgba(242, 229, 212, 0.94));
    border: 1px solid rgba(106, 77, 42, 0.12);
    color: #453629;
    font-size: 0.92rem;
    line-height: 1.65;
}

.footer-note {
    margin-top: 16px;
    padding: 14px 18px 0;
    color: var(--muted);
    font-size: 0.83rem;
    letter-spacing: 0.02em;
}

@media (max-width: 1080px) {
    .hero-panel {
        grid-template-columns: 1fr;
    }

    .metric-grid,
    .tip-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 12px 12px 36px !important;
    }

    .app-shell {
        border-radius: 24px;
        padding: 14px;
    }

    .hero-card,
    .hero-note,
    .section-card {
        border-radius: 22px;
    }

    .hero-title {
        font-size: 2.35rem;
    }
}
"""


def build_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="amber",
        secondary_hue="orange",
        neutral_hue="stone",
    ).set(
        body_background_fill="#f5efe6",
        body_text_color="#221c16",
        block_background_fill="rgba(255, 250, 243, 0.82)",
        block_border_color="rgba(76, 58, 37, 0.12)",
        input_background_fill="#fffaf4",
        input_border_color="rgba(95, 72, 46, 0.12)",
        button_primary_background_fill="#b06a28",
        button_primary_background_fill_hover="#8a511d",
        button_primary_text_color="#fff8f0",
        checkbox_background_color="#fffaf4",
        slider_color="#b06a28",
    )


def run_enhance(
    image: Optional[Image.Image],
    prompt: str,
    negative_prompt: str,
    upscale_factor: float,
    strength: float,
    conditioning_scale: float,
    guidance_scale: float,
    steps: int,
    seed: Optional[float],
    tile_size: int,
    tile_overlap: int,
    model_id: str,
    controlnet_id: str,
    device: str,
    dtype: str,
    use_xformers: bool,
    progress=gr.Progress(track_tqdm=False),
):
    if image is None:
        return None, "Error: Please upload an input image."

    if tile_overlap >= tile_size:
        return None, "Error: tile_overlap must be smaller than tile_size."

    if tile_size % 8 != 0:
        return None, "Error: tile_size must be a multiple of 8."

    seed_value = int(seed) if seed is not None else None

    work_dir = Path(tempfile.mkdtemp(prefix="sd_restoration_"))
    input_path = work_dir / "input.png"
    output_path = work_dir / "output.png"
    logs = io.StringIO()

    try:
        progress(0.05, desc="Preparing input image")
        image.convert("RGB").save(input_path)

        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype, resolved_device)

        config = EnhanceConfig(
            image_path=input_path,
            output_path=output_path,
            prompt=prompt.strip() if prompt.strip() else DEFAULT_PROMPT,
            negative_prompt=(
                negative_prompt.strip() if negative_prompt.strip() else DEFAULT_NEGATIVE_PROMPT
            ),
            model_id=model_id.strip() if model_id.strip() else DEFAULT_MODEL_ID,
            controlnet_id=controlnet_id.strip() if controlnet_id.strip() else DEFAULT_CONTROLNET_ID,
            upscale_factor=float(upscale_factor),
            strength=float(strength),
            conditioning_scale=float(conditioning_scale),
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=seed_value,
            device=resolved_device,
            dtype=resolved_dtype,
            use_xformers=bool(use_xformers),
            overwrite=True,
            tile_size=int(tile_size),
            tile_overlap=int(tile_overlap),
        )

        progress(0.15, desc="Running enhancement")
        with contextlib.redirect_stdout(logs):
            enhance_image(config)

        progress(0.95, desc="Loading output")
        with Image.open(output_path) as enhanced:
            output_image = enhanced.convert("RGB").copy()

        progress(1.0, desc="Done")
        output_logs = logs.getvalue().strip() or "Enhancement finished."
        return output_image, output_logs
    except SafetyCheckerTriggeredError as exc:
        output_logs = logs.getvalue().strip()
        message = (
            f"Generation blocked by the model safety checker.\n\n{exc}\n\n"
            "Try a different prompt or a different checkpoint, then run again."
        )
        if output_logs:
            message = f"{message}\n\nLogs:\n{output_logs}"
        return None, message
    except Exception as exc:
        output_logs = logs.getvalue().strip()
        if output_logs:
            return None, f"Error: {exc}\n\nLogs:\n{output_logs}"
        return None, f"Error: {exc}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Lustre Restore Studio") as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                """
                <section class="hero-panel">
                  <div class="hero-card">
                    <span class="eyebrow">Premium Restoration Suite</span>
                    <h1 class="hero-title">Lustre Restore Studio</h1>
                    <p class="hero-copy">
                      Tile-aware image enhancement with a more refined presentation layer. Upload a frame,
                      dial in the aesthetic, and render a cleaner upscale pass without the usual tool-room feel.
                    </p>
                    <div class="metric-grid">
                      <div class="metric-card">
                        <div class="metric-label">Render Mode</div>
                        <div class="metric-value">Tiled Precision</div>
                      </div>
                      <div class="metric-card">
                        <div class="metric-label">Output Style</div>
                        <div class="metric-value">Photo Finish</div>
                      </div>
                      <div class="metric-card">
                        <div class="metric-label">Runtime Guardrails</div>
                        <div class="metric-value">VRAM Aware</div>
                      </div>
                    </div>
                  </div>
                  <aside class="hero-note">
                    <h3 class="note-title">Studio Notes</h3>
                    <p class="note-copy">
                      The interface is tuned to feel more like a boutique rendering product: warmer palette,
                      stronger typography, clearer hierarchy, and calmer parameter grouping.
                    </p>
                    <ul class="note-list">
                      <li>Use smaller tile sizes when memory is tight.</li>
                      <li>Keep overlap around 48-80 for smoother seams.</li>
                      <li>Switch to fp32 only when a specific image needs it.</li>
                    </ul>
                  </aside>
                </section>
                """
            )

            with gr.Row(elem_classes="workspace-grid"):
                with gr.Column(scale=11, elem_classes="pane-stack"):
                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Source Frame", elem_classes="section-heading")
                        gr.Markdown(
                            "Upload the image you want to restore. The engine will resize, tile, and blend it into a single polished output.",
                            elem_classes="section-kicker",
                        )
                        input_image = gr.Image(type="pil", label="Input Image", elem_classes="lux-image")
                        run_button = gr.Button(
                            "Render Premium Pass",
                            variant="primary",
                            elem_id="enhance-button",
                        )
                        gr.HTML(
                            "<p class='micro-copy'>Best results usually come from clean source frames, moderate guidance, and a tile size that matches your available VRAM.</p>"
                        )

                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Creative Direction", elem_classes="section-heading")
                        gr.Markdown(
                            "Shape the tone of the restoration pass. Positive prompt defines the finish; negative prompt trims artifacts and unwanted traits.",
                            elem_classes="section-kicker",
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=5,
                            value=DEFAULT_PROMPT,
                            elem_classes="lux-text",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            lines=4,
                            value=DEFAULT_NEGATIVE_PROMPT,
                            elem_classes="lux-text",
                        )

                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes="section-card"):
                                gr.Markdown("### Render Controls", elem_classes="section-heading")
                                gr.Markdown(
                                    "Tune the visual intensity and denoising behavior.",
                                    elem_classes="section-kicker",
                                )
                                upscale_factor = gr.Slider(
                                    minimum=1.0,
                                    maximum=4.0,
                                    value=2.0,
                                    step=0.1,
                                    label="Upscale Factor",
                                )
                                strength = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.35,
                                    step=0.01,
                                    label="Strength",
                                )
                                conditioning_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.05,
                                    label="ControlNet Conditioning Scale",
                                )
                                guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=15.0,
                                    value=7.5,
                                    step=0.5,
                                    label="Guidance Scale",
                                )
                                steps = gr.Slider(
                                    minimum=5,
                                    maximum=80,
                                    value=25,
                                    step=1,
                                    label="Inference Steps",
                                )
                                seed = gr.Number(label="Seed (optional)", precision=0, value=None)

                        with gr.Column():
                            with gr.Group(elem_classes="section-card"):
                                gr.Markdown("### Tiling & Runtime", elem_classes="section-heading")
                                gr.Markdown(
                                    "Balance memory usage and seam quality for larger renders.",
                                    elem_classes="section-kicker",
                                )
                                tile_size = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Tile Size",
                                )
                                tile_overlap = gr.Slider(
                                    minimum=0,
                                    maximum=256,
                                    value=64,
                                    step=8,
                                    label="Tile Overlap",
                                )
                                model_id = gr.Textbox(label="Model ID", value=DEFAULT_MODEL_ID)
                                controlnet_id = gr.Textbox(label="ControlNet ID", value=DEFAULT_CONTROLNET_ID)
                                device = gr.Dropdown(
                                    choices=["auto", "cuda", "cpu"],
                                    value="auto",
                                    label="Device",
                                )
                                dtype = gr.Dropdown(
                                    choices=["auto", "fp16", "fp32"],
                                    value="auto",
                                    label="DType",
                                )
                                use_xformers = gr.Checkbox(
                                    value=True,
                                    label="Use xFormers if available",
                                )

                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Render Guidance", elem_classes="section-heading")
                        gr.HTML(
                            """
                            <div class="tip-grid">
                              <div class="tip-card">
                                <h4>For Cleaner Skin & Detail</h4>
                                <p>Keep strength conservative and let the prompt describe texture rather than extreme realism.</p>
                              </div>
                              <div class="tip-card">
                                <h4>For Large Frames</h4>
                                <p>Drop tile size first when memory gets tight. 384 with overlap 48 is a good fallback.</p>
                              </div>
                              <div class="tip-card">
                                <h4>For Safer Iteration</h4>
                                <p>Use a fixed seed while tuning prompt and guidance, then remove it once the look is locked.</p>
                              </div>
                            </div>
                            """
                        )

                with gr.Column(scale=9, elem_classes="pane-stack"):
                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Output Preview", elem_classes="section-heading")
                        gr.Markdown(
                            "The final composited frame appears here after tiled processing completes.",
                            elem_classes="section-kicker",
                        )
                        output_image = gr.Image(type="pil", label="Enhanced Image", elem_classes="lux-output")
                        gr.HTML(
                            "<div class='status-banner'>Premium pass output is assembled from overlapping tiles to keep memory predictable while preserving a more seamless finish.</div>"
                        )

                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Process Log", elem_classes="section-heading")
                        gr.Markdown(
                            "Runtime diagnostics, tile fallback notes, and backend messages appear below.",
                            elem_classes="section-kicker",
                        )
                        logs = gr.Textbox(
                            label="Runtime Logs",
                            lines=22,
                            interactive=False,
                            elem_classes="lux-log",
                        )

            gr.HTML(
                "<div class='footer-note'>Lustre Restore Studio keeps the original CLI backend intact while presenting it as a cleaner, more product-like image restoration workspace.</div>"
            )

        run_button.click(
            fn=run_enhance,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                upscale_factor,
                strength,
                conditioning_scale,
                guidance_scale,
                steps,
                seed,
                tile_size,
                tile_overlap,
                model_id,
                controlnet_id,
                device,
                dtype,
                use_xformers,
            ],
            outputs=[output_image, logs],
            api_name="enhance",
        )

    return demo


def parse_launch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio UI for tiled image enhancement.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Enable a public Gradio share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the Gradio URL in a browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_launch_args()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=8)
    theme = build_theme()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        theme=theme,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
