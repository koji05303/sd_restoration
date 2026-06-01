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
:root {
    --canvas: #11141a;
    --canvas-deep: #11141a;
    --ink: #f6ead9;
    --muted: #b2a38f;
    --panel: rgba(17, 20, 26, 0.96);
    --panel-strong: rgba(17, 20, 26, 0.96);
    --line: rgba(235, 214, 185, 0.09);
    --line-strong: rgba(235, 214, 185, 0.2);
    --accent: #d79a51;
    --accent-deep: #f2bb72;
    --accent-soft: rgba(215, 154, 81, 0.14);
    --shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
    --shadow-soft: 0 10px 28px rgba(0, 0, 0, 0.16);
}

body, .gradio-container {
    font-family: 'Manrope', ui-sans-serif, sans-serif;
    color: var(--ink);
    color-scheme: dark;
    background: var(--canvas) !important;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 auto !important;
    box-sizing: border-box !important;
    padding: 24px 24px 60px !important;
    background: var(--canvas) !important;
}

.gradio-container > .app,
.gradio-container > .app > .wrap,
.gradio-container > .app > .wrap > .contain,
.gradio-container > .app > .wrap > .contain > .column {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
}

.app-shell {
    position: relative;
    width: 100%;
    margin: 0 auto;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 34px;
    background: var(--panel-strong);
    box-shadow: var(--shadow);
    padding: 22px;
}

.app-shell::before,
.app-shell::after {
    display: none;
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
    background: var(--panel-strong);
    box-shadow: var(--shadow-soft);
    backdrop-filter: none;
}

.hero-card {
    padding: 28px 30px;
}

.hero-note {
    padding: 24px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.04);
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
    background: linear-gradient(180deg, #f0b66c, #b56d24);
    box-shadow: 0 0 0 6px rgba(215, 154, 81, 0.14);
}

.hero-title {
    margin: 18px 0 14px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(2.5rem, 5vw, 4.8rem);
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: #fff0dc;
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
    background: var(--panel-strong);
    border: 1px solid rgba(235, 214, 185, 0.08);
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
    color: #ffe7c7;
}

.note-title {
    margin: 0 0 10px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.55rem;
    color: #ffe7c7;
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
    align-items: flex-start;
}

.pane-stack {
    gap: 18px;
}

.controls-grid {
    gap: 18px;
}

.workspace-grid > .column,
.controls-grid > .column {
    min-width: 0;
}

.section-card {
    padding: 18px;
    background: var(--panel-strong);
}

.section-card .gr-form,
.section-card .form,
.section-card .gr-group,
.section-card .gr-box,
.section-card .gr-block,
.section-card .styler {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.section-card .prose,
.section-card .prose *,
.section-card .gr-markdown,
.section-card .gr-markdown * {
    background: transparent !important;
}

.section-heading {
    margin: 0 0 10px !important;
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.42rem !important;
    letter-spacing: -0.03em;
    color: #ffe7c7 !important;
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
    border: 1px solid rgba(235, 214, 185, 0.08);
    background: var(--panel-strong);
}

.lux-image .image-container,
.lux-output .image-container,
.lux-image .empty,
.lux-output .empty {
    aspect-ratio: 4 / 3;
    min-height: 0 !important;
}

.lux-image img,
.lux-image canvas,
.lux-output img,
.lux-output canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.lux-text textarea,
.lux-log textarea,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 18px !important;
    border: 1px solid rgba(235, 214, 185, 0.1) !important;
    background: rgba(255, 255, 255, 0.04) !important;
    color: var(--ink) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: rgba(178, 163, 143, 0.82) !important;
}

.gradio-container label,
.gradio-container .block_label {
    font-weight: 700 !important;
    color: #e6d5c0 !important;
}

.gradio-container label.float,
.gradio-container .float {
    background: var(--panel-strong) !important;
    color: #e6d5c0 !important;
    border: 1px solid rgba(235, 214, 185, 0.08) !important;
    box-shadow: none !important;
}

.gradio-container .wrap.svelte-1ipelgc,
.gradio-container .wrap.svelte-13io5gv {
    border-radius: 18px !important;
}

.gradio-container .wrap,
.gradio-container .gradio-dropdown,
.gradio-container .gradio-dropdown > div,
.gradio-container .gradio-slider,
.gradio-container .gradio-number,
.gradio-container .gradio-textbox {
    background: transparent !important;
}

.gradio-container .icon-btn,
.gradio-container button.secondary,
.gradio-container .lg.secondary {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(235, 214, 185, 0.08) !important;
    color: var(--ink) !important;
}

.gradio-container .options,
.gradio-container [role="listbox"].options,
.gradio-container ul.options {
    background: var(--panel-strong) !important;
    border: 1px solid rgba(235, 214, 185, 0.1) !important;
    border-radius: 18px !important;
    box-shadow: var(--shadow-soft) !important;
}

.gradio-container .options .item,
.gradio-container [role="option"] {
    color: var(--ink) !important;
}

.gradio-container .options .item:hover,
.gradio-container .options .item.selected,
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {
    background: var(--accent-soft) !important;
    color: #ffe7c7 !important;
}

.lux-log textarea {
    min-height: 360px !important;
}

.compare-kicker {
    margin: 16px 2px 12px;
    color: #e6d5c0;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.compare-grid {
    gap: 12px;
}

.compare-image,
.compare-image .image-container {
    border-radius: 20px !important;
    overflow: hidden !important;
}

.compare-image {
    border: 1px solid rgba(235, 214, 185, 0.08);
    background: var(--panel-strong);
}

.compare-image .image-container,
.compare-image .empty {
    aspect-ratio: 4 / 3;
    min-height: 0 !important;
}

.compare-image img,
.compare-image canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.tip-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}

.tip-card {
    padding: 16px;
    border-radius: 20px;
    background: var(--panel-strong);
    border: 1px solid rgba(235, 214, 185, 0.08);
}

.tip-card h4 {
    margin: 0 0 8px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.1rem;
    color: #ffe4bf;
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
    background: linear-gradient(135deg, #a55b16, #e0a35b) !important;
    color: #130e09 !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 18px 42px rgba(224, 163, 91, 0.2);
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
    background: var(--panel-strong);
    border: 1px solid rgba(235, 214, 185, 0.08);
    color: #d8c7b2;
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

.gradio-container footer {
    display: none !important;
}

.gradio-container .gradio-image .upload-container,
.gradio-container .gr-image .upload-container,
.gradio-container .image-container,
.gradio-container .empty,
.gradio-container .gr-panel {
    background: var(--panel-strong) !important;
}

.gradio-container .prose,
.gradio-container .prose * {
    color: inherit;
}

@media (min-width: 1201px) {
    .workspace-grid > .column:last-child > .gr-group.section-card:first-child {
        position: sticky;
        top: 24px;
        z-index: 3;
    }
}

@media (min-width: 1600px) {
    .gradio-container {
        padding: 30px 30px 72px !important;
    }

    .app-shell {
        padding: 28px;
        border-radius: 38px;
    }

    .hero-panel {
        grid-template-columns: minmax(0, 1.72fr) minmax(360px, 0.92fr);
        gap: 24px;
        padding: 14px 10px 28px;
    }

    .workspace-grid,
    .controls-grid,
    .pane-stack {
        gap: 24px;
    }

    .hero-card {
        padding: 34px 36px;
    }

    .hero-note,
    .section-card {
        padding: 24px;
    }

    .hero-title {
        font-size: clamp(3.2rem, 4vw, 5.4rem);
    }

    .hero-copy {
        max-width: 60ch;
        font-size: 1.08rem;
    }

    .metric-grid {
        gap: 14px;
    }

    .metric-card,
    .tip-card {
        padding: 18px;
    }

    .workspace-grid > .column:first-child {
        flex: 1.15 1 0% !important;
    }

    .workspace-grid > .column:last-child {
        flex: 0.9 1 0% !important;
    }

    .workspace-grid > .column:last-child > .gr-group.section-card:first-child {
        top: 30px;
    }
}

@media (max-width: 1200px) {
    .gradio-container {
        padding: 18px 16px 40px !important;
    }

    .app-shell {
        padding: 18px;
    }

    .hero-panel {
        grid-template-columns: 1fr;
        gap: 16px;
        padding: 4px 2px 18px;
    }

    .workspace-grid,
    .controls-grid {
        flex-direction: column !important;
    }

    .workspace-grid > .column,
    .controls-grid > .column {
        width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    .hero-title {
        font-size: clamp(2.35rem, 6vw, 4.1rem);
    }

    .hero-copy,
    .note-copy,
    .note-list,
    .section-kicker {
        max-width: 100%;
    }
}

@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .tip-grid {
        grid-template-columns: 1fr;
    }

    .compare-grid {
        flex-direction: column !important;
    }

    .hero-card,
    .hero-note,
    .section-card {
        border-radius: 24px;
    }

    .hero-card {
        padding: 22px;
    }

    .hero-note,
    .section-card {
        padding: 18px;
    }

    .hero-title {
        font-size: clamp(2.1rem, 8vw, 3.2rem);
        line-height: 0.98;
    }

    .metric-value {
        font-size: 1.2rem;
    }

    .lux-image,
    .lux-output,
    .lux-image .image-container,
    .lux-output .image-container {
        border-radius: 20px !important;
    }

    .lux-log textarea {
        min-height: 260px !important;
    }
}

@media (max-width: 640px) {
    .gradio-container {
        padding: 10px 10px 28px !important;
    }

    .app-shell {
        border-radius: 20px;
        padding: 12px;
    }

    .hero-card,
    .hero-note,
    .section-card {
        border-radius: 18px;
        padding: 14px;
    }

    .hero-title {
        font-size: 1.95rem;
    }

    .hero-copy,
    .section-kicker,
    .note-copy,
    .note-list,
    .micro-copy,
    .status-banner {
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .eyebrow {
        padding: 7px 10px;
        font-size: 0.76rem;
    }

    .metric-card,
    .tip-card {
        padding: 14px;
        border-radius: 16px;
    }

    .lux-text textarea,
    .lux-log textarea,
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
        border-radius: 14px !important;
    }

    #enhance-button {
        min-height: 58px;
        border-radius: 16px !important;
        font-size: 0.95rem !important;
    }

    .gradio-container .image-container,
    .gradio-container .empty {
        min-height: 220px;
    }

    .lux-image .image-container,
    .lux-output .image-container,
    .lux-image .empty,
    .lux-output .empty,
    .compare-image .image-container,
    .compare-image .empty {
        aspect-ratio: 1 / 1;
    }

    .lux-log textarea {
        min-height: 150px !important;
        max-height: 150px !important;
    }
}
"""


APP_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""


def build_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="amber",
        secondary_hue="zinc",
        neutral_hue="slate",
    ).set(
        body_background_fill="#11141a",
        body_text_color="#f6ead9",
        block_background_fill="rgba(17, 20, 26, 0.96)",
        block_border_color="rgba(235, 214, 185, 0.08)",
        panel_background_fill="rgba(17, 20, 26, 0.96)",
        panel_border_color="rgba(235, 214, 185, 0.08)",
        input_background_fill="rgba(255, 255, 255, 0.04)",
        input_border_color="rgba(235, 214, 185, 0.1)",
        input_placeholder_color="#9f917d",
        body_text_color_subdued="#b2a38f",
        button_primary_background_fill="#d79a51",
        button_primary_background_fill_hover="#f2bb72",
        button_primary_text_color="#130e09",
        button_secondary_background_fill="rgba(255, 255, 255, 0.05)",
        button_secondary_border_color="rgba(235, 214, 185, 0.08)",
        checkbox_background_color="rgba(255, 255, 255, 0.04)",
        checkbox_border_color="rgba(235, 214, 185, 0.12)",
        slider_color="#d79a51",
        color_accent="#d79a51",
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
        return None, None, None, "Error: Please upload an input image."

    if tile_overlap >= tile_size:
        return None, None, None, "Error: tile_overlap must be smaller than tile_size."

    if tile_size % 8 != 0:
        return None, None, None, "Error: tile_size must be a multiple of 8."

    seed_value = int(seed) if seed is not None else None
    source_image = image.convert("RGB").copy()

    work_dir = Path(tempfile.mkdtemp(prefix="sd_restoration_"))
    input_path = work_dir / "input.png"
    output_path = work_dir / "output.png"
    logs = io.StringIO()

    try:
        progress(0.05, desc="Preparing input image")
        source_image.save(input_path)

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
        return output_image, source_image, output_image.copy(), output_logs
    except SafetyCheckerTriggeredError as exc:
        output_logs = logs.getvalue().strip()
        message = (
            f"Generation blocked by the model safety checker.\n\n{exc}\n\n"
            "Try a different prompt or a different checkpoint, then run again."
        )
        if output_logs:
            message = f"{message}\n\nLogs:\n{output_logs}"
        return None, source_image, None, message
    except Exception as exc:
        output_logs = logs.getvalue().strip()
        if output_logs:
            return None, source_image, None, f"Error: {exc}\n\nLogs:\n{output_logs}"
        return None, source_image, None, f"Error: {exc}"
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

                    with gr.Row(elem_classes="controls-grid"):
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
                        gr.HTML("<div class='compare-kicker'>Before / After</div>")
                        with gr.Row(elem_classes="compare-grid"):
                            compare_before = gr.Image(
                                type="pil",
                                label="Before",
                                interactive=False,
                                elem_classes="compare-image",
                            )
                            compare_after = gr.Image(
                                type="pil",
                                label="After",
                                interactive=False,
                                elem_classes="compare-image",
                            )
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
            outputs=[output_image, compare_before, compare_after, logs],
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
        head=APP_HEAD,
    )


if __name__ == "__main__":
    main()
