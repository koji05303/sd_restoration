from __future__ import annotations

import argparse
import contextlib
import io
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from sd_enhancer.config import (
    DEFAULT_CONTROLNET_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    EnhanceConfig,
    OFFLOAD_MODES,
    SKIN_PROTECT_MODES,
    TILE_SEED_MODES,
)
from sd_enhancer.presets import PRESETS, get_preset


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
    grid-template-columns: 1fr;
    gap: 0;
    padding: 6px 6px 16px;
}

.hero-card,
.section-card {
    border: 1px solid var(--line);
    border-radius: 28px;
    background: var(--panel-strong);
    box-shadow: var(--shadow-soft);
    backdrop-filter: none;
}

.hero-card {
    padding: 22px 24px;
}

.hero-title {
    margin: 0;
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(2.1rem, 4vw, 3.4rem);
    line-height: 1;
    letter-spacing: -0.04em;
    color: #fff0dc;
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
        grid-template-columns: 1fr;
        gap: 0;
        padding: 10px 10px 22px;
    }

    .workspace-grid,
    .controls-grid,
    .pane-stack {
        gap: 24px;
    }

    .hero-card {
        padding: 34px 36px;
    }

    .section-card {
        padding: 24px;
    }

    .hero-title {
        font-size: clamp(2.5rem, 4vw, 3.8rem);
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

    .section-kicker {
        max-width: 100%;
    }
}

@media (max-width: 900px) {
    .compare-grid {
        flex-direction: column !important;
    }

    .hero-card,
    .section-card {
        border-radius: 24px;
    }

    .hero-card {
        padding: 22px;
    }

    .section-card {
        padding: 18px;
    }

    .hero-title {
        font-size: clamp(2.1rem, 8vw, 3.2rem);
        line-height: 0.98;
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
    .section-card {
        border-radius: 18px;
        padding: 14px;
    }

    .hero-title {
        font-size: 1.95rem;
    }

    .section-kicker {
        font-size: 0.9rem;
        line-height: 1.6;
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


def preset_to_ui_values(preset_name: str):
    preset = get_preset(preset_name)
    skin_tile_size = preset.skin_tile_size if preset.skin_tile_size is not None else preset.tile_size
    return (
        preset.prompt,
        preset.negative_prompt,
        preset.upscale_factor,
        preset.strength,
        preset.conditioning_scale,
        preset.guidance_scale,
        preset.steps,
        preset.tile_size,
        preset.tile_overlap,
        preset.tile_seed_mode,
        preset.tile_batch_size,
        preset.skin_protect,
        preset.skin_protect_mode,
        preset.skin_strength,
        preset.skin_guidance_scale,
        preset.skin_conditioning_scale,
        skin_tile_size,
        preset.skin_texture_guard,
        preset.skin_texture_guard_strength,
        preset.model_id,
        preset.controlnet_id,
        preset.offload_mode,
    )


def run_enhance(
    image: Optional[Image.Image],
    preset_name: str,
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
    tile_seed_mode: str,
    tile_batch_size: int,
    skin_protect: bool,
    skin_protect_mode: str,
    skin_strength: float,
    skin_guidance_scale: float,
    skin_conditioning_scale: float,
    skin_tile_size: int,
    skin_texture_guard: bool,
    skin_texture_guard_strength: float,
    sharpen: bool,
    contrast: bool,
    match_color_input: bool,
    model_id: str,
    controlnet_id: str,
    offload_mode: str,
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

    if int(tile_batch_size) <= 0:
        return None, None, None, "Error: tile_batch_size must be a positive integer."

    if int(skin_tile_size) % 8 != 0:
        return None, None, None, "Error: skin_tile_size must be a multiple of 8."

    seed_value = int(seed) if seed is not None else None
    source_image = image.convert("RGB").copy()

    work_dir = Path(tempfile.mkdtemp(prefix="sd_restoration_"))
    input_path = work_dir / "input.png"
    output_path = work_dir / "output.png"
    logs = io.StringIO()
    log_handler = logging.StreamHandler(logs)
    log_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    enhancer_logger = logging.getLogger("sd_enhancer")
    previous_log_level = enhancer_logger.level
    previous_propagate = enhancer_logger.propagate
    enhancer_logger.addHandler(log_handler)
    enhancer_logger.setLevel(logging.INFO)
    enhancer_logger.propagate = False

    try:
        progress(0.05, desc="Preparing input image")
        source_image.save(input_path)

        from sd_enhancer.pipeline import enhance_image, resolve_device, resolve_dtype

        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype, resolved_device)
        preset = get_preset(preset_name)

        config = EnhanceConfig(
            image_path=input_path,
            output_path=output_path,
            prompt=prompt.strip() if prompt.strip() else preset.prompt,
            negative_prompt=(
                negative_prompt.strip() if negative_prompt.strip() else preset.negative_prompt
            ),
            model_id=model_id.strip() if model_id.strip() else preset.model_id,
            controlnet_id=controlnet_id.strip() if controlnet_id.strip() else preset.controlnet_id,
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
            tile_seed_mode=tile_seed_mode,
            tile_batch_size=int(tile_batch_size),
            preset=preset_name,
            skin_protect=bool(skin_protect),
            skin_protect_mode=skin_protect_mode,
            skin_strength=float(skin_strength),
            skin_guidance_scale=float(skin_guidance_scale),
            skin_conditioning_scale=float(skin_conditioning_scale),
            skin_tile_size=int(skin_tile_size),
            offload_mode=offload_mode,
            sharpen=bool(sharpen),
            contrast=bool(contrast),
            match_color_input=bool(match_color_input),
            skin_texture_guard=bool(skin_texture_guard),
            skin_texture_guard_strength=float(skin_texture_guard_strength),
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
    except Exception as exc:
        output_logs = logs.getvalue().strip()
        if exc.__class__.__name__ == "SafetyCheckerTriggeredError":
            message = (
                f"Generation blocked by the model safety checker.\n\n{exc}\n\n"
                "Try a different prompt or a different checkpoint, then run again."
            )
            if output_logs:
                message = f"{message}\n\nLogs:\n{output_logs}"
            return None, source_image, None, message
        if output_logs:
            return None, source_image, None, f"Error: {exc}\n\nLogs:\n{output_logs}"
        return None, source_image, None, f"Error: {exc}"
    finally:
        enhancer_logger.removeHandler(log_handler)
        enhancer_logger.setLevel(previous_log_level)
        enhancer_logger.propagate = previous_propagate
        log_handler.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Lustre Restore Studio") as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                """
                <section class="hero-panel">
                  <div class="hero-card">
                    <h1 class="hero-title">Lustre Restore Studio</h1>
                  </div>
                </section>
                """
            )

            with gr.Row(elem_classes="workspace-grid"):
                with gr.Column(scale=11, elem_classes="pane-stack"):
                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Source Frame", elem_classes="section-heading")
                        preset_name = gr.Dropdown(
                            choices=sorted(PRESETS),
                            value="photo",
                            label="Preset",
                        )
                        input_image = gr.Image(
                            type="pil",
                            format="png",
                            label="Input Image",
                            elem_classes="lux-image",
                        )
                        run_button = gr.Button(
                            "Run Enhancement",
                            variant="primary",
                            elem_id="enhance-button",
                        )

                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Creative Direction", elem_classes="section-heading")
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
                                    value=0.2,
                                    step=0.01,
                                    label="Strength",
                                )
                                conditioning_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=0.8,
                                    step=0.05,
                                    label="ControlNet Conditioning Scale",
                                )
                                guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=15.0,
                                    value=4.2,
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

                            with gr.Group(elem_classes="section-card"):
                                gr.Markdown("### Skin & Finish", elem_classes="section-heading")
                                skin_protect = gr.Checkbox(
                                    value=True,
                                    label="Skin Protect",
                                )
                                skin_protect_mode = gr.Dropdown(
                                    choices=list(SKIN_PROTECT_MODES),
                                    value="tone",
                                    label="Skin Protect Mode",
                                )
                                skin_strength = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.16,
                                    step=0.01,
                                    label="Skin Strength",
                                )
                                skin_guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=8.0,
                                    value=3.8,
                                    step=0.1,
                                    label="Skin Guidance Scale",
                                )
                                skin_conditioning_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.5,
                                    value=0.65,
                                    step=0.05,
                                    label="Skin ControlNet Scale",
                                )
                                skin_tile_size = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=640,
                                    step=64,
                                    label="Skin Tile Size",
                                )
                                skin_texture_guard = gr.Checkbox(
                                    value=True,
                                    label="Skin Texture Guard",
                                )
                                skin_texture_guard_strength = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.72,
                                    step=0.01,
                                    label="Skin Texture Guard Strength",
                                )
                                sharpen = gr.Checkbox(
                                    value=False,
                                    label="Sharpen",
                                )
                                contrast = gr.Checkbox(
                                    value=False,
                                    label="Contrast",
                                )
                                match_color_input = gr.Checkbox(
                                    value=False,
                                    label="Match Color To Input",
                                )

                        with gr.Column():
                            with gr.Group(elem_classes="section-card"):
                                gr.Markdown("### Tiling & Runtime", elem_classes="section-heading")
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
                                    value=128,
                                    step=8,
                                    label="Tile Overlap",
                                )
                                tile_seed_mode = gr.Dropdown(
                                    choices=list(TILE_SEED_MODES),
                                    value="same",
                                    label="Tile Seed Mode",
                                )
                                tile_batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=8,
                                    value=2,
                                    step=1,
                                    label="Tile Batch Size",
                                )
                                model_id = gr.Textbox(label="Model ID", value=DEFAULT_MODEL_ID)
                                controlnet_id = gr.Textbox(label="ControlNet ID", value=DEFAULT_CONTROLNET_ID)
                                offload_mode = gr.Dropdown(
                                    choices=list(OFFLOAD_MODES),
                                    value="none",
                                    label="CPU Offload",
                                )
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

                with gr.Column(scale=9, elem_classes="pane-stack"):
                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Output Preview", elem_classes="section-heading")
                        output_image = gr.Image(
                            type="pil",
                            format="png",
                            label="Enhanced Image",
                            elem_classes="lux-output",
                        )
                        gr.HTML("<div class='compare-kicker'>Before / After</div>")
                        with gr.Row(elem_classes="compare-grid"):
                            compare_before = gr.Image(
                                type="pil",
                                format="png",
                                label="Before",
                                interactive=False,
                                elem_classes="compare-image",
                            )
                            compare_after = gr.Image(
                                type="pil",
                                format="png",
                                label="After",
                                interactive=False,
                                elem_classes="compare-image",
                            )

                    with gr.Group(elem_classes="section-card"):
                        gr.Markdown("### Process Log", elem_classes="section-heading")
                        logs = gr.Textbox(
                            label="Runtime Logs",
                            lines=22,
                            interactive=False,
                            elem_classes="lux-log",
                        )
        preset_name.change(
            fn=preset_to_ui_values,
            inputs=[preset_name],
            outputs=[
                prompt,
                negative_prompt,
                upscale_factor,
                strength,
                conditioning_scale,
                guidance_scale,
                steps,
                tile_size,
                tile_overlap,
                tile_seed_mode,
                tile_batch_size,
                skin_protect,
                skin_protect_mode,
                skin_strength,
                skin_guidance_scale,
                skin_conditioning_scale,
                skin_tile_size,
                skin_texture_guard,
                skin_texture_guard_strength,
                model_id,
                controlnet_id,
                offload_mode,
            ],
        )
        run_button.click(
            fn=run_enhance,
            inputs=[
                input_image,
                preset_name,
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
                tile_seed_mode,
                tile_batch_size,
                skin_protect,
                skin_protect_mode,
                skin_strength,
                skin_guidance_scale,
                skin_conditioning_scale,
                skin_tile_size,
                skin_texture_guard,
                skin_texture_guard_strength,
                sharpen,
                contrast,
                match_color_input,
                model_id,
                controlnet_id,
                offload_mode,
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
