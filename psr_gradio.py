from __future__ import annotations

import argparse
import html
import inspect
import threading
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from pure_psr import DEFAULT_CUDA_DEVICE, PureSREngine


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=JetBrains+Mono:wght@500;600&family=Manrope:wght@500;600;700;800;900&display=swap');

:root {
    --canvas: #f7f3ee;
    --panel: #fffaf7;
    --panel-raised: #ffffff;
    --panel-soft: #f8efe7;
    --ink: #2c2824;
    --muted: #81756d;
    --quiet: #a5968d;
    --line: rgba(78, 55, 42, 0.12);
    --line-strong: rgba(78, 55, 42, 0.2);
    --accent: #bd7256;
    --accent-strong: #9d563e;
    --accent-soft: rgba(189, 114, 86, 0.13);
    --sage: #4f8f82;
    --gold: #b78a43;
    --danger: #c85a50;
    --shadow: 0 24px 70px rgba(93, 67, 53, 0.14);
    --shadow-soft: 0 12px 34px rgba(93, 67, 53, 0.09);
}

html,
html.dark,
body,
body.dark,
.dark,
.gradio-container,
.gradio-container.dark {
    color-scheme: light !important;
    --body-background-fill: #f7f3ee !important;
    --body-text-color: #2c2824 !important;
    --background-fill-primary: #fffaf7 !important;
    --background-fill-secondary: #f8efe7 !important;
    --block-background-fill: rgba(255, 255, 255, 0.76) !important;
    --block-border-color: rgba(78, 55, 42, 0.12) !important;
    --border-color-primary: rgba(78, 55, 42, 0.12) !important;
    --border-color-accent: rgba(189, 114, 86, 0.28) !important;
    --input-background-fill: #f8efe7 !important;
    --input-border-color: rgba(78, 55, 42, 0.12) !important;
    --button-secondary-background-fill: #f8efe7 !important;
    --button-secondary-text-color: #2c2824 !important;
    --link-text-color: #9d563e !important;
}

body,
.gradio-container {
    background: var(--canvas) !important;
    color: var(--ink) !important;
    color-scheme: light !important;
    font-family: Manrope, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 20px !important;
    background:
        radial-gradient(circle at 10% 0%, rgba(189, 114, 86, 0.14), rgba(189, 114, 86, 0) 28%),
        linear-gradient(180deg, rgba(255, 251, 246, 0.9), rgba(247, 243, 238, 1) 44%),
        var(--canvas) !important;
}

.app-shell {
    width: 100%;
    min-height: calc(100vh - 36px);
    display: grid;
    grid-template-rows: auto 1fr;
    gap: 14px;
    padding: 14px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 250, 245, 0.88);
    box-shadow: var(--shadow);
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    min-height: 72px;
    padding: 12px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.78);
    box-shadow: var(--shadow-soft);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}

.brand-mark {
    display: grid;
    width: 44px;
    height: 44px;
    place-items: center;
    flex: 0 0 auto;
    border: 1px solid rgba(189, 114, 86, 0.24);
    border-radius: 8px;
    background: #fff2e9;
    color: var(--accent-strong);
    font-size: 0.74rem;
    font-weight: 900;
}

.brand-title {
    margin: 0;
    color: var(--ink);
    font-family: Fraunces, Georgia, serif;
    font-size: 1.28rem;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.05;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 700;
    white-space: nowrap;
    margin-top: 5px;
}

.top-actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 10px;
    min-width: 0;
    flex: 1 1 360px;
}

.mode-chip {
    padding: 8px 10px;
    border: 1px solid rgba(93, 155, 141, 0.26);
    border-radius: 8px;
    background: rgba(93, 155, 141, 0.13);
    color: #416f66;
    font-size: 0.74rem;
    font-weight: 900;
    letter-spacing: 0;
}

.gpu-pill {
    width: clamp(240px, 38vw, 420px);
    min-width: 0;
    max-width: 100%;
    padding: 10px 12px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel-raised);
    color: var(--ink);
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1.45;
}

.gpu-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 10px;
    min-width: 0;
}

.gpu-name {
    overflow: hidden;
    color: var(--ink);
    text-overflow: ellipsis;
    white-space: nowrap;
}

.gpu-memory {
    color: var(--muted);
    font-size: 0.74rem;
    white-space: nowrap;
}

.status-meter {
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(80, 57, 48, 0.08);
}

.status-meter-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--sage), var(--accent));
}

.hero-panel {
    display: grid !important;
    grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.86fr);
    gap: 14px;
    align-items: stretch;
}

.hero-panel > .gr-html {
    min-width: 0;
}

.hero-card,
.hero-note {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.7);
    box-shadow: var(--shadow-soft);
}

.hero-card {
    padding: 24px 26px;
}

.hero-note {
    padding: 22px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border: 1px solid rgba(189, 114, 86, 0.16);
    border-radius: 8px;
    background: var(--accent-soft);
    color: var(--accent-strong);
    font-size: 0.74rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.eyebrow::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(180deg, #d99972, #9d563e);
    box-shadow: 0 0 0 5px rgba(189, 114, 86, 0.13);
}

.hero-title {
    margin: 16px 0 12px;
    color: var(--ink);
    font-family: Fraunces, Georgia, serif;
    font-size: clamp(2.6rem, 5vw, 4.6rem);
    font-weight: 700;
    line-height: 0.95;
    letter-spacing: 0;
}

.hero-copy,
.note-copy {
    margin: 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.75;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 22px;
}

.metric-card {
    padding: 13px 14px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 250, 246, 0.84);
}

.metric-label {
    color: var(--quiet);
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 7px;
    color: var(--ink);
    font-family: Fraunces, Georgia, serif;
    font-size: 1.25rem;
    line-height: 1.1;
}

.note-title {
    margin: 0 0 10px;
    color: var(--ink);
    font-family: Fraunces, Georgia, serif;
    font-size: 1.38rem;
    line-height: 1.12;
}

.note-list {
    margin: 14px 0 0;
    padding-left: 18px;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.72;
}

.workbench {
    display: grid;
    grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
    gap: 14px;
    min-height: 0;
}

.control-panel,
.viewer-panel,
.status-panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.76);
    box-shadow: var(--shadow-soft);
}

.control-panel {
    padding: 16px;
    align-self: start;
}

.viewer-panel {
    padding: 14px;
}

.status-panel {
    padding: 12px;
}

.panel-heading {
    margin: 0 0 12px !important;
    color: var(--ink) !important;
    font-family: Fraunces, Georgia, serif !important;
    font-size: 1.22rem !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
}

.viewer-grid {
    gap: 14px;
    align-items: stretch;
}

.viewer-grid > .column {
    min-width: 0;
}

.gradio-container .gr-form,
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-block,
.gradio-container .styler {
    background: transparent !important;
    border-color: var(--line) !important;
    box-shadow: none !important;
}

.gradio-container .gr-panel,
.gradio-container .block {
    border-radius: 8px !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: var(--panel-soft) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}

.gradio-container label,
.gradio-container .block_label {
    color: var(--muted) !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
}

.gradio-container button.primary,
#run-button {
    min-height: 48px !important;
    border: 0 !important;
    border-radius: 8px !important;
    background: linear-gradient(135deg, var(--accent-strong), var(--accent)) !important;
    color: #fffaf4 !important;
    font-weight: 900 !important;
    box-shadow: 0 14px 30px rgba(157, 86, 62, 0.22) !important;
}

#run-button:hover {
    background: var(--accent-strong) !important;
}

.gradio-container button.secondary,
#refresh-button {
    border-radius: 8px !important;
    background: var(--panel-soft) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}

.gradio-container .image-container,
.gradio-container .empty {
    background: #fffdf9 !important;
    border-radius: 8px !important;
    border-color: var(--line) !important;
}

.gradio-container textarea {
    min-height: 124px !important;
    font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.55 !important;
}

.gradio-container .gr-file,
.gradio-container .file-preview,
.gradio-container .download {
    border-radius: 8px !important;
}

.gradio-container .wrap,
.gradio-container .gradio-dropdown,
.gradio-container .gradio-radio,
.gradio-container .gradio-slider {
    min-width: 0 !important;
}

.compact-note {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.5;
}

@media (max-width: 920px) {
    .topbar {
        align-items: flex-start;
        flex-direction: column;
    }

    .top-actions,
    .gpu-pill {
        width: 100%;
        justify-content: flex-start;
    }

    .gpu-pill {
        text-align: left;
        min-width: 0;
    }

    .workbench {
        grid-template-columns: 1fr;
    }

    .hero-panel {
        grid-template-columns: 1fr;
    }

    .metric-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 580px) {
    .gradio-container {
        padding: 10px !important;
    }

    .app-shell {
        padding: 10px;
    }

    .brand-subtitle {
        white-space: normal;
    }

    .top-actions {
        flex-direction: column;
        align-items: stretch;
    }

    .hero-card,
    .hero-note {
        padding: 18px;
    }

    .hero-title {
        font-size: clamp(2.1rem, 13vw, 3.1rem);
    }

    .gpu-row {
        grid-template-columns: 1fr;
        gap: 4px;
    }

    .gpu-memory {
        white-space: normal;
    }
}
"""

LIGHT_MODE_HEAD = """
<script>
(() => {
    const forceLight = () => {
        document.documentElement.classList.remove("dark");
        document.documentElement.classList.add("light");
        document.body?.classList.remove("dark");
        document.body?.classList.add("light");
        try {
            localStorage.setItem("theme", "light");
            localStorage.setItem("__theme", "light");
            localStorage.setItem("gradio.theme", "light");
        } catch (_) {}
    };

    forceLight();
    document.addEventListener("DOMContentLoaded", forceLight);
    new MutationObserver(forceLight).observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class"],
    });
})();
</script>
"""


@dataclass
class EngineSlot:
    device: str
    engine: PureSREngine


_ENGINE_SLOT: Optional[EngineSlot] = None
_ENGINE_LOCK = threading.Lock()


def resolve_device(device: str) -> str:
    if device == "auto":
        return DEFAULT_CUDA_DEVICE if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")
    return device


def get_engine(device: str) -> PureSREngine:
    global _ENGINE_SLOT

    resolved_device = resolve_device(device)
    with _ENGINE_LOCK:
        if _ENGINE_SLOT is None or _ENGINE_SLOT.device != resolved_device:
            _ENGINE_SLOT = EngineSlot(
                device=resolved_device,
                engine=PureSREngine(device=resolved_device),
            )
        return _ENGINE_SLOT.engine


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def gpu_status_html() -> str:
    if not torch.cuda.is_available():
        return (
            "<div class='gpu-pill'>"
            "<div class='gpu-row'>"
            "<span class='gpu-name'>CPU mode</span>"
            "<span class='gpu-memory'>VRAM unavailable</span>"
            "</div>"
            "<div class='status-meter'><div class='status-meter-fill' style='width: 0%'></div></div>"
            "</div>"
        )

    device_index = torch.cuda.current_device()
    name = html.escape(torch.cuda.get_device_name(device_index))
    props = torch.cuda.get_device_properties(device_index)
    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3
    total = props.total_memory / 1024**3
    used_ratio = max(0.0, min(100.0, reserved / total * 100.0))

    return (
        "<div class='gpu-pill'>"
        "<div class='gpu-row'>"
        f"<span class='gpu-name'>{name}</span>"
        f"<span class='gpu-memory'>{allocated:.2f} / {reserved:.2f} / {total:.1f} GB</span>"
        "</div>"
        "<div class='status-meter'>"
        f"<div class='status-meter-fill' style='width: {used_ratio:.1f}%'></div>"
        "</div>"
        "</div>"
    )


def image_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_image(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def make_preview_image(label: str, height: int) -> gr.Image:
    kwargs = {
        "label": label,
        "type": "pil",
        "height": height,
    }
    image_params = inspect.signature(gr.Image).parameters
    if "format" in image_params:
        kwargs["format"] = "png"
    if "show_download_button" in image_params:
        kwargs["show_download_button"] = False
    return gr.Image(**kwargs)


def save_download_image(image: Image.Image, output_format: str) -> str:
    normalized_format = output_format.strip().upper()
    if normalized_format not in {"PNG", "JPG"}:
        normalized_format = "PNG"

    suffix = ".jpg" if normalized_format == "JPG" else ".png"
    with tempfile.NamedTemporaryFile(
        prefix="pure_psr_restore_",
        suffix=suffix,
        delete=False,
    ) as file:
        output_path = file.name

    if normalized_format == "JPG":
        image.convert("RGB").save(output_path, format="JPEG", quality=96, subsampling=0)
    else:
        image.save(output_path, format="PNG", optimize=True)

    return output_path


def run_psr(
    image: Optional[Image.Image],
    enhance_detail: bool,
    output_format: str,
    tile_size: int,
    tile_pad: int,
    device: str,
    progress=gr.Progress(track_tqdm=False),
):
    if image is None:
        return None, None, None, None, "Error: Please upload an image.", gpu_status_html()

    if tile_size < 64 or tile_size % 8 != 0:
        return None, None, None, None, "Error: tile size must be at least 64 and divisible by 8.", gpu_status_html()

    if tile_pad < 0 or tile_pad >= tile_size:
        return None, None, None, None, "Error: tile padding must be smaller than tile size.", gpu_status_html()

    selected_format = (output_format or "PNG").strip().upper()
    if selected_format not in {"PNG", "JPG"}:
        selected_format = "PNG"

    source = image.convert("RGB").copy()
    start_time = time.monotonic()
    last_update = 0.0

    try:
        progress(0.02, desc="Loading model")
        engine = get_engine(device)
        engine.tile_size = int(tile_size)
        engine.tile_pad = int(tile_pad)

        input_bgr = image_to_bgr(source)
        height, width = input_bgr.shape[:2]
        output_size = f"{width * engine.scale} x {height * engine.scale}"

        def update_tile_progress(done: int, total: int) -> None:
            nonlocal last_update
            now = time.monotonic()
            if done < total and now - last_update < 0.35:
                return
            last_update = now
            ratio = done / max(total, 1)
            elapsed = now - start_time
            eta = elapsed * (1.0 - ratio) / max(ratio, 1e-6)
            progress(
                0.08 + ratio * 0.86,
                desc=f"Tile {done}/{total} | ETA {format_seconds(eta)}",
            )

        progress(0.08, desc="Preparing tiles")
        output_bgr = engine.enhance(
            input_bgr,
            enhance_detail=bool(enhance_detail),
            progress_callback=update_tile_progress,
        )

        progress(0.96, desc="Finalizing output")
        output_image = bgr_to_image(output_bgr)
        download_path = save_download_image(output_image, selected_format)
        elapsed = format_seconds(time.monotonic() - start_time)
        log = (
            f"Done in {elapsed}\n"
            f"Input: {width} x {height}\n"
            f"Output: {output_size}\n"
            f"Download format: {selected_format}\n"
            f"Tile size: {tile_size}\n"
            f"Tile padding: {tile_pad}\n"
            f"Device: {resolve_device(device)}\n"
            f"Detail enhancement: {'on' if enhance_detail else 'off'}"
        )
        progress(1.0, desc="Done")
        return output_image, source, output_image.copy(), download_path, log, gpu_status_html()
    except Exception as exc:
        return None, source, None, None, f"Error: {exc}", gpu_status_html()


def build_ui() -> gr.Blocks:
    default_device = "auto"
    device_choices = ["auto", "cpu"]
    if torch.cuda.is_available():
        device_choices.insert(1, DEFAULT_CUDA_DEVICE)

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="teal",
        neutral_hue="stone",
    )

    blocks_kwargs = {
        "title": "Lumi Restore Studio",
        "css": APP_CSS,
        "theme": theme,
    }
    if "head" in inspect.signature(gr.Blocks).parameters:
        blocks_kwargs["head"] = LIGHT_MODE_HEAD

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_classes="app-shell"):
            with gr.Row(elem_classes="topbar"):
                gr.HTML(
                    """
                    <div class="brand">
                        <div class="brand-mark">LR</div>
                        <div>
                            <h1 class="brand-title">Lumi Restore Studio</h1>
                            <div class="brand-subtitle">Pure x4 photo restoration</div>
                        </div>
                    </div>
                    """
                )
                with gr.Row(elem_classes="top-actions"):
                    gr.HTML("<div class='mode-chip'>Ready to restore</div>")
                    gpu_status = gr.HTML(gpu_status_html())

            with gr.Row(elem_classes="hero-panel"):
                gr.HTML(
                    """
                    <section class="hero-card">
                        <div class="eyebrow">Pure SR Workspace</div>
                        <h2 class="hero-title">Lumi Restore Studio</h2>
                        <p class="hero-copy">
                            A bright restoration desk for Real-ESRGAN x4 upscaling,
                            tile-safe rendering, soft detail polish, and clean before-after review.
                        </p>
                        <div class="metric-grid">
                            <div class="metric-card">
                                <div class="metric-label">Scale</div>
                                <div class="metric-value">4x SR</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Engine</div>
                                <div class="metric-value">RRDBNet</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Output</div>
                                <div class="metric-value">PNG / JPG</div>
                            </div>
                        </div>
                    </section>
                    """
                )
                gr.HTML(
                    """
                    <aside class="hero-note">
                        <h3 class="note-title">Restoration Notes</h3>
                        <p class="note-copy">
                            Upload an image, choose the output format, and run the restore pass.
                            GPU usage stays visible while tiled inference keeps larger images manageable.
                        </p>
                        <ul class="note-list">
                            <li>Detail polish uses a light unsharp mask after upscaling.</li>
                            <li>Tile controls are available for memory-sensitive renders.</li>
                            <li>Before and after panes keep the review path direct.</li>
                        </ul>
                    </aside>
                    """
                )

            with gr.Row(elem_classes="workbench"):
                with gr.Column(elem_classes="control-panel", scale=0, min_width=300):
                    gr.Markdown("### Your Photo", elem_classes="panel-heading")
                    input_image = make_preview_image("Upload", 300)
                    enhance_detail = gr.Checkbox(label="Soft detail polish", value=True)
                    output_format = gr.Radio(
                        label="Save as",
                        choices=["PNG", "JPG"],
                        value="PNG",
                    )
                    with gr.Accordion("Fine details", open=False):
                        tile_size = gr.Slider(
                            label="Render tile size",
                            minimum=64,
                            maximum=512,
                            value=256,
                            step=8,
                        )
                        tile_pad = gr.Slider(
                            label="Edge blending",
                            minimum=0,
                            maximum=64,
                            value=16,
                            step=1,
                        )
                        device = gr.Dropdown(
                            label="Processing device",
                            choices=device_choices,
                            value=default_device,
                            allow_custom_value=True,
                        )
                    run_button = gr.Button("Restore photo", variant="primary", elem_id="run-button")
                    refresh_button = gr.Button("Refresh status", variant="secondary", elem_id="refresh-button")

                with gr.Column(elem_classes="viewer-panel"):
                    gr.Markdown("### Before / After", elem_classes="panel-heading")
                    with gr.Row(elem_classes="viewer-grid"):
                        before_image = make_preview_image("Before", 520)
                        after_image = make_preview_image("After", 520)
                    with gr.Row():
                        output_image = make_preview_image("Restored", 360)
                    with gr.Column(elem_classes="status-panel"):
                        download_file = gr.File(label="Download")
                        logs = gr.Textbox(label="Session notes", lines=7, interactive=False)

            run_button.click(
                fn=run_psr,
                inputs=[input_image, enhance_detail, output_format, tile_size, tile_pad, device],
                outputs=[output_image, before_image, after_image, download_file, logs, gpu_status],
                api_name="upscale",
            )
            refresh_button.click(fn=gpu_status_html, outputs=gpu_status, api_name="gpu_status")
            demo.load(fn=gpu_status_html, outputs=gpu_status)

            if hasattr(gr, "Timer"):
                timer = gr.Timer(2.0)
                timer.tick(fn=gpu_status_html, outputs=gpu_status)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Pure PSR Gradio app.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7861, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Enable a public Gradio share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the app in a browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=4)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
