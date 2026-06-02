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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --canvas: #f5efe8;
    --panel: #fffaf4;
    --panel-raised: #ffffff;
    --panel-soft: #f7eee7;
    --ink: #2f2a27;
    --muted: #8b7f77;
    --quiet: #aa9d94;
    --line: rgba(80, 57, 48, 0.12);
    --line-strong: rgba(80, 57, 48, 0.2);
    --accent: #d9798f;
    --accent-strong: #c85f78;
    --sage: #5d9b8d;
    --gold: #c39a4e;
    --danger: #c85a50;
    --shadow: 0 24px 70px rgba(94, 64, 49, 0.16);
    --shadow-soft: 0 12px 34px rgba(94, 64, 49, 0.1);
}

body,
.gradio-container {
    background: var(--canvas) !important;
    color: var(--ink) !important;
    color-scheme: light;
    font-family: Manrope, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 20px !important;
    background:
        radial-gradient(circle at 10% 0%, rgba(217, 121, 143, 0.16), rgba(217, 121, 143, 0) 28%),
        linear-gradient(180deg, rgba(255, 250, 244, 0.86), rgba(245, 239, 232, 1) 44%),
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
    background: rgba(255, 250, 244, 0.86);
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
    border: 1px solid rgba(84, 214, 189, 0.3);
    border-radius: 8px;
    background: #fff1f3;
    color: var(--accent-strong);
    font-size: 0.74rem;
    font-weight: 900;
}

.brand-title {
    margin: 0;
    color: var(--ink);
    font-size: 1.18rem;
    font-weight: 800;
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
    background: rgba(93, 155, 141, 0.12);
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
    font-size: 0.92rem !important;
    font-weight: 800 !important;
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
    background: var(--accent) !important;
    color: #fffaf4 !important;
    font-weight: 900 !important;
    box-shadow: 0 14px 30px rgba(217, 121, 143, 0.26) !important;
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

    .gpu-row {
        grid-template-columns: 1fr;
        gap: 4px;
    }

    .gpu-memory {
        white-space: normal;
    }
}
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

    with gr.Blocks(title="Lumi Restore Studio", css=APP_CSS) as demo:
        with gr.Column(elem_classes="app-shell"):
            with gr.Row(elem_classes="topbar"):
                gr.HTML(
                    """
                    <div class="brand">
                        <div class="brand-mark">LR</div>
                        <div>
                            <h1 class="brand-title">Lumi Restore Studio</h1>
                            <div class="brand-subtitle">Soft x4 photo restoration</div>
                        </div>
                    </div>
                    """
                )
                with gr.Row(elem_classes="top-actions"):
                    gr.HTML("<div class='mode-chip'>Ready to restore</div>")
                    gpu_status = gr.HTML(gpu_status_html())

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
