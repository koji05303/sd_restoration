import json
import logging
from pathlib import Path
from typing import Any, Iterable

from .config import EnhanceConfig, VALID_IMAGE_EXTENSIONS


logger = logging.getLogger(__name__)
OUTPUT_SUFFIX = ".png"


def png_output_path(path: Path) -> Path:
    return path.with_suffix(OUTPUT_SUFFIX)


def is_supported_image(path: Path) -> bool:
    logger.debug("Checking if %s is a supported image", path)
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def read_text_prompt(path: Path, prompt_name: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    logger.debug("Read %s from %s: %s%s", prompt_name, path, text[:30], "..." if len(text) > 30 else "")
    if not text:
        raise ValueError(f"{prompt_name} file is empty: {path}")
    return text


def collect_input_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted((path for path in iterator if is_supported_image(path)), key=lambda item: str(item).lower())
    logger.debug("Collected %d images from %s (recursive=%s)", len(images), input_dir, recursive)
    return images


def ensure_available_output(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")


def resolve_single_output_path(output_arg: Path, input_image: Path, overwrite: bool) -> Path:
    if output_arg.exists() and output_arg.is_dir():
        output_path = output_arg / f"{input_image.stem}_enhanced{OUTPUT_SUFFIX}"
    elif output_arg.suffix == "":
        output_arg.mkdir(parents=True, exist_ok=True)
        output_path = output_arg / f"{input_image.stem}_enhanced{OUTPUT_SUFFIX}"
    else:
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        output_path = png_output_path(output_arg)

    ensure_available_output(output_path, overwrite)
    return output_path


def resolve_batch_output_path(output_dir: Path, input_root: Path, input_image: Path, overwrite: bool) -> Path:
    relative_path = input_image.relative_to(input_root)
    output_subdir = output_dir / relative_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / f"{relative_path.stem}_enhanced{OUTPUT_SUFFIX}"
    ensure_available_output(output_path, overwrite)
    return output_path


def save_image(image: Any, output_path: Path) -> Path:
    output_path = png_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    logger.debug("Saved PNG image to %s", output_path)
    return output_path


def metadata_path_for(output_path: Path) -> Path:
    logger.debug("Generating metadata path for %s", output_path)
    return output_path.with_suffix(".json")


def write_metadata_sidecar(
    config: EnhanceConfig,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    fallback_tiles: list[tuple[int, int, str]],
) -> Path:
    metadata = {
        "input_path": str(config.image_path),
        "output_path": str(config.output_path),
        "input_size": {"width": input_size[0], "height": input_size[1]},
        "output_size": {"width": output_size[0], "height": output_size[1]},
        "preset": config.preset,
        "model_id": config.model_id,
        "controlnet_id": config.controlnet_id,
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "seed": config.seed,
        "tile_seed_mode": config.tile_seed_mode,
        "strength": config.strength,
        "conditioning_scale": config.conditioning_scale,
        "guidance": config.guidance_scale,
        "guidance_scale": config.guidance_scale,
        "steps": config.steps,
        "tile_size": config.tile_size,
        "tile_overlap": config.tile_overlap,
        "upscale_factor": config.upscale_factor,
        "skin_protect": config.skin_protect,
        "skin_protect_mode": config.skin_protect_mode,
        "skin_strength": config.skin_strength,
        "skin_texture_guard": config.skin_texture_guard,
        "skin_texture_guard_strength": config.skin_texture_guard_strength,
        "skin_guidance_scale": config.skin_guidance_scale,
        "skin_conditioning_scale": config.skin_conditioning_scale,
        "skin_tile_size": config.skin_tile_size,
        "offload": config.offload_mode,
        "device": config.device,
        "dtype": str(config.dtype).replace("torch.", ""),
        "use_xformers": config.use_xformers,
        "postprocess": {
            "sharpen": config.sharpen,
            "contrast": config.contrast,
            "match_color_input": config.match_color_input,
        },
        "fallback_tiles": [
            {"x": x, "y": y, "reason": reason}
            for x, y, reason in fallback_tiles
        ],
    }

    sidecar_path = metadata_path_for(config.output_path)
    sidecar_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Wrote metadata to %s", sidecar_path)
    return sidecar_path
