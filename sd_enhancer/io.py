import json
from pathlib import Path
from typing import Any, Iterable

from .config import EnhanceConfig, VALID_IMAGE_EXTENSIONS


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def read_text_prompt(path: Path, prompt_name: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{prompt_name} file is empty: {path}")
    return text


def collect_input_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted((path for path in iterator if is_supported_image(path)), key=lambda item: str(item).lower())


def ensure_available_output(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")


def resolve_single_output_path(output_arg: Path, input_image: Path, overwrite: bool) -> Path:
    if output_arg.exists() and output_arg.is_dir():
        output_path = output_arg / f"{input_image.stem}_enhanced{input_image.suffix}"
    elif output_arg.suffix == "":
        output_arg.mkdir(parents=True, exist_ok=True)
        output_path = output_arg / f"{input_image.stem}_enhanced{input_image.suffix}"
    else:
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_arg

    ensure_available_output(output_path, overwrite)
    return output_path


def resolve_batch_output_path(output_dir: Path, input_root: Path, input_image: Path, overwrite: bool) -> Path:
    relative_path = input_image.relative_to(input_root)
    output_subdir = output_dir / relative_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / f"{relative_path.stem}_enhanced{relative_path.suffix}"
    ensure_available_output(output_path, overwrite)
    return output_path


def save_image(image: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95, "subsampling": 0}
    image.save(output_path, **save_kwargs)


def metadata_path_for(output_path: Path) -> Path:
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
    return sidecar_path
