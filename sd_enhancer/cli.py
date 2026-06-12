import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import (
    EnhanceConfig,
    OFFLOAD_MODES,
    SKIN_PROTECT_MODES,
    TILE_SEED_MODES,
    VALID_IMAGE_EXTENSIONS,
)
from .io import (
    collect_input_images,
    read_text_prompt,
    resolve_batch_output_path,
    resolve_single_output_path,
)
from .presets import PRESETS, get_preset


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return number


def float_in_range(min_value: float, max_value: float):
    def validator(value: str) -> float:
        number = float(value)
        if number < min_value or number > max_value:
            raise argparse.ArgumentTypeError(f"Value must be in [{min_value}, {max_value}].")
        return number

    return validator


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return number


def existing_image_file(value: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError(f"Input image not found: {path}")
    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(VALID_IMAGE_EXTENSIONS))
        raise argparse.ArgumentTypeError(f"Unsupported image type '{path.suffix}'. Supported: {supported}")
    return path


def existing_input_dir(value: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_dir():
        raise argparse.ArgumentTypeError(f"Input directory not found: {path}")
    return path


def existing_text_file(value: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance images with Stable Diffusion + ControlNet Tile.",
        formatter_class=HelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--image", type=existing_image_file, help="Input image path.")
    input_group.add_argument("--input-dir", type=existing_input_dir, help="Directory of input images.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan --input-dir recursively.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output file path for --image, or output directory for batch mode.",
    )

    preset_group = parser.add_argument_group("Preset options")
    preset_group.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="photo",
        help="Preset values for prompt and generation settings.",
    )

    prompt_group = parser.add_argument_group("Prompt options")
    positive_prompt_group = prompt_group.add_mutually_exclusive_group()
    positive_prompt_group.add_argument("--prompt", type=str, default=None, help="Positive prompt text.")
    positive_prompt_group.add_argument(
        "--prompt-file",
        type=existing_text_file,
        help="Path to a text file for positive prompt.",
    )

    negative_prompt_group = prompt_group.add_mutually_exclusive_group()
    negative_prompt_group.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt text.",
    )
    negative_prompt_group.add_argument(
        "--negative-prompt-file",
        type=existing_text_file,
        help="Path to a text file for negative prompt.",
    )

    model_group = parser.add_argument_group("Model options")
    model_group.add_argument("--model-id", type=str, default=None, help="Base model ID.")
    model_group.add_argument(
        "--controlnet-id",
        type=str,
        default=None,
        help="ControlNet model ID.",
    )

    generation_group = parser.add_argument_group("Generation options")
    generation_group.add_argument(
        "--upscale-factor",
        type=positive_float,
        default=None,
        help="Resize multiplier before generation.",
    )
    generation_group.add_argument(
        "--strength",
        type=float_in_range(0.0, 1.0),
        default=None,
        help="How much to redraw the image.",
    )
    generation_group.add_argument(
        "--conditioning-scale",
        type=positive_float,
        default=None,
        help="ControlNet conditioning scale.",
    )
    generation_group.add_argument(
        "--guidance-scale",
        type=positive_float,
        default=None,
        help="Classifier-free guidance scale.",
    )
    generation_group.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help="Number of denoising inference steps.",
    )
    generation_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    generation_group.add_argument(
        "--tile-size",
        type=positive_int,
        default=None,
        help="Tile size for tiled inference. Lower values reduce VRAM usage.",
    )
    generation_group.add_argument(
        "--tile-overlap",
        type=non_negative_int,
        default=None,
        help="Overlap (pixels) between adjacent tiles to reduce seams.",
    )
    generation_group.add_argument(
        "--tile-seed-mode",
        choices=TILE_SEED_MODES,
        default=None,
        help="How tile seeds are derived from --seed.",
    )
    generation_group.add_argument(
        "--skin-protect",
        action="store_true",
        help="Protect detected skin with a full-image feathered mask.",
    )
    generation_group.add_argument(
        "--skin-protect-mode",
        choices=SKIN_PROTECT_MODES,
        default="tone",
        help="Skin protection strategy. tone is faster; dual-pass runs an extra low-strength SD pass.",
    )
    generation_group.add_argument(
        "--skin-strength",
        type=float_in_range(0.0, 1.0),
        default=0.18,
        help="Denoising strength used in dual-pass skin regions when --skin-protect is enabled.",
    )

    postprocess_group = parser.add_argument_group("Postprocess options")
    postprocess_group.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply a light unsharp-mask pass after tiled generation.",
    )
    postprocess_group.add_argument(
        "--contrast",
        action="store_true",
        help="Apply a subtle contrast pass after tiled generation.",
    )
    postprocess_group.add_argument(
        "--match-color-input",
        action="store_true",
        help="Match output RGB mean/std to the upscaled input image.",
    )

    runtime_group = parser.add_argument_group("Runtime options")
    runtime_group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    runtime_group.add_argument(
        "--dtype",
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="Torch dtype for model loading.",
    )
    runtime_group.add_argument(
        "--disable-xformers",
        action="store_true",
        help="Disable xFormers memory efficient attention.",
    )
    runtime_group.add_argument(
        "--offload",
        choices=OFFLOAD_MODES,
        default=None,
        help="CPU offload mode. sequential saves the most VRAM but is slow.",
    )
    runtime_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    parser.epilog = (
        "Examples:\n"
        "  python enhancer.py -i input/example.jpg -o output/ --preset photo --seed 42\n"
        "  python enhancer.py -i input/example.jpg -o output/enhanced.png --prompt-file prompt.txt\n"
        "  python enhancer.py --input-dir input/ --recursive -o output/ --preset low-vram --overwrite"
    )

    return parser


def build_config(args: argparse.Namespace, image_path: Path, output_path: Path) -> EnhanceConfig:
    preset = get_preset(args.preset)

    prompt = read_text_prompt(args.prompt_file, "Prompt") if args.prompt_file else args.prompt
    if prompt is None:
        prompt = preset.prompt

    negative_prompt = (
        read_text_prompt(args.negative_prompt_file, "Negative prompt")
        if args.negative_prompt_file
        else args.negative_prompt
    )
    if negative_prompt is None:
        negative_prompt = preset.negative_prompt

    return EnhanceConfig(
        image_path=image_path,
        output_path=output_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=args.model_id or preset.model_id,
        controlnet_id=args.controlnet_id or preset.controlnet_id,
        upscale_factor=args.upscale_factor if args.upscale_factor is not None else preset.upscale_factor,
        strength=args.strength if args.strength is not None else preset.strength,
        conditioning_scale=(
            args.conditioning_scale
            if args.conditioning_scale is not None
            else preset.conditioning_scale
        ),
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else preset.guidance_scale,
        steps=args.steps if args.steps is not None else preset.steps,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        use_xformers=not args.disable_xformers,
        overwrite=args.overwrite,
        tile_size=args.tile_size if args.tile_size is not None else preset.tile_size,
        tile_overlap=args.tile_overlap if args.tile_overlap is not None else preset.tile_overlap,
        tile_seed_mode=(
            args.tile_seed_mode
            if args.tile_seed_mode is not None
            else preset.tile_seed_mode
        ),
        preset=args.preset,
        skin_protect=args.skin_protect,
        skin_protect_mode=args.skin_protect_mode,
        skin_strength=args.skin_strength,
        offload_mode=args.offload if args.offload is not None else preset.offload_mode,
        sharpen=args.sharpen,
        contrast=args.contrast,
        match_color_input=args.match_color_input,
    )


def parse_args(argv: Optional[list[str]] = None) -> list[EnhanceConfig]:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.recursive and args.input_dir is None:
        parser.error("--recursive requires --input-dir.")

    preset = get_preset(args.preset)
    tile_size = args.tile_size if args.tile_size is not None else preset.tile_size
    tile_overlap = args.tile_overlap if args.tile_overlap is not None else preset.tile_overlap

    if tile_overlap >= tile_size:
        parser.error("--tile-overlap must be smaller than --tile-size.")

    if tile_size % 8 != 0:
        parser.error("--tile-size must be a multiple of 8.")

    if args.input_dir is not None:
        if args.output.exists() and not args.output.is_dir():
            parser.error("--output must be a directory when using --input-dir.")
        if not args.output.exists() and args.output.suffix:
            parser.error("--output must be a directory when using --input-dir.")

        images = collect_input_images(args.input_dir, args.recursive)
        if not images:
            supported = ", ".join(sorted(VALID_IMAGE_EXTENSIONS))
            parser.error(f"No supported images found in {args.input_dir}. Supported: {supported}")

        output_paths = [
            resolve_batch_output_path(args.output, args.input_dir, image, args.overwrite)
            for image in images
        ]
    else:
        images = [args.image]
        output_paths = [resolve_single_output_path(args.output, args.image, args.overwrite)]

    resolved_outputs = [str(path.resolve()) for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("Multiple inputs resolve to the same output path.")

    return [
        build_config(args, image_path=image_path, output_path=output_path)
        for image_path, output_path in zip(images, output_paths)
    ]


def main(argv: Optional[list[str]] = None) -> int:
    try:
        configs = parse_args(argv)

        from .pipeline import create_pipeline, enhance_image

        pipe = create_pipeline(configs[0])
        for config in configs[1:]:
            config.device = configs[0].device
            config.dtype = configs[0].dtype

        for index, config in enumerate(configs, start=1):
            if len(configs) > 1:
                print(f"[Batch {index}/{len(configs)}] {config.image_path} -> {config.output_path}")
            enhance_image(config, pipe=pipe)
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
