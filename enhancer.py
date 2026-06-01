import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageStat
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline


DEFAULT_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11f1e_sd15_tile"

DEFAULT_PROMPT = (
    "(masterpiece, best quality:1.2), "
    "photorealistic, "
    "8k resolution, "
    "ultra-detailed skin texture, "
    "soft studio lighting, "
    "sharp focus, "
    "volumetric shadow, "
    "hyper-realistic, "
    "raw photo, "
    "subsurface scattering"
)

DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality, low quality:1.4), "
    "blurry, "
    "bad anatomy, "
    "noise, "
    "painting, "
    "cartoon, "
    "3d render, "
    "cg, "
    "digital art, "
    "compression artifacts, "
    "lowres, "
    "text, "
    "watermark, "
    "bad hands, "
    "missing fingers, "
)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class SafetyCheckerTriggeredError(RuntimeError):
    pass

@dataclass
class EnhanceConfig:
    image_path: Path
    output_path: Path
    prompt: str
    negative_prompt: str
    model_id: str
    controlnet_id: str
    upscale_factor: float
    strength: float
    conditioning_scale: float
    guidance_scale: float
    steps: int
    seed: Optional[int]
    device: str
    dtype: torch.dtype
    use_xformers: bool
    overwrite: bool
    tile_size: int
    tile_overlap: int


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


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


def existing_text_file(value: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def read_text_prompt(path: Path, prompt_name: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{prompt_name} file is empty: {path}")
    return text


def resolve_device(device_choice: str) -> str:
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-capable GPU is available.")

    return device_choice


def resolve_dtype(dtype_choice: str, device: str) -> torch.dtype:
    if dtype_choice == "auto":
        return torch.float16 if device == "cuda" else torch.float32

    if dtype_choice == "fp16" and device == "cpu":
        print("[Warning] fp16 on CPU is not supported by many ops. Falling back to fp32.")
        return torch.float32

    return torch.float16 if dtype_choice == "fp16" else torch.float32


def round_up_to_multiple(value: float, base: int = 64) -> int:
    return max(base, int(math.ceil(value / base) * base))


def resolve_output_path(output_arg: Path, input_image: Path, overwrite: bool) -> Path:
    if output_arg.exists() and output_arg.is_dir():
        output_path = output_arg / f"{input_image.stem}_enhanced{input_image.suffix}"
    elif output_arg.suffix == "":
        output_arg.mkdir(parents=True, exist_ok=True)
        output_path = output_arg / f"{input_image.stem}_enhanced{input_image.suffix}"
    else:
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_arg

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance an image with Stable Diffusion + ControlNet Tile.",
        formatter_class=HelpFormatter,
    )

    parser.add_argument("-i", "--image", required=True, type=existing_image_file, help="Input image path.")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output file path, or output directory.",
    )

    prompt_group = parser.add_argument_group("Prompt options")
    positive_prompt_group = prompt_group.add_mutually_exclusive_group()
    positive_prompt_group.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Positive prompt text.")
    positive_prompt_group.add_argument(
        "--prompt-file",
        type=existing_text_file,
        help="Path to a text file for positive prompt.",
    )

    negative_prompt_group = prompt_group.add_mutually_exclusive_group()
    negative_prompt_group.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt text.",
    )
    negative_prompt_group.add_argument(
        "--negative-prompt-file",
        type=existing_text_file,
        help="Path to a text file for negative prompt.",
    )

    model_group = parser.add_argument_group("Model options")
    model_group.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base model ID.")
    model_group.add_argument(
        "--controlnet-id",
        type=str,
        default=DEFAULT_CONTROLNET_ID,
        help="ControlNet model ID.",
    )

    generation_group = parser.add_argument_group("Generation options")
    generation_group.add_argument(
        "--upscale-factor",
        type=positive_float,
        default=2.0,
        help="Resize multiplier before generation.",
    )
    generation_group.add_argument(
        "--strength",
        type=float_in_range(0.0, 1.0),
        default=0.35,
        help="How much to redraw the image.",
    )
    generation_group.add_argument(
        "--conditioning-scale",
        type=positive_float,
        default=1.0,
        help="ControlNet conditioning scale.",
    )
    generation_group.add_argument(
        "--guidance-scale",
        type=positive_float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    generation_group.add_argument(
        "--steps",
        type=positive_int,
        default=25,
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
        default=512,
        help="Tile size for tiled inference. Lower values reduce VRAM usage.",
    )
    generation_group.add_argument(
        "--tile-overlap",
        type=non_negative_int,
        default=64,
        help="Overlap (pixels) between adjacent tiles to reduce seams.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    parser.epilog = (
        "Examples:\n"
        "  python enhancer.py -i input/example.jpg -o output/ --seed 42 --steps 30\n"
        "  python enhancer.py -i input/example.jpg -o output/enhanced.png --prompt-file prompt.txt"
    )

    return parser


def parse_args(argv: Optional[list[str]] = None) -> EnhanceConfig:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.tile_overlap >= args.tile_size:
        parser.error("--tile-overlap must be smaller than --tile-size.")

    if args.tile_size % 8 != 0:
        parser.error("--tile-size must be a multiple of 8.")

    prompt = read_text_prompt(args.prompt_file, "Prompt") if args.prompt_file else args.prompt
    negative_prompt = (
        read_text_prompt(args.negative_prompt_file, "Negative prompt")
        if args.negative_prompt_file
        else args.negative_prompt
    )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_path = resolve_output_path(args.output, args.image, args.overwrite)

    return EnhanceConfig(
        image_path=args.image,
        output_path=output_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        upscale_factor=args.upscale_factor,
        strength=args.strength,
        conditioning_scale=args.conditioning_scale,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        device=device,
        dtype=dtype,
        use_xformers=not args.disable_xformers,
        overwrite=args.overwrite,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )


def enable_low_vram_optimizations(pipe: StableDiffusionControlNetImg2ImgPipeline) -> None:
    # Keep attention memory bounded when xFormers is unavailable or insufficient.
    pipe.enable_attention_slicing()

    # Prefer non-deprecated VAE APIs when available.
    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
            print("Enabled VAE tiling.")
        elif hasattr(pipe, "enable_vae_tiled_decode"):
            pipe.enable_vae_tiled_decode()
            print("Enabled VAE tiled decode.")
        elif hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
            print("Enabled VAE tiling (compat mode).")
        else:
            print("[Warning] This diffusers version does not expose VAE tiled decode/tiling APIs.")
    except Exception as exc:
        print(f"[Warning] Failed to enable VAE tiled decode/tiling: {exc}")

    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
            print("Enabled VAE slicing.")
        else:
            pipe.enable_vae_slicing()
            print("Enabled VAE slicing (compat mode).")
    except Exception as exc:
        print(f"[Warning] Failed to enable VAE slicing: {exc}")


def is_near_black_image(image: Image.Image, mean_threshold: float = 2.0) -> bool:
    rgb_image = image.convert("RGB")
    channel_means = ImageStat.Stat(rgb_image).mean
    return max(channel_means) <= mean_threshold


def upgrade_vae_decode_precision(pipe: StableDiffusionControlNetImg2ImgPipeline) -> bool:
    try:
        if hasattr(pipe, "upcast_vae"):
            pipe.upcast_vae()
        else:
            pipe.vae.to(dtype=torch.float32)
        print("[Info] Upgraded VAE decode precision to FP32.")
        return True
    except Exception as exc:
        print(f"[Warning] Failed to upgrade VAE decode precision: {exc}")
        return False


def compute_tile_starts(total_size: int, tile_size: int, overlap: int) -> list[int]:
    effective_tile = min(tile_size, total_size)
    if total_size <= effective_tile:
        return [0]

    stride = max(effective_tile - overlap, 1)
    starts = list(range(0, total_size - effective_tile + 1, stride))
    final_start = total_size - effective_tile
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def build_blend_mask(
    tile_width: int,
    tile_height: int,
    overlap_x: int,
    overlap_y: int,
    has_left: bool,
    has_right: bool,
    has_top: bool,
    has_bottom: bool,
) -> np.ndarray:
    mask_x = np.ones(tile_width, dtype=np.float32)
    mask_y = np.ones(tile_height, dtype=np.float32)

    x_ramp = min(overlap_x, tile_width // 2)
    y_ramp = min(overlap_y, tile_height // 2)

    if x_ramp > 0:
        if has_left:
            mask_x[:x_ramp] = np.linspace(0.0, 1.0, num=x_ramp, endpoint=False, dtype=np.float32)
        if has_right:
            mask_x[-x_ramp:] = np.linspace(1.0, 0.0, num=x_ramp, endpoint=False, dtype=np.float32)

    if y_ramp > 0:
        if has_top:
            mask_y[:y_ramp] = np.linspace(0.0, 1.0, num=y_ramp, endpoint=False, dtype=np.float32)
        if has_bottom:
            mask_y[-y_ramp:] = np.linspace(1.0, 0.0, num=y_ramp, endpoint=False, dtype=np.float32)

    return np.outer(mask_y, mask_x)


def enhance_image(config: EnhanceConfig) -> None:
    print("Initializing model...")
    if config.device == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    controlnet = ControlNetModel.from_pretrained(config.controlnet_id, torch_dtype=config.dtype)

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        config.model_id,
        controlnet=controlnet,
        torch_dtype=config.dtype,
        use_safetensors=True,
    )

    # Force low-VRAM decode path before moving to device.
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    enable_low_vram_optimizations(pipe)
    pipe = pipe.to(config.device)

    if config.device == "cuda" and config.use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xFormers memory efficient attention.")
        except Exception as exc:
            print(f"[Warning] Failed to enable xFormers: {exc}")
            print("[Info] Falling back to attention slicing + VAE tiling/slicing.")

    init_image = Image.open(config.image_path).convert("RGB")
    width, height = init_image.size

    target_width = round_up_to_multiple(width * config.upscale_factor, base=64)
    target_height = round_up_to_multiple(height * config.upscale_factor, base=64)
    resized_image = init_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    print(f"Enhancing image from {width}x{height} to {target_width}x{target_height} ...")

    tile_width = min(config.tile_size, target_width)
    tile_height = min(config.tile_size, target_height)
    overlap_x = min(config.tile_overlap, max(tile_width - 1, 0))
    overlap_y = min(config.tile_overlap, max(tile_height - 1, 0))

    x_starts = compute_tile_starts(target_width, tile_width, overlap_x)
    y_starts = compute_tile_starts(target_height, tile_height, overlap_y)
    total_tiles = len(x_starts) * len(y_starts)

    print(
        f"[Info] Tiled inference: tile={tile_width}x{tile_height}, "
        f"overlap={overlap_x}x{overlap_y}, total_tiles={total_tiles}"
    )

    accumulator = np.zeros((target_height, target_width, 3), dtype=np.float32)
    weights = np.zeros((target_height, target_width), dtype=np.float32)

    def run_generation_once(tile_image: Image.Image, tile_index: int):
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=config.device).manual_seed(config.seed + tile_index)

        with torch.inference_mode():
            return pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=tile_image,
                control_image=tile_image,
                controlnet_conditioning_scale=config.conditioning_scale,
                strength=config.strength,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )

    if config.seed is not None:
        print(f"Using base seed: {config.seed}")

    vae_upgraded = False
    tile_counter = 0
    fallback_tiles: list[tuple[int, int, str]] = []

    for y in y_starts:
        for x in x_starts:
            tile_counter += 1
            box = (x, y, x + tile_width, y + tile_height)
            tile_image = resized_image.crop(box)
            use_input_tile_fallback = False
            fallback_reason = ""

            result = run_generation_once(tile_image, tile_counter)
            output_tile = result.images[0]
            nsfw_flags = getattr(result, "nsfw_content_detected", None)

            if nsfw_flags and any(flag is True for flag in nsfw_flags):
                use_input_tile_fallback = True
                fallback_reason = "safety-checker"

            if not use_input_tile_fallback and is_near_black_image(output_tile):
                print(f"[Warning] Near-black tile detected at ({x}, {y}).")
                if config.dtype == torch.float16 and not vae_upgraded:
                    print("[Info] Retrying tile once with FP32 VAE decode...")
                    vae_upgraded = upgrade_vae_decode_precision(pipe)
                    if vae_upgraded:
                        result = run_generation_once(tile_image, tile_counter)
                        output_tile = result.images[0]
                        nsfw_flags = getattr(result, "nsfw_content_detected", None)

            if nsfw_flags and any(flag is True for flag in nsfw_flags):
                use_input_tile_fallback = True
                fallback_reason = "safety-checker-retry"

            if not use_input_tile_fallback and is_near_black_image(output_tile):
                use_input_tile_fallback = True
                fallback_reason = "near-black-retry"

            if use_input_tile_fallback:
                output_tile = tile_image.copy()
                fallback_tiles.append((x, y, fallback_reason))
                print(
                    f"[Warning] Reusing original tile at ({x}, {y}) due to {fallback_reason}."
                )

            output_tile_arr = np.asarray(output_tile.convert("RGB"), dtype=np.float32)
            blend_mask = build_blend_mask(
                tile_width,
                tile_height,
                overlap_x,
                overlap_y,
                has_left=(x > 0),
                has_right=(x + tile_width < target_width),
                has_top=(y > 0),
                has_bottom=(y + tile_height < target_height),
            )

            accumulator[y : y + tile_height, x : x + tile_width, :] += output_tile_arr * blend_mask[:, :, None]
            weights[y : y + tile_height, x : x + tile_width] += blend_mask

            print(f"[Tile {tile_counter}/{total_tiles}] Completed at ({x}, {y})")

            if config.device == "cuda":
                torch.cuda.empty_cache()

    safe_weights = np.maximum(weights, 1e-6)
    output_array = np.clip(accumulator / safe_weights[:, :, None], 0.0, 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_array, mode="RGB")

    if fallback_tiles:
        print(f"[Warning] Reused original content for {len(fallback_tiles)} tile(s).")

    if is_near_black_image(output_image):
        raise RuntimeError(
            "Output remains near-black after tiled generation. "
            "Try --dtype fp32, lower --guidance-scale, or different model weights."
        )

    output_image.save(config.output_path)
    print(f"Enhanced image saved to: {config.output_path}")


def main(argv: Optional[list[str]] = None) -> int:
    try:
        config = parse_args(argv)
        enhance_image(config)
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

