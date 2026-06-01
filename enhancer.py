import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
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
    )


def enable_low_vram_optimizations(pipe: StableDiffusionControlNetImg2ImgPipeline) -> None:
    # Keep attention memory bounded when xFormers is unavailable or insufficient.
    pipe.enable_attention_slicing()

    # Newer diffusers may expose enable_vae_tiled_decode; older versions use enable_vae_tiling.
    try:
        if hasattr(pipe, "enable_vae_tiled_decode"):
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
        pipe.enable_vae_slicing()
        print("Enabled VAE slicing.")
    except Exception as exc:
        print(f"[Warning] Failed to enable VAE slicing: {exc}")


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

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=config.device).manual_seed(config.seed)
        print(f"Using seed: {config.seed}")

    with torch.inference_mode():
        output_image = pipe(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            image=resized_image,
            control_image=resized_image,
            controlnet_conditioning_scale=config.conditioning_scale,
            strength=config.strength,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        ).images[0]

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

