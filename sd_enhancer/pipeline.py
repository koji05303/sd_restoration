import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

from .config import EnhanceConfig
from .io import save_image, write_metadata_sidecar
from .tiling import build_blend_mask, compute_tile_starts, round_up_to_multiple


class SafetyCheckerTriggeredError(RuntimeError):
    pass


def resolve_device(device_choice: str) -> str:
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-capable GPU is available.")

    return device_choice


def resolve_dtype(dtype_choice, device: str):
    if not isinstance(dtype_choice, str):
        return dtype_choice

    if dtype_choice == "auto":
        return torch.float16 if device == "cuda" else torch.float32

    if dtype_choice == "fp16" and device == "cpu":
        print("[Warning] fp16 on CPU is not supported by many ops. Falling back to fp32.")
        return torch.float32

    return torch.float16 if dtype_choice == "fp16" else torch.float32


def enable_low_vram_optimizations(pipe: StableDiffusionControlNetImg2ImgPipeline) -> None:
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


def enable_attention_backend(
    pipe: StableDiffusionControlNetImg2ImgPipeline,
    config: EnhanceConfig,
) -> None:
    if config.device == "cuda" and config.use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xFormers memory efficient attention.")
            return
        except Exception as exc:
            print(f"[Warning] Failed to enable xFormers: {exc}")

    try:
        pipe.enable_attention_slicing()
        print("Enabled attention slicing.")
    except Exception as exc:
        print(f"[Warning] Failed to enable attention slicing: {exc}")


def place_pipeline(
    pipe: StableDiffusionControlNetImg2ImgPipeline,
    config: EnhanceConfig,
) -> StableDiffusionControlNetImg2ImgPipeline:
    if config.device != "cuda" and config.offload_mode != "none":
        print("[Warning] CPU offload requires CUDA. Falling back to --offload none.")
        config.offload_mode = "none"

    if config.offload_mode == "sequential":
        if not hasattr(pipe, "enable_sequential_cpu_offload"):
            raise RuntimeError("This diffusers version does not support sequential CPU offload.")
        pipe.enable_sequential_cpu_offload()
        print("Enabled sequential CPU offload.")
        return pipe

    if config.offload_mode == "model":
        if not hasattr(pipe, "enable_model_cpu_offload"):
            raise RuntimeError("This diffusers version does not support model CPU offload.")
        pipe.enable_model_cpu_offload()
        print("Enabled model CPU offload.")
        return pipe

    return pipe.to(config.device)


def create_pipeline(config: EnhanceConfig) -> StableDiffusionControlNetImg2ImgPipeline:
    config.device = resolve_device(config.device)
    config.dtype = resolve_dtype(config.dtype, config.device)

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

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    enable_low_vram_optimizations(pipe)
    enable_attention_backend(pipe, config)
    pipe = place_pipeline(pipe, config)

    return pipe


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


def derive_tile_seed(
    config: EnhanceConfig,
    tile_index: int,
    random_seed_source: Optional[random.Random],
) -> Optional[int]:
    if config.seed is None:
        return None

    if config.tile_seed_mode == "same":
        return config.seed

    if config.tile_seed_mode == "offset":
        return config.seed + tile_index

    if random_seed_source is None:
        return None

    return random_seed_source.randrange(0, 2**32 - 1)


def make_generator(config: EnhanceConfig, tile_seed: Optional[int]) -> Optional[torch.Generator]:
    if tile_seed is None:
        return None
    generator_device = "cpu" if config.offload_mode != "none" else config.device
    return torch.Generator(device=generator_device).manual_seed(tile_seed)


def pad_image_to_size(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = image.size
    if width == target_width and height == target_height:
        return image

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pad_width = target_width - width
    pad_height = target_height - height
    if pad_width < 0 or pad_height < 0:
        raise ValueError("Target tile size must be >= image size.")

    padded = np.pad(
        arr,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded, mode="RGB")


def pad_mask_to_size(mask: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = mask.shape
    if width == target_width and height == target_height:
        return mask

    pad_width = target_width - width
    pad_height = target_height - height
    if pad_width < 0 or pad_height < 0:
        raise ValueError("Target mask size must be >= mask size.")

    return np.pad(mask, ((0, pad_height), (0, pad_width)), mode="edge")


def detect_skin_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.float32)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    channel_span = rgb.max(axis=2) - rgb.min(axis=2)

    ycbcr_skin = (cb >= 77.0) & (cb <= 135.0) & (cr >= 133.0) & (cr <= 180.0)
    rgb_skin = (
        (red > 60.0)
        & (green > 35.0)
        & (blue > 20.0)
        & (red >= green * 0.95)
        & (red >= blue * 1.05)
        & (channel_span > 10.0)
    )
    skin = ycbcr_skin & rgb_skin

    if not skin.any():
        return np.zeros((image.height, image.width), dtype=np.float32)

    mask_image = Image.fromarray((skin.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(7))
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=18))
    return np.asarray(mask_image, dtype=np.float32) / 255.0


def skin_mask_coverage(mask: np.ndarray, threshold: float = 0.15) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask > threshold) / mask.size)


def blend_skin_tiles(normal_tile: Image.Image, skin_tile: Image.Image, skin_mask: np.ndarray) -> Image.Image:
    if skin_tile.size != normal_tile.size:
        skin_tile = skin_tile.resize(normal_tile.size, Image.Resampling.LANCZOS)

    normal_arr = np.asarray(normal_tile.convert("RGB"), dtype=np.float32)
    skin_arr = np.asarray(skin_tile.convert("RGB"), dtype=np.float32)
    mask = np.clip(skin_mask, 0.0, 1.0)[:, :, None]
    blended = (skin_arr * mask) + (normal_arr * (1.0 - mask))
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


def apply_skin_tone_correction(
    output_tile: Image.Image,
    reference_tile: Image.Image,
    skin_mask: np.ndarray,
    radius: float = 16.0,
) -> Image.Image:
    reference_low = reference_tile.filter(ImageFilter.GaussianBlur(radius=radius))
    output_low = output_tile.filter(ImageFilter.GaussianBlur(radius=radius))

    output_arr = np.asarray(output_tile.convert("RGB"), dtype=np.float32)
    reference_low_arr = np.asarray(reference_low.convert("RGB"), dtype=np.float32)
    output_low_arr = np.asarray(output_low.convert("RGB"), dtype=np.float32)
    mask = np.clip(skin_mask, 0.0, 1.0)[:, :, None]

    corrected = output_arr + (reference_low_arr - output_low_arr) * mask
    return Image.fromarray(np.clip(corrected, 0.0, 255.0).astype(np.uint8), mode="RGB")


def match_color_to_reference(image: Image.Image, reference: Image.Image) -> Image.Image:
    reference = reference.resize(image.size, Image.Resampling.LANCZOS)
    reference_low = reference.filter(ImageFilter.GaussianBlur(radius=24))
    image_low = image.filter(ImageFilter.GaussianBlur(radius=24))

    image_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    reference_low_arr = np.asarray(reference_low.convert("RGB"), dtype=np.float32)
    image_low_arr = np.asarray(image_low.convert("RGB"), dtype=np.float32)

    matched = image_arr + (reference_low_arr - image_low_arr)
    return Image.fromarray(np.clip(matched, 0.0, 255.0).astype(np.uint8), mode="RGB")


def apply_postprocess(
    output_image: Image.Image,
    reference_image: Image.Image,
    config: EnhanceConfig,
) -> Image.Image:
    if config.match_color_input:
        print("[Postprocess] Matching output color to input.")
        output_image = match_color_to_reference(output_image, reference_image)

    if config.contrast:
        print("[Postprocess] Applying contrast.")
        output_image = ImageEnhance.Contrast(output_image).enhance(1.06)

    if config.sharpen:
        print("[Postprocess] Applying sharpen.")
        output_image = output_image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    return output_image


def enhance_image(
    config: EnhanceConfig,
    pipe: Optional[StableDiffusionControlNetImg2ImgPipeline] = None,
) -> Path:
    if pipe is None:
        pipe = create_pipeline(config)

    init_image = Image.open(config.image_path).convert("RGB")
    width, height = init_image.size

    scaled_width = max(1, round(width * config.upscale_factor))
    scaled_height = max(1, round(height * config.upscale_factor))
    target_width = round_up_to_multiple(scaled_width, base=64)
    target_height = round_up_to_multiple(scaled_height, base=64)
    scaled_image = init_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    resized_image = pad_image_to_size(scaled_image, target_width, target_height)

    print(f"Enhancing image from {width}x{height} to {scaled_width}x{scaled_height} ...")
    if (target_width, target_height) != (scaled_width, scaled_height):
        print(f"[Info] Padded inference canvas to {target_width}x{target_height}.")

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

    if config.seed is not None:
        print(f"Using seed: {config.seed} (tile mode: {config.tile_seed_mode})")
    if config.skin_protect:
        print(
            f"[Info] Skin protect enabled: mode={config.skin_protect_mode}, "
            f"skin_strength={config.skin_strength}"
        )

    accumulator = np.zeros((target_height, target_width, 3), dtype=np.float32)
    weights = np.zeros((target_height, target_width), dtype=np.float32)
    random_seed_source = random.Random(config.seed) if config.tile_seed_mode == "random" and config.seed is not None else None
    full_skin_mask = detect_skin_mask(resized_image) if config.skin_protect else None

    vae_upgraded = False
    fallback_tiles: list[tuple[int, int, str]] = []
    progress_line_length = 0

    def clear_progress_line() -> None:
        nonlocal progress_line_length
        if progress_line_length == 0:
            return
        sys.stdout.write("\r" + (" " * progress_line_length) + "\r")
        sys.stdout.flush()
        progress_line_length = 0

    def write_progress(message: str) -> None:
        nonlocal progress_line_length
        padding = max(progress_line_length - len(message), 0)
        sys.stdout.write("\r" + message + (" " * padding))
        sys.stdout.flush()
        progress_line_length = len(message)

    def finish_progress_line() -> None:
        nonlocal progress_line_length
        if progress_line_length > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            progress_line_length = 0

    def run_generation_once(tile_image: Image.Image, tile_seed: Optional[int], strength: float):
        generator = make_generator(config, tile_seed)

        with torch.inference_mode():
            return pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=tile_image,
                control_image=tile_image,
                controlnet_conditioning_scale=config.conditioning_scale,
                strength=strength,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )

    def generate_tile_checked(
        tile_image: Image.Image,
        tile_seed: Optional[int],
        x: int,
        y: int,
        strength: float,
        pass_name: str,
    ) -> Image.Image:
        nonlocal vae_upgraded

        result = run_generation_once(tile_image, tile_seed, strength)
        output_tile = result.images[0]
        nsfw_flags = getattr(result, "nsfw_content_detected", None)

        if nsfw_flags and any(flag is True for flag in nsfw_flags):
            clear_progress_line()
            raise SafetyCheckerTriggeredError(
                f"Safety checker triggered during {pass_name} pass at tile ({x}, {y})."
            )

        if is_near_black_image(output_tile):
            clear_progress_line()
            print(f"[Warning] Near-black {pass_name} tile detected at ({x}, {y}).")
            if config.dtype == torch.float16 and not vae_upgraded:
                print("[Info] Retrying tile once with FP32 VAE decode...")
                if vae_upgraded:
                    result = run_generation_once(tile_image, tile_seed, strength)
                    output_tile = result.images[0]
                    nsfw_flags = getattr(result, "nsfw_content_detected", None)

        if nsfw_flags and any(flag is True for flag in nsfw_flags):
            clear_progress_line()
            raise SafetyCheckerTriggeredError(
                f"Safety checker triggered during {pass_name} retry at tile ({x}, {y})."
            )

        if is_near_black_image(output_tile):
            clear_progress_line()
            raise RuntimeError(
                f"Near-black {pass_name} tile detected at tile ({x}, {y}). "
                "Try --dtype fp32, --offload sequential, lower --guidance-scale, "
                "or reduce --strength."
            )

        if output_tile.size != tile_image.size:
            output_tile = output_tile.resize(tile_image.size, Image.Resampling.LANCZOS)

        return output_tile

    tile_counter = 0

    for y in y_starts:
        for x in x_starts:
            tile_counter += 1
            tile_seed = derive_tile_seed(config, tile_counter, random_seed_source)
            right = min(x + tile_width, target_width)
            bottom = min(y + tile_height, target_height)
            valid_width = right - x
            valid_height = bottom - y
            tile_image = resized_image.crop((x, y, right, bottom))
            tile_image = pad_image_to_size(tile_image, tile_width, tile_height)

            skin_mask = None
            skin_coverage = 0.0
            if full_skin_mask is not None:
                skin_mask_valid = full_skin_mask[y:bottom, x:right]
                skin_coverage = skin_mask_coverage(skin_mask_valid)
                skin_mask = pad_mask_to_size(skin_mask_valid, tile_width, tile_height)

            if (
                skin_mask is not None
                and config.skin_protect_mode == "dual-pass"
                and skin_coverage >= 0.98
            ):
                output_tile = generate_tile_checked(
                    tile_image,
                    tile_seed,
                    x,
                    y,
                    config.skin_strength,
                    "skin",
                )
            else:
                output_tile = generate_tile_checked(
                    tile_image,
                    tile_seed,
                    x,
                    y,
                    config.strength,
                    "normal",
                )

                if skin_mask is not None and skin_coverage > 0.01:
                    if config.skin_protect_mode == "tone":
                        output_tile = apply_skin_tone_correction(output_tile, tile_image, skin_mask)
                    else:
                        skin_tile = generate_tile_checked(
                            tile_image,
                            tile_seed,
                            x,
                            y,
                            config.skin_strength,
                            "skin",
                        )
                        output_tile = blend_skin_tiles(output_tile, skin_tile, skin_mask)

            output_tile_arr = np.asarray(output_tile.convert("RGB"), dtype=np.float32)[:valid_height, :valid_width]
            blend_mask = build_blend_mask(
                valid_width,
                valid_height,
                overlap_x,
                overlap_y,
                has_left=(x > 0),
                has_right=(right < target_width),
                has_top=(y > 0),
                has_bottom=(bottom < target_height),
            )

            accumulator[y:bottom, x:right, :] += output_tile_arr * blend_mask[:, :, None]
            weights[y:bottom, x:right] += blend_mask

            progress_percent = tile_counter / total_tiles * 100.0
            write_progress(
                f"[Tile {tile_counter}/{total_tiles} | {progress_percent:5.1f}%] "
                f"Completed at ({x}, {y})"
            )

            if config.device == "cuda":
                torch.cuda.empty_cache()

    finish_progress_line()

    safe_weights = np.maximum(weights, 1e-6)
    output_array = np.clip(accumulator / safe_weights[:, :, None], 0.0, 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_array, mode="RGB")
    output_image = output_image.crop((0, 0, scaled_width, scaled_height))
    output_image = apply_postprocess(output_image, scaled_image, config)

    if fallback_tiles:
        print(f"[Warning] Reused original content for {len(fallback_tiles)} tile(s).")

    if is_near_black_image(output_image):
        raise RuntimeError(
            "Output remains near-black after tiled generation. "
            "Try --dtype fp32, lower --guidance-scale, or different model weights."
        )

    save_image(output_image, config.output_path)
    sidecar_path = write_metadata_sidecar(
        config=config,
        input_size=(width, height),
        output_size=(scaled_width, scaled_height),
        fallback_tiles=fallback_tiles,
    )
    print(f"Enhanced image saved to: {config.output_path}")
    print(f"Metadata saved to: {sidecar_path}")
    return config.output_path
