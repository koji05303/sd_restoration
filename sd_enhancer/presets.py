from dataclasses import dataclass

from .config import DEFAULT_CONTROLNET_ID, DEFAULT_MODEL_ID, DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT


@dataclass(frozen=True)
class EnhancementPreset:
    prompt: str
    negative_prompt: str
    model_id: str = DEFAULT_MODEL_ID
    controlnet_id: str = DEFAULT_CONTROLNET_ID
    upscale_factor: float = 2.0
    strength: float = 0.25
    conditioning_scale: float = 1.3
    guidance_scale: float = 5.5
    steps: int = 25
    tile_size: int = 512
    tile_overlap: int = 128
    tile_seed_mode: str = "same"
    offload_mode: str = "none"


PHOTO_PRESET = EnhancementPreset(
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
)

ANIME_PRESET = EnhancementPreset(
    prompt=(
        "(masterpiece, best quality:1.2), "
        "anime illustration, "
        "clean lineart, "
        "detailed eyes, "
        "refined hair detail, "
        "smooth cel shading, "
        "vibrant but balanced color, "
        "crisp high-resolution finish"
    ),
    negative_prompt=(
        "(worst quality, low quality:1.4), "
        "blurry, "
        "muddy color, "
        "noise, "
        "jpeg artifacts, "
        "bad anatomy, "
        "extra fingers, "
        "missing fingers, "
        "text, "
        "watermark, "
        "photorealistic, "
        "3d render"
    ),
    strength=0.32,
    guidance_scale=8.0,
    steps=28,
)

DENOISE_PRESET = EnhancementPreset(
    prompt=(
        "(best quality:1.15), "
        "clean natural detail, "
        "reduced noise, "
        "preserved original colors, "
        "realistic texture, "
        "sharp but natural edges"
    ),
    negative_prompt=(
        "(worst quality, low quality:1.4), "
        "heavy noise, "
        "blurry, "
        "overprocessed, "
        "plastic skin, "
        "oversharpened, "
        "compression artifacts, "
        "color shift, "
        "text, "
        "watermark"
    ),
    upscale_factor=1.0,
    strength=0.22,
    guidance_scale=5.5,
    steps=20,
)

UPSCALE_PRESET = EnhancementPreset(
    prompt=(
        "(masterpiece, best quality:1.2), "
        "high-resolution detail, "
        "faithful upscale, "
        "natural texture, "
        "clean edges, "
        "sharp focus, "
        "preserved composition"
    ),
    negative_prompt=(
        "(worst quality, low quality:1.4), "
        "blurry, "
        "noise, "
        "overprocessed, "
        "hallucinated detail, "
        "color shift, "
        "compression artifacts, "
        "lowres, "
        "text, "
        "watermark"
    ),
    upscale_factor=4.0,
    strength=0.25,
    guidance_scale=5.5,
    steps=24,
)

LOW_VRAM_PRESET = EnhancementPreset(
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    upscale_factor=2.0,
    strength=0.25,
    guidance_scale=5.5,
    steps=20,
    tile_size=384,
    offload_mode="sequential",
)

PRESETS = {
    "photo": PHOTO_PRESET,
    "anime": ANIME_PRESET,
    "denoise": DENOISE_PRESET,
    "upscale": UPSCALE_PRESET,
    "low-vram": LOW_VRAM_PRESET,
}


def get_preset(name: str) -> EnhancementPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid_names}") from exc
