from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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
TILE_SEED_MODES = ("same", "offset", "random")


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
    dtype: Any
    use_xformers: bool
    overwrite: bool
    tile_size: int
    tile_overlap: int
    tile_seed_mode: str
    preset: str
    sharpen: bool
    contrast: bool
    match_color_input: bool
