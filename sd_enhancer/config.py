from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11f1e_sd15_tile"


DEFAULT_PROMPT = (
    "(best quality, high fidelity:1.15), "
    "faithful photographic enhancement, "
    "preserved identity and anatomy, "
    "natural skin tone, "
    "smooth skin color transition, "
    "subtle realistic detail, "
    "preserved lighting, "
    "clean edges"
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
    "mottled skin, "
    "patchy skin tone, "
    "uneven skin texture, "
    "excessive pores, "
    "repeating texture, "
    "tile pattern, "
    "overprocessed skin, "
    "plastic skin"
)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TILE_SEED_MODES = ("same", "offset", "random")
SKIN_PROTECT_MODES = ("tone", "dual-pass")
OFFLOAD_MODES = ("none", "model", "sequential")


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
    skin_protect: bool
    skin_protect_mode: str
    skin_strength: float
    offload_mode: str
    sharpen: bool
    contrast: bool
    match_color_input: bool
