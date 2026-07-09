import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11f1e_sd15_tile"


DEFAULT_PROMPT = (
    "(best quality, faithful restoration:1.1), "
    "restrained photographic enhancement, "
    "preserved identity and anatomy, "
    "natural tone transitions, "
    "preserved natural skin texture, "
    "subtle original detail, "
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
    "synthetic texture, "
    "repeating texture, "
    "tile pattern, "
    "oversharpening artifacts, "
    "overprocessed"
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
    tile_batch_size: int
    preset: str
    skin_protect: bool
    skin_protect_mode: str
    skin_strength: float
    offload_mode: str
    sharpen: bool
    contrast: bool
    match_color_input: bool
    skin_texture_guard: bool = True
    skin_texture_guard_strength: float = 0.72
    skin_guidance_scale: float = 4.0
    skin_conditioning_scale: float = 0.7
    skin_tile_size: Optional[int] = None


def print_config(config: EnhanceConfig) -> None:
    logger.debug("EnhanceConfig:")
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if isinstance(value, Path):
            logger.debug("  %s: %s (exists=%s)", field, value, value.exists())
        else:
            logger.debug("  %s: %s", field, value)
