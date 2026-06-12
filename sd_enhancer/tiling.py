from __future__ import annotations

import math


def round_up_to_multiple(value: float, base: int = 64) -> int:
    return max(base, int(math.ceil(value / base) * base))


def compute_tile_starts(total_size: int, tile_size: int, overlap: int) -> list[int]:
    effective_tile = min(tile_size, total_size)
    if total_size <= effective_tile:
        return [0]

    stride = effective_tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    starts = [0]
    while starts[-1] + effective_tile < total_size:
        starts.append(starts[-1] + stride)
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
    import numpy as np

    mask_x = np.ones(tile_width, dtype=np.float32)
    mask_y = np.ones(tile_height, dtype=np.float32)

    x_ramp = min(overlap_x, tile_width // 2)
    y_ramp = min(overlap_y, tile_height // 2)

    def cosine_fade(length: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, num=length, endpoint=False, dtype=np.float32)
        return 0.5 - 0.5 * np.cos(np.pi * t)

    if x_ramp > 0:
        x_fade = cosine_fade(x_ramp)
        if has_left:
            mask_x[:x_ramp] = x_fade
        if has_right:
            mask_x[-x_ramp:] = 1.0 - x_fade

    if y_ramp > 0:
        y_fade = cosine_fade(y_ramp)
        if has_top:
            mask_y[:y_ramp] = y_fade
        if has_bottom:
            mask_y[-y_ramp:] = 1.0 - y_fade

    return np.outer(mask_y, mask_x)
