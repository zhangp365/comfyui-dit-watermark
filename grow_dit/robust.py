"""Geometry-alignment candidates for robust watermark extraction."""

from __future__ import annotations

from collections.abc import Iterator
import math

import torch
import torch.nn.functional as F


ROTATION_ANGLES = (0, 10, -10, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)
ROTATION_SCALES = (1.0, 0.8, 0.9, 1.1, 1.2)
CROP_SCALES = (1.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.2)
ROBUST_MODES = ("none", "rotation", "crop_scale", "rotation_crop_scale")


def _affine_candidate(image: torch.Tensor, angle: float, scale: float) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("image must be a ComfyUI [B,H,W,C] tensor")
    radians = math.radians(angle)
    cosine = math.cos(radians) / scale
    sine = math.sin(radians) / scale
    batch = image.shape[0]
    theta = image.new_tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]]
    ).unsqueeze(0).expand(batch, -1, -1)
    channels_first = image.movedim(-1, 1)
    grid = F.affine_grid(theta, channels_first.shape, align_corners=False)
    aligned = F.grid_sample(
        channels_first,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return aligned.movedim(1, -1).clamp(0.0, 1.0)


def alignment_candidates(
    image: torch.Tensor, mode: str
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield identity first, followed by original-GROW-style search candidates."""
    if mode not in ROBUST_MODES:
        raise ValueError(f"robust mode must be one of {ROBUST_MODES}")
    yield "identity", image
    if mode in ("rotation", "rotation_crop_scale"):
        for scale in ROTATION_SCALES:
            for angle in ROTATION_ANGLES:
                if angle == 0 and scale == 1.0:
                    continue
                yield (
                    f"rotation(angle={angle:g},scale={scale:g})",
                    _affine_candidate(image, angle, scale),
                )
    if mode in ("crop_scale", "rotation_crop_scale"):
        for scale in CROP_SCALES:
            if scale == 1.0:
                continue
            yield f"crop_scale(scale={scale:g})", _affine_candidate(image, 0.0, scale)
