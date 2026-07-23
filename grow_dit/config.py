"""Shared layout configuration for GROW embedding and detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrowConfig:
    """Parameters that must match when embedding and detecting a watermark."""

    secret_key: str = "watermark"
    dct_min: float = 0.15
    dct_max: float = 0.45
    max_channels: int = 8
    channel_start: int = 4
    center_ratio: float = 1.0

    def validate(self) -> None:
        if not self.secret_key:
            raise ValueError("secret_key must not be empty")
        if not 0.0 <= self.dct_min < self.dct_max <= 1.0:
            raise ValueError("require 0 <= dct_min < dct_max <= 1")
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if self.channel_start < 0:
            raise ValueError("channel_start must be non-negative")
        if not 0.0 < self.center_ratio <= 1.0:
            raise ValueError("center_ratio must be in (0, 1]")
