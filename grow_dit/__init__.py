"""Model-agnostic GROW latent watermark primitives."""

from .codec import bits_to_message, message_to_bits
from .ecc import FrameDecodeResult, decode_frame, encode_frame
from .core import (
    DetectionResult,
    WatermarkLayout,
    build_layout,
    extract_bits,
    frequency_transform,
    guide_denoised,
)

__all__ = [
    "DetectionResult",
    "FrameDecodeResult",
    "WatermarkLayout",
    "bits_to_message",
    "build_layout",
    "extract_bits",
    "frequency_transform",
    "guide_denoised",
    "message_to_bits",
    "decode_frame",
    "encode_frame",
]
