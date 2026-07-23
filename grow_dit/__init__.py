"""Model-agnostic GROW latent watermark primitives."""

from .config import GrowConfig
from .codec import bits_to_message, message_to_bits
from .ecc import (
    FrameDecodeResult,
    decode_frame,
    encode_frame,
    frame_bytes_for_payload_length,
)
from .core import (
    DetectionResult,
    WatermarkLayout,
    build_layout,
    build_layout_for_bits,
    detect_payload,
    extract_bits,
    frequency_transform,
    guide_denoised,
)

__all__ = [
    "DetectionResult",
    "FrameDecodeResult",
    "GrowConfig",
    "WatermarkLayout",
    "bits_to_message",
    "build_layout",
    "build_layout_for_bits",
    "detect_payload",
    "extract_bits",
    "frequency_transform",
    "guide_denoised",
    "message_to_bits",
    "decode_frame",
    "frame_bytes_for_payload_length",
    "encode_frame",
]
