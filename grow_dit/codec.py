"""Bit serialization for the fixed Reed-Solomon protected GROW frame."""

from __future__ import annotations

from .ecc import FRAME_BYTES, decode_frame, encode_frame


def message_to_bits(message: str) -> list[int]:
    """Serialize a message to a fixed 256-bit ECC frame."""
    return bytes_to_bits(encode_frame(message))


def bits_to_message(bits: list[int]) -> str:
    """Correct and deserialize one complete fixed frame."""
    result = decode_frame(bits_to_bytes(bits))
    if not result.valid:
        raise ValueError("ECC frame is invalid or has more than two symbol errors")
    return result.message


def bytes_to_bits(values: bytes) -> list[int]:
    bits: list[int] = []
    for value in values:
        bits.extend(int(bit) for bit in f"{value:08b}")
    return bits


def bits_to_bytes(bits: list[int] | tuple[int, ...]) -> bytes:
    if len(bits) != FRAME_BYTES * 8:
        raise ValueError(f"frame must contain exactly {FRAME_BYTES * 8} bits")
    return bytes(
        int("".join(str(bit) for bit in bits[offset : offset + 8]), 2)
        for offset in range(0, len(bits), 8)
    )
