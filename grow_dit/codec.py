"""UTF-8 payload serialization used by the GROW layout."""

from __future__ import annotations


def message_to_bits(message: str) -> list[int]:
    """Serialize a non-empty UTF-8 string to MSB-first bits."""
    if not message:
        raise ValueError("message must not be empty")
    bits: list[int] = []
    for value in message.encode("utf-8"):
        bits.extend(int(bit) for bit in f"{value:08b}")
    return bits


def bits_to_message(bits: list[int]) -> str:
    """Deserialize MSB-first bits; replacement characters expose corruption."""
    if not bits or len(bits) % 8:
        raise ValueError("bit count must be a non-zero multiple of 8")
    values = bytearray(
        int("".join(str(bit) for bit in bits[offset : offset + 8]), 2)
        for offset in range(0, len(bits), 8)
    )
    return values.decode("utf-8", errors="replace")
