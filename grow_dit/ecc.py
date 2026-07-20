"""Small fixed-frame Reed-Solomon codec over GF(256).

The 32-byte frame contains 28 data symbols and four parity symbols, so it can
correct any two corrupted byte symbols. The implementation is intentionally
self-contained to avoid adding a package dependency to ComfyUI.
"""

from __future__ import annotations

from dataclasses import dataclass


FRAME_BYTES = 32
PARITY_BYTES = 4
DATA_BYTES = FRAME_BYTES - PARITY_BYTES
MAX_MESSAGE_BYTES = DATA_BYTES - 1
PRIMITIVE_POLYNOMIAL = 0x11D

_EXP = [0] * 512
_LOG = [0] * 256
_value = 1
for _index in range(255):
    _EXP[_index] = _value
    _LOG[_value] = _index
    _value <<= 1
    if _value & 0x100:
        _value ^= PRIMITIVE_POLYNOMIAL
for _index in range(255, 512):
    _EXP[_index] = _EXP[_index - 255]


def _mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return _EXP[_LOG[left] + _LOG[right]]


def _div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if numerator == 0:
        return 0
    return _EXP[(_LOG[numerator] - _LOG[denominator]) % 255]


def _pow(value: int, exponent: int) -> int:
    if exponent == 0:
        return 1
    if value == 0:
        return 0
    return _EXP[(_LOG[value] * exponent) % 255]


def _poly_mul(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            result[i + j] ^= _mul(left_value, right_value)
    return result


def _poly_eval(polynomial: bytes | bytearray | list[int], value: int) -> int:
    result = 0
    for coefficient in polynomial:
        result = _mul(result, value) ^ coefficient
    return result


def _generator() -> list[int]:
    result = [1]
    for root in range(PARITY_BYTES):
        result = _poly_mul(result, [1, _EXP[root]])
    return result


_GENERATOR = _generator()


def rs_encode(data: bytes) -> bytes:
    """Return a systematic RS codeword with four parity symbols."""
    if len(data) != DATA_BYTES:
        raise ValueError(f"RS data must be exactly {DATA_BYTES} bytes")
    remainder = list(data) + [0] * PARITY_BYTES
    for offset in range(DATA_BYTES):
        coefficient = remainder[offset]
        if coefficient:
            for generator_offset in range(1, len(_GENERATOR)):
                remainder[offset + generator_offset] ^= _mul(
                    _GENERATOR[generator_offset], coefficient
                )
    return data + bytes(remainder[-PARITY_BYTES:])


def _syndromes(codeword: bytes | bytearray) -> list[int]:
    return [_poly_eval(codeword, _EXP[root]) for root in range(PARITY_BYTES)]


@dataclass(frozen=True)
class RSDecodeResult:
    codeword: bytes
    valid: bool
    corrected_symbols: int


def rs_decode(codeword: bytes) -> RSDecodeResult:
    """Correct up to two byte-symbol errors by solving syndrome equations."""
    if len(codeword) != FRAME_BYTES:
        raise ValueError(f"RS codeword must be exactly {FRAME_BYTES} bytes")
    syndromes = _syndromes(codeword)
    if not any(syndromes):
        return RSDecodeResult(codeword, True, 0)

    positions = range(FRAME_BYTES)
    location_values = [_EXP[(FRAME_BYTES - 1 - position) % 255] for position in positions]

    error = syndromes[0]
    if error:
        for position, location in zip(positions, location_values):
            if all(
                syndromes[index] == _mul(error, _pow(location, index))
                for index in range(PARITY_BYTES)
            ):
                corrected = bytearray(codeword)
                corrected[position] ^= error
                if not any(_syndromes(corrected)):
                    return RSDecodeResult(bytes(corrected), True, 1)

    for first in range(FRAME_BYTES - 1):
        first_location = location_values[first]
        for second in range(first + 1, FRAME_BYTES):
            second_location = location_values[second]
            denominator = first_location ^ second_location
            first_error = _div(
                syndromes[1] ^ _mul(syndromes[0], second_location), denominator
            )
            second_error = syndromes[0] ^ first_error
            if first_error == 0 or second_error == 0:
                continue
            if not all(
                syndromes[index]
                == (
                    _mul(first_error, _pow(first_location, index))
                    ^ _mul(second_error, _pow(second_location, index))
                )
                for index in range(PARITY_BYTES)
            ):
                continue
            corrected = bytearray(codeword)
            corrected[first] ^= first_error
            corrected[second] ^= second_error
            if not any(_syndromes(corrected)):
                return RSDecodeResult(bytes(corrected), True, 2)

    return RSDecodeResult(codeword, False, 0)


@dataclass(frozen=True)
class FrameDecodeResult:
    message: str
    valid: bool
    corrected_symbols: int
    corrected_codeword: bytes


def encode_frame(message: str) -> bytes:
    """Encode a UTF-8 message into the fixed 32-byte protected frame."""
    payload = message.encode("utf-8")
    if not payload:
        raise ValueError("message must not be empty")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"message is {len(payload)} UTF-8 bytes; maximum is {MAX_MESSAGE_BYTES}"
        )
    data = bytes([len(payload)]) + payload
    data += bytes(DATA_BYTES - len(data))
    return rs_encode(data)


def decode_frame(codeword: bytes) -> FrameDecodeResult:
    """Correct and validate a fixed frame without knowing the message."""
    decoded = rs_decode(codeword)
    if not decoded.valid:
        return FrameDecodeResult("", False, 0, codeword)
    data = decoded.codeword[:DATA_BYTES]
    length = data[0]
    if length == 0 or length > MAX_MESSAGE_BYTES:
        return FrameDecodeResult("", False, decoded.corrected_symbols, decoded.codeword)
    if any(data[1 + length :]):
        return FrameDecodeResult("", False, decoded.corrected_symbols, decoded.codeword)
    try:
        message = data[1 : 1 + length].decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return FrameDecodeResult("", False, decoded.corrected_symbols, decoded.codeword)
    return FrameDecodeResult(message, True, decoded.corrected_symbols, decoded.codeword)
