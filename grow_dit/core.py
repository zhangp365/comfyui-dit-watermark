"""GROW progressive frequency guidance and inversion-free extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn.functional as F

from .codec import bits_to_bytes, message_to_bits
from .ecc import decode_frame


@dataclass(frozen=True)
class ChannelLayout:
    channel: int
    bit_offset: int
    bit_count: int
    repetitions: int
    coordinates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class WatermarkLayout:
    """A keyed target and its bit-to-frequency-coordinate mapping."""

    target: torch.Tensor
    mask: torch.Tensor
    message_bits: tuple[int, ...]
    channels: tuple[ChannelLayout, ...]
    center_ratio: float


@dataclass(frozen=True)
class DetectionResult:
    decoded_message: str
    ecc_valid: bool
    corrected_symbols: int
    min_vote_margin: float
    bits: tuple[int, ...]
    raw_codeword_hex: str


def frequency_transform(latent: torch.Tensor) -> torch.Tensor:
    """Real-part orthonormal FFT used as the differentiable DCT proxy in GROW."""
    return torch.fft.fft2(latent.float(), norm="ortho").real


def _center_patch(latent: torch.Tensor, center_ratio: float) -> torch.Tensor:
    if not 0.0 < center_ratio <= 1.0:
        raise ValueError("center_ratio must be in (0, 1]")
    height, width = latent.shape[-2:]
    patch_h = max(1, round(height * center_ratio))
    patch_w = max(1, round(width * center_ratio))
    top = (height - patch_h) // 2
    left = (width - patch_w) // 2
    return latent[..., top : top + patch_h, left : left + patch_w]


def _key_seed(secret_key: str) -> int:
    if not secret_key:
        raise ValueError("secret_key must not be empty")
    digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def build_layout(
    latent: torch.Tensor,
    message: str,
    secret_key: str,
    dct_min: float = 0.15,
    dct_max: float = 0.45,
    max_channels: int = 8,
    strength: float = 0.02,
    center_ratio: float = 1.0,
) -> WatermarkLayout:
    """Build a deterministic, repeated signed-frequency target."""
    if latent.ndim != 4:
        raise ValueError(f"expected a 4D [B,C,H,W] latent, got {tuple(latent.shape)}")
    if not 0.0 <= dct_min < dct_max <= 1.0:
        raise ValueError("require 0 <= dct_min < dct_max <= 1")
    if max_channels <= 0:
        raise ValueError("max_channels must be positive")
    if strength <= 0:
        raise ValueError("strength must be positive")

    bits = message_to_bits(message)
    patch = _center_patch(latent, center_ratio)
    _, channel_count, height, width = patch.shape
    used_channels = min(max_channels, channel_count, len(bits))
    if used_channels <= 0:
        raise ValueError("latent has no usable channels")

    h0, h1 = int(height * dct_min), int(height * dct_max)
    w0, w1 = int(width * dct_min), int(width * dct_max)
    coordinates = [(h, w) for h in range(h0, h1) for w in range(w0, w1)]
    if not coordinates:
        raise ValueError("selected frequency band is empty")

    generator = torch.Generator(device="cpu").manual_seed(_key_seed(secret_key))
    order = torch.randperm(len(coordinates), generator=generator).tolist()
    shuffled = [coordinates[index] for index in order]

    target = torch.zeros(
        (1, channel_count, height, width), device=latent.device, dtype=torch.float32
    )
    mask = torch.zeros_like(target, dtype=torch.bool)
    channel_layouts: list[ChannelLayout] = []

    base_count, remainder = divmod(len(bits), used_channels)
    bit_offset = 0
    for channel in range(used_channels):
        bit_count = base_count + (1 if channel < remainder else 0)
        repetitions = len(shuffled) // bit_count
        # An odd repetition count guarantees that hard majority voting cannot
        # end in a tie. Discarding one repetition is preferable to a secret
        # dependence on how ties happen to be resolved.
        if repetitions > 1 and repetitions % 2 == 0:
            repetitions -= 1
        if repetitions < 1:
            raise ValueError(
                f"payload needs {bit_count} coefficients in channel {channel}, "
                f"but the selected band provides only {len(shuffled)}"
            )
        used_coordinates = tuple(shuffled[: repetitions * bit_count])
        channel_bits = bits[bit_offset : bit_offset + bit_count]
        for repeat_index in range(repetitions):
            for local_bit, bit in enumerate(channel_bits):
                h, w = used_coordinates[repeat_index * bit_count + local_bit]
                target[0, channel, h, w] = strength if bit else -strength
                mask[0, channel, h, w] = True
        channel_layouts.append(
            ChannelLayout(
                channel=channel,
                bit_offset=bit_offset,
                bit_count=bit_count,
                repetitions=repetitions,
                coordinates=used_coordinates,
            )
        )
        bit_offset += bit_count

    return WatermarkLayout(
        target=target,
        mask=mask,
        message_bits=tuple(bits),
        channels=tuple(channel_layouts),
        center_ratio=center_ratio,
    )


def _frequency_loss(latent: torch.Tensor, layout: WatermarkLayout) -> torch.Tensor:
    transformed = frequency_transform(_center_patch(latent, layout.center_ratio))
    # Layouts are cached by the sampler and may themselves have been created
    # under ComfyUI inference mode. Clone them here so boolean indexing does
    # not ask autograd to save an inference tensor for backward.
    target = layout.target.detach().to(device=transformed.device).clone()
    mask = layout.mask.detach().to(device=transformed.device).clone()
    if transformed.shape[1:] != target.shape[1:]:
        raise ValueError(
            f"layout shape {tuple(target.shape)} does not match latent frequency "
            f"shape {tuple(transformed.shape)}"
        )
    expanded_target = target.expand(transformed.shape[0], -1, -1, -1)
    expanded_mask = mask.expand_as(expanded_target)
    coefficients = transformed[expanded_mask]
    signed_targets = expanded_target[expanded_mask]
    target_sign = torch.sign(signed_targets)
    target_margin = torch.abs(signed_targets)
    # The extractor only reads coefficient signs. A one-sided margin loss is
    # the quality-preserving GROW objective for short distilled schedules: it
    # moves wrong/weak coefficients but leaves already-correct strong ones
    # untouched instead of dragging every coefficient toward a fixed value.
    violations = F.relu(target_margin - target_sign * coefficients)
    return torch.mean(violations.square())


def guide_denoised(
    denoised: torch.Tensor,
    layout: WatermarkLayout,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one fp32 GROW gradient step to a predicted clean latent."""
    if guidance_scale <= 0:
        raise ValueError("guidance_scale must be positive")
    original_dtype = denoised.dtype
    # ComfyUI may call samplers under `torch.inference_mode()`. Merely enabling
    # gradients does not turn inference tensors back into autograd tensors, so
    # disable inference mode locally before cloning the tiny latent to fp32.
    with torch.inference_mode(False), torch.enable_grad():
        candidate = denoised.detach().float().clone().requires_grad_(True)
        before = _frequency_loss(candidate, layout)
        gradient = torch.autograd.grad(before, candidate)[0]
        guided = candidate - guidance_scale * gradient
        after = _frequency_loss(guided, layout)
    return guided.detach().to(original_dtype), before.detach(), after.detach()


@torch.no_grad()
def extract_bits(latent: torch.Tensor, layout: WatermarkLayout) -> DetectionResult:
    """Read keyed frequency signs and majority-vote each repeated payload bit."""
    if latent.ndim != 4:
        raise ValueError(f"expected a 4D [B,C,H,W] latent, got {tuple(latent.shape)}")
    transformed = frequency_transform(_center_patch(latent[:1], layout.center_ratio))
    decoded = [0] * len(layout.message_bits)
    margins = [0.0] * len(layout.message_bits)

    for channel_layout in layout.channels:
        for local_bit in range(channel_layout.bit_count):
            votes: list[int] = []
            for repeat_index in range(channel_layout.repetitions):
                index = repeat_index * channel_layout.bit_count + local_bit
                h, w = channel_layout.coordinates[index]
                votes.append(int(transformed[0, channel_layout.channel, h, w] > 0))
            ones = sum(votes)
            zeros = len(votes) - ones
            global_bit = channel_layout.bit_offset + local_bit
            decoded[global_bit] = int(ones > zeros)
            margins[global_bit] = abs(ones - zeros) / len(votes)

    raw_codeword = bits_to_bytes(decoded)
    frame = decode_frame(raw_codeword)
    return DetectionResult(
        decoded_message=frame.message,
        ecc_valid=frame.valid,
        corrected_symbols=frame.corrected_symbols,
        min_vote_margin=min(margins),
        bits=tuple(decoded),
        raw_codeword_hex=raw_codeword.hex(),
    )
