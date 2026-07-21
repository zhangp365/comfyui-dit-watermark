"""GROW progressive frequency guidance and inversion-free extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn.functional as F

from .codec import bits_to_bytes, message_to_bits
from .ecc import MAX_MESSAGE_BYTES, frame_bytes_for_payload_length, decode_frame


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


def image_latent_4d(latent: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize Flux 4D or single-frame Qwen 5D latents to [B,C,H,W]."""
    if latent.ndim == 4:
        return latent, False
    if latent.ndim == 5 and latent.shape[2] == 1:
        return latent.squeeze(2), True
    raise ValueError(
        "expected a 4D [B,C,H,W] latent or single-frame 5D "
        f"[B,C,1,H,W] latent, got {tuple(latent.shape)}"
    )


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
    channel_start: int = 0,
) -> WatermarkLayout:
    """Build a deterministic, repeated signed-frequency target."""
    return build_layout_for_bits(
        latent,
        message_to_bits(message),
        secret_key,
        dct_min=dct_min,
        dct_max=dct_max,
        max_channels=max_channels,
        strength=strength,
        center_ratio=center_ratio,
        channel_start=channel_start,
    )


def build_layout_for_bits(
    latent: torch.Tensor,
    bits: list[int] | tuple[int, ...],
    secret_key: str,
    dct_min: float = 0.15,
    dct_max: float = 0.45,
    max_channels: int = 8,
    strength: float = 0.02,
    center_ratio: float = 1.0,
    channel_start: int = 0,
) -> WatermarkLayout:
    """Build a deterministic layout for an already encoded elastic frame."""
    latent, _ = image_latent_4d(latent)
    if not 0.0 <= dct_min < dct_max <= 1.0:
        raise ValueError("require 0 <= dct_min < dct_max <= 1")
    if max_channels <= 0:
        raise ValueError("max_channels must be positive")
    if channel_start < 0:
        raise ValueError("channel_start must be non-negative")
    if strength <= 0:
        raise ValueError("strength must be positive")
    if not bits or len(bits) % 8 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must be a non-empty byte-aligned binary sequence")

    bits = list(bits)
    patch = _center_patch(latent, center_ratio)
    _, channel_count, height, width = patch.shape
    if channel_start >= channel_count:
        raise ValueError(
            f"channel_start {channel_start} is outside latent with "
            f"{channel_count} channels"
        )
    used_channels = min(max_channels, channel_count - channel_start, len(bits))
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
    for local_channel in range(used_channels):
        channel = channel_start + local_channel
        bit_count = base_count + (1 if local_channel < remainder else 0)
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
    denoised_4d, was_5d = image_latent_4d(denoised)
    original_dtype = denoised.dtype
    # ComfyUI may call samplers under `torch.inference_mode()`. Merely enabling
    # gradients does not turn inference tensors back into autograd tensors, so
    # disable inference mode locally before cloning the tiny latent to fp32.
    with torch.inference_mode(False), torch.enable_grad():
        candidate = denoised_4d.detach().float().clone().requires_grad_(True)
        before = _frequency_loss(candidate, layout)
        gradient = torch.autograd.grad(before, candidate)[0]
        guided = candidate - guidance_scale * gradient
        after = _frequency_loss(guided, layout)
    guided = guided.detach().to(original_dtype)
    if was_5d:
        guided = guided.unsqueeze(2)
    return guided, before.detach(), after.detach()


@torch.no_grad()
def extract_bits(latent: torch.Tensor, layout: WatermarkLayout) -> DetectionResult:
    """Read keyed frequency signs and majority-vote each repeated payload bit."""
    latent, _ = image_latent_4d(latent)
    transformed = frequency_transform(_center_patch(latent[:1], layout.center_ratio))
    decoded = [0] * len(layout.message_bits)
    margins = [0.0] * len(layout.message_bits)

    for channel_layout in layout.channels:
        coordinate_tensor = torch.tensor(
            channel_layout.coordinates,
            device=transformed.device,
            dtype=torch.long,
        )
        votes = (
            transformed[
                0,
                channel_layout.channel,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
            ]
            .gt(0)
            .reshape(channel_layout.repetitions, channel_layout.bit_count)
        )
        ones = votes.sum(dim=0)
        channel_decoded = ones.gt(channel_layout.repetitions // 2).to(torch.uint8)
        channel_margins = (
            (2 * ones - channel_layout.repetitions).abs().float()
            / channel_layout.repetitions
        )
        start = channel_layout.bit_offset
        stop = start + channel_layout.bit_count
        decoded[start:stop] = channel_decoded.cpu().tolist()
        margins[start:stop] = channel_margins.cpu().tolist()

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


@torch.no_grad()
def detect_payload(
    latent: torch.Tensor,
    secret_key: str,
    dct_min: float = 0.15,
    dct_max: float = 0.45,
    max_channels: int = 8,
    center_ratio: float = 1.0,
    max_watermark_bytes: int = 64,
    channel_start: int = 0,
) -> DetectionResult:
    """Blindly detect a self-describing elastic ECC frame.

    The detector does not receive the expected watermark. It enumerates frame
    lengths and accepts only a candidate whose RS parity, embedded byte length,
    and UTF-8 payload all validate.
    """
    if not 1 <= max_watermark_bytes <= MAX_MESSAGE_BYTES:
        raise ValueError(
            f"max_watermark_bytes must be in [1, {MAX_MESSAGE_BYTES}]"
        )
    last_result: DetectionResult | None = None
    for payload_bytes in range(1, max_watermark_bytes + 1):
        frame_bits = [0] * (frame_bytes_for_payload_length(payload_bytes) * 8)
        try:
            layout = build_layout_for_bits(
                latent,
                frame_bits,
                secret_key,
                dct_min=dct_min,
                dct_max=dct_max,
                max_channels=max_channels,
                strength=1.0,
                center_ratio=center_ratio,
                channel_start=channel_start,
            )
        except ValueError as error:
            if "payload needs" in str(error):
                break
            raise
        result = extract_bits(latent, layout)
        if result.ecc_valid:
            return result
        last_result = result
    if last_result is None:
        raise ValueError("selected latent frequency band cannot hold a watermark frame")
    # Returning the longest attempted frame makes failed detections reproducible
    # and lets callers that intentionally set a payload bound inspect its raw bits.
    return last_result
