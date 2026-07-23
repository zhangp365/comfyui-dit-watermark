"""ComfyUI node adapters for GROW DiT watermarking and detection."""

from __future__ import annotations

import math

import torch

from .grow_dit.config import GrowConfig
from .grow_dit.core import DetectionResult, detect_payload
from .grow_dit.robust import ROBUST_MODES, alignment_candidates
from .grow_dit.sampler import GrowSamplerWrapper, GrowSettings


def _settings(
    watermark,
    strength,
    guidance_scale,
    start_ratio,
    config: GrowConfig,
) -> GrowSettings:
    config.validate()
    settings = GrowSettings(
        watermark=watermark,
        secret_key=config.secret_key,
        strength=strength,
        guidance_scale=guidance_scale,
        start_ratio=start_ratio,
        dct_min=config.dct_min,
        dct_max=config.dct_max,
        max_channels=config.max_channels,
        channel_start=config.channel_start,
        center_ratio=config.center_ratio,
    )
    settings.validate()
    return settings


class GROWWatermarkConfig:
    """Build one validated layout config for embedding and detection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "secret_key": ("STRING", {"default": "watermark"}),
                "dct_min": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 0.95, "step": 0.01},
                ),
                "dct_max": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "max_channels": (
                    "INT",
                    {"default": 8, "min": 1, "max": 256, "step": 1},
                ),
                "channel_start": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "First latent channel in the contiguous profile.",
                    },
                ),
                "center_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("GROW_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"
    CATEGORY = "GROW/watermark"
    DESCRIPTION = "Shares one keyed frequency layout between embedding and detection."

    def build(
        self,
        secret_key,
        dct_min,
        dct_max,
        max_channels,
        channel_start,
        center_ratio,
    ):
        config = GrowConfig(
            secret_key=secret_key,
            dct_min=dct_min,
            dct_max=dct_max,
            max_channels=max_channels,
            channel_start=channel_start,
            center_ratio=center_ratio,
        )
        config.validate()
        return (config,)


class GROWDiTSampler:
    """Wrap any ComfyUI sampler with GROW x0 frequency guidance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "config": ("GROW_CONFIG",),
                "watermark": ("STRING", {"default": "zhangp36512345"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.01,
                        "max": 5.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 4000.0,
                        "min": 1.0,
                        "max": 20000.0,
                        "step": 1.0,
                        "round": 1.0,
                    },
                ),
                "start_ratio": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "wrap"
    CATEGORY = "GROW/watermark"
    DESCRIPTION = "Applies GROW progressive frequency guidance to DiT x0 predictions."

    def wrap(self, sampler, **kwargs):
        return (GrowSamplerWrapper(sampler, _settings(**kwargs)),)


class GROWWatermarkDetect:
    """Detect a GROW payload by VAE encoding, without diffusion inversion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "config": ("GROW_CONFIG",),
                "max_watermark_bytes": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 250,
                        "step": 1,
                        "tooltip": (
                            "Blind detection checks candidate UTF-8 byte lengths up to "
                            "this value; a larger limit is slower."
                        ),
                    },
                ),
                "robust_mode": (ROBUST_MODES, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "decoded_message",
        "ecc_valid",
        "corrected_symbols",
        "min_vote_margin",
        "raw_codeword_hex",
    )
    FUNCTION = "detect"
    CATEGORY = "GROW/watermark"
    DESCRIPTION = "VAE-encodes an image and extracts the keyed GROW payload."
    OUTPUT_NODE = True

    def detect(
        self,
        image,
        vae,
        config: GrowConfig,
        max_watermark_bytes,
        robust_mode,
    ):
        if image.ndim != 4:
            raise ValueError("image must be a ComfyUI [B,H,W,C] tensor")
        config.validate()
        result: DetectionResult | None = None
        alignment = "identity"
        for candidate_alignment, candidate in alignment_candidates(image, robust_mode):
            candidate_result = detect_payload(
                vae.encode(candidate),
                secret_key=config.secret_key,
                dct_min=config.dct_min,
                dct_max=config.dct_max,
                max_channels=config.max_channels,
                center_ratio=config.center_ratio,
                max_watermark_bytes=max_watermark_bytes,
                channel_start=config.channel_start,
            )
            if result is None or candidate_result.min_vote_margin > result.min_vote_margin:
                result = candidate_result
                alignment = candidate_alignment
            if candidate_result.ecc_valid:
                result = candidate_result
                alignment = candidate_alignment
                break
        assert result is not None
        values = (
            result.decoded_message,
            result.ecc_valid,
            result.corrected_symbols,
            result.min_vote_margin,
            result.raw_codeword_hex,
        )
        summary = (
            f"decoded={result.decoded_message!r}; ecc_valid={result.ecc_valid}; "
            f"corrected_symbols={result.corrected_symbols}; "
            f"min_vote_margin={result.min_vote_margin:.6f}; "
            f"alignment={alignment}; "
            f"raw_codeword_hex={result.raw_codeword_hex}"
        )
        return {"ui": {"text": [summary]}, "result": values}


class GROWImagePSNR:
    """Measure RGB PSNR between a clean reference and watermarked image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reference": ("IMAGE",), "watermarked": ("IMAGE",)}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("psnr_db",)
    FUNCTION = "calculate"
    CATEGORY = "GROW/metrics"
    OUTPUT_NODE = True

    def calculate(self, reference, watermarked):
        if reference.shape != watermarked.shape:
            raise ValueError(
                f"image shapes must match, got {tuple(reference.shape)} and "
                f"{tuple(watermarked.shape)}"
            )
        mse = torch.mean((reference.float() - watermarked.float()) ** 2).item()
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        return {
            "ui": {"text": [f"PSNR={psnr:.6f} dB"]},
            "result": (psnr,),
        }


NODE_CLASS_MAPPINGS = {
    "GROWWatermarkConfig": GROWWatermarkConfig,
    "GROWDiTSampler": GROWDiTSampler,
    "GROWWatermarkDetect": GROWWatermarkDetect,
    "GROWImagePSNR": GROWImagePSNR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GROWWatermarkConfig": "GROW Watermark Config",
    "GROWDiTSampler": "GROW DiT Sampler",
    "GROWWatermarkDetect": "GROW Watermark Detect",
    "GROWImagePSNR": "GROW Image PSNR",
}
