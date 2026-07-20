"""ComfyUI node adapters for GROW DiT watermarking and detection."""

from __future__ import annotations

import math

import torch

from .grow_dit.core import build_layout, extract_bits
from .grow_dit.sampler import GrowSamplerWrapper, GrowSettings


def _settings(
    message,
    secret_key,
    strength,
    guidance_scale,
    start_ratio,
    dct_min,
    dct_max,
    max_channels,
    center_ratio,
) -> GrowSettings:
    settings = GrowSettings(
        message=message,
        secret_key=secret_key,
        strength=strength,
        guidance_scale=guidance_scale,
        start_ratio=start_ratio,
        dct_min=dct_min,
        dct_max=dct_max,
        max_channels=max_channels,
        center_ratio=center_ratio,
    )
    settings.validate()
    return settings


class GROWDiTSampler:
    """Wrap any ComfyUI sampler with GROW x0 frequency guidance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "message": ("STRING", {"default": "zhangp365123456"}),
                "secret_key": ("STRING", {"default": "watermark"}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0001, "max": 2.0, "step": 0.0001},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 2000.0, "min": 0.01, "max": 10000.0, "step": 0.01},
                ),
                "start_ratio": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01},
                ),
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
                "center_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05},
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
                "message": ("STRING", {"default": "zhangp365123456"}),
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
                "center_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "FLOAT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "decoded_message",
        "exact_match",
        "bit_accuracy",
        "correct_bits",
        "total_bits",
        "min_vote_margin",
    )
    FUNCTION = "detect"
    CATEGORY = "GROW/watermark"
    DESCRIPTION = "VAE-encodes an image and extracts the keyed GROW payload."
    OUTPUT_NODE = True

    def detect(
        self,
        image,
        vae,
        message,
        secret_key,
        dct_min,
        dct_max,
        max_channels,
        center_ratio,
    ):
        if image.ndim != 4:
            raise ValueError("image must be a ComfyUI [B,H,W,C] tensor")
        latent = vae.encode(image)
        layout = build_layout(
            latent,
            message=message,
            secret_key=secret_key,
            dct_min=dct_min,
            dct_max=dct_max,
            max_channels=max_channels,
            strength=1.0,
            center_ratio=center_ratio,
        )
        result = extract_bits(latent, layout)
        values = (
            result.decoded_message,
            result.exact_match,
            result.bit_accuracy,
            result.correct_bits,
            result.total_bits,
            result.min_vote_margin,
        )
        summary = (
            f"decoded={result.decoded_message!r}; exact={result.exact_match}; "
            f"bits={result.correct_bits}/{result.total_bits}; "
            f"accuracy={result.bit_accuracy:.6f}; "
            f"min_vote_margin={result.min_vote_margin:.6f}"
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
    "GROWDiTSampler": GROWDiTSampler,
    "GROWWatermarkDetect": GROWWatermarkDetect,
    "GROWImagePSNR": GROWImagePSNR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GROWDiTSampler": "GROW DiT Sampler",
    "GROWWatermarkDetect": "GROW Watermark Detect",
    "GROWImagePSNR": "GROW Image PSNR",
}
