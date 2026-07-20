"""Transparent ComfyUI sampler wrapper for progressive GROW guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import WatermarkLayout, build_layout, guide_denoised


@dataclass(frozen=True)
class GrowSettings:
    message: str = "watermark"
    secret_key: str = "watermark"
    strength: float = 1.2
    guidance_scale: float = 4000.0
    start_ratio: float = 0.5
    dct_min: float = 0.15
    dct_max: float = 0.45
    max_channels: int = 8
    center_ratio: float = 1.0

    def validate(self) -> None:
        if not self.message:
            raise ValueError("message must not be empty")
        if not self.secret_key:
            raise ValueError("secret_key must not be empty")
        if self.strength <= 0:
            raise ValueError("strength must be positive")
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be positive")
        if not 0.0 <= self.start_ratio < 1.0:
            raise ValueError("start_ratio must be in [0, 1)")
        if not 0.0 <= self.dct_min < self.dct_max <= 1.0:
            raise ValueError("require 0 <= dct_min < dct_max <= 1")
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if not 0.0 < self.center_ratio <= 1.0:
            raise ValueError("center_ratio must be in (0, 1]")


class GrowDenoiserProxy:
    """Intercept denoised x0 predictions while delegating every other attribute."""

    def __init__(
        self,
        denoiser,
        settings: GrowSettings,
        sigmas: torch.Tensor,
    ) -> None:
        settings.validate()
        self._denoiser = denoiser
        self.settings = settings
        self.sigmas = sigmas.detach()
        self.total_steps = max(1, len(sigmas) - 1)
        self.start_step = int(self.total_steps * settings.start_ratio)
        self.guided_calls = 0
        self.last_loss_before: float | None = None
        self.last_loss_after: float | None = None
        self._layout: WatermarkLayout | None = None
        self._layout_signature: tuple[tuple[int, ...], torch.device] | None = None

    def __getattr__(self, name):
        return getattr(self._denoiser, name)

    def _step_index(self, sigma: torch.Tensor) -> int:
        schedule = self.sigmas[:-1].to(device=sigma.device, dtype=torch.float32)
        value = sigma.reshape(-1)[0].float()
        return int(torch.argmin(torch.abs(schedule - value)).item())

    def _get_layout(self, denoised: torch.Tensor) -> WatermarkLayout:
        signature = (tuple(denoised.shape), denoised.device)
        if self._layout is None or self._layout_signature != signature:
            self._layout = build_layout(
                denoised,
                message=self.settings.message,
                secret_key=self.settings.secret_key,
                dct_min=self.settings.dct_min,
                dct_max=self.settings.dct_max,
                max_channels=self.settings.max_channels,
                strength=self.settings.strength,
                center_ratio=self.settings.center_ratio,
            )
            self._layout_signature = signature
        return self._layout

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs):
        denoised = self._denoiser(x, sigma, **kwargs)
        if not isinstance(denoised, torch.Tensor) or denoised.ndim != 4:
            raise ValueError(
                "GROW requires the sampler denoiser to return a 4D [B,C,H,W] tensor"
            )
        if self._step_index(sigma) < self.start_step:
            return denoised
        guided, before, after = guide_denoised(
            denoised, self._get_layout(denoised), self.settings.guidance_scale
        )
        self.guided_calls += 1
        self.last_loss_before = float(before.item())
        self.last_loss_after = float(after.item())
        return guided


class GrowSamplerWrapper:
    """ComfyUI `SAMPLER` object that delegates to a selected base sampler."""

    def __init__(self, base_sampler, settings: GrowSettings) -> None:
        settings.validate()
        self.base_sampler = base_sampler
        self.settings = settings

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        proxy = GrowDenoiserProxy(model_wrap, self.settings, sigmas)
        return self.base_sampler.sample(
            proxy,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    def max_denoise(self, model_wrap, sigmas):
        return self.base_sampler.max_denoise(model_wrap, sigmas)
