from __future__ import annotations

import unittest

import torch

from grow_dit.sampler import GrowDenoiserProxy, GrowSettings


class FakeDenoiser:
    def __call__(self, x, sigma, **kwargs):
        return x * 0.5


class SamplerProxyTests(unittest.TestCase):
    def test_proxy_only_guides_after_start_ratio(self) -> None:
        settings = GrowSettings(
            message="grow",
            secret_key="secret",
            strength=0.02,
            guidance_scale=50.0,
            start_ratio=0.5,
            max_channels=8,
        )
        proxy = GrowDenoiserProxy(
            FakeDenoiser(), settings, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
        )
        x = torch.randn(1, 32, 64, 64)
        for sigma in (4.0, 3.0, 2.0, 1.0):
            proxy(x, torch.tensor([sigma]))
        self.assertEqual(proxy.guided_calls, 2)
        self.assertIsNotNone(proxy.last_loss_before)
        self.assertLess(proxy.last_loss_after, proxy.last_loss_before)

    def test_invalid_settings_fail_before_sampling(self) -> None:
        settings = GrowSettings(message="", secret_key="key")
        with self.assertRaisesRegex(ValueError, "message must not be empty"):
            GrowDenoiserProxy(FakeDenoiser(), settings, torch.tensor([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
