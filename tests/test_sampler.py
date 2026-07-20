from __future__ import annotations

import unittest
import warnings

import torch

from grow_dit.sampler import GrowDenoiserProxy, GrowSettings


class FakeDenoiser:
    def __call__(self, x, sigma, **kwargs):
        return x * 0.5


class SamplerProxyTests(unittest.TestCase):
    def test_default_guides_from_first_step(self) -> None:
        settings = GrowSettings(
            watermark="grow",
            secret_key="secret",
            strength=0.02,
            guidance_scale=50.0,
            max_channels=8,
        )
        proxy = GrowDenoiserProxy(
            FakeDenoiser(), settings, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
        )
        x = torch.randn(1, 32, 64, 64)
        for sigma in (4.0, 3.0, 2.0, 1.0):
            proxy(x, torch.tensor([sigma]))
        self.assertEqual(proxy.guided_calls, 4)

    def test_proxy_only_guides_after_start_ratio(self) -> None:
        settings = GrowSettings(
            watermark="grow",
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
        settings = GrowSettings(watermark="", secret_key="key")
        with self.assertRaisesRegex(ValueError, "watermark must not be empty"):
            GrowDenoiserProxy(FakeDenoiser(), settings, torch.tensor([1.0, 0.0]))

    def test_long_watermark_warns_about_robustness_and_detection_speed(self) -> None:
        settings = GrowSettings(watermark="x" * 33, secret_key="key")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            settings.validate()
        self.assertEqual(len(caught), 1)
        self.assertIn("reduce frequency repetitions", str(caught[0].message))

    def test_negative_channel_start_is_rejected(self) -> None:
        settings = GrowSettings(
            watermark="grow", secret_key="key", channel_start=-1
        )
        with self.assertRaisesRegex(ValueError, "channel_start"):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
