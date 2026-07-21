from __future__ import annotations

import unittest

import torch

from grow_dit.codec import bits_to_message, message_to_bits
from grow_dit.core import build_layout, detect_payload, extract_bits, guide_denoised


class CodecTests(unittest.TestCase):
    def test_utf8_round_trip(self) -> None:
        message = "水印-grow"
        self.assertEqual(bits_to_message(message_to_bits(message)), message)

    def test_empty_message_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            message_to_bits("")


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.latent = torch.randn(1, 32, 64, 64)

    def test_layout_is_deterministic_and_keyed(self) -> None:
        first = build_layout(self.latent, "grow", "secret-a")
        second = build_layout(self.latent, "grow", "secret-a")
        other = build_layout(self.latent, "grow", "secret-b")
        self.assertTrue(torch.equal(first.mask, second.mask))
        self.assertTrue(torch.equal(first.target, second.target))
        self.assertNotEqual(first.channels[0].coordinates, other.channels[0].coordinates)
        self.assertTrue(all(channel.repetitions % 2 == 1 for channel in first.channels))

    def test_layout_uses_requested_contiguous_channel_profile(self) -> None:
        layout = build_layout(
            self.latent,
            "grow",
            "secret",
            max_channels=8,
            channel_start=8,
        )
        self.assertEqual(
            [channel.channel for channel in layout.channels], list(range(8, 16))
        )
        self.assertFalse(layout.mask[:, :8].any().item())
        self.assertFalse(layout.mask[:, 16:].any().item())

    def test_layout_rejects_channel_start_outside_latent(self) -> None:
        with self.assertRaisesRegex(ValueError, "channel_start"):
            build_layout(
                self.latent,
                "grow",
                "secret",
                max_channels=8,
                channel_start=32,
            )

    def test_guidance_reduces_keyed_frequency_loss(self) -> None:
        layout = build_layout(
            self.latent, "grow", "secret", 0.15, 0.45, 8, 0.01
        )
        guided, before, after = guide_denoised(self.latent, layout, 50.0)
        self.assertLess(after.item(), before.item())
        self.assertEqual(guided.dtype, self.latent.dtype)
        self.assertEqual(guided.shape, self.latent.shape)

    def test_guidance_works_inside_comfyui_inference_mode(self) -> None:
        with torch.inference_mode():
            inference_latent = self.latent.clone()
            layout = build_layout(
                inference_latent, "grow", "secret", 0.15, 0.45, 8, 0.01
            )
            guided, before, after = guide_denoised(inference_latent, layout, 50.0)
        self.assertLess(after.item(), before.item())
        self.assertFalse(guided.is_inference())

    def test_qwen_single_frame_5d_latent_round_trip(self) -> None:
        qwen_latent = self.latent.unsqueeze(2)
        layout = build_layout(
            qwen_latent, "grow", "secret", 0.15, 0.45, 8, 0.01
        )
        guided, before, after = guide_denoised(qwen_latent, layout, 50.0)
        self.assertEqual(guided.shape, qwen_latent.shape)
        self.assertLess(after.item(), before.item())

    def test_multiframe_5d_latent_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-frame"):
            build_layout(self.latent.unsqueeze(2).expand(-1, -1, 2, -1, -1), "grow", "secret")

    def test_extraction_recovers_a_strongly_guided_payload(self) -> None:
        layout = build_layout(
            self.latent, "grow", "secret", 0.10, 0.50, 8, 0.1
        )
        guided = self.latent
        for _ in range(20):
            guided, _, _ = guide_denoised(guided, layout, 500.0)
        result = extract_bits(guided, layout)
        self.assertTrue(result.ecc_valid)
        self.assertEqual(result.decoded_message, "grow")
        self.assertLessEqual(result.corrected_symbols, 2)

    def test_payload_capacity_is_validated(self) -> None:
        small = torch.zeros(1, 1, 8, 8)
        with self.assertRaisesRegex(ValueError, "payload needs"):
            build_layout(small, "ok", "key", 0.1, 0.2, 1, 0.1)

    def test_blind_detection_enumerates_elastic_frame_lengths(self) -> None:
        layout = build_layout(
            self.latent, "grow", "secret", 0.10, 0.50, 8, 0.1
        )
        guided = self.latent
        for _ in range(20):
            guided, _, _ = guide_denoised(guided, layout, 500.0)
        result = detect_payload(
            guided, "secret", 0.10, 0.50, 8, 1.0, max_watermark_bytes=8
        )
        self.assertTrue(result.ecc_valid)
        self.assertEqual(result.decoded_message, "grow")

    def test_blind_detection_bound_can_exclude_longer_payload(self) -> None:
        layout = build_layout(
            self.latent, "longer", "secret", 0.10, 0.50, 8, 0.1
        )
        guided = self.latent
        for _ in range(20):
            guided, _, _ = guide_denoised(guided, layout, 500.0)
        result = detect_payload(
            guided, "secret", 0.10, 0.50, 8, 1.0, max_watermark_bytes=4
        )
        self.assertFalse(result.ecc_valid)


if __name__ == "__main__":
    unittest.main()
