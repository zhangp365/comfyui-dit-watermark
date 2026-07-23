from __future__ import annotations

import unittest

from scripts.comfy_api import build_flux2_prompt


class ApiPromptTests(unittest.TestCase):
    def test_clean_prompt_uses_base_sampler(self) -> None:
        prompt = build_flux2_prompt(watermarked=False, filename_prefix="clean")
        self.assertNotIn("16", prompt)
        self.assertEqual(prompt["18"]["inputs"]["sampler"], ["15", 0])
        self.assertNotIn("21", prompt)

    def test_watermarked_prompt_adds_sampler_and_detector(self) -> None:
        prompt = build_flux2_prompt(
            watermarked=True,
            filename_prefix="marked",
            strength=0.005,
            guidance_scale=50.0,
        )
        self.assertEqual(prompt["16"]["class_type"], "GROWDiTSampler")
        self.assertEqual(prompt["23"]["class_type"], "GROWWatermarkConfig")
        self.assertEqual(prompt["16"]["inputs"]["watermark"], "zhangp36512345")
        self.assertNotIn("message", prompt["16"]["inputs"])
        self.assertEqual(prompt["18"]["inputs"]["sampler"], ["16", 0])
        self.assertEqual(prompt["21"]["class_type"], "GROWWatermarkDetect")
        self.assertEqual(prompt["16"]["inputs"]["config"], ["23", 0])
        self.assertEqual(prompt["21"]["inputs"]["config"], ["23", 0])
        self.assertNotIn("message", prompt["21"]["inputs"])
        self.assertEqual(prompt["16"]["inputs"]["strength"], 0.005)
        self.assertEqual(prompt["16"]["inputs"]["guidance_scale"], 50.0)
        self.assertEqual(prompt["16"]["inputs"]["start_ratio"], 0.0)
        self.assertEqual(prompt["21"]["inputs"]["max_watermark_bytes"], 64)
        self.assertEqual(prompt["21"]["inputs"]["robust_mode"], "none")

    def test_channel_profile_is_shared_by_sampler_and_detector(self) -> None:
        prompt = build_flux2_prompt(
            watermarked=True,
            filename_prefix="profile_2",
            channel_start=8,
            max_channels=8,
        )
        self.assertEqual(prompt["23"]["inputs"]["channel_start"], 8)
        self.assertEqual(prompt["23"]["inputs"]["max_channels"], 8)
        self.assertNotIn("channel_start", prompt["16"]["inputs"])
        self.assertNotIn("channel_start", prompt["21"]["inputs"])

    def test_img2img_prompt_uses_input_latent_and_low_sigmas(self) -> None:
        prompt = build_flux2_prompt(
            watermarked=True,
            filename_prefix="identity",
            scheduler_steps=20,
            img2img_denoise=0.1,
        )
        self.assertEqual(prompt["14"]["inputs"]["steps"], 20)
        self.assertEqual(prompt["18"]["inputs"]["latent_image"], ["9", 0])
        self.assertEqual(prompt["18"]["inputs"]["sigmas"], ["22", 1])
        self.assertEqual(prompt["22"]["class_type"], "SplitSigmasDenoise")


if __name__ == "__main__":
    unittest.main()
