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
        self.assertEqual(prompt["18"]["inputs"]["sampler"], ["16", 0])
        self.assertEqual(prompt["21"]["class_type"], "GROWWatermarkDetect")
        self.assertEqual(prompt["16"]["inputs"]["strength"], 0.005)
        self.assertEqual(prompt["16"]["inputs"]["guidance_scale"], 50.0)


if __name__ == "__main__":
    unittest.main()
