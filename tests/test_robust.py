from __future__ import annotations

import unittest

import torch

from grow_dit.robust import alignment_candidates


class RobustAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = torch.zeros((1, 32, 48, 3))
        self.image[:, 8:24, 12:36, :] = 1.0

    def test_none_only_returns_identity(self) -> None:
        candidates = list(alignment_candidates(self.image, "none"))
        self.assertEqual([name for name, _ in candidates], ["identity"])
        self.assertIs(candidates[0][1], self.image)

    def test_rotation_candidates_preserve_shape_and_include_inverse_angles(self) -> None:
        candidates = list(alignment_candidates(self.image, "rotation"))
        names = [name for name, _ in candidates]
        self.assertEqual(names[0], "identity")
        self.assertIn("rotation(angle=10,scale=1)", names)
        self.assertIn("rotation(angle=-10,scale=1)", names)
        self.assertTrue(all(candidate.shape == self.image.shape for _, candidate in candidates))

    def test_crop_scale_candidates_preserve_shape(self) -> None:
        candidates = list(alignment_candidates(self.image, "crop_scale"))
        names = [name for name, _ in candidates]
        self.assertIn("crop_scale(scale=0.75)", names)
        self.assertIn("crop_scale(scale=1.2)", names)
        self.assertTrue(all(candidate.shape == self.image.shape for _, candidate in candidates))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "robust mode"):
            list(alignment_candidates(self.image, "unknown"))


if __name__ == "__main__":
    unittest.main()
