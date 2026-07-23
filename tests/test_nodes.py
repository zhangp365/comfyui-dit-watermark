from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_dit_watermark_test"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)

nodes = sys.modules[f"{PACKAGE_NAME}.nodes"]
GrowConfig = sys.modules[f"{PACKAGE_NAME}.grow_dit.config"].GrowConfig


class GrowConfigNodeTests(unittest.TestCase):
    def test_config_node_returns_validated_immutable_config(self) -> None:
        (config,) = nodes.GROWWatermarkConfig().build(
            "secret", 0.1, 0.5, 6, 2, 0.75
        )
        self.assertEqual(
            config,
            GrowConfig(
                secret_key="secret",
                dct_min=0.1,
                dct_max=0.5,
                max_channels=6,
                channel_start=2,
                center_ratio=0.75,
            ),
        )
        with self.assertRaisesRegex(ValueError, "dct_min"):
            nodes.GROWWatermarkConfig().build("secret", 0.5, 0.1, 6, 2, 0.75)

    def test_config_node_is_registered_with_grow_config_output(self) -> None:
        self.assertIs(
            nodes.NODE_CLASS_MAPPINGS["GROWWatermarkConfig"],
            nodes.GROWWatermarkConfig,
        )
        self.assertEqual(nodes.GROWWatermarkConfig.RETURN_TYPES, ("GROW_CONFIG",))

    def test_sampler_and_detector_require_only_the_shared_layout_input(self) -> None:
        shared_fields = {
            "secret_key",
            "dct_min",
            "dct_max",
            "max_channels",
            "channel_start",
            "center_ratio",
        }
        for node_class in (nodes.GROWDiTSampler, nodes.GROWWatermarkDetect):
            required = node_class.INPUT_TYPES()["required"]
            self.assertEqual(required["config"], ("GROW_CONFIG",))
            self.assertTrue(shared_fields.isdisjoint(required))


if __name__ == "__main__":
    unittest.main()
