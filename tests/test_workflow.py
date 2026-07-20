from __future__ import annotations

import unittest

from scripts.inject_workflow import inject_workflow


def node(node_id, node_type, inputs=None, outputs=None):
    return {
        "id": node_id,
        "type": node_type,
        "pos": [node_id * 10, 100],
        "order": node_id,
        "inputs": inputs or [],
        "outputs": outputs or [],
    }


class WorkflowInjectionTests(unittest.TestCase):
    def test_sampler_link_is_split_and_detector_is_connected(self) -> None:
        graph = {
            "state": {"lastNodeId": 4, "lastLinkId": 12},
            "nodes": [
                node(1, "KSamplerSelect", outputs=[{"links": [10]}]),
                node(2, "SamplerCustomAdvanced", inputs=[
                    {"name": "noise", "link": None},
                    {"name": "guider", "link": None},
                    {"name": "sampler", "link": 10},
                ]),
                node(3, "VAEDecode", outputs=[{"links": [11]}]),
                node(4, "VAELoader", outputs=[{"links": [12]}]),
            ],
            "links": [
                {"id": 10, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 2, "type": "SAMPLER"},
                {"id": 11, "origin_id": 3, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                {"id": 12, "origin_id": 4, "origin_slot": 0, "target_id": 3, "target_slot": 1, "type": "VAE"},
            ],
        }
        updated = inject_workflow(graph)
        grow = next(item for item in updated["nodes"] if item["type"] == "GROWDiTSampler")
        detector = next(item for item in updated["nodes"] if item["type"] == "GROWWatermarkDetect")
        old_link = next(item for item in updated["links"] if item["id"] == 10)
        new_link = next(item for item in updated["links"] if item["origin_id"] == grow["id"])
        self.assertEqual(old_link["target_id"], grow["id"])
        self.assertEqual(new_link["target_id"], 2)
        self.assertEqual(detector["inputs"][0]["link"], 14)
        self.assertEqual(detector["inputs"][1]["link"], 15)
        self.assertEqual(detector["widgets_values"][0], "watermark")
        self.assertEqual(len(detector["widgets_values"]), 8)
        self.assertEqual(detector["widgets_values"][4], 0)
        self.assertEqual(detector["widgets_values"][-2:], [64, "none"])
        self.assertEqual(grow["widgets_values"][0], "zhangp36512345")
        self.assertEqual(grow["widgets_values"][4], 0.0)
        self.assertEqual(grow["widgets_values"][8], 0)

    def test_existing_grow_node_is_not_duplicated(self) -> None:
        workflow = {"nodes": [{"id": 1, "type": "GROWDiTSampler"}], "links": []}
        with self.assertRaisesRegex(ValueError, "injected 0"):
            inject_workflow(workflow)

    def test_legacy_root_links_are_skipped_before_object_link_subgraph(self) -> None:
        subgraph = {
            "state": {"lastNodeId": 4, "lastLinkId": 12},
            "nodes": [
                node(1, "KSamplerSelect", outputs=[{"links": [10]}]),
                node(2, "SamplerCustomAdvanced", inputs=[
                    {"name": "sampler", "link": 10}
                ]),
                node(3, "VAEDecode", outputs=[{"links": [11]}]),
                node(4, "VAELoader", outputs=[{"links": [12]}]),
            ],
            "links": [
                {"id": 10, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 2, "type": "SAMPLER"},
                {"id": 11, "origin_id": 3, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                {"id": 12, "origin_id": 4, "origin_slot": 0, "target_id": 3, "target_slot": 1, "type": "VAE"},
            ],
        }
        workflow = {
            "nodes": [],
            "links": [[1, 10, 0, 11, 0, "IMAGE"]],
            "definitions": {"subgraphs": [subgraph]},
        }
        updated = inject_workflow(workflow)
        types = [item["type"] for item in updated["definitions"]["subgraphs"][0]["nodes"]]
        self.assertEqual(types.count("GROWDiTSampler"), 1)
        self.assertEqual(types.count("GROWWatermarkDetect"), 1)


if __name__ == "__main__":
    unittest.main()
