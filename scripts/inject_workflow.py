"""Insert GROW sampler and detector nodes into a ComfyUI workflow JSON."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_WIDGETS = [
    "watermark",
    "watermark",
    1.2,
    4000.0,
    0.5,
    0.15,
    0.45,
    8,
    1.0,
]


def _graphs(workflow: dict[str, Any]):
    yield workflow
    definitions = workflow.get("definitions", {})
    for subgraph in definitions.get("subgraphs", []):
        yield subgraph


def _node(graph: dict[str, Any], node_id: int) -> dict[str, Any]:
    return next(node for node in graph["nodes"] if node["id"] == node_id)


def _append_output_link(node: dict[str, Any], slot: int, link_id: int) -> None:
    links = node["outputs"][slot].setdefault("links", [])
    if link_id not in links:
        links.append(link_id)


def _state_max(graph: dict[str, Any], name: str, observed: int) -> int:
    state = graph.setdefault("state", {})
    current = int(state.get(name, observed))
    return max(current, observed)


def inject_graph(
    graph: dict[str, Any],
    message: str = "watermark",
    secret_key: str = "watermark",
    strength: float = 1.2,
    guidance_scale: float = 4000.0,
    start_ratio: float = 0.5,
    dct_min: float = 0.15,
    dct_max: float = 0.45,
    max_channels: int = 8,
    center_ratio: float = 1.0,
    add_detector: bool = True,
    identity_prompt: str = (
        "Keep the input image exactly unchanged. Preserve every pixel, color, "
        "texture, composition, and detail. Add only the invisible watermark."
    ),
) -> bool:
    if any(node.get("type") == "GROWDiTSampler" for node in graph.get("nodes", [])):
        return False

    sampler_links = [
        link
        for link in graph.get("links", [])
        if isinstance(link, dict)
        and link.get("type") == "SAMPLER"
        and _node(graph, link["origin_id"]).get("type") == "KSamplerSelect"
        and _node(graph, link["target_id"]).get("type") == "SamplerCustomAdvanced"
    ]
    if not sampler_links:
        return False
    if len(sampler_links) != 1:
        raise ValueError(f"expected one sampler link, found {len(sampler_links)}")

    node_ids = [int(node["id"]) for node in graph["nodes"] if int(node["id"]) >= 0]
    link_ids = [int(link["id"]) for link in graph["links"]]
    last_node = _state_max(graph, "lastNodeId", max(node_ids, default=0))
    last_link = _state_max(graph, "lastLinkId", max(link_ids, default=0))

    link = sampler_links[0]
    source = _node(graph, link["origin_id"])
    target = _node(graph, link["target_id"])
    grow_id = last_node + 1
    grow_to_sampler = last_link + 1
    source_pos = source.get("pos", [0, 0])
    target_pos = target.get("pos", [400, 0])
    grow_pos = [
        (float(source_pos[0]) + float(target_pos[0])) / 2,
        (float(source_pos[1]) + float(target_pos[1])) / 2,
    ]

    link["target_id"] = grow_id
    link["target_slot"] = 0
    for input_socket in target.get("inputs", []):
        if input_socket.get("name") == "sampler":
            input_socket["link"] = grow_to_sampler

    graph["links"].append(
        {
            "id": grow_to_sampler,
            "origin_id": grow_id,
            "origin_slot": 0,
            "target_id": target["id"],
            "target_slot": 2,
            "type": "SAMPLER",
        }
    )
    widgets = [
        message,
        secret_key,
        strength,
        guidance_scale,
        start_ratio,
        dct_min,
        dct_max,
        max_channels,
        center_ratio,
    ]
    grow_node = {
        "id": grow_id,
        "type": "GROWDiTSampler",
        "pos": grow_pos,
        "size": [330, 330],
        "flags": {},
        "order": max((node.get("order", 0) for node in graph["nodes"]), default=0) + 1,
        "mode": 0,
        "inputs": [
            {"name": "sampler", "type": "SAMPLER", "link": link["id"]}
        ],
        "outputs": [
            {"name": "sampler", "type": "SAMPLER", "links": [grow_to_sampler]}
        ],
        "properties": {
            "Node name for S&R": "GROWDiTSampler",
            "cnr_id": "comfyui-dit-watermark",
            "ver": "0.2.0",
        },
        "widgets_values": widgets,
    }
    graph["nodes"].append(grow_node)
    graph["state"]["lastNodeId"] = grow_id
    graph["state"]["lastLinkId"] = grow_to_sampler

    if add_detector:
        vae_decoders = [node for node in graph["nodes"] if node.get("type") == "VAEDecode"]
        vae_loaders = [node for node in graph["nodes"] if node.get("type") == "VAELoader"]
        if len(vae_decoders) != 1 or len(vae_loaders) != 1:
            raise ValueError("detector injection requires one VAEDecode and one VAELoader")
        decoder, vae_loader = vae_decoders[0], vae_loaders[0]
        detector_id = grow_id + 1
        image_link = grow_to_sampler + 1
        vae_link = grow_to_sampler + 2
        decoder_pos = decoder.get("pos", [1200, 200])
        detector = {
            "id": detector_id,
            "type": "GROWWatermarkDetect",
            "pos": [float(decoder_pos[0]) + 320, float(decoder_pos[1]) + 180],
            "size": [350, 310],
            "flags": {},
            "order": grow_node["order"] + 1,
            "mode": 0,
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": image_link},
                {"name": "vae", "type": "VAE", "link": vae_link},
            ],
            "outputs": [
                {"name": "decoded_message", "type": "STRING", "links": []},
                {"name": "ecc_valid", "type": "BOOLEAN", "links": []},
                {"name": "corrected_symbols", "type": "INT", "links": []},
                {"name": "min_vote_margin", "type": "FLOAT", "links": []},
                {"name": "raw_codeword_hex", "type": "STRING", "links": []},
            ],
            "properties": {
                "Node name for S&R": "GROWWatermarkDetect",
                "cnr_id": "comfyui-dit-watermark",
                "ver": "0.2.0",
            },
            "widgets_values": [
                secret_key,
                dct_min,
                dct_max,
                max_channels,
                center_ratio,
            ],
        }
        graph["nodes"].append(detector)
        graph["links"].extend(
            [
                {
                    "id": image_link,
                    "origin_id": decoder["id"],
                    "origin_slot": 0,
                    "target_id": detector_id,
                    "target_slot": 0,
                    "type": "IMAGE",
                },
                {
                    "id": vae_link,
                    "origin_id": vae_loader["id"],
                    "origin_slot": 0,
                    "target_id": detector_id,
                    "target_slot": 1,
                    "type": "VAE",
                },
            ]
        )
        _append_output_link(decoder, 0, image_link)
        _append_output_link(vae_loader, 0, vae_link)
        graph["state"]["lastNodeId"] = detector_id
        graph["state"]["lastLinkId"] = vae_link

    text_nodes = [node for node in graph["nodes"] if node.get("type") == "CLIPTextEncode"]
    if len(text_nodes) == 1:
        text_nodes[0]["widgets_values"] = [identity_prompt]
    return True


def inject_workflow(workflow: dict[str, Any], **kwargs) -> dict[str, Any]:
    result = copy.deepcopy(workflow)
    injected = [graph for graph in _graphs(result) if inject_graph(graph, **kwargs)]
    if len(injected) != 1:
        raise ValueError(f"expected to inject one graph, injected {len(injected)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--message", default="watermark")
    parser.add_argument("--secret-key", default="watermark")
    parser.add_argument("--strength", type=float, default=1.2)
    parser.add_argument("--guidance-scale", type=float, default=4000.0)
    parser.add_argument("--start-ratio", type=float, default=0.5)
    parser.add_argument("--dct-min", type=float, default=0.15)
    parser.add_argument("--dct-max", type=float, default=0.45)
    parser.add_argument("--max-channels", type=int, default=8)
    parser.add_argument("--center-ratio", type=float, default=1.0)
    parser.add_argument("--without-detector", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        backup = source.with_suffix(source.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(source, backup)
    workflow = json.loads(source.read_text(encoding="utf-8"))
    updated = inject_workflow(
        workflow,
        message=args.message,
        secret_key=args.secret_key,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        start_ratio=args.start_ratio,
        dct_min=args.dct_min,
        dct_max=args.dct_max,
        max_channels=args.max_channels,
        center_ratio=args.center_ratio,
        add_detector=not args.without_detector,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
