"""Build and run deterministic Flux2 Klein clean/GROW ComfyUI API prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def build_flux2_prompt(
    *,
    watermarked: bool,
    filename_prefix: str,
    watermark: str = "zhangp36512345",
    secret_key: str = "watermark",
    strength: float = 1.2,
    guidance_scale: float = 4000.0,
    start_ratio: float = 0.0,
    dct_min: float = 0.15,
    dct_max: float = 0.45,
    max_channels: int = 8,
    channel_start: int = 4,
    center_ratio: float = 1.0,
    max_watermark_bytes: int = 64,
    robust_mode: str = "none",
    seed: int = 167626463082108,
    prompt: str = (
        "Keep the input image exactly unchanged. Preserve every pixel, color, "
        "texture, composition, and detail."
    ),
    input_image: str = "generation-b1e59042-91a9-4338-8308-5acb024f7c5a.png",
    scheduler_steps: int = 4,
    img2img_denoise: float | None = None,
) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_image}},
        "2": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 1,
            },
        },
        "3": {"class_type": "GetImageSize", "inputs": {"image": ["2", 0]}},
        "4": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux-2-klein-4b-fp8.safetensors",
                "weight_dtype": "default",
            },
        },
        "5": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "flux2",
                "device": "default",
            },
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "flux2-vae.safetensors"},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["5", 0]},
        },
        "8": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["7", 0]},
        },
        "9": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["2", 0], "vae": ["6", 0]},
        },
        "10": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["7", 0], "latent": ["9", 0]},
        },
        "11": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["8", 0], "latent": ["9", 0]},
        },
        "12": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["4", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "cfg": 1.0,
            },
        },
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": scheduler_steps,
                "width": ["3", 0],
                "height": ["3", 1],
            },
        },
        "15": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "17": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": ["3", 0], "height": ["3", 1], "batch_size": 1},
        },
        "18": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0],
                "guider": ["12", 0],
                "sampler": ["16", 0] if watermarked else ["15", 0],
                "sigmas": ["14", 0],
                "latent_image": ["17", 0],
            },
        },
        "19": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["18", 0], "vae": ["6", 0]},
        },
        "20": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["19", 0],
                "filename_prefix": filename_prefix,
                "filename_suffix": "png",
                "grayscale": False,
            },
        },
    }
    if watermarked:
        graph["23"] = {
            "class_type": "GROWWatermarkConfig",
            "inputs": {
                "secret_key": secret_key,
                "dct_min": dct_min,
                "dct_max": dct_max,
                "max_channels": max_channels,
                "channel_start": channel_start,
                "center_ratio": center_ratio,
            },
        }
        graph["16"] = {
            "class_type": "GROWDiTSampler",
            "inputs": {
                "sampler": ["15", 0],
                "config": ["23", 0],
                "watermark": watermark,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "start_ratio": start_ratio,
            },
        }
        graph["21"] = {
            "class_type": "GROWWatermarkDetect",
            "inputs": {
                "image": ["19", 0],
                "vae": ["6", 0],
                "config": ["23", 0],
                "max_watermark_bytes": max_watermark_bytes,
                "robust_mode": robust_mode,
            },
        }
    if img2img_denoise is not None:
        if not 0.0 < img2img_denoise <= 1.0:
            raise ValueError("img2img_denoise must be in (0, 1]")
        graph["22"] = {
            "class_type": "SplitSigmasDenoise",
            "inputs": {"sigmas": ["14", 0], "denoise": img2img_denoise},
        }
        graph["18"]["inputs"]["sigmas"] = ["22", 1]
        graph["18"]["inputs"]["latent_image"] = ["9", 0]
    return graph


def _json_request(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def queue_and_wait(server: str, prompt: dict[str, Any], timeout: int) -> tuple[str, Any]:
    queued = _json_request(f"{server.rstrip('/')}/prompt", {"prompt": prompt})
    if queued.get("node_errors"):
        raise RuntimeError(f"prompt validation failed: {queued['node_errors']}")
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = _json_request(f"{server.rstrip('/')}/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"prompt execution failed: {status}")
            return prompt_id, record
        time.sleep(2)
    raise TimeoutError(f"prompt {prompt_id} did not finish within {timeout}s")


def download_images(server: str, record: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for node_output in record.get("outputs", {}).values():
        for image in node_output.get("images", []):
            query = urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
            )
            with urlopen(f"{server.rstrip('/')}/view?{query}", timeout=60) as response:
                destination = output_dir / image["filename"]
                destination.write_bytes(response.read())
                downloaded.append(destination)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://10.168.1.168:8189")
    parser.add_argument("--mode", choices=("clean", "watermarked"), required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-api", type=Path)
    parser.add_argument("--watermark", default="zhangp36512345")
    parser.add_argument("--secret-key", default="watermark")
    parser.add_argument("--strength", type=float, default=1.2)
    parser.add_argument("--guidance-scale", type=float, default=4000.0)
    parser.add_argument("--start-ratio", type=float, default=0.0)
    parser.add_argument("--dct-min", type=float, default=0.15)
    parser.add_argument("--dct-max", type=float, default=0.45)
    parser.add_argument("--max-channels", type=int, default=8)
    parser.add_argument("--channel-start", type=int, default=4)
    parser.add_argument("--center-ratio", type=float, default=1.0)
    parser.add_argument("--max-watermark-bytes", type=int, default=64)
    parser.add_argument(
        "--robust-mode",
        choices=("none", "rotation", "crop_scale", "rotation_crop_scale"),
        default="none",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--scheduler-steps", type=int, default=4)
    parser.add_argument("--img2img-denoise", type=float)
    args = parser.parse_args()
    prompt = build_flux2_prompt(
        watermarked=args.mode == "watermarked",
        filename_prefix=args.prefix,
        watermark=args.watermark,
        secret_key=args.secret_key,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        start_ratio=args.start_ratio,
        dct_min=args.dct_min,
        dct_max=args.dct_max,
        max_channels=args.max_channels,
        channel_start=args.channel_start,
        center_ratio=args.center_ratio,
        max_watermark_bytes=args.max_watermark_bytes,
        robust_mode=args.robust_mode,
        scheduler_steps=args.scheduler_steps,
        img2img_denoise=args.img2img_denoise,
    )
    if args.save_api:
        args.save_api.parent.mkdir(parents=True, exist_ok=True)
        args.save_api.write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")
    prompt_id, record = queue_and_wait(args.server, prompt, args.timeout)
    paths = download_images(args.server, record, args.output_dir)
    result = {
        "prompt_id": prompt_id,
        "images": [str(path.resolve()) for path in paths],
        "detector": record.get("outputs", {}).get("21", {}),
        "status": record.get("status", {}),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
