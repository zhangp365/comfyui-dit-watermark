"""Compare clean/GROW outputs for Flux2 Klein 4B and 9B over a directory."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
from urllib.request import Request, urlopen
import uuid

import numpy as np
from PIL import Image

if __package__:
    from scripts.comfy_api import build_flux2_prompt, download_images, queue_and_wait
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.comfy_api import build_flux2_prompt, download_images, queue_and_wait


MODELS = {
    "4b": ("flux-2-klein-4b-fp8.safetensors", "qwen_3_4b.safetensors"),
    "9b": ("flux-2-klein-9b-fp8.safetensors", "qwen_3_8b_fp8mixed.safetensors"),
}


def upload_image(server: str, source: Path, remote_name: str) -> None:
    boundary = f"----codex-{uuid.uuid4().hex}"
    payload = source.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{remote_name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + payload + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    request = Request(
        f"{server.rstrip('/')}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    if result.get("name") != remote_name:
        raise RuntimeError(f"unexpected upload result: {result}")


def parse_detector(output: dict[str, Any]) -> dict[str, Any]:
    text = output.get("text", [""])[0]
    match = re.fullmatch(
        r"decoded=(.*); ecc_valid=(True|False); corrected_symbols=(\d+); "
        r"min_vote_margin=([0-9.]+); alignment=(.*); raw_codeword_hex=([0-9a-f]*)",
        text,
    )
    if not match:
        raise RuntimeError(f"cannot parse detector output: {output}")
    return {
        "decoded": ast.literal_eval(match.group(1)),
        "ecc_valid": match.group(2) == "True",
        "corrected_symbols": int(match.group(3)),
        "min_vote_margin": float(match.group(4)),
        "alignment": match.group(5),
        "raw_codeword_hex": match.group(6),
    }


def psnr(
    reference: Path, marked: Path, *, center_crop_reference: bool = False
) -> tuple[float, list[int]]:
    with Image.open(reference) as clean_image, Image.open(marked) as marked_image:
        clean = np.asarray(clean_image.convert("RGB"), dtype=np.float64) / 255.0
        watermarked = np.asarray(marked_image.convert("RGB"), dtype=np.float64) / 255.0
        size = list(clean_image.size)
    if center_crop_reference and clean.shape != watermarked.shape:
        extra_height = clean.shape[0] - watermarked.shape[0]
        extra_width = clean.shape[1] - watermarked.shape[1]
        if extra_height < 0 or extra_width < 0:
            raise ValueError(
                f"reference is smaller than target: {clean.shape} != {watermarked.shape}"
            )
        top = extra_height // 2
        left = extra_width // 2
        clean = clean[
            top : top + watermarked.shape[0],
            left : left + watermarked.shape[1],
        ]
    if clean.shape != watermarked.shape:
        raise ValueError(f"shape mismatch: {clean.shape} != {watermarked.shape}")
    mse = float(np.mean((clean - watermarked) ** 2))
    return (float("inf") if mse == 0 else -10.0 * math.log10(mse)), size


def save_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://10.168.1.168:8189")
    parser.add_argument("--input-dir", type=Path, default=Path("images/test"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/flux2_4b_9b_ten_images_channel4"),
    )
    parser.add_argument("--watermark", default="zhangp365123456")
    parser.add_argument("--strength", type=float, default=1.2)
    parser.add_argument("--guidance-scale", type=float, default=4000.0)
    parser.add_argument("--channel-start", type=int, default=4)
    parser.add_argument(
        "--preserve-input-size",
        action="store_true",
        help="Bypass ImageScaleToTotalPixels and use the original input dimensions.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    result_path = args.output_dir / "results.json"
    settings = {
        "watermark": args.watermark,
        "secret_key": "watermark",
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "start_ratio": 0.0,
        "dct_min": 0.15,
        "dct_max": 0.45,
        "max_channels": 8,
        "channel_start": args.channel_start,
        "center_ratio": 1.0,
        "scheduler_steps": 4,
        "seed": 167626463082108,
        "vae": "flux2-vae.safetensors",
        "preserve_input_size": args.preserve_input_size,
    }
    results: dict[str, Any] = {
        "settings": settings,
        "models": {
            key: {"unet": value[0], "clip": value[1], "images": {}}
            for key, value in MODELS.items()
        },
    }
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("settings") != settings:
            raise RuntimeError("existing result settings differ; choose another output directory")
        results = prior

    sources = sorted(
        path for path in args.input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    for source in sources:
        remote_name = f"flux2_compare_{source.name}"
        upload_image(args.server, source, remote_name)
        for model_key, (unet_name, clip_name) in MODELS.items():
            model_results = results["models"][model_key]["images"]
            if source.name in model_results:
                print(f"skip {model_key} {source.name}", flush=True)
                continue
            item_dir = args.output_dir / model_key / source.stem
            item_dir.mkdir(parents=True, exist_ok=True)
            prompt_ids: dict[str, str] = {}
            output_paths: dict[str, Path] = {}
            detector: dict[str, Any] | None = None
            for mode in ("clean", "watermarked"):
                marked = mode == "watermarked"
                print(f"run {model_key} {source.name} {mode}", flush=True)
                prompt = build_flux2_prompt(
                    watermarked=marked,
                    filename_prefix=f"flux2_compare/{model_key}/{source.stem}/{mode}",
                    watermark=args.watermark,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    channel_start=args.channel_start,
                    input_image=remote_name,
                    unet_name=unet_name,
                    clip_name=clip_name,
                )
                source_node = "2"
                if args.preserve_input_size:
                    source_node = "1"
                    prompt["3"]["inputs"]["image"] = [source_node, 0]
                    prompt["9"]["inputs"]["pixels"] = [source_node, 0]
                if not marked:
                    prompt["23"] = {
                        "class_type": "SaveImage",
                        "inputs": {
                            "images": [source_node, 0],
                            "filename_prefix": (
                                f"flux2_compare/{model_key}/{source.stem}/source_reference"
                            ),
                            "filename_suffix": "png",
                            "grayscale": False,
                        },
                    }
                prompt_id, record = queue_and_wait(args.server, prompt, args.timeout)
                downloads = download_images(args.server, record, item_dir)
                expected_downloads = 2 if not marked else 1
                if len(downloads) != expected_downloads:
                    raise RuntimeError(
                        f"expected {expected_downloads} output images, got {downloads}"
                    )
                destination = item_dir / f"{mode}.png"
                generated = next(
                    path for path in downloads if "source_reference" not in path.name
                )
                generated.replace(destination)
                if not marked:
                    reference = next(
                        path for path in downloads if "source_reference" in path.name
                    )
                    reference.replace(item_dir / "source_reference.png")
                prompt_ids[mode] = prompt_id
                output_paths[mode] = destination
                if marked:
                    detector = parse_detector(record.get("outputs", {}).get("21", {}))
            psnr_db, output_size = psnr(output_paths["clean"], output_paths["watermarked"])
            source_clean_psnr_db, source_size = psnr(
                item_dir / "source_reference.png",
                output_paths["clean"],
                center_crop_reference=True,
            )
            model_results[source.name] = {
                "source": str(source),
                "clean_prompt_id": prompt_ids["clean"],
                "watermarked_prompt_id": prompt_ids["watermarked"],
                "clean_image": str(output_paths["clean"]),
                "watermarked_image": str(output_paths["watermarked"]),
                "source_reference_image": str(item_dir / "source_reference.png"),
                "source_to_clean_psnr_db": source_clean_psnr_db,
                "psnr_db": psnr_db,
                "output_size": output_size,
                "source_size": source_size,
                "detector": detector,
                "watermark_success": bool(
                    detector
                    and detector["ecc_valid"]
                    and detector["decoded"] == args.watermark
                ),
            }
            save_results(result_path, results)
            print(
                f"done {model_key} {source.name}: PSNR={psnr_db:.6f}, detector={detector}",
                flush=True,
            )

    for model_key in MODELS:
        items = list(results["models"][model_key]["images"].values())
        results["models"][model_key]["summary"] = {
            "count": len(items),
            "watermark_successes": sum(item["watermark_success"] for item in items),
            "mean_psnr_db": float(np.mean([item["psnr_db"] for item in items])),
            "min_psnr_db": min(item["psnr_db"] for item in items),
            "max_psnr_db": max(item["psnr_db"] for item in items),
            "mean_source_to_clean_psnr_db": float(
                np.mean([item["source_to_clean_psnr_db"] for item in items])
            ),
            "min_source_to_clean_psnr_db": min(
                item["source_to_clean_psnr_db"] for item in items
            ),
            "max_source_to_clean_psnr_db": max(
                item["source_to_clean_psnr_db"] for item in items
            ),
            "mean_min_vote_margin": float(
                np.mean([item["detector"]["min_vote_margin"] for item in items])
            ),
            "min_vote_margin": min(item["detector"]["min_vote_margin"] for item in items),
            "max_vote_margin": max(item["detector"]["min_vote_margin"] for item in items),
        }
    save_results(result_path, results)
    print(json.dumps({key: results["models"][key]["summary"] for key in MODELS}, indent=2))


if __name__ == "__main__":
    main()
