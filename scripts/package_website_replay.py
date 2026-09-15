"""Package an explicitly selected nuScenes prediction export for static Pages.

No inference or training occurs here. Input images are copied byte-for-byte.
Only an allowlist of predictions and camera calibration enters the public bundle.
Raw labels, LiDAR, optimizer state, weights and local paths never enter the bundle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_child(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes the configured data root")
    return path


def encode_runs(values):
    flat = np.asarray(values, dtype=np.uint8).ravel().tolist()
    runs = []
    for value in flat:
        if runs and runs[-2] == value:
            runs[-1] += 1
        else:
            runs.extend((value, 1))
    return runs


def decode_runs(runs):
    if len(runs) % 2:
        raise ValueError("Unpaired run")
    return [value for value, count in zip(runs[::2], runs[1::2]) for _ in range(count)]


def display_road(record):
    """Match viewer/app.js's prediction-only visibility + confidence policy."""
    road = np.asarray(record["road"])
    confidence = np.asarray(record.get("road_confidence", np.ones_like(road)))
    visible = np.asarray(record.get("road_visible", np.ones_like(road)), dtype=bool)
    if road.ndim != 2 or not road.size or max(road.shape) > 512:
        raise ValueError("Invalid road shape")
    if road.shape != confidence.shape or road.shape != visible.shape:
        raise ValueError("Road/visibility/confidence shapes differ")
    mask = visible & np.isfinite(confidence) & (confidence >= 0.45) & np.isin(road, (0, 1, 2))
    result = np.where(mask, road, 255).astype(np.uint8)
    return {"width": road.shape[1], "height": road.shape[0], "runs": encode_runs(result)}, result


def predicted_objects(record):
    keys = ("boxes", "scores", "labels", "track_ids")
    result = {key: record[key] for key in keys}
    count = len(result["boxes"])
    if count > 100 or any(len(result[key]) != count for key in keys):
        raise ValueError("Object arrays do not match")
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        if len(box) != 7 or not all(math.isfinite(x) for x in box) or min(box[3:6]) <= 0:
            raise ValueError("Invalid predicted box")
        if not math.isfinite(score) or not 0 <= score <= 1 or label not in (0, 1):
            raise ValueError("Invalid predicted class or confidence")
    return result


def write_json(path, value):
    # Hash the same bytes Git publishes on Linux, even when packaging on Windows.
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def package(export: Path, config: Path, output: Path, expected_checkpoint: str, model_label: str, limit=40):
    if output.exists():
        raise ValueError("Output exists; preserve it and choose a new bundle directory")
    settings = json.loads(config.read_text(encoding="utf-8-sig"))
    metadata = json.loads((export / "replay-metadata.json").read_text(encoding="utf-8-sig"))
    identity = metadata["identity"]
    if identity["checkpoint_sha256"].lower() != expected_checkpoint.lower():
        raise ValueError("Prediction export is not from the selected checkpoint")
    if settings["model"].get("camera_geometry") != "pinhole":
        raise ValueError("This package supports the six-camera pinhole nuScenes export")
    root = Path(settings["data"]["root"]).resolve()
    manifest_path = safe_child(root, settings["data"].get("manifest", "manifest.json"))
    if sha256(manifest_path) != identity["manifest_sha256"]:
        raise ValueError("Prepared dataset manifest identity changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = {(r["sequence"], r["frame_id"]): r for r in manifest["samples"]}
    records = []
    with (export / "scenes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
            if len(records) == limit:
                break
    if not records:
        raise ValueError("No completed prediction frames")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    (output / "images").mkdir()
    frame_index, assets = [], []
    for index, record in enumerate(records):
        row = rows[(record["sequence"], record["frame_id"])]
        if abs(row["timestamp"] - record["timestamp"]) > 1e-6:
            raise ValueError("Image and prediction timestamps differ")
        if record["image_paths"] != row["images"] or len(row["images"]) != 6:
            raise ValueError("Image paths/camera count differ from the source manifest")
        if row["scene_name"] not in manifest["splits"][metadata["split"]]:
            raise ValueError("Frame does not belong to the declared split")
        if index and record["sequence"] == records[index-1]["sequence"] and record["timestamp"] <= records[index-1]["timestamp"]:
            raise ValueError("Replay frames are not chronological")
        cached = safe_child(root, row["cache"])
        if sha256(cached) != row["cache_sha256"]:
            raise ValueError("Calibration cache hash changed")
        # Explicit read allowlist: no boxes, road targets, depth labels or IDs.
        with np.load(cached, allow_pickle=False) as stored:
            intrinsics = stored["intrinsics"]
            transforms = stored["camera_to_ego"]
        if intrinsics.shape != (6, 3, 3) or transforms.shape != (6, 4, 4):
            raise ValueError("Expected six calibrated cameras")
        if not np.isfinite(intrinsics).all() or not np.isfinite(transforms).all():
            raise ValueError("Nonfinite camera calibration")
        images = []
        for camera, name in enumerate(row["images"]):
            source = safe_child(root, name)
            if source.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                raise ValueError("Unsupported source image format")
            target = output / "images" / f"frame-{index:03d}-camera-{camera}{source.suffix.lower()}"
            with Image.open(source) as image:
                width, height = image.size
            if [height, width] != settings["model"]["image_size"]:
                raise ValueError("Image size and projection calibration differ")
            shutil.copyfile(source, target)
            source_hash = sha256(source)
            if sha256(target) != source_hash:
                raise ValueError("Published image bytes differ from the source")
            relative = target.relative_to(output).as_posix()
            images.append(relative)
            assets.append({"file": relative, "sha256": source_hash, "bytes": target.stat().st_size})
        road, expected = display_road(record)
        if decode_runs(road["runs"]) != expected.ravel().tolist():
            raise AssertionError("Road display compression must be lossless")
        frame = {
            "index": index, "sequence": record["sequence"], "scene_name": row["scene_name"],
            "frame_id": record["frame_id"], "timestamp": record["timestamp"],
            **predicted_objects(record),
            "road_display": road,
            "ground_z": record.get("road_ground_z_m", metadata["ground_z"]),
            "images": images, "image_size": [width, height],
            "intrinsics": intrinsics.tolist(), "camera_to_ego": transforms.tolist(),
        }
        frame_path = output / "frames" / f"{index:03d}.json"
        write_json(frame_path, frame)
        assets.append({"file": frame_path.relative_to(output).as_posix(), "sha256": sha256(frame_path), "bytes": frame_path.stat().st_size})
        frame_index.append({"file": frame_path.relative_to(output).as_posix(), "timestamp": record["timestamp"], "sequence": record["sequence"], "scene_name": row["scene_name"], "source_frame": record["frame_id"]})
    public = {
        "schema_version": 1, "model": model_label, "dataset": "nuScenes v1.0-trainval",
        "split": metadata["split"], "checkpoint_sha256": identity["checkpoint_sha256"],
        "source_export_sha256": sha256(export / "scenes.jsonl"),
        "source_manifest_sha256": identity["manifest_sha256"],
        "selection": "First 40 chronological validation keyframes, selected before inspecting predictions.",
        "runtime": metadata["runtime_backend"], "refinement_enabled": bool(metadata["refine"]),
        "cameras": metadata["camera_names"], "frames": frame_index,
        "bev_min": metadata["bev_min"], "bev_step": metadata["bev_step"], "ground_z": metadata["ground_z"],
        "confidence_default": 0.55, "source_keyframe_rate_hz": 2,
        "bytes": sum(asset["bytes"] for asset in assets),
        "limitations": [
            "Recorded checkpoint predictions, not live browser inference.",
            "Stage A only; spatial/temporal refinement is bypassed.",
            "Pedestrian detection and scene reconstruction remain unreliable.",
            "This clip illustrates behaviour; it is not a benchmark or a representative accuracy sample.",
            "Earlier article results describe a different KITTI-360 experiment.",
            "Camera overlays are projections of predicted cuboids, not annotations or visible silhouettes.",
            "Road colours use predicted classes, predicted confidence >= 0.45, and calibration-based field of view.",
        ],
        "attribution": {
            "dataset": "nuScenes, Motional",
            "paper": "Caesar et al., nuScenes: A multimodal dataset for autonomous driving, CVPR 2020.",
            "url": "https://www.nuscenes.org/nuscenes",
            "terms": "https://www.nuscenes.org/terms-of-use",
            "license": "CC BY-NC-SA 4.0 with the nuScenes Dataset Terms",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "changes": "Inputs were resized by the existing model preparation pipeline to 704x256; this bundle copies those processed images without further modification. Model predictions and projected overlays are derived visualizations.",
            "endorsement": "Independent research demonstration; no endorsement or affiliation is implied.",
        },
    }
    write_json(output / "manifest.json", public)
    write_json(output / "asset-integrity.json", {"algorithm": "SHA256", "assets": assets})
    return {"frames": len(records), "images": len(records)*6, "bytes": public["bytes"], "checkpoint_sha256": public["checkpoint_sha256"], "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--model-label", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.export, args.config, args.output, args.checkpoint_sha256, args.model_label), indent=2))
