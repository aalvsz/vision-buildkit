#!/usr/bin/env python3
"""Screen an Axelera-quantized raw YOLO26 graph against its FP32 source.

The raw graph has no end-to-end top-k/NMS head. Both source and candidate use
the same deterministic external decoder so this test isolates quantization
error in the detector graph. Candidate execution is the Axelera CPU simulator,
not the Metis AIPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import eval_quantized as common
import numpy as np
import onnxruntime as ort
import torch
from axelera.compiler.quantized_model import AxeleraQuantizedModel
from axelera.types.manifest import Manifest

EXPECTED_RAW_SOURCE_SHA256 = "db42840c9ae5f1490c3d748f0bc72253d1509ed2c0f5d20b3f3847102596e9a9"


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    intersection_xy1 = np.maximum(one[:2], many[:, :2])
    intersection_xy2 = np.minimum(one[2:], many[:, 2:])
    intersection_wh = np.maximum(0.0, intersection_xy2 - intersection_xy1)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    one_area = max(0.0, float(one[2] - one[0])) * max(0.0, float(one[3] - one[1]))
    many_area = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(0.0, many[:, 3] - many[:, 1])
    union = one_area + many_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> list[int]:
    retained: list[int] = []
    for class_index in sorted(int(value) for value in np.unique(classes)):
        indices = np.flatnonzero(classes == class_index)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        while order.size:
            current = int(order[0])
            retained.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            order = remaining[box_iou(boxes[current], boxes[remaining]) <= iou_threshold]
    retained.sort(key=lambda index: (-float(scores[index]), index))
    return retained[:300]


def raw_predictions(
    output: np.ndarray,
    image_id: int,
    ratio: float,
    padding: tuple[int, int],
    original_size: tuple[int, int],
    threshold: float,
) -> list[dict[str, Any]]:
    tensor = np.asarray(output, dtype=np.float32)
    if tensor.shape != (1, 84, 8400):
        raise ValueError(f"expected exact raw output shape (1, 84, 8400), got {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise ValueError("model output contains a non-finite value")
    rows = tensor[0].T
    if np.any(rows[:, 2:4] < 0.0):
        raise ValueError("raw output contains a negative box width or height")
    class_scores = rows[:, 4:]
    if np.any(class_scores < 0.0) or np.any(class_scores > 1.0):
        raise ValueError("raw class score outside [0, 1]")
    classes = np.argmax(class_scores, axis=1).astype(np.int64)
    scores = class_scores[np.arange(class_scores.shape[0]), classes]
    selected = scores >= threshold
    rows, scores, classes = rows[selected], scores[selected], classes[selected]
    boxes = np.empty((len(rows), 4), dtype=np.float32)
    boxes[:, 0] = rows[:, 0] - rows[:, 2] / 2
    boxes[:, 1] = rows[:, 1] - rows[:, 3] / 2
    boxes[:, 2] = rows[:, 0] + rows[:, 2] / 2
    boxes[:, 3] = rows[:, 1] + rows[:, 3] / 2
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.7)
    width, height = original_size
    result: list[dict[str, Any]] = []
    for index in keep:
        box = boxes[index].copy()
        box[[0, 2]] = (box[[0, 2]] - padding[0]) / ratio
        box[[1, 3]] = (box[[1, 3]] - padding[1]) / ratio
        box[[0, 2]] = box[[0, 2]].clip(0, width)
        box[[1, 3]] = box[[1, 3]].clip(0, height)
        result.append(
            {
                "image_id": image_id,
                "category_id": common.COCO_CATEGORY_IDS[int(classes[index])],
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(max(0.0, box[2] - box[0])),
                    float(max(0.0, box[3] - box[1])),
                ],
                "score": float(scores[index]),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--quantized-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    source_path = args.source_onnx.resolve()
    manifest_path = args.quantized_manifest.resolve()
    dataset_path = args.dataset.resolve()
    annotation_path = dataset_path / "instances.json"
    annotation = json.loads(annotation_path.read_text())
    if common.sha256(source_path) != EXPECTED_RAW_SOURCE_SHA256:
        raise ValueError("source ONNX does not match the frozen raw YOLO26 profile")
    accepted = dict(common.EXPECTED_SOURCE_PROFILES)
    common.EXPECTED_SOURCE_PROFILES[EXPECTED_RAW_SOURCE_SHA256] = "raw-opset17-end2end-false"
    try:
        image_ids, source_profile = common.verify_frozen_inputs(
            source_path, annotation_path, annotation, args.threshold
        )
    finally:
        common.EXPECTED_SOURCE_PROFILES.clear()
        common.EXPECTED_SOURCE_PROFILES.update(accepted)

    raw_manifest = json.loads(manifest_path.read_text())
    dependencies = common.manifest_dependencies(manifest_path, raw_manifest)
    session = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    with common.working_directory(manifest_path.parent):
        candidate_model = AxeleraQuantizedModel.load(Manifest.from_json(manifest_path.read_text()))

    source_predictions: list[dict[str, Any]] = []
    candidate_predictions: list[dict[str, Any]] = []
    for image in sorted(annotation["images"], key=lambda item: int(item["id"])):
        tensor, ratio, padding, original_size = common.letterbox(dataset_path / image["file_name"])
        source_output = session.run(None, {input_name: tensor})[0]
        with torch.inference_mode():
            candidate_output = candidate_model(torch.from_numpy(tensor)).detach().cpu().numpy()
        source_predictions.extend(
            raw_predictions(source_output, int(image["id"]), ratio, padding, original_size, args.threshold)
        )
        candidate_predictions.extend(
            raw_predictions(candidate_output, int(image["id"]), ratio, padding, original_size, args.threshold)
        )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source-predictions.json").write_text(
        json.dumps(source_predictions, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (args.output / "candidate-predictions.json").write_text(
        json.dumps(candidate_predictions, sort_keys=True, separators=(",", ":")) + "\n"
    )
    source_map = common.coco_map(annotation_path, source_predictions, image_ids)
    candidate_map = common.coco_map(annotation_path, candidate_predictions, image_ids)
    comparison = common.parity(source_predictions, candidate_predictions, image_ids)
    result = {
        "schema_version": 1,
        "evidence_kind": "axelera_cpu_quantized_simulator_raw_output_screening",
        "source": {
            "path": str(source_path),
            "profile": source_profile,
            "sha256": common.sha256(source_path),
            "map50_95": source_map,
        },
        "candidate": {
            "manifest": str(manifest_path),
            "manifest_sha256": common.sha256(manifest_path),
            "dependencies": dependencies,
            "map50_95": candidate_map,
            "map_drop": source_map - candidate_map,
        },
        "dataset": {
            "annotations": str(annotation_path),
            "sha256": common.sha256(annotation_path),
            "images": len(annotation["images"]),
        },
        "decoder": {
            "box_format": "cxcywh",
            "class_selection": "argmax",
            "nms": "class-aware",
            "nms_iou_threshold": 0.7,
            "max_detections": 300,
        },
        "threshold": args.threshold,
        "parity": comparison,
        "gates": {
            "map_drop": source_map - candidate_map <= 0.002,
            "coverage": comparison["coverage"] >= 0.98,
            "max_score_delta": comparison["max_score_delta"] is not None
            and comparison["max_score_delta"] <= 0.10,
            "class_mismatches": comparison["class_mismatches"] == 0,
        },
        "non_claim": "Candidate execution used the Axelera CPU quantized-model simulator, not the Metis AIPU.",
    }
    result["decision"] = "PASS" if all(result["gates"].values()) else "FAIL"
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
