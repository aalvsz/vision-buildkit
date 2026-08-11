#!/usr/bin/env python3
"""Evaluate an Axelera quantized model against its exact FP32 ONNX source.

This is a screening harness. Candidate inference uses Axelera's CPU-side
quantized model simulator; it must not be reported as an AIPU result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from axelera.compiler.quantized_model import AxeleraQuantizedModel
from axelera.types.manifest import Manifest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

COCO_CATEGORY_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)
EXPECTED_SOURCE_SHA256 = "82849fd17775d3584c7d917f73464b43a54bb2001d30ff213efc95047e7c0e7c"
EXPECTED_ANNOTATIONS_SHA256 = "f685a47efdcc0fbbcb22194e1d0e3a2a2d34dc565fe30f3f204b2bd5ed862f8c"
EXPECTED_IMAGES = {
    6818: ("000000006818.jpg", "99479f61ef2931ea70e161db0859e8706c6e48f13e8cb0798959f52433b01b98"),
    37777: ("000000037777.jpg", "fe513b68ba46bc146114c469fdff52c288226073f03d73d3300d4fa739937369"),
    87038: ("000000087038.jpg", "c6cfc7e454a432c31ac3c3a997d5f33ccebe120fc6b9465965609ce12995ede9"),
    174482: ("000000174482.jpg", "41fab00aad97dad3eb7d035420b8cc61ed0ca8f4f0b66a4fbc1bd0bba6f8a5b3"),
    252219: ("000000252219.jpg", "1cf48bddb1bcab80d9f4d18514ae438c3e4f57d18e80c4fbea9d5d22396f514a"),
    331352: ("000000331352.jpg", "8b69da4f3b8f7657105ef920d1084d657a939a763c728e2178fc6a007a1e762f"),
    397133: ("000000397133.jpg", "09e1d25c75f7879bdaa69c327fece5cabacd53939c8c2ef9e87f1c97a2e478c4"),
    403385: ("000000403385.jpg", "11632ed3fb470d62f7fe5f0445c4f10ec91225c4a820c95d1c3946af9426d4c7"),
    458054: ("000000458054.jpg", "b05c1e0c2219ac20707c4a47014f03d7fe149d8d3d25ff504837da48896c6bf7"),
    480985: ("000000480985.jpg", "3ac1735a37375f41662744d2f729d1c04c2b8460b8b1035f4e0c024d8d08d7eb"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def letterbox(path: Path, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"unable to read image: {path}")
    height, width = bgr.shape[:2]
    ratio = min(size / height, size / width)
    resized = (round(width * ratio), round(height * ratio))
    pad_w = (size - resized[0]) / 2
    pad_h = (size - resized[1]) / 2
    if resized != (width, height):
        bgr = cv2.resize(bgr, resized, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    bgr = cv2.copyMakeBorder(bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, ratio, (left, top), (width, height)


def predictions(
    output: np.ndarray,
    image_id: int,
    ratio: float,
    padding: tuple[int, int],
    original_size: tuple[int, int],
    threshold: float,
) -> list[dict[str, Any]]:
    rows = np.asarray(output, dtype=np.float32)
    if rows.shape != (1, 300, 6):
        raise ValueError(f"expected exact output shape (1, 300, 6), got {rows.shape}")
    rows = rows[0]
    if not np.isfinite(rows).all():
        raise ValueError("model output contains a non-finite value")
    if np.any(rows[:, 2] < rows[:, 0]) or np.any(rows[:, 3] < rows[:, 1]):
        raise ValueError("model output contains an inverted box")
    if np.any(rows[:, 4] < 0.0) or np.any(rows[:, 4] > 1.0):
        raise ValueError("model output contains a score outside [0, 1]")
    if np.any(np.abs(rows[:, 5] - np.rint(rows[:, 5])) > 1e-4):
        raise ValueError("model output contains a non-integral class index")
    if np.any(rows[:, 5] < 0) or np.any(rows[:, 5] >= len(COCO_CATEGORY_IDS)):
        raise ValueError("model output contains an out-of-range class index")
    rows = rows[rows[:, 4] >= threshold]
    width, height = original_size
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    result: list[dict[str, Any]] = []
    for x1, y1, x2, y2, score, class_index in rows_with_boxes(rows, boxes):
        class_id = int(round(float(class_index)))
        if class_id < 0 or class_id >= len(COCO_CATEGORY_IDS):
            raise ValueError(f"invalid class index {class_id}")
        result.append(
            {
                "image_id": image_id,
                "category_id": COCO_CATEGORY_IDS[class_id],
                "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                "score": float(score),
            }
        )
    return result


def rows_with_boxes(rows: np.ndarray, boxes: np.ndarray) -> Iterator[tuple[float, ...]]:
    for row, box in zip(rows, boxes, strict=True):
        yield float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(row[4]), float(row[5])


def coco_map(annotation_path: Path, detections: list[dict[str, Any]], image_ids: list[int]) -> float:
    if not detections:
        return 0.0
    ground_truth = COCO(str(annotation_path))
    result = ground_truth.loadRes(detections)
    evaluator = COCOeval(ground_truth, result, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0])


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, aw, ah = a["bbox"]
    bx1, by1, bw, bh = b["bbox"]
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def parity(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], image_ids: list[int]) -> dict[str, Any]:
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for image_id in image_ids:
        ref_image = [item for item in reference if item["image_id"] == image_id]
        cand_image = [item for item in candidate if item["image_id"] == image_id]
        if not ref_image or not cand_image:
            continue
        matrix = np.array([[iou(ref, cand) for cand in cand_image] for ref in ref_image], dtype=np.float64)
        costs = np.where(matrix >= 0.5, 1.0 - matrix, 1_000_000.0)
        ref_indices, cand_indices = linear_sum_assignment(costs)
        for ref_index, cand_index in zip(ref_indices, cand_indices, strict=True):
            overlap = float(matrix[ref_index, cand_index])
            if overlap >= 0.5:
                matches.append((ref_image[ref_index], cand_image[cand_index], overlap))
    matched_ref_ids = {id(ref) for ref, _, _ in matches}
    matched_cand_ids = {id(cand) for _, cand, _ in matches}
    unmatched_ref = [item for item in reference if id(item) not in matched_ref_ids]
    unmatched_cand = [item for item in candidate if id(item) not in matched_cand_ids]
    score_deltas = [abs(float(ref["score"]) - float(cand["score"])) for ref, cand, _ in matches]
    overlaps = [overlap for _, _, overlap in matches]
    high_confidence_lost = sum(float(item["score"]) >= 0.5 for item in unmatched_ref)
    high_confidence_added = sum(float(item["score"]) >= 0.5 for item in unmatched_cand)
    class_mismatches = sum(ref["category_id"] != cand["category_id"] for ref, cand, _ in matches)
    return {
        "reference_detections": len(reference),
        "candidate_detections": len(candidate),
        "matched": len(matches),
        "coverage": len(matches) / len(reference) if reference else 1.0,
        "unmatched_reference": len(unmatched_ref),
        "unmatched_candidate": len(unmatched_cand),
        "high_confidence_lost": high_confidence_lost,
        "high_confidence_added": high_confidence_added,
        "class_mismatches": class_mismatches,
        "mean_iou": float(np.mean(overlaps)) if overlaps else None,
        "min_iou": float(np.min(overlaps)) if overlaps else None,
        "mean_score_delta": float(np.mean(score_deltas)) if score_deltas else None,
        "max_score_delta": float(np.max(score_deltas)) if score_deltas else None,
    }


def verify_frozen_inputs(
    source_path: Path, annotation_path: Path, annotation: dict[str, Any], threshold: float
) -> list[int]:
    if threshold != 0.25:
        raise ValueError("the frozen screening threshold is exactly 0.25")
    if sha256(source_path) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source ONNX does not match the frozen Axelera evidence model")
    if sha256(annotation_path) != EXPECTED_ANNOTATIONS_SHA256:
        raise ValueError("annotation file does not match the frozen ten-image subset")
    if len(annotation.get("images", [])) != 10 or len(annotation.get("annotations", [])) != 97:
        raise ValueError("frozen dataset cardinality mismatch")
    if tuple(item["id"] for item in annotation.get("categories", [])) != COCO_CATEGORY_IDS:
        raise ValueError("frozen COCO category mapping mismatch")
    observed = {int(item["id"]): item["file_name"] for item in annotation["images"]}
    if observed != {image_id: record[0] for image_id, record in EXPECTED_IMAGES.items()}:
        raise ValueError("frozen image membership mismatch")
    for image_id, (filename, expected_sha) in EXPECTED_IMAGES.items():
        image_path = annotation_path.parent / filename
        if image_path.is_symlink() or not image_path.is_file() or sha256(image_path) != expected_sha:
            raise ValueError(f"frozen image identity mismatch: {image_id}")
    return sorted(observed)


def manifest_dependencies(manifest_path: Path, raw_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    for field in (
        "quantized_model_file",
        "preprocess_graph",
        "postprocess_graph",
        "model_lib_file",
        "model_params_file",
    ):
        relative = raw_manifest.get(field)
        if relative is None:
            continue
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"unsafe manifest dependency in {field}")
        dependency = manifest_path.parent / relative
        if dependency.is_symlink() or not dependency.is_file():
            raise ValueError(f"missing regular manifest dependency: {dependency}")
        dependencies.append({"field": field, "path": str(dependency), "sha256": sha256(dependency)})
    if not any(item["field"] == "quantized_model_file" for item in dependencies):
        raise ValueError("manifest does not bind a quantized model")
    return dependencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--quantized-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="directory containing instances.json and images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    source_path = args.source_onnx.resolve()
    manifest_path = args.quantized_manifest.resolve()
    dataset_path = args.dataset.resolve()
    annotation_path = dataset_path / "instances.json"
    annotation = json.loads(annotation_path.read_text())
    image_ids = verify_frozen_inputs(source_path, annotation_path, annotation, args.threshold)
    raw_manifest = json.loads(manifest_path.read_text())
    dependencies = manifest_dependencies(manifest_path, raw_manifest)
    session = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    with working_directory(manifest_path.parent):
        candidate_model = AxeleraQuantizedModel.load(Manifest.from_json(manifest_path.read_text()))

    source_predictions: list[dict[str, Any]] = []
    candidate_predictions: list[dict[str, Any]] = []
    for image in sorted(annotation["images"], key=lambda item: int(item["id"])):
        tensor, ratio, padding, original_size = letterbox(dataset_path / image["file_name"])
        source_output = session.run(None, {input_name: tensor})[0]
        with torch.inference_mode():
            candidate_output = candidate_model(torch.from_numpy(tensor)).detach().cpu().numpy()
        source_predictions.extend(
            predictions(source_output, int(image["id"]), ratio, padding, original_size, args.threshold)
        )
        candidate_predictions.extend(
            predictions(candidate_output, int(image["id"]), ratio, padding, original_size, args.threshold)
        )

    args.output.mkdir(parents=True, exist_ok=False)
    source_file = args.output / "source-predictions.json"
    candidate_file = args.output / "candidate-predictions.json"
    source_file.write_text(json.dumps(source_predictions, sort_keys=True, separators=(",", ":")) + "\n")
    candidate_file.write_text(json.dumps(candidate_predictions, sort_keys=True, separators=(",", ":")) + "\n")
    source_map = coco_map(annotation_path, source_predictions, image_ids)
    candidate_map = coco_map(annotation_path, candidate_predictions, image_ids)
    comparison = parity(source_predictions, candidate_predictions, image_ids)
    result = {
        "schema_version": 1,
        "evidence_kind": "axelera_cpu_quantized_simulator_screening",
        "source": {"path": str(source_path), "sha256": sha256(source_path), "map50_95": source_map},
        "candidate": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "dependencies": dependencies,
            "map50_95": candidate_map,
            "map_drop": source_map - candidate_map,
        },
        "dataset": {
            "annotations": str(annotation_path),
            "sha256": sha256(annotation_path),
            "images": len(annotation["images"]),
        },
        "threshold": args.threshold,
        "parity": comparison,
        "gates": {
            "map_drop": source_map - candidate_map <= 0.002,
            "coverage": comparison["coverage"] >= 0.98,
            "max_score_delta": comparison["max_score_delta"] is not None and comparison["max_score_delta"] <= 0.10,
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
