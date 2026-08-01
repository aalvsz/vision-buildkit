"""Offline, reproducible CPU adapter parity gate for the three supported families.

All inputs and exports are frozen by SHA256 below.  This never downloads,
exports, benchmarks, or silently marks a missing dependency as a pass.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

import vision
from vision.manifest import VisionManifest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "autoresearch/adapters-complete-260730"
ASSETS = {
    "bus": (
        ROOT / "autoresearch/adapters-repair-260730/integration/bus.jpg",
        "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
    ),
    "yolo_pt": (ROOT / "yolo26n.pt", "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    "yolo_onnx": (ROOT / "yolo26n.onnx", "81caddbbc9a312e36c31382e61a76e554abc0f2dc38fb2c07cc4b0a3c73bccdf"),
    "rt_pt": (ROOT / "rtdetr-l.pt", "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"),
    "rt_onnx": (ROOT / "rtdetr-l.onnx", "a5386605945d18ccbc17ce94a2765d51dada2cb20f0804c364e2051cc5066ee6"),
    "rf_pt": (
        Path.home() / ".roboflow/models/rf-detr-nano.pth",
        "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d",
    ),
    "rf_onnx": (
        ROOT / "autoresearch/adapters-repair-260730/rfdetr-export/rfdetr-nano.onnx",
        "af8b4355afe69e566bf73749bb7a357a1d830deb12ff698d694ba7374f5986a1",
    ),
    "wide": (OUT / "wide.jpg", "d3597d8136548b290e784e4c8d1f215500f87576cef436a99d691b2f44e31b88"),
    "tall": (OUT / "tall.jpg", "bcf8e7c4c026ad34623ec2897ba3ea39b31350a8faeace99259204f3500476e6"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(name: str) -> Path:
    path, expected = ASSETS[name]
    if not path.is_file():
        raise RuntimeError(f"frozen asset missing: {path}")
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"frozen asset hash mismatch: {path}: {actual}")
    return path


def manifest(family: str, source: Path, *, onnx: bool) -> VisionManifest:
    if family == "yolo26":
        return VisionManifest(
            source=str(source),
            family="onnx" if onnx else "yolo26",
            output={"boxes_name": "output0"},
            decoder={"kind": "yolo_end2end", "score_threshold": 0.25},
        )
    if family == "rtdetr":
        return VisionManifest(
            source=str(source),
            family="onnx" if onnx else "rtdetr",
            preprocessing={"resize": "resize", "implementation": "opencv_linear_scale_fill"},
            output={"box_format": "cxcywh", "boxes_name": "output0"},
            decoder={"kind": "rtdetr_combined", "boxes_normalized": True, "score_threshold": 0.4},
        )
    return VisionManifest(
        source=str(source),
        family="onnx" if onnx else "rfdetr",
        input={"shape": [1, 3, 384, 384], "input_name": "input" if onnx else "images"},
        preprocessing={
            "resize": "resize",
            "implementation": "torchvision_bilinear_antialias_false",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        output={"box_format": "cxcywh", "boxes_name": "dets", "logits_name": "labels"},
        decoder={"kind": "rfdetr", "boxes_normalized": True, "score_threshold": 0.3},
    )


def execute(family: str, pt_name: str, onnx_name: str, threshold: float) -> dict:
    pt, onnx = checked(pt_name), checked(onnx_name)
    native_manifest, onnx_manifest = manifest(family, pt, onnx=False), manifest(family, onnx, onnx=True)
    native_manifest.to_yaml(OUT / f"{family}-native-manifest.yaml")
    onnx_manifest.to_yaml(OUT / f"{family}-onnx-manifest.yaml")
    native = vision.load(str(pt), device="cpu", manifest=native_manifest)
    candidate = vision.load(str(onnx), device="cpu", manifest=onnx_manifest)
    variants = {"bus": checked("bus"), "wide": checked("wide"), "tall": checked("tall")}
    outcomes = {}
    for name, image in variants.items():
        reference, observed = native(image), candidate(image)
        gate = {
            "yolo26": (0.95, 0.85, 0.05, 0.12),
            "rtdetr": (0.98, 0.97, 0.01, 0.01),
            "rfdetr": (0.99, 0.97, 0.01, 0.03),
        }[family]
        detail = vision.validate(reference, observed, iou_threshold=0.80, score_tolerance=gate[3], high_confidence=0.5)
        detail.update(
            {
                "reference_count": len(reference.boxes),
                "candidate_count": len(observed.boxes),
                "reference_metadata": reference.metadata,
                "candidate_metadata": observed.metadata,
                "image_sha256": sha256(image),
                "image_size": Image.open(image).size,
            }
        )
        if (
            not reference.boxes
            or detail["lost"]
            or detail["added"]
            or detail["high_conf_lost"]
            or detail["high_conf_added"]
            or detail["class_mismatches"]
            or detail["mean_iou"] < gate[0]
            or detail["min_iou"] < gate[1]
            or detail["mean_score_delta"] > gate[2]
            or detail["max_score_delta"] > gate[3]
        ):
            raise RuntimeError(f"{family}/{name} parity gate failed: {detail}")
        outcomes[name] = detail
    return {
        "status": "passed",
        "family": family,
        "native": str(pt),
        "native_sha256": sha256(pt),
        "onnx": str(onnx),
        "onnx_sha256": sha256(onnx),
        "threshold": threshold,
        "variants": outcomes,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "at": datetime.now(UTC).isoformat(),
        "hardware": platform.platform(),
        "python": sys.version,
        "mode": "offline-frozen-assets",
        "models": {},
    }
    for family, pt, onnx, threshold in (
        ("yolo26", "yolo_pt", "yolo_onnx", 0.25),
        ("rtdetr", "rt_pt", "rt_onnx", 0.4),
        ("rfdetr", "rf_pt", "rf_onnx", 0.3),
    ):
        try:
            result["models"][family] = execute(family, pt, onnx, threshold)
        except Exception as error:
            result["models"][family] = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if any(model["status"] != "passed" for model in result["models"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
