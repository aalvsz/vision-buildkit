"""Run an offline parity gate from explicit, checksum-pinned inputs.

The command never downloads or exports a model.  Every model, manifest, and
image must be supplied in a JSON configuration with a SHA-256 digest.  Run
``python scripts/integration_models.py --help`` for the configuration shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

import vision
from vision.manifest import VisionManifest

_DIGEST = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {"name", "native", "candidate", "images", "gates"}
_MODEL_KEYS = {"model", "model_sha256", "manifest", "manifest_sha256"}
_IMAGE_KEYS = {"name", "path", "sha256"}
_GATE_KEYS = {
    "iou_threshold",
    "score_tolerance",
    "high_confidence",
    "min_mean_iou",
    "min_iou",
    "max_mean_score_delta",
    "max_score_delta",
}


@dataclass(frozen=True)
class PinnedFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ModelInput:
    model: PinnedFile
    manifest: PinnedFile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}")


def _pinned(base: Path, path_value: Any, digest_value: Any, label: str) -> PinnedFile:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path must be a non-empty string")
    if not isinstance(digest_value, str) or _DIGEST.fullmatch(digest_value) is None:
        raise ValueError(f"{label} sha256 must be 64 lowercase hexadecimal characters")
    path = Path(path_value)
    if not path.is_absolute():
        path = base / path
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular, non-symlink file")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular, non-symlink file")
    actual = _sha256(path)
    if actual != digest_value:
        raise ValueError(f"{label} sha256 mismatch: expected {digest_value}, observed {actual}")
    return PinnedFile(path, actual)


def _model_input(base: Path, value: Any, label: str) -> ModelInput:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _exact_keys(value, _MODEL_KEYS, label)
    return ModelInput(
        model=_pinned(base, value["model"], value["model_sha256"], f"{label}.model"),
        manifest=_pinned(base, value["manifest"], value["manifest_sha256"], f"{label}.manifest"),
    )


def _load_config(path: Path) -> tuple[str, ModelInput, ModelInput, dict[str, PinnedFile], dict[str, float]]:
    config_path = path.resolve(strict=True)
    raw = json.loads(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    _exact_keys(raw, _TOP_LEVEL_KEYS, "configuration")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError("configuration name must be a non-empty string")
    base = config_path.parent
    native = _model_input(base, raw["native"], "native")
    candidate = _model_input(base, raw["candidate"], "candidate")

    image_rows = raw["images"]
    if not isinstance(image_rows, list) or not image_rows:
        raise ValueError("images must be a non-empty list")
    images: dict[str, PinnedFile] = {}
    for index, row in enumerate(image_rows):
        if not isinstance(row, dict):
            raise ValueError(f"images[{index}] must be an object")
        _exact_keys(row, _IMAGE_KEYS, f"images[{index}]")
        name = row["name"]
        if not isinstance(name, str) or not name or name in images:
            raise ValueError("image names must be non-empty and unique")
        images[name] = _pinned(base, row["path"], row["sha256"], f"images[{index}]")

    gates = raw["gates"]
    if not isinstance(gates, dict):
        raise ValueError("gates must be an object")
    _exact_keys(gates, _GATE_KEYS, "gates")
    parsed_gates: dict[str, float] = {}
    for key, value in gates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"gates.{key} must be numeric")
        parsed_gates[key] = float(value)
    if not all(0.0 <= value <= 1.0 for value in parsed_gates.values()):
        raise ValueError("all gate values must be within [0,1]")
    return raw["name"], native, candidate, images, parsed_gates


def _manifest(pinned: PinnedFile, model: PinnedFile) -> VisionManifest:
    manifest = VisionManifest.from_yaml(pinned.path)
    return manifest.model_copy(update={"source": str(model.path)})


def execute(config: Path) -> dict[str, Any]:
    name, native_input, candidate_input, images, gates = _load_config(config)
    native_manifest = _manifest(native_input.manifest, native_input.model)
    candidate_manifest = _manifest(candidate_input.manifest, candidate_input.model)
    native = vision.load(str(native_input.model.path), device="cpu", manifest=native_manifest)
    candidate = vision.load(str(candidate_input.model.path), device="cpu", manifest=candidate_manifest)

    outcomes: dict[str, Any] = {}
    for image_name, pinned_image in images.items():
        reference = native(pinned_image.path)
        observed = candidate(pinned_image.path)
        detail = vision.validate(
            reference,
            observed,
            iou_threshold=gates["iou_threshold"],
            score_tolerance=gates["score_tolerance"],
            high_confidence=gates["high_confidence"],
        )
        detail.update(
            {
                "reference_count": len(reference.boxes),
                "candidate_count": len(observed.boxes),
                "reference_metadata": reference.metadata,
                "candidate_metadata": observed.metadata,
                "image_sha256": pinned_image.sha256,
                "image_size": Image.open(pinned_image.path).size,
            }
        )
        failed = (
            not reference.boxes
            or detail["lost"]
            or detail["added"]
            or detail["high_conf_lost"]
            or detail["high_conf_added"]
            or detail["class_mismatches"]
            or detail["mean_iou"] < gates["min_mean_iou"]
            or detail["min_iou"] < gates["min_iou"]
            or detail["mean_score_delta"] > gates["max_mean_score_delta"]
            or detail["max_score_delta"] > gates["max_score_delta"]
        )
        if failed:
            raise RuntimeError(f"{name}/{image_name} parity gate failed: {detail}")
        outcomes[image_name] = detail

    return {
        "status": "passed",
        "name": name,
        "at": datetime.now(timezone.utc).isoformat(),
        "hardware": platform.platform(),
        "python": sys.version,
        "mode": "offline-checksum-pinned",
        "native_sha256": native_input.model.sha256,
        "candidate_sha256": candidate_input.model.sha256,
        "images": outcomes,
        "gates": gates,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two local vision models using checksum-pinned manifests and images."
    )
    parser.add_argument("--config", required=True, type=Path, help="JSON configuration; all input paths need SHA-256")
    parser.add_argument("--output", required=True, type=Path, help="new JSON result path (never overwritten)")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = execute(args.config)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
