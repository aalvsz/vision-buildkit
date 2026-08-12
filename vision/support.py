"""Machine-readable capability and repository support reporting."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_MATRIX = _ROOT / "docs" / "support-matrix.yml"
_STATUSES = {"api_only", "build_only", "development_verified", "certified", "unsupported"}


def load_support_matrix(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the repository support matrix."""
    matrix_path = Path(path) if path else _MATRIX
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("support matrix must be a mapping with schema_version=1")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("support matrix entries must be a non-empty list")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("support matrix entries must be mappings")
        entry_id = entry.get("id")
        status = entry.get("status")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("support matrix entry ids must be non-empty strings")
        if entry_id in seen:
            raise ValueError(f"duplicate support matrix id: {entry_id}")
        if status not in _STATUSES:
            raise ValueError(f"unsupported status {status!r} for {entry_id}")
        if not isinstance(entry.get("evidence"), list):
            raise ValueError(f"evidence must be a list for {entry_id}")
        if status in {"development_verified", "certified"} and not entry["evidence"]:
            raise ValueError(f"verified entry {entry_id} must bind evidence")
        for evidence in entry["evidence"]:
            if not isinstance(evidence, str) or not evidence.startswith("docs/evidence/"):
                raise ValueError(f"evidence path is not repository-relative: {evidence!r}")
            if not (_ROOT / evidence).is_file():
                raise ValueError(f"evidence path does not exist: {evidence}")
        seen.add(entry_id)
    return payload


def support_report(path: str | Path | None = None) -> dict[str, Any]:
    matrix = load_support_matrix(path)
    return {"schema_version": matrix["schema_version"], "entries": matrix["entries"], "policy": matrix["policy"]}


def doctor_report() -> dict[str, Any]:
    """Report local runtime capabilities without claiming execution support."""
    optional = {
        name: bool(importlib.util.find_spec(name))
        for name in ("onnxruntime", "onnx", "openvino", "torch", "ultralytics", "cv2", "pycocotools")
    }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "optional_modules": optional,
        "support_matrix": str(_MATRIX.relative_to(_ROOT)),
        "note": "Installed modules and device detection do not constitute paired runtime verification.",
    }
