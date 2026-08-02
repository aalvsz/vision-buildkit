import hashlib
import json
from pathlib import Path

import pytest

from scripts.integration_models import _load_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configuration(tmp_path: Path) -> dict:
    for name in ("native.bin", "native.yaml", "candidate.onnx", "candidate.yaml", "image.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    return {
        "name": "portable-offline-check",
        "native": {
            "model": "native.bin",
            "model_sha256": _sha(tmp_path / "native.bin"),
            "manifest": "native.yaml",
            "manifest_sha256": _sha(tmp_path / "native.yaml"),
        },
        "candidate": {
            "model": "candidate.onnx",
            "model_sha256": _sha(tmp_path / "candidate.onnx"),
            "manifest": "candidate.yaml",
            "manifest_sha256": _sha(tmp_path / "candidate.yaml"),
        },
        "images": [{"name": "sample", "path": "image.jpg", "sha256": _sha(tmp_path / "image.jpg")}],
        "gates": {
            "iou_threshold": 0.8,
            "score_tolerance": 0.01,
            "high_confidence": 0.5,
            "min_mean_iou": 0.99,
            "min_iou": 0.97,
            "max_mean_score_delta": 0.01,
            "max_score_delta": 0.02,
        },
    }


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path


def test_portable_integration_config_requires_pinned_local_inputs(tmp_path):
    name, native, candidate, images, gates = _load_config(_write_config(tmp_path, _configuration(tmp_path)))
    assert name == "portable-offline-check"
    assert native.model.path == (tmp_path / "native.bin").resolve()
    assert candidate.model.sha256 == _sha(tmp_path / "candidate.onnx")
    assert tuple(images) == ("sample",)
    assert gates["min_mean_iou"] == 0.99


def test_portable_integration_config_rejects_unknown_keys_and_digest_drift(tmp_path):
    value = _configuration(tmp_path)
    value["private_cache"] = "hidden"
    with pytest.raises(ValueError, match="configuration keys must be exactly"):
        _load_config(_write_config(tmp_path, value))

    value = _configuration(tmp_path)
    value["candidate"]["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _load_config(_write_config(tmp_path, value))


def test_portable_integration_config_rejects_symlink_inputs(tmp_path):
    value = _configuration(tmp_path)
    link = tmp_path / "linked.jpg"
    link.symlink_to(tmp_path / "image.jpg")
    value["images"][0] = {"name": "sample", "path": "linked.jpg", "sha256": _sha(link)}
    with pytest.raises(ValueError, match="non-symlink"):
        _load_config(_write_config(tmp_path, value))
