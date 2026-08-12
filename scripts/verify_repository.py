"""Fail-closed repository integrity checks used by local development and CI."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.experiment_evidence import load_experiment_report  # noqa: E402
from vision.support import load_support_matrix  # noqa: E402

_LEGACY_STATUSES = {"PASS", "FAIL", "PARTIAL_PASS"}
_LEGACY_KINDS = {"observed_unsigned_development"}
_BINARY_SUFFIXES = (".onnx", ".pt", ".pth", ".engine", ".vcap", ".safetensors")
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
)
_BENCHMARK_STATUSES = {"executed", "planned", "tooling_ready"}


def _tracked_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")[:-1]


def _verify_evidence() -> tuple[int, int]:
    legacy = canonical = 0
    for path in sorted((ROOT / "docs" / "evidence").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema_version") if isinstance(payload, dict) else None
        if schema == 2:
            load_experiment_report(path)
            canonical += 1
            continue
        if schema != 1 or payload.get("evidence_kind") not in _LEGACY_KINDS:
            raise ValueError(f"unsupported evidence schema or kind: {path}")
        if payload.get("status") not in _LEGACY_STATUSES:
            raise ValueError(f"invalid legacy evidence status: {path}")
        if payload.get("publishable") is not False or payload.get("certified") is not False:
            raise ValueError(f"legacy evidence must remain non-publishable and uncertified: {path}")
        legacy += 1
    return legacy, canonical


def _verify_no_secrets(tracked: list[str]) -> None:
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 4_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"secret-like material detected in tracked file: {relative}")


def _verify_benchmark_plan() -> None:
    payload = yaml.safe_load((ROOT / "docs" / "benchmark-plan.yml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("benchmark plan must have schema_version=1")
    scopes = payload.get("scopes")
    if not isinstance(scopes, list) or not scopes:
        raise ValueError("benchmark plan must contain scopes")
    ids: set[str] = set()
    for scope in scopes:
        if not isinstance(scope, dict) or not isinstance(scope.get("id"), str):
            raise ValueError("benchmark scopes require string ids")
        if scope["id"] in ids:
            raise ValueError(f"duplicate benchmark scope: {scope['id']}")
        if scope.get("status") not in _BENCHMARK_STATUSES:
            raise ValueError(f"invalid benchmark status: {scope['id']}")
        if not isinstance(scope.get("metrics"), list) or not scope["metrics"]:
            raise ValueError(f"benchmark scope has no metrics: {scope['id']}")
        ids.add(scope["id"])


def main() -> int:
    load_support_matrix()
    _verify_benchmark_plan()
    legacy, canonical = _verify_evidence()
    tracked = _tracked_files()
    _verify_no_secrets(tracked)
    forbidden = [p for p in tracked if p.lower().endswith(_BINARY_SUFFIXES)]
    if forbidden:
        raise ValueError("generated model/runtime artifacts must not be tracked: " + ", ".join(forbidden))
    print(
        json.dumps(
            {
                "support_matrix": "valid",
                "evidence_files": legacy + canonical,
                "legacy_evidence": legacy,
                "canonical_evidence": canonical,
                "tracked_artifact_policy": "valid",
                "secret_scan": "clean",
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"repository verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
