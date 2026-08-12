"""Create and verify a content-bound local release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _tracked() -> list[str]:
    return _git("ls-files", "-z").split("\0")[:-1]


def build_manifest(*, allow_dirty: bool = False) -> dict[str, Any]:
    status = _git("status", "--porcelain")
    if status and not allow_dirty:
        raise ValueError("release manifest requires a clean working tree; use --allow-dirty for a local diagnostic")
    tracked = _tracked()
    files = [{"path": path, "sha256": _sha256(ROOT / path)} for path in tracked if (ROOT / path).is_file()]
    lock_digest = _sha256(ROOT / "uv.lock")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vision-buildkit-release-manifest",
        "project": tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"],
        "version": tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"],
        "git_commit": _git("rev-parse", "HEAD"),
        "working_tree_clean": not bool(status),
        "lockfile_sha256": lock_digest,
        "tracked_file_count": len(files),
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != "vision-buildkit-release-manifest":
        raise ValueError("unsupported release manifest")
    expected = dict(payload)
    recorded = expected.pop("manifest_sha256", None)
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if recorded != hashlib.sha256(canonical).hexdigest():
        raise ValueError("release manifest self-digest mismatch")
    if payload["git_commit"] != _git("rev-parse", "HEAD"):
        raise ValueError("release manifest is bound to a different Git commit")
    current = {item["path"]: item["sha256"] for item in payload["files"]}
    actual = {relative: _sha256(ROOT / relative) for relative in _tracked() if (ROOT / relative).is_file()}
    if current != actual:
        raise ValueError("tracked file inventory or digest changed since manifest creation")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("output", type=Path)
    create.add_argument("--allow-dirty", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        payload = build_manifest(allow_dirty=args.allow_dirty)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        payload = verify_manifest(args.manifest)
    print(
        json.dumps(
            {"valid": True, "manifest_sha256": payload["manifest_sha256"], "files": payload["tracked_file_count"]}
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"release manifest failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
