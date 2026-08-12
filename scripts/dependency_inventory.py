"""Emit a deterministic locked-dependency inventory for release provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def build_inventory() -> dict:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = []
    for package in lock.get("package", []):
        entry = {"name": package["name"], "version": package["version"]}
        if "source" in package:
            entry["source"] = package["source"]
        packages.append(entry)
    packages.sort(key=lambda item: (item["name"], item["version"]))
    payload = {
        "schema_version": 1,
        "format": "locked-python-dependency-inventory",
        "lockfile": "uv.lock",
        "lockfile_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "packages": packages,
    }
    payload["inventory_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_inventory()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    print(
        json.dumps(
            {"valid": True, "packages": len(payload["packages"]), "inventory_sha256": payload["inventory_sha256"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
