"""Run and verify a small, content-bound local runtime benchmark suite.

This command deliberately produces development evidence under ``autoresearch/``.
It does not certify a backend, compare unlike hardware, or upload artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import load  # noqa: E402
from vision.benchmark import benchmark_model, verify_benchmark_report  # noqa: E402
from vision.manifest import VisionManifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--target",
        action="append",
        nargs=4,
        metavar=("NAME", "SOURCE", "MANIFEST", "DEVICE"),
        required=True,
        help="target group; repeat this option for each runtime",
    )
    args = parser.parse_args()
    if args.image.is_symlink() or not args.image.is_file():
        raise SystemExit("image must be a regular file")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, source, manifest_path, device in args.target:
        manifest = VisionManifest.from_yaml(manifest_path)
        model = load(source, device=device, manifest=manifest)
        report = benchmark_model(model, args.image, repeats=args.repeats, warmup=args.warmup)
        verified = verify_benchmark_report(report)
        path = args.output_dir / f"benchmark-{name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / f"benchmark-{name}-verification.json").write_text(
            json.dumps(verified, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append({"name": name, **verified, "runtime": report["runtime"]})
    print(json.dumps({"valid": True, "targets": reports}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
