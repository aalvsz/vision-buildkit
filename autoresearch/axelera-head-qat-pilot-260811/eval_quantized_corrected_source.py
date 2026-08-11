#!/usr/bin/env python3
"""Corrected-source fork of the canonical raw CPU simulator harness.

The only behavioral/provenance change is the expected source ONNX digest; all
decoder, parity, threshold, CLI, and gate logic is imported from the canonical
harness unchanged.
"""
from pathlib import Path
import sys
import hashlib

CANONICAL = Path('/home/ubuntu/voyager-sdk/experiments/cv-engine-94527ce/eval_quantized_raw.py')
CORRECTED_SOURCE_SHA256 = '1b7c38d3da0003e94f5ba5ff13ab9de59b48c7c395c3fe208cfe8b829f09fc5b'
CANONICAL_SHA256 = '0d7552e9a0e2a09645e248b8202f9cc4628d5bd2d98f343cf9f91a687a84ba3a'
QUANTIZED_MANIFEST_SHA256 = 'bdb83c0ba59e480793b48ea09512023218209e56b6ff79ee7d5528186c787949'
if not CANONICAL.is_file():
    raise SystemExit('FAIL_CLOSED: canonical harness path is missing')
if hashlib.sha256(CANONICAL.read_bytes()).hexdigest() != CANONICAL_SHA256:
    raise SystemExit('FAIL_CLOSED: canonical harness SHA mismatch')
sys.path.insert(0, str(CANONICAL.parent))
import eval_quantized_raw as canonical  # noqa: E402
canonical.EXPECTED_RAW_SOURCE_SHA256 = CORRECTED_SOURCE_SHA256
if __name__ == '__main__':
    if '--output' not in sys.argv or Path(sys.argv[sys.argv.index('--output') + 1]).exists():
        raise SystemExit('FAIL_CLOSED: require a fresh --output directory')
    if '--quantized-manifest' not in sys.argv:
        raise SystemExit('FAIL_CLOSED: quantized manifest argument required')
    manifest = Path(sys.argv[sys.argv.index('--quantized-manifest') + 1])
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != QUANTIZED_MANIFEST_SHA256:
        raise SystemExit('FAIL_CLOSED: quantized manifest receipt SHA mismatch')
    canonical.main()
