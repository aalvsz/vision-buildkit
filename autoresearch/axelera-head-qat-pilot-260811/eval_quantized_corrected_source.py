#!/usr/bin/env python3
"""Corrected-source fork of the canonical raw CPU simulator harness.

The only behavioral/provenance change is the expected source ONNX digest; all
decoder, parity, threshold, CLI, and gate logic is imported from the canonical
harness unchanged.
"""
from pathlib import Path
import sys

CANONICAL = Path('/home/ubuntu/vision-buildkit/autoresearch/axelera-int8-accuracy-260811/eval_quantized_raw.py')
CORRECTED_SOURCE_SHA256 = '1b7c38d3da0003e94f5ba5ff13ab9de59b48c7c395c3fe208cfe8b829f09fc5b'
if not CANONICAL.is_file():
    raise SystemExit('FAIL_CLOSED: canonical harness path is missing')
sys.path.insert(0, str(CANONICAL.parent))
import eval_quantized_raw as canonical  # noqa: E402
canonical.EXPECTED_RAW_SOURCE_SHA256 = CORRECTED_SOURCE_SHA256
if __name__ == '__main__':
    canonical.main()
