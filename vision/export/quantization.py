from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..manifest import CalibrationManifest


def calibration_arrays(manifest: CalibrationManifest) -> Iterator[np.ndarray]:
    """Stream validated samples; never accepts in-memory bypass batches."""
    expected_dtype = np.dtype(manifest.dtype)
    for sample in manifest.samples:
        path = Path(sample.path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"calibration sample is not a regular file: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample.sha256:
            raise ValueError(f"calibration sample digest mismatch: {path}")
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != tuple(manifest.shape):
            raise ValueError(f"calibration sample shape mismatch: {path}")
        if array.dtype != expected_dtype:
            raise ValueError(f"calibration sample dtype mismatch: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"calibration sample contains non-finite values: {path}")
        yield array


def quantize_ort_static(onnx_path: str, output: str, calibration: CalibrationManifest) -> str:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._items = iter({calibration.input_name: array} for array in calibration_arrays(calibration))

        def get_next(self):
            return next(self._items, None)

    quantize_static(onnx_path, output, Reader(), quant_format=QuantFormat.QDQ)
    import onnx

    ops = {node.op_type for node in onnx.load(output).graph.node}
    if not {"QuantizeLinear", "DequantizeLinear"}.issubset(ops):
        raise RuntimeError("ORT static quantization did not emit QDQ nodes")
    return output
