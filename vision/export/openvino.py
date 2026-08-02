from __future__ import annotations

from pathlib import Path

from ..manifest import CalibrationManifest
from .quantization import calibration_arrays


def _compatibility_source(source: Path, directory: Path) -> tuple[Path, list[dict[str, str | int]]]:
    """Translate narrowly equivalent ONNX spellings required by OpenVINO.

    ONNX GridSample calls the 2D linear interpolation mode ``linear`` while
    OpenVINO's operator enum calls the same operation ``bilinear``.  The
    rewritten intermediate is OpenVINO-only; parity is still measured against
    the untouched ONNX source before an artifact is accepted.
    """

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("OpenVINO compatibility inspection requires onnx") from exc
    model = onnx.load(str(source))
    rewrites: list[dict[str, str | int]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "GridSample":
            continue
        for attribute in node.attribute:
            if attribute.name == "mode" and attribute.s == b"linear":
                attribute.s = b"bilinear"
                rewrites.append(
                    {
                        "node_index": node_index,
                        "op_type": "GridSample",
                        "attribute": "mode",
                        "from": "linear",
                        "to": "bilinear",
                    }
                )
    if not rewrites:
        return source, rewrites
    candidate = directory / "openvino-compat.onnx"
    onnx.save(model, str(candidate))
    return candidate, rewrites


def build_openvino(source: Path, output: Path, precision: str, calibration: CalibrationManifest | None):
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("OpenVINO unavailable; install vision-buildkit[openvino]") from exc
    compatible_source, rewrites = _compatibility_source(source, output.parent)
    model = ov.convert_model(str(compatible_source))
    try:
        import onnx

        onnx_model = onnx.load(str(source), load_external_data=False)
        source_inputs = [item.name for item in onnx_model.graph.input]
        if len(source_inputs) != len(model.inputs):
            raise RuntimeError("OpenVINO conversion changed the model input count")
        for port, name in zip(model.inputs, source_inputs, strict=True):
            port.get_tensor().set_names({name})
    except ImportError as exc:  # pragma: no cover - openvino extra includes onnx
        raise RuntimeError("OpenVINO conversion requires onnx for stable input names") from exc
    if precision == "int8":
        if calibration is None:
            raise ValueError("OpenVINO INT8 requires calibration_manifest_path")
        try:
            import nncf
        except ImportError as exc:
            raise RuntimeError("OpenVINO INT8 requires nncf>=2.19") from exc
        model = nncf.quantize(
            model, nncf.Dataset({calibration.input_name: array} for array in calibration_arrays(calibration))
        )
    ov.save_model(model, str(output), compress_to_fp16=precision == "fp16")
    if not output.is_file() or not output.with_suffix(".bin").is_file():
        raise RuntimeError("OpenVINO did not create a complete IR artifact")
    saved = ov.Core().read_model(str(output))
    type_names = [str(port.get_element_type()).lower() for port in [*saved.inputs, *saved.outputs]]
    xml = output.read_text(errors="ignore").lower()
    operations = list(saved.get_ops())
    fake_quantize_count = sum(op.get_type_name() == "FakeQuantize" for op in operations)
    quantized_constant_count = sum(
        op.get_type_name() == "Constant"
        and any(
            str(output.get_element_type()).lower() in {"<type: 'int8_t'>", "<type: 'uint8_t'>"}
            for output in op.outputs()
        )
        for op in operations
    )
    # Integer result tensors and ordinary integer constants are valid in FP32
    # detectors (for example class IDs).  NNCF PTQ is accepted only when the
    # saved IR contains both calibrated FakeQuantize structure and quantized
    # constants; incidental i8/u8 types alone are not precision evidence.
    has_fake_quantize = fake_quantize_count > 0
    has_quant_ops = has_fake_quantize and quantized_constant_count > 0
    has_f16 = "f16" in xml or any("f16" in name for name in type_names)
    inspection = {
        "io_element_types": type_names,
        "fake_quantize": has_fake_quantize,
        "quantization_ops": has_quant_ops,
        "fake_quantize_count": fake_quantize_count,
        "quantized_constant_count": quantized_constant_count,
        "int8_constants": quantized_constant_count > 0,
        "f16_tensors": has_f16,
    }
    effective = "int8" if has_quant_ops else "fp16" if has_f16 else "fp32"
    if effective != precision:
        raise RuntimeError(f"OpenVINO IR inspection found {effective}, requested {precision}: {inspection}")
    metadata = {
        "inputs": [
            {"name": port.get_any_name(), "shape": [str(dim) for dim in port.partial_shape]} for port in model.inputs
        ],
        "outputs": [port.get_any_name() for port in model.outputs],
        "compatibility_rewrites": rewrites,
    }
    metadata["precision_inspection"] = inspection
    return output, metadata, effective


def convert_openvino(source, output, precision="fp32"):
    artifact, _, _ = build_openvino(Path(source), Path(output).with_suffix(".xml"), precision, None)
    return str(artifact)
