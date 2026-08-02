import hashlib
import json
import os
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from pydantic import ValidationError

import vision.plugins as plugins_module
from vision.buildkit import BuildPlanner
from vision.contract import VisionContract
from vision.manifest import CalibrationManifest
from vision.plugins import (
    DEFAULT_REGISTRY,
    ManualOnnxSource,
    OnnxRuntimeTargetPlugin,
    OpenVinoTargetPlugin,
    make_openvino_target,
)
from vision.preprocess import PreparedInput


def _tiny_onnx(path: Path, *, score: float = 0.9) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 1])
    c = helper.make_tensor_value_info("c", TensorProto.INT64, [1, 1])
    nodes = [
        helper.make_node(
            "Constant", [], ["b"], value=helper.make_tensor("bb", TensorProto.FLOAT, [1, 1, 4], [0, 0, 1, 1])
        ),
        helper.make_node("Constant", [], ["s"], value=helper.make_tensor("ss", TensorProto.FLOAT, [1, 1], [score])),
        helper.make_node("Constant", [], ["c"], value=helper.make_tensor("cc", TensorProto.INT64, [1, 1], [0])),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "tiny", [x], [b, s, c]), opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, path)


def _tiny_quantizable_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 1])
    c = helper.make_tensor_value_info("c", TensorProto.INT64, [1, 1])
    weights_b = helper.make_tensor("weights_b", TensorProto.FLOAT, [12, 4], [0.01] * 48)
    weights_s = helper.make_tensor("weights_s", TensorProto.FLOAT, [12, 1], [0.02] * 12)
    boxes_shape = helper.make_tensor("boxes_shape", TensorProto.INT64, [3], [1, 1, 4])
    scores_shape = helper.make_tensor("scores_shape", TensorProto.INT64, [2], [1, 1])
    nodes = [
        helper.make_node("Flatten", ["x"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "weights_b"], ["boxes_flat"]),
        helper.make_node("Reshape", ["boxes_flat", "boxes_shape"], ["b"]),
        helper.make_node("MatMul", ["flat", "weights_s"], ["score_logit"]),
        helper.make_node("Sigmoid", ["score_logit"], ["score_flat"]),
        helper.make_node("Reshape", ["score_flat", "scores_shape"], ["s"]),
        helper.make_node("Constant", [], ["c"], value=helper.make_tensor("cc", TensorProto.INT64, [1, 1], [0])),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "tiny-quantizable", [x], [b, s, c], [weights_b, weights_s, boxes_shape, scores_shape]),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    onnx.save(model, path)


def _tiny_uint8_classes_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 1])
    c = helper.make_tensor_value_info("c", TensorProto.UINT8, [1, 1])
    nodes = [
        helper.make_node(
            "Constant", [], ["b"], value=helper.make_tensor("bb", TensorProto.FLOAT, [1, 1, 4], [0, 0, 1, 1])
        ),
        helper.make_node("Constant", [], ["s"], value=helper.make_tensor("ss", TensorProto.FLOAT, [1, 1], [0.9])),
        helper.make_node("Constant", [], ["c"], value=helper.make_tensor("cc", TensorProto.UINT8, [1, 1], [0])),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "tiny-uint8-classes", [x], [b, s, c]), opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, path)


def _calibration_manifest(tmp_path: Path, contract: VisionContract) -> CalibrationManifest:
    samples = []
    for index, value in enumerate((0.0, 0.5, 1.0)):
        path = tmp_path / f"calibration-{index}.npy"
        np.save(path, np.full((1, 3, 2, 2), value, dtype=np.float32))
        samples.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    preprocessing_digest = OpenVinoTargetPlugin._preprocessing_digest(contract)
    body = {
        "samples": samples,
        "sample_count": len(samples),
        "preprocessing_digest": preprocessing_digest,
        "input_name": contract.input.input_name,
        "shape": list(contract.input.shape),
        "dtype": contract.input.dtype,
    }
    body["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CalibrationManifest.model_validate(body)


def _changed_calibration(calibration: CalibrationManifest, **updates) -> CalibrationManifest:
    body = calibration.model_dump(exclude={"digest"}, mode="json")
    body.update(updates)
    body["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CalibrationManifest.model_validate(body)


def _contract(digest: str) -> VisionContract:
    return VisionContract.model_validate(
        {
            "input": {"color": "RGB", "layout": "NCHW", "dtype": "float32", "input_name": "x", "shape": [1, 3, 2, 2]},
            "preprocessing": [{"kind": "resize", "height": 2, "width": 2, "interpolation": "bilinear"}],
            "raw_output": {
                "semantics": "boxes_scores_classes",
                "tensors": [
                    {"name": "b", "dtype": "float32", "shape": [1, 1, 4]},
                    {"name": "s", "dtype": "float32", "shape": [1, 1]},
                    {"name": "c", "dtype": "int64", "shape": [1, 1]},
                ],
                "roles": {"boxes": "b", "scores": "s", "classes": "c"},
            },
            "result": {
                "box_format": "xyxy",
                "coordinate_space": "original_image",
                "nms_owner": "source",
                "nms_iou_threshold": 0.5,
                "ordering": "score_desc",
                "tie_policy": "stable",
                "score_threshold": 0.0,
                "class_aware_nms": True,
                "clip_boxes": True,
                "max_detections": 10,
                "suppression": "greedy_iou",
            },
            "labels": ["person"],
            "coco_category_ids": [1],
            "tolerances": {"box_iou": 0.9, "score_delta": 0.01},
            "provenance": {"source": "tiny", "source_sha256": digest},
        }
    )


def _uint8_classes_contract(digest: str) -> VisionContract:
    body = _contract(digest).model_dump(mode="json")
    body["raw_output"]["tensors"][2]["dtype"] = "uint8"
    return VisionContract.model_validate(body)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_openvino_real_neutral_fp32_fp16_matches_ort(tmp_path: Path, precision: str):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    target = OpenVinoTargetPlugin(tmp_path / f"{precision}.xml", precision=precision)
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    sample = PreparedInput(np.zeros((1, 3, 2, 2), dtype=np.float32), (2, 2), "a" * 64)
    result = target.run(artifact, sample)
    ort = OnnxRuntimeTargetPlugin().prepare(tmp_path / "neutral.onnx", contract)
    expected = OnnxRuntimeTargetPlugin().run(ort, sample)
    assert result.boxes == expected.boxes and result.classes == expected.classes
    assert result.scores == pytest.approx(expected.scores, abs=contract.tolerances.score_delta)
    assert result.metadata["execution_device"] == "CPU" and target.verify_artifact(artifact, contract)


def test_openvino_real_neutral_int8_matches_ort_within_contract_tolerance(tmp_path: Path):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _calibration_manifest(tmp_path, contract)
    target = OpenVinoTargetPlugin(tmp_path / "int8.xml", precision="int8", calibration=calibration)
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    sample = PreparedInput(np.zeros((1, 3, 2, 2), dtype=np.float32), (2, 2), "b" * 64)
    result = target.run(artifact, sample)
    ort_target = OnnxRuntimeTargetPlugin()
    expected = ort_target.run(ort_target.prepare(tmp_path / "neutral.onnx", contract), sample)
    assert result.boxes == pytest.approx(expected.boxes, abs=0.01)
    assert result.scores == pytest.approx(expected.scores, abs=contract.tolerances.score_delta)
    assert result.classes == expected.classes
    assert dict(artifact.descriptors)["effective_precision"] == "int8"


def test_openvino_fp32_allows_contract_valid_uint8_class_output(tmp_path: Path):
    source = tmp_path / "source.onnx"
    _tiny_uint8_classes_onnx(source)
    contract = _uint8_classes_contract(hashlib.sha256(source.read_bytes()).hexdigest())
    target = OpenVinoTargetPlugin(tmp_path / "fp32.xml", precision="fp32")
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    sample = PreparedInput(np.zeros((1, 3, 2, 2), dtype=np.float32), (2, 2), "f" * 64)
    result = target.run(artifact, sample)
    receipt = json.loads(Path(dict(artifact.descriptors)["receipt_path"]).read_text())
    assert result.classes == [0]
    assert receipt["precision"] == "fp32"
    assert receipt["metadata"]["precision_inspection"]["quantization_ops"] is False


def test_openvino_int8_rejects_nonquantized_ir_even_with_calibration(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _calibration_manifest(tmp_path, contract)
    import nncf

    monkeypatch.setattr(nncf, "quantize", lambda model, dataset: model)
    output = tmp_path / "int8.xml"
    with pytest.raises(RuntimeError, match="found fp32, requested int8"):
        BuildPlanner().materialize(
            ManualOnnxSource(),
            OpenVinoTargetPlugin(output, precision="int8", calibration=calibration),
            source,
            contract,
            tmp_path / "neutral.onnx",
        )
    assert not any(path.exists() for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json")))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_name", "wrong", "contradicts contract"),
        ("shape", [1, 3, 4, 4], "contradicts contract"),
        ("dtype", "float16", "contradicts contract"),
        ("preprocessing_digest", "0" * 64, "contradicts contract"),
    ],
)
def test_openvino_int8_calibration_contract_mismatch_fails_before_publication(
    tmp_path: Path, field: str, value, message: str
):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _changed_calibration(_calibration_manifest(tmp_path, contract), **{field: value})
    output = tmp_path / "int8.xml"
    with pytest.raises(ValueError, match=message):
        BuildPlanner().materialize(
            ManualOnnxSource(),
            OpenVinoTargetPlugin(output, precision="int8", calibration=calibration),
            source,
            contract,
            tmp_path / "neutral.onnx",
        )
    assert not any(path.exists() or path.is_symlink() for path in (output, output.with_suffix(".bin")))


def test_openvino_int8_calibration_sample_hash_mismatch_fails_closed(tmp_path: Path):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _calibration_manifest(tmp_path, contract)
    samples = calibration.model_dump(mode="json")["samples"]
    samples[0]["sha256"] = "0" * 64
    calibration = _changed_calibration(calibration, samples=samples)
    output = tmp_path / "int8.xml"
    with pytest.raises(ValueError, match="calibration sample digest mismatch"):
        BuildPlanner().materialize(
            ManualOnnxSource(),
            OpenVinoTargetPlugin(output, precision="int8", calibration=calibration),
            source,
            contract,
            tmp_path / "neutral.onnx",
        )
    assert not any(
        path.exists() or path.is_symlink()
        for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json"))
    )
    assert not list(tmp_path.glob(".int8.openvino-*"))


def test_openvino_int8_calibration_load_is_bound_to_verified_bytes(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _calibration_manifest(tmp_path, contract)
    sample = Path(calibration.samples[0].path)
    declared_sha = calibration.samples[0].sha256
    from vision.export import quantization

    original_load = quantization.np.load
    observed: dict[str, str] = {}

    def replace_path_before_decode(file, *args, **kwargs):
        if not observed:
            observed["loaded_sha"] = hashlib.sha256(file.getvalue()).hexdigest()
            np.save(sample, np.full((1, 3, 2, 2), 0.75, dtype=np.float32))
            observed["replacement_sha"] = hashlib.sha256(sample.read_bytes()).hexdigest()
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(quantization.np, "load", replace_path_before_decode)
    arrays = list(quantization.calibration_arrays(calibration))
    assert observed["loaded_sha"] == declared_sha
    assert observed["replacement_sha"] != declared_sha
    assert np.array_equal(arrays[0], np.zeros((1, 3, 2, 2), dtype=np.float32))


def test_openvino_int8_owns_deep_frozen_calibration_snapshot(tmp_path: Path):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _calibration_manifest(tmp_path, contract)
    target = OpenVinoTargetPlugin(tmp_path / "int8.xml", precision="int8", calibration=calibration)
    assert target.calibration is not calibration
    assert target.calibration is not None
    assert target.calibration.digest == target.calibration.canonical_digest()
    with pytest.raises(ValidationError):
        calibration.samples[0].path = str(tmp_path / "replacement.npy")
    with pytest.raises(TypeError):
        calibration.samples[0] = calibration.samples[0]
    with pytest.raises(TypeError):
        calibration.shape[0] = calibration.shape[0]


def test_openvino_int8_concurrent_caller_manifest_mutation_cannot_change_build(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _calibration_manifest(tmp_path, contract)
    original_digest = calibration.digest
    original_sample_sha = calibration.samples[0].sha256
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, np.full((1, 3, 2, 2), 37.0, dtype=np.float32))
    replacement_sample = type(calibration.samples[0])(
        path=str(replacement), sha256=hashlib.sha256(replacement.read_bytes()).hexdigest()
    )
    target = OpenVinoTargetPlugin(tmp_path / "int8.xml", precision="int8", calibration=calibration)
    from vision.export import openvino as exporter

    original_build = exporter.build_openvino

    def mutate_caller_during_build(snapshot, output, precision, frozen_calibration):
        thread = threading.Thread(
            target=lambda: object.__setattr__(calibration, "samples", (replacement_sample, *calibration.samples[1:]))
        )
        thread.start()
        thread.join()
        return original_build(snapshot, output, precision, frozen_calibration)

    monkeypatch.setattr(exporter, "build_openvino", mutate_caller_during_build)
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    receipt = json.loads(Path(dict(artifact.descriptors)["receipt_path"]).read_text())
    assert calibration.samples[0].sha256 != original_sample_sha
    assert receipt["calibration"]["digest"] == original_digest
    assert target.calibration is not None and target.calibration.samples[0].sha256 == original_sample_sha
    assert target.verify_artifact(artifact, contract)


def test_openvino_target_configuration_reassignment_is_rejected(tmp_path: Path):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    calibration = _calibration_manifest(tmp_path, contract)
    target = OpenVinoTargetPlugin(tmp_path / "int8.xml", precision="int8", calibration=calibration)
    with pytest.raises(AttributeError, match="configuration is immutable"):
        target.calibration = calibration
    with pytest.raises(AttributeError, match="configuration is immutable"):
        target.precision = "fp32"
    with pytest.raises(AttributeError, match="configuration is immutable"):
        target.output = tmp_path / "other.xml"


def test_openvino_int8_build_uses_one_local_snapshot_under_forced_target_race(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_quantizable_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    original_calibration = _calibration_manifest(tmp_path, contract)
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, np.full((1, 3, 2, 2), 37.0, dtype=np.float32))
    samples = original_calibration.model_dump(mode="json")["samples"]
    samples[0] = {"path": str(replacement), "sha256": hashlib.sha256(replacement.read_bytes()).hexdigest()}
    alternate_calibration = _changed_calibration(original_calibration, samples=samples)
    target = OpenVinoTargetPlugin(tmp_path / "int8.xml", precision="int8", calibration=original_calibration)
    original_identity, changed = target._identity_at, False

    def force_target_race(directory_fd, name):
        nonlocal changed
        identity = original_identity(directory_fd, name)
        if name == ".verified-source.onnx" and not changed:
            changed = True
            object.__setattr__(target, "calibration", alternate_calibration)
        return identity

    monkeypatch.setattr(target, "_identity_at", force_target_race)
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    object.__setattr__(target, "calibration", original_calibration)
    receipt = json.loads(Path(dict(artifact.descriptors)["receipt_path"]).read_text())
    assert changed
    assert receipt["calibration"]["digest"] == original_calibration.digest
    assert receipt["calibration"]["digest"] != alternate_calibration.digest
    assert target.verify_artifact(artifact, contract)


def test_openvino_receipt_tamper_and_collision_fail_closed(tmp_path: Path):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    receipt = Path(dict(artifact.descriptors)["receipt_path"])
    receipt.write_bytes(b"{}")
    assert not target.verify_artifact(artifact, contract)
    with pytest.raises(FileExistsError):
        BuildPlanner().materialize(
            ManualOnnxSource(), OpenVinoTargetPlugin(output), source, contract, tmp_path / "other.onnx"
        )


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_openvino_link_failure_leaves_zero_owned_residue(tmp_path: Path, monkeypatch, failure_at: int):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    original, calls = plugins_module.os.link, 0

    def injected(src, dst, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected link")
        return original(src, dst, **kwargs)

    monkeypatch.setattr(plugins_module.os, "link", injected)
    with pytest.raises(OSError, match="injected link"):
        BuildPlanner().materialize(
            ManualOnnxSource(), OpenVinoTargetPlugin(output), source, contract, tmp_path / "neutral.onnx"
        )
    assert not any(
        path.exists() or path.is_symlink()
        for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json"))
    )
    assert not list(tmp_path.glob(".model.openvino-*"))


@pytest.mark.parametrize("failure_at", range(1, 4))
def test_openvino_file_fsync_failure_leaves_zero_owned_residue(tmp_path: Path, monkeypatch, failure_at: int):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, calls = target._fsync_public_at, 0

    def injected(parent_fd, name):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected file fsync")
        return original(parent_fd, name)

    monkeypatch.setattr(target, "_fsync_public_at", injected)
    with pytest.raises(OSError, match="injected file fsync"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert not any(
        path.exists() or path.is_symlink()
        for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json"))
    )
    assert not list(tmp_path.glob(".model.openvino-*"))


@pytest.mark.parametrize("failure_at", range(1, 4))
def test_openvino_staged_fsync_failure_leaves_zero_private_residue(tmp_path: Path, monkeypatch, failure_at: int):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, calls = target._fsync_at, 0

    def injected(directory_fd, name):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected staged fsync")
        return original(directory_fd, name)

    monkeypatch.setattr(target, "_fsync_at", injected)
    with pytest.raises(OSError, match="injected staged fsync"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert not any(
        path.exists() or path.is_symlink()
        for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json"))
    )
    assert not list(tmp_path.glob(".model.openvino-*"))


@pytest.mark.parametrize("failure_at", range(1, 5))
def test_openvino_directory_fsync_failure_leaves_zero_owned_residue(tmp_path: Path, monkeypatch, failure_at: int):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, calls = target._fsync_parent, 0

    def injected(parent_fd):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected directory fsync")
        return original(parent_fd)

    monkeypatch.setattr(target, "_fsync_parent", injected)
    with pytest.raises(OSError, match="injected directory fsync"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert not any(
        path.exists() or path.is_symlink()
        for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json"))
    )
    assert not list(tmp_path.glob(".model.openvino-*"))


def test_openvino_rehash_race_removes_owned_publication(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, calls = target._fsync_parent, 0

    def mutate_after_commit_marker(parent_fd):
        nonlocal calls
        calls += 1
        original(parent_fd)
        if calls == 3:
            output.write_bytes(output.read_bytes() + b"tamper")

    monkeypatch.setattr(target, "_fsync_parent", mutate_after_commit_marker)
    with pytest.raises(RuntimeError, match="changed before commit"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert not any(
        path.exists() or path.is_symlink()
        for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json"))
    )


def test_openvino_cleanup_never_deletes_attacker_replacement(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original = target._fsync_public_at

    def replace_first_final(parent_fd, name):
        if name == output.name:
            os.unlink(name, dir_fd=parent_fd)
            output.write_bytes(b"attacker-owned")
            raise OSError("attacker race")
        return original(parent_fd, name)

    monkeypatch.setattr(target, "_fsync_public_at", replace_first_final)
    with pytest.raises(OSError, match="attacker race"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert output.read_bytes() == b"attacker-owned"
    assert not output.with_suffix(".bin").exists()
    assert not output.with_suffix(".receipt.json").exists()
    assert not list(tmp_path.glob(".model.openvino-*"))


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_openvino_parent_replacement_at_each_link_fails_closed(tmp_path: Path, monkeypatch, failure_at: int):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    parent = tmp_path / "out"
    parent.mkdir()
    moved = tmp_path / "moved-original-parent"
    output = parent / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, calls = plugins_module.os.link, 0

    def replace_parent(src, dst, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            parent.rename(moved)
            parent.mkdir()
            (parent / "attacker-owned").write_bytes(b"attacker")
        return original(src, dst, **kwargs)

    monkeypatch.setattr(plugins_module.os, "link", replace_parent)
    with pytest.raises(RuntimeError, match="output parent identity changed"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert {path.name for path in parent.iterdir()} == {"attacker-owned"}
    assert (parent / "attacker-owned").read_bytes() == b"attacker"
    assert moved.is_dir() and not list(moved.iterdir())
    assert target._artifact_receipts == {}


def test_openvino_parent_replacement_during_private_cleanup_rolls_back_publication(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    parent = tmp_path / "out"
    parent.mkdir()
    moved = tmp_path / "moved-original-parent"
    output = parent / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, calls = target._fsync_stage, 0

    def replace_parent_on_cleanup(stage_fd):
        nonlocal calls
        calls += 1
        original(stage_fd)
        if calls == 2:
            parent.rename(moved)
            parent.mkdir()
            (parent / "attacker-owned").write_bytes(b"attacker")

    monkeypatch.setattr(target, "_fsync_stage", replace_parent_on_cleanup)
    with pytest.raises(RuntimeError, match="output parent identity changed"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert {path.name for path in parent.iterdir()} == {"attacker-owned"}
    assert (parent / "attacker-owned").read_bytes() == b"attacker"
    assert moved.is_dir() and not list(moved.iterdir())
    assert target._artifact_receipts == {}


def test_openvino_parent_replacement_after_preissuance_check_revokes_token(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    parent = tmp_path / "out"
    parent.mkdir()
    moved = tmp_path / "moved-original-parent"
    output = parent / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, calls = target._assert_parent_path, 0

    def replace_after_preissuance_check(path, expected):
        nonlocal calls
        calls += 1
        original(path, expected)
        if calls == 8:
            parent.rename(moved)
            parent.mkdir()
            (parent / "attacker-owned").write_bytes(b"attacker")

    monkeypatch.setattr(target, "_assert_parent_path", replace_after_preissuance_check)
    with pytest.raises(RuntimeError, match="output parent identity changed"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    assert calls == 9
    assert {path.name for path in parent.iterdir()} == {"attacker-owned"}
    assert moved.is_dir() and not list(moved.iterdir())
    assert target._artifact_receipts == {}


def test_openvino_stage_replacement_during_private_cleanup_prevents_token(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    moved = tmp_path / "moved-original-stage"
    original, calls = target._fsync_stage, 0

    def replace_stage_on_cleanup(stage_fd):
        nonlocal calls
        calls += 1
        original(stage_fd)
        if calls == 2:
            stage = next(tmp_path.glob(".model.openvino-*"))
            stage.rename(moved)
            stage.mkdir(mode=0o700)
            (stage / "attacker-owned").write_bytes(b"attacker")

    monkeypatch.setattr(target, "_fsync_stage", replace_stage_on_cleanup)
    with pytest.raises(RuntimeError, match="stage identity changed during cleanup"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    replacement = next(tmp_path.glob(".model.openvino-*"))
    assert (replacement / "attacker-owned").read_bytes() == b"attacker"
    assert moved.is_dir() and not list(moved.iterdir())
    assert not any(path.exists() for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json")))
    assert target._artifact_receipts == {}


def test_openvino_stage_replacement_preserves_attacker_and_erases_private_members(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, moved = target._fsync_at, tmp_path / "moved-original-stage"
    replaced = False

    def replace_stage(directory_fd, name):
        nonlocal replaced
        if not replaced:
            replaced = True
            stage = next(tmp_path.glob(".model.openvino-*"))
            stage.rename(moved)
            stage.mkdir(mode=0o700)
            (stage / "attacker-owned").write_bytes(b"attacker")
        return original(directory_fd, name)

    monkeypatch.setattr(target, "_fsync_at", replace_stage)
    with pytest.raises(RuntimeError, match="stage was replaced"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    replacement = next(tmp_path.glob(".model.openvino-*"))
    assert (replacement / "attacker-owned").read_bytes() == b"attacker"
    assert moved.is_dir() and not list(moved.iterdir())
    assert not any(path.exists() for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json")))


def test_openvino_snapshot_replacement_preserves_attacker_and_erases_moved_source(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    original, replaced = target._identity_at, False

    def replace_snapshot(directory_fd, name):
        nonlocal replaced
        if name == ".verified-source.onnx" and not replaced:
            replaced = True
            stage = next(tmp_path.glob(".model.openvino-*"))
            (stage / name).rename(stage / "moved-private-source")
            (stage / name).write_bytes(b"attacker-owned")
        return original(directory_fd, name)

    monkeypatch.setattr(target, "_identity_at", replace_snapshot)
    with pytest.raises(RuntimeError, match="source snapshot identity changed"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    stage = next(tmp_path.glob(".model.openvino-*"))
    assert (stage / ".verified-source.onnx").read_bytes() == b"attacker-owned"
    assert (stage / "moved-private-source").stat().st_size == 0
    assert not any(path.exists() for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json")))


def test_openvino_unowned_stage_injection_survives_without_private_ir_or_publication(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    from vision.export import openvino as exporter

    original = exporter.build_openvino

    def inject_unowned(snapshot, staged_output, precision, calibration):
        result = original(snapshot, staged_output, precision, calibration)
        (staged_output.parent / "attacker-owned").write_bytes(b"attacker")
        return result

    monkeypatch.setattr(exporter, "build_openvino", inject_unowned)
    with pytest.raises(RuntimeError, match="unexpected private stage members"):
        BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    stage = next(tmp_path.glob(".model.openvino-*"))
    assert (stage / "attacker-owned").read_bytes() == b"attacker"
    assert {path.name for path in stage.iterdir()} == {"attacker-owned"}
    assert not any(path.exists() for path in (output, output.with_suffix(".bin"), output.with_suffix(".receipt.json")))


def test_openvino_build_consumes_verified_snapshot_not_mutated_provider_path(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source, score=0.9)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    neutral = tmp_path / "neutral.onnx"
    from vision.export import openvino as exporter

    original = exporter.build_openvino

    def mutate_provider_path(snapshot, output, precision, calibration):
        _tiny_onnx(neutral, score=0.1)
        return original(snapshot, output, precision, calibration)

    monkeypatch.setattr(exporter, "build_openvino", mutate_provider_path)
    target = OpenVinoTargetPlugin(tmp_path / "model.xml")
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, neutral)
    result = target.run(artifact, PreparedInput(np.zeros((1, 3, 2, 2), dtype=np.float32), (2, 2), "c" * 64))
    assert result.scores == pytest.approx([0.9], abs=contract.tolerances.score_delta)
    assert target.verify_artifact(artifact, contract)


def test_openvino_runtime_path_mutation_after_verify_fails_closed(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    original = target.verify_artifact

    def mutate_after_verification(candidate, expected_contract):
        valid = original(candidate, expected_contract)
        output.write_bytes(output.read_bytes() + b"mutant")
        return valid

    monkeypatch.setattr(target, "verify_artifact", mutate_after_verification)
    with pytest.raises(ValueError, match="changed before in-memory compilation"):
        target.run(artifact, PreparedInput(np.zeros((1, 3, 2, 2), dtype=np.float32), (2, 2), "d" * 64))


def test_openvino_coherent_triple_replacement_after_verify_fails_closed(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    mutant = tmp_path / "mutant.onnx"
    _tiny_onnx(source, score=0.9)
    _tiny_onnx(mutant, score=0.1)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    mutant_contract = _contract(hashlib.sha256(mutant.read_bytes()).hexdigest())
    target = OpenVinoTargetPlugin(tmp_path / "original.xml")
    mutant_target = OpenVinoTargetPlugin(tmp_path / "mutant.xml")
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    BuildPlanner().materialize(
        ManualOnnxSource(), mutant_target, mutant, mutant_contract, tmp_path / "mutant-neutral.onnx"
    )
    original_verify = target.verify_artifact

    def replace_all_after_verification(candidate, expected_contract):
        valid = original_verify(candidate, expected_contract)
        for suffix in (".xml", ".bin", ".receipt.json"):
            source_member = (tmp_path / "mutant.xml").with_suffix(suffix)
            target_member = (tmp_path / "original.xml").with_suffix(suffix)
            target_member.write_bytes(source_member.read_bytes())
        return valid

    monkeypatch.setattr(target, "verify_artifact", replace_all_after_verification)
    with pytest.raises(ValueError, match="receipt changed before in-memory compilation"):
        target.run(artifact, PreparedInput(np.zeros((1, 3, 2, 2), dtype=np.float32), (2, 2), "f" * 64))


@pytest.mark.parametrize(("dtype", "shape"), [("float16", (1, 3, 2, 2)), ("float32", (1, 3, 3, 3))])
def test_openvino_compiled_input_dtype_and_shape_are_exact(tmp_path: Path, monkeypatch, dtype: str, shape: tuple):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    target = OpenVinoTargetPlugin(tmp_path / "model.xml")
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")

    class ElementType:
        def to_dtype(self):
            return np.dtype(dtype)

    class Input:
        def get_any_name(self):
            return "x"

        def get_element_type(self):
            return ElementType()

        def get_shape(self):
            return shape

    class Compiled:
        inputs = [Input()]
        outputs = []

    class Core:
        def read_model(self, xml, binary):
            return object()

        def compile_model(self, model, device):
            return Compiled()

    fake_openvino = type("FakeOpenVino", (), {"Core": Core})
    monkeypatch.setattr(target, "_ov", lambda: fake_openvino)
    with pytest.raises(ValueError, match="input ABI mismatch"):
        target.run(artifact, PreparedInput(np.zeros((1, 3, 2, 2), dtype=np.float32), (2, 2), "e" * 64))


@pytest.mark.parametrize("member", ["xml", "bin", "receipt"])
@pytest.mark.parametrize("kind", ["file", "symlink", "directory"])
def test_openvino_collision_matrix_is_no_clobber(tmp_path: Path, member: str, kind: str):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    collision = {
        "xml": output,
        "bin": output.with_suffix(".bin"),
        "receipt": output.with_suffix(".receipt.json"),
    }[member]
    if kind == "file":
        collision.write_bytes(b"existing")
    elif kind == "symlink":
        collision.symlink_to(tmp_path / "missing-target")
    else:
        collision.mkdir()
    with pytest.raises(FileExistsError):
        BuildPlanner().materialize(
            ManualOnnxSource(), OpenVinoTargetPlugin(output), source, contract, tmp_path / "neutral.onnx"
        )
    assert collision.exists() or collision.is_symlink()
    assert not list(tmp_path.glob(".model.openvino-*"))


def test_openvino_descriptor_receipt_and_member_tamper_matrix(tmp_path: Path):
    source = tmp_path / "source.onnx"
    _tiny_onnx(source)
    contract = _contract(hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "model.xml"
    target = OpenVinoTargetPlugin(output)
    _, _, artifact = BuildPlanner().materialize(ManualOnnxSource(), target, source, contract, tmp_path / "neutral.onnx")
    for index, (key, value) in enumerate(artifact.descriptors):
        changed = list(artifact.descriptors)
        changed[index] = (key, value + "-tampered")
        assert not target.verify_artifact(replace(artifact, descriptors=tuple(changed)), contract)
    assert not target.verify_artifact(
        replace(artifact, descriptors=artifact.descriptors + (("unexpected", "value"),)), contract
    )
    receipt_path = Path(dict(artifact.descriptors)["receipt_path"])
    original_receipt = receipt_path.read_bytes()
    body = json.loads(original_receipt)
    for key in tuple(body):
        changed = dict(body)
        changed[key] = "tampered"
        receipt_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")))
        assert not target.verify_artifact(artifact, contract), key
    receipt_path.write_bytes(original_receipt)
    binary = output.with_suffix(".bin")
    original_binary = binary.read_bytes()
    binary.unlink()
    binary.symlink_to(source)
    assert not target.verify_artifact(artifact, contract)
    binary.unlink()
    binary.write_bytes(original_binary)
    original_receipt_copy = tmp_path / "receipt-copy.json"
    original_receipt_copy.write_bytes(original_receipt)
    receipt_path.unlink()
    receipt_path.symlink_to(original_receipt_copy)
    assert not target.verify_artifact(artifact, contract)


def test_openvino_target_requires_explicit_xml_and_calibration():
    with pytest.raises(ValueError, match="explicit .xml"):
        OpenVinoTargetPlugin("artifact")
    with pytest.raises(ValueError, match="CalibrationManifest"):
        OpenVinoTargetPlugin("artifact.xml", precision="int8")


def test_openvino_factory_preserves_explicit_output(tmp_path: Path):
    target = make_openvino_target(output=tmp_path / "model.xml", precision="fp16")
    assert isinstance(target, OpenVinoTargetPlugin)
    assert target.output == tmp_path / "model.xml"
    assert target.precision == "fp16"
    assert target.capabilities.modes == frozenset({"static"})
    assert "openvino" not in DEFAULT_REGISTRY.targets


def test_openvino_rejects_unsealed_target_request(tmp_path: Path):
    with pytest.raises(TypeError, match="sealed NeutralOnnxRequest"):
        OpenVinoTargetPlugin(tmp_path / "model.xml").build(None, object())
