"""Pure/mock contract tests for the dependency-free TensorRT BuildKit plugin."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vision.backends import TensorRTBackend
from vision.contract import VisionContract
from vision.plugins import (
    DEFAULT_REGISTRY,
    MaterializedOnnx,
    PreparedSource,
    SourceAdapter,
    TensorRTTargetPlugin,
    make_tensorrt_target,
    neutral_onnx_request,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(source_digest: str) -> VisionContract:
    return VisionContract.model_validate(
        {
            "task": "detection",
            "input": {
                "color": "RGB",
                "layout": "NCHW",
                "dtype": "float32",
                "input_name": "images",
                "shape": [1, 3, 32, 32],
            },
            "preprocessing": [
                {"kind": "letterbox", "height": 32, "width": 32, "pad_value": 114, "interpolation": "bilinear"},
                {"kind": "normalize", "mean": [0, 0, 0], "std": [255, 255, 255]},
                {"kind": "transpose", "order": [0, 3, 1, 2]},
            ],
            "raw_output": {
                "semantics": "yolo_end2end",
                "tensors": [{"name": "output0", "dtype": "float32", "shape": [1, 2, 6]}],
            },
            "result": {
                "box_format": "xyxy",
                "coordinate_space": "original_image",
                "nms_owner": "source",
                "nms_iou_threshold": 0.5,
                "score_threshold": 0.25,
                "class_aware_nms": True,
                "clip_boxes": True,
                "max_detections": 20,
                "suppression": "greedy_iou",
                "ordering": "score_desc",
                "tie_policy": "stable",
            },
            "labels": ["object"],
            "coco_category_ids": [1],
            "tolerances": {"box_iou": 0.99, "score_delta": 0.01},
            "provenance": {"source": "test", "source_sha256": source_digest},
        }
    )


class Source(SourceAdapter):
    plugin_id, plugin_version = "test-source", "1"

    def capture(self, source, contract):
        return source

    def run(self, captured, sample):
        raise AssertionError("not used")


def materialized(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    value = contract(digest(checkpoint))
    captured = PreparedSource(checkpoint, digest(checkpoint), "test-source", "1", value.digest)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"mock-onnx")
    lineage = onnx.with_suffix(".onnx.lineage.json")
    lineage.write_text(
        json.dumps(
            {
                "schema": 1,
                "source_sha256": captured.sha256,
                "onnx_sha256": digest(onnx),
                "source_plugin": "test-source",
                "source_plugin_version": "1",
                "contract_digest": value.digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    source = Source()
    raw = MaterializedOnnx(
        onnx, digest(onnx), captured.sha256, "test-source", "1", value.digest, lineage, digest(lineage), ""
    )
    return source._issue_materialization(raw, captured, value), captured, source, value


def request(material, captured, source, value):
    return neutral_onnx_request(material, captured, source, value)


def install_build_mock(monkeypatch):
    import vision.plugins as plugins

    monkeypatch.setattr(plugins, "_validate_onnx_contract", lambda *args: None)
    monkeypatch.setattr(
        "vision.backends.TensorRTBackend.resolve_trtexec", classmethod(lambda cls, path=None: str(Path(__file__)))
    )

    def build(cls, source, output, request):
        assert request.precision in {"fp32", "fp16"} and not request.dynamic and not request.profiles
        output.write_bytes(b"engine")
        return (
            output,
            [
                request.trtexec_path,
                "--onnx=<bound>",
                "--saveEngine=<bound>",
                *(["--noTF32"] if request.precision == "fp32" else ["--fp16"]),
                "--skipInference",
            ],
            "build-log",
            {"trtexec": "TensorRT 10.0"},
        )

    monkeypatch.setattr("vision.backends.TensorRTBackend.build_atomic", classmethod(build))


def test_registry_does_not_advertise_an_unconfigured_tensorrt_target():
    assert "tensorrt" not in DEFAULT_REGISTRY.targets
    assert isinstance(make_tensorrt_target(engine_output="x.engine"), TensorRTTargetPlugin)


def test_tensorrt_rejects_legacy_four_argument_prepare_before_verification(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("vision.plugins._verify_neutral_request", lambda request: calls.append(request))
    plugin = TensorRTTargetPlugin(engine_output=tmp_path / "artifact.engine")
    with pytest.raises(TypeError, match=r"0\.2 migration.*exactly one sealed NeutralOnnxRequest"):
        plugin.prepare_materialized(object(), object(), object(), object())
    assert calls == []
    assert not (tmp_path / "artifact.engine").exists()


def test_build_binds_engine_log_lineage_precision_and_tamper(tmp_path, monkeypatch):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    plugin = TensorRTTargetPlugin(
        engine_output=tmp_path / "artifact.engine",
        precision="fp16",
        trtexec_path="/fake/trtexec",
    )
    artifact = plugin.prepare_materialized(request(material, captured, source, value))
    fields = dict(artifact.descriptors)
    assert artifact.material_path.read_bytes() == b"engine"
    assert fields["requested_precision"] == "fp16"
    assert fields["effective_precision"] == "mixed_fp16_fp32"
    assert fields["onnx_sha256"] == material.sha256 and fields["build_log_sha256"]
    receipt = json.loads(fields["build_receipt_json"])
    assert "--fp16" in receipt["build_command"] and "/" not in " ".join(receipt["build_command"])
    assert plugin.verify_artifact(artifact, value)
    mutated = replace(artifact, descriptors=artifact.descriptors + (("duplicate", "bad"),))
    assert not plugin.verify_artifact(mutated, value)


def test_rejects_existing_output_bad_precision_and_no_receipt(tmp_path, monkeypatch):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    with pytest.raises(ValueError, match="fp32 or fp16"):
        TensorRTTargetPlugin(engine_output=tmp_path / "x.engine", precision="int8")
    output = tmp_path / "x.engine"
    output.write_bytes(b"stale")
    with pytest.raises(ValueError, match="must not already exist"):
        TensorRTTargetPlugin(engine_output=output).prepare_materialized(request(material, captured, source, value))
    clean = tmp_path / "clean.engine"
    with pytest.raises(ValueError, match="receipt"):
        TensorRTTargetPlugin(engine_output=clean).prepare_materialized(
            request(replace(material, receipt="forged"), captured, source, value)
        )
    with pytest.raises(ValueError, match="centrally verified"):
        TensorRTTargetPlugin(engine_output=tmp_path / "dict.engine").prepare_materialized({})


def test_rejects_symbolic_shapes_and_target_nms(tmp_path, monkeypatch):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    symbolic = value.model_copy(update={"input": value.input.model_copy(update={"shape": ("N", 3, 32, 32)})})
    with pytest.raises(ValueError, match="symbolic/dynamic"):
        TensorRTTargetPlugin(engine_output=tmp_path / "dynamic.engine")._prepare_from_materialized(material, symbolic)
    target_nms = value.model_copy(update={"result": value.result.model_copy(update={"nms_owner": "target"})})
    with pytest.raises(ValueError, match="does not execute"):
        TensorRTTargetPlugin(engine_output=tmp_path / "nms.engine")._prepare_from_materialized(material, target_nms)


@pytest.mark.parametrize("shape", ((1, 32, 32, 3), (1, 3, 32, 32)))
def test_tensorrt_rejects_nhwc_contract_even_when_shape_looks_plausible(tmp_path, monkeypatch, shape):
    install_build_mock(monkeypatch)
    material, captured, _source, value = materialized(tmp_path)
    nhwc = value.model_copy(update={"input": value.input.model_copy(update={"layout": "NHWC", "shape": shape})})
    with pytest.raises(ValueError, match="NCHW"):
        TensorRTTargetPlugin(engine_output=tmp_path / f"{shape[-1]}.engine")._prepare_from_materialized(material, nhwc)


def test_fp32_tactics_no_tf32_command_is_build_only():
    from vision.backends import TensorRTBackend

    command = TensorRTBackend.build_command("in.onnx", "out.engine", fp32_tactics_no_tf32=True)
    assert "--noTF32" in command and "--skipInference" in command


def test_atomic_failure_leaves_no_engine_or_log(tmp_path, monkeypatch):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    monkeypatch.setattr(
        "vision.backends.TensorRTBackend.build_atomic",
        classmethod(lambda *args: (_ for _ in ()).throw(RuntimeError("fail"))),
    )
    output = tmp_path / "failed.engine"
    with pytest.raises(RuntimeError, match="fail"):
        TensorRTTargetPlugin(engine_output=output).prepare_materialized(request(material, captured, source, value))
    assert not output.exists() and not output.with_suffix(".engine.build.log").exists()


def test_artifact_receipt_is_relocatable_and_engine_tamper_is_rejected(tmp_path, monkeypatch):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    plugin = TensorRTTargetPlugin(engine_output=tmp_path / "safe.engine")
    artifact = plugin.prepare_materialized(request(material, captured, source, value))
    moved = tmp_path / "relocated.engine"
    moved.write_bytes(artifact.material_path.read_bytes())
    relocated = replace(artifact, material_path=moved)
    assert plugin.verify_artifact(relocated, value)
    moved.write_bytes(b"tampered")
    assert not plugin.verify_artifact(relocated, value)


@pytest.mark.parametrize("field", ["engine_sha256", "source_sha256", "build_receipt_json", "target_receipt"])
def test_process_attested_receipt_rejects_four_descriptor_forgeries(tmp_path, monkeypatch, field):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    plugin = TensorRTTargetPlugin(engine_output=tmp_path / "forgery.engine")
    artifact = plugin.prepare_materialized(request(material, captured, source, value))
    descriptors = dict(artifact.descriptors)
    descriptors[field] = "0" * 64 if field != "target_receipt" else "forged"
    forged = replace(artifact, descriptors=tuple(descriptors.items()))
    assert not plugin.verify_artifact(forged, value)


@pytest.mark.parametrize(
    "axis",
    (
        "runtime-resolution",
        "runtime-batch",
        "runtime-precision",
        "receipt-requested-precision",
        "receipt-effective-precision",
        "command-no-tf32",
        "command-skip-inference",
        "command-fp16-on-fp32",
        "receipt-dynamic",
        "runtime-backend",
        "runtime-provider",
        "runtime-device",
        "runtime-version",
        "host-os",
        "host-architecture",
        "host-builder-target",
        "host-trtexec-sha256",
        "host-trtexec-version",
        "host-cuda-device",
        "host-cuda-compute-capability",
        "host-cuda-runtime",
        "host-tensorrt-python-version",
        "host-driver-or-jetpack",
    ),
)
def test_real_tensorrt_attestation_negative_matrix_rejects_portable_verification(tmp_path, monkeypatch, axis):
    """Each certified TensorRT axis is bound by an issued, not hand-written, receipt."""
    from vision.capsule import _verify_tensorrt_build_receipt

    install_build_mock(monkeypatch)
    identity = {
        "cuda_device": "Mock GPU",
        "cuda_compute_capability": "8.7",
        "cuda_runtime": "12.4",
        "tensorrt_python_version": "10.0",
        "driver_or_jetpack": "JP6",
    }
    monkeypatch.setattr(
        TensorRTTargetPlugin,
        "_build_host_metadata",
        staticmethod(
            lambda executable, version: {
                "os": "mock",
                "architecture": "arm64",
                "builder_target": "mock-arm64",
                "trtexec_sha256": digest(Path(executable)),
                "trtexec_version": version,
                **identity,
            }
        ),
    )
    material, captured, source, value = materialized(tmp_path)
    plugin = TensorRTTargetPlugin(engine_output=tmp_path / "matrix.engine")
    artifact = plugin.prepare_materialized(request(material, captured, source, value))
    receipt = json.loads(dict(artifact.descriptors)["build_receipt_json"])
    runtime = SimpleNamespace(
        backend="tensorrt",
        provider="TensorRTExecutionProvider",
        device="cuda",
        runtime_version="10.0",
        precision="fp32",
        resolution=(32, 32),
        batch=1,
        dynamic=False,
        hardware=identity.copy(),
    )
    report = SimpleNamespace(
        target_plugin_version="1",
        source_plugin="test-source",
        source_plugin_version="1",
        source_digest=captured.sha256,
        runtime=runtime,
    )
    if axis == "runtime-resolution":
        runtime.resolution = (33, 32)
    elif axis == "runtime-batch":
        runtime.batch = 2
    elif axis == "runtime-precision":
        runtime.precision = "fp16"
    elif axis == "receipt-requested-precision":
        receipt["requested_precision"] = "fp16"
    elif axis == "receipt-effective-precision":
        receipt["effective_precision"] = "mixed_fp16_fp32"
    elif axis == "command-no-tf32":
        receipt["build_command"].remove("--noTF32")
    elif axis == "command-skip-inference":
        receipt["build_command"].remove("--skipInference")
    elif axis == "command-fp16-on-fp32":
        receipt["build_command"].insert(-1, "--fp16")
    elif axis == "receipt-dynamic":
        receipt["dynamic"] = True
    elif axis.startswith("runtime-"):
        field = axis.removeprefix("runtime-").replace("version", "runtime_version")
        setattr(runtime, field, "forged")
    else:
        field = (
            axis.removeprefix("host-")
            .replace("trtexec-sha256", "trtexec_sha256")
            .replace("trtexec-version", "trtexec_version")
            .replace("cuda-device", "cuda_device")
            .replace("cuda-compute-capability", "cuda_compute_capability")
            .replace("cuda-runtime", "cuda_runtime")
            .replace("tensorrt-python-version", "tensorrt_python_version")
            .replace("driver-or-jetpack", "driver_or_jetpack")
            .replace("builder-target", "builder_target")
        )
        receipt["build_host"][field] = "forged"
    with pytest.raises(ValueError):
        _verify_tensorrt_build_receipt(receipt, report, value.digest, artifact.sha256)


def test_runtime_requires_tensor_cuda_and_decodes_contract_output(tmp_path, monkeypatch):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    plugin = TensorRTTargetPlugin(engine_output=tmp_path / "run.engine")
    artifact = plugin.prepare_materialized(request(material, captured, source, value))
    image = tmp_path / "input.png"
    Image.new("RGB", (10, 8), "white").save(image)
    prepared = plugin.prepare_input(artifact, image, value)

    class Runtime:
        input_names = ("images",)
        output_names = ("output0",)
        trt = type("T", (), {"__version__": "10"})()
        input_specs = {"images": ((1, 3, 32, 32), "float32")}
        output_specs = {"output0": ((1, 2, 6), "float32")}

        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, tensor):
            assert tensor.shape == (1, 3, 32, 32)
            return {"output0": np.asarray([[[1, 2, 9, 7, 0.9, 0], [0, 0, 1, 1, 0.1, 0]]], dtype=np.float32)}

    monkeypatch.setattr("vision.backends.TensorRTBackend", Runtime)
    monkeypatch.setattr(
        plugin,
        "_runtime_provenance",
        lambda runtime: {
            "execution_provider": "TensorRTExecutionProvider",
            "execution_device": "cuda",
            "cuda_device": "mock",
        },
    )
    result = plugin.run(artifact, prepared)
    assert result.classes == [0] and result.names == {0: "object"}
    assert result.metadata["execution_provider"] == "TensorRTExecutionProvider"


def test_runtime_rejects_engine_interface_shape_or_dtype(tmp_path, monkeypatch):
    install_build_mock(monkeypatch)
    material, captured, source, value = materialized(tmp_path)
    plugin = TensorRTTargetPlugin(engine_output=tmp_path / "wrong.engine")
    artifact = plugin.prepare_materialized(request(material, captured, source, value))
    image = tmp_path / "input.png"
    Image.new("RGB", (10, 8), "white").save(image)
    prepared = plugin.prepare_input(artifact, image, value)

    class WrongRuntime:
        input_names, output_names = ("images",), ("output0",)
        input_specs = {"images": ((1, 3, 32, 32), "float32")}
        output_specs = {"output0": ((1, 3, 6), "float16")}

        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("vision.backends.TensorRTBackend", WrongRuntime)
    with pytest.raises(ValueError, match="output shape/dtype"):
        plugin.run(artifact, prepared)


def _backend_with_fake_torch(monkeypatch, *, v3: bool, context, engine):
    class Tensor:
        shape = (1, 3, 32, 32)

        def contiguous(self):
            return self

        def data_ptr(self):
            return 11

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 1, 6), dtype=np.float32)

    class Stream:
        cuda_stream = 7

        def synchronize(self):
            pass

    fake_torch = type(
        "Torch",
        (),
        {
            "cuda": type(
                "Cuda", (), {"is_available": staticmethod(lambda: True), "current_stream": staticmethod(Stream)}
            )(),
            "as_tensor": staticmethod(lambda *_args, **_kwargs: Tensor()),
            "empty": staticmethod(lambda *_args, **_kwargs: Tensor()),
            "from_numpy": staticmethod(lambda *_args: type("D", (), {"dtype": "float32"})()),
        },
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    backend = object.__new__(TensorRTBackend)
    backend.trt = type("Trt", (), {"nptype": staticmethod(lambda _value: np.float32)})()
    backend.engine, backend.context, backend._engine_object = "engine", context, engine
    backend.input_name, backend.input_names, backend.output_names = "images", ("images",), ("out",)
    return backend


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("input_address", "input address"),
        ("unresolved", "shape unresolved"),
        ("output_address", "output address"),
        ("execute", "execute_async_v3"),
    ],
)
def test_tensorrt_backend_v3_rejects_address_shape_and_execute_failures(monkeypatch, mode, expected):
    class Engine:
        num_io_tensors = 2

        @staticmethod
        def get_tensor_dtype(_name):
            return 0

    class Context:
        def set_input_shape(self, *_args):
            return True

        def set_tensor_address(self, name, _ptr):
            return not ((mode == "input_address" and name == "images") or (mode == "output_address" and name == "out"))

        def get_tensor_shape(self, _name):
            return (-1, 1, 6) if mode == "unresolved" else (1, 1, 6)

        def execute_async_v3(self, _stream):
            return mode != "execute"

    with pytest.raises(RuntimeError, match=expected):
        _backend_with_fake_torch(monkeypatch, v3=True, context=Context(), engine=Engine()).run(np.zeros((1, 3, 32, 32)))


@pytest.mark.parametrize(
    "mode,expected",
    [("shape", "rejected input shape"), ("unresolved", "shape unresolved"), ("execute", "execute_async_v2")],
)
def test_tensorrt_backend_v2_rejects_binding_shape_and_execute_failures(monkeypatch, mode, expected):
    class Engine:
        num_bindings = 2

        @staticmethod
        def get_binding_index(name):
            return {"images": 1, "out": 0}[name]

        @staticmethod
        def get_binding_shape(_index):
            return (1, 1, 6)

        @staticmethod
        def get_binding_dtype(_index):
            return 0

    class Context:
        def set_binding_shape(self, index, _shape):
            assert index == 1  # input binding is deliberately not assumed first
            return mode != "shape"

        def get_binding_shape(self, _index):
            return (-1, 1, 6) if mode == "unresolved" else (1, 1, 6)

        def execute_async_v2(self, bindings, _stream):
            assert len(bindings) == 2 and all(bindings)  # ordered/completely populated bindings
            return mode != "execute"

    with pytest.raises((RuntimeError, ValueError), match=expected):
        _backend_with_fake_torch(monkeypatch, v3=False, context=Context(), engine=Engine()).run(
            np.zeros((1, 3, 32, 32))
        )
