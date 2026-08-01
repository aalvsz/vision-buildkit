import numpy as np
import pytest
from PIL import Image

from vision.adapters.backend import BackendAdapter
from vision.manifest import VisionManifest


def manifest():
    return VisionManifest(
        source="artifact",
        family="onnx",
        input={"shape": [1, 3, 8, 8], "input_name": "images"},
        output={"boxes_name": "detections"},
        decoder={"kind": "yolo_end2end", "score_threshold": 0.25},
    )


def test_onnxruntime_backend_adapter_composes_preprocess_run_decode(monkeypatch):
    class FakeOrt:
        def __init__(self, source, providers, *, input_name):
            assert source == "model.onnx"
            assert providers == ["CPUExecutionProvider"]
            assert input_name == "images"
            self.input = "images"
            self.outputs = ("detections",)

        def run(self, tensor):
            assert tensor.shape == (1, 3, 8, 8)
            return {"detections": np.array([[[1, 2, 7, 6, 0.9, 3]]], dtype=np.float32)}

    monkeypatch.setattr("vision.adapters.backend.OrtBackend", FakeOrt)
    model = BackendAdapter(manifest(), "onnxruntime", precision="int8").load("model.onnx", "cpu")
    result = model(Image.new("RGB", (8, 8)))
    assert result.classes == [3]
    assert result.boxes[0].as_xyxy() == pytest.approx([1, 2, 7, 6])
    assert result.metadata == {
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "precision": "int8",
        "input_shape": [1, 3, 8, 8],
        "input_name": "images",
    }


def test_openvino_backend_adapter_uses_explicit_provider_without_fallback(monkeypatch):
    class FakeOpenVINO:
        def __init__(self, source, device, *, input_name):
            assert (source, device, input_name) == ("model.xml", "CPU", "images")
            self.input_name = "images"
            self.outputs = ("detections",)

        def run(self, tensor):
            return {"detections": np.empty((1, 0, 6), dtype=np.float32)}

    monkeypatch.setattr("vision.adapters.backend.OpenVINOBackend", FakeOpenVINO)
    model = BackendAdapter(manifest(), "openvino", "CPU", "fp16").load("model.xml", "cpu")
    result = model(Image.new("RGB", (8, 8)))
    assert result.boxes == []
    assert model.runtime_identity() == {
        "backend": "openvino",
        "provider": "CPU",
        "device": "cpu",
        "precision": "fp16",
        "input_shape": [1, 3, 8, 8],
    }


def test_tensorrt_adapter_rejects_cpu_before_execution():
    with pytest.raises(ValueError, match="requires device"):
        BackendAdapter(manifest(), "tensorrt").load("model.engine", "cpu")


def test_compiled_backend_cannot_reverse_export():
    with pytest.raises(RuntimeError, match="reverse-exported"):
        BackendAdapter(manifest(), "openvino").export_onnx("x.onnx")


def test_backend_adapter_executes_real_fixed_batches_and_pads_tail(monkeypatch):
    calls = []

    class FakeOrt:
        def __init__(self, source, providers, *, input_name):
            self.input = input_name
            self.outputs = ("detections",)

        def run(self, tensor):
            calls.append(tensor.shape)
            rows = [[[index, 0, index + 1, 1, 0.9, index]] for index in range(tensor.shape[0])]
            return {"detections": np.asarray(rows, dtype=np.float32)}

    contract = VisionManifest.model_validate(
        manifest().model_dump(mode="json") | {"input": {"shape": [2, 3, 8, 8], "input_name": "images"}}
    )
    monkeypatch.setattr("vision.adapters.backend.OrtBackend", FakeOrt)
    model = BackendAdapter(contract, "onnxruntime").load("model.onnx", "cpu")
    results = model.predict_batch([Image.new("RGB", (8, 8)) for _ in range(3)])
    assert calls == [(2, 3, 8, 8), (2, 3, 8, 8)]
    assert [result.classes for result in results] == [[0], [1], [0]]
