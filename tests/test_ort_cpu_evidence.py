"""ORT CPU execution evidence must be observed, not merely requested."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import vision.plugins as plugins_module
from vision.contract import VisionContract
from vision.plugins import OnnxRuntimeTargetPlugin
from vision.preprocess import PreparedInput
from vision.processing import GeometryTrace


def _contract(source_sha256, *, semantics="boxes_scores_classes"):
    raw_output = {
        "semantics": "boxes_scores_classes",
        "tensors": [
            {"name": "b", "dtype": "float32", "shape": [1, "N", 4]},
            {"name": "s", "dtype": "float32", "shape": [1, "N"]},
            {"name": "c", "dtype": "int64", "shape": [1, "N"]},
        ],
        "roles": {"boxes": "b", "scores": "s", "classes": "c"},
    }
    if semantics == "yolo_end2end":
        raw_output = {"semantics": semantics, "tensors": [{"name": "output", "dtype": "float32", "shape": [1, "N", 6]}]}
    return VisionContract.model_validate(
        {
            "input": {"color": "RGB", "layout": "NCHW", "dtype": "float32", "input_name": "x", "shape": [1, 3, 32, 32]},
            "preprocessing": [
                {
                    "kind": "letterbox",
                    "height": 32,
                    "width": 32,
                    "pad_value": 114,
                    "interpolation": "bilinear",
                    "placement": "center",
                    "rounding": "round_half_up",
                    "scale": "min_ratio",
                }
            ],
            "raw_output": raw_output,
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
            "provenance": {"source": "fake", "source_sha256": source_sha256},
        }
    )


class _FakeSession:
    def __init__(self, path, providers, observed_providers, semantics):
        assert providers == ["CPUExecutionProvider"]
        self.path = path
        self.observed_providers = observed_providers
        self.inputs = [SimpleNamespace(name="x", shape=[1, 3, 32, 32])]
        self.outputs = (
            [SimpleNamespace(name="output", type="tensor(float)", shape=[1, "N", 6])]
            if semantics == "yolo_end2end"
            else [
                SimpleNamespace(name="b", type="tensor(float)", shape=[1, "N", 4]),
                SimpleNamespace(name="s", type="tensor(float)", shape=[1, "N"]),
                SimpleNamespace(name="c", type="tensor(int64)", shape=[1, "N"]),
            ]
        )

    def get_providers(self):
        return self.observed_providers

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, unused_output_names, inputs):
        assert unused_output_names is None
        assert set(inputs) == {"x"}
        if len(self.outputs) == 1:
            return [np.array([[[0, 0, 2, 2, 0.9, 0]]], dtype=np.float32)]
        return [
            np.array([[[0, 0, 2, 2]]], dtype=np.float32),
            np.array([[0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        ]


def _plugin(monkeypatch, observed_providers, *, semantics="boxes_scores_classes"):
    plugin = OnnxRuntimeTargetPlugin()
    monkeypatch.setattr(plugins_module, "_validate_onnx_contract", lambda path, contract: None)
    monkeypatch.setattr(plugin, "_validate_optional_ultralytics_metadata", lambda path, contract: None)
    monkeypatch.setattr(
        plugin,
        "_ort",
        lambda: SimpleNamespace(
            InferenceSession=lambda path, providers: _FakeSession(path, providers, observed_providers, semantics)
        ),
    )
    return plugin


@pytest.mark.parametrize("semantics", ["boxes_scores_classes", "yolo_end2end"])
def test_ort_cpu_provider_is_observed_and_recorded(monkeypatch, tmp_path, semantics):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake-onnx")
    contract = _contract(hashlib.sha256(path.read_bytes()).hexdigest(), semantics=semantics)
    plugin = _plugin(monkeypatch, ["CPUExecutionProvider"], semantics=semantics)

    artifact = plugin.prepare(path, contract)
    sample = PreparedInput(
        tensor=np.zeros((1, 3, 32, 32), dtype=np.float32),
        image_size=(8, 8),
        input_sha256="a" * 64,
        trace=GeometryTrace((8, 8), (32, 32), 1, 1),
    )
    result = plugin.run(artifact, sample)

    assert result.metadata == {
        "execution_provider": "CPUExecutionProvider",
        "execution_device": "cpu",
        "input_sha256": "a" * 64,
    }
    if semantics == "yolo_end2end":
        assert result.names == {0: "person"}


@pytest.mark.parametrize("observed", [["CUDAExecutionProvider"], ["CPUExecutionProvider", "CUDAExecutionProvider"]])
def test_ort_prepare_rejects_nonexclusive_cpu_provider(monkeypatch, tmp_path, observed):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake-onnx")
    plugin = _plugin(monkeypatch, observed)

    with pytest.raises(ValueError, match=r"exactly \['CPUExecutionProvider'\]"):
        plugin.prepare(path, _contract(hashlib.sha256(path.read_bytes()).hexdigest()))


@pytest.mark.parametrize("observed", [["CUDAExecutionProvider"], ["CPUExecutionProvider", "CUDAExecutionProvider"]])
def test_ort_run_rejects_nonexclusive_cpu_provider(monkeypatch, tmp_path, observed):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake-onnx")
    contract = _contract(hashlib.sha256(path.read_bytes()).hexdigest())
    plugin = _plugin(monkeypatch, ["CPUExecutionProvider"])
    artifact = plugin.prepare(path, contract)
    plugin = _plugin(monkeypatch, observed)
    sample = PreparedInput(np.zeros((1, 3, 32, 32), dtype=np.float32), (8, 8), "a" * 64)

    with pytest.raises(ValueError, match=r"exactly \['CPUExecutionProvider'\]"):
        plugin.run(artifact, sample)
