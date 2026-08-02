"""ORT-mocked integration coverage for the frozen RT-DETR target path."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from vision.adapters.onnx import OnnxAdapter
from vision.contract import VisionContract
from vision.decoders import RTDETR_COMBINED_ABI
from vision.manifest import VisionManifest, manifest_from_contract
from vision.plugins import OnnxRuntimeTargetPlugin
from vision.preprocess import PreparedInput, preprocess
from vision.processing import GeometryTrace, direct_resize
from vision.types import Box


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _onnx(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    values = np.zeros((1, 300, 6), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["output0"],
                value=onnx.numpy_helper.from_array(values, name="output0-value"),
            )
        ],
        "rtdetr-combined",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 300, 6])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def _contract(source_sha, *, clip_boxes=True, top_k=300):
    return VisionContract.model_validate(
        {
            "input": {
                "color": "RGB",
                "layout": "NCHW",
                "dtype": "float32",
                "input_name": "images",
                "shape": [1, 3, 640, 640],
            },
            "preprocessing": [
                {
                    "kind": "resize",
                    "height": 640,
                    "width": 640,
                    "interpolation": "bilinear",
                    "implementation": "opencv_linear_scale_fill",
                },
                {"kind": "normalize", "mean": [0, 0, 0], "std": [255, 255, 255]},
                {"kind": "transpose", "order": [0, 3, 1, 2]},
            ],
            "raw_output": {
                "semantics": "rtdetr_combined",
                "tensors": [{"name": "output0", "dtype": "float32", "shape": [1, 300, 6]}],
            },
            "result": {
                "box_format": "xyxy",
                "coordinate_space": "original_image",
                "nms_owner": "none",
                "nms_iou_threshold": None,
                "ordering": "score_desc",
                "tie_policy": "class_then_box",
                "score_threshold": 0.4,
                "class_aware_nms": False,
                "clip_boxes": clip_boxes,
                "top_k": top_k,
                "max_detections": 300,
                "suppression": "none",
            },
            "labels": ["cat", "dog"],
            "coco_category_ids": [1, 18],
            "tolerances": {"box_iou": 0.99, "score_delta": 0.01},
            "provenance": {"source": "rtdetr.onnx", "source_sha256": source_sha},
        }
    )


def _runtime(output, *, output_shape=(1, 300, 6), output_type="tensor(float)"):
    sessions = []

    class Session:
        def __init__(self, path, providers):
            assert path.endswith(".onnx")
            assert providers == ["CPUExecutionProvider"]
            self.run_calls = 0
            sessions.append(self)

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 640, 640], type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output0", shape=list(output_shape), type=output_type)]

        def run(self, names, feed):
            self.run_calls += 1
            assert names is None
            assert set(feed) == {"images"}
            assert feed["images"].shape == (1, 3, 640, 640)
            return [output.copy()]

    return SimpleNamespace(InferenceSession=Session), sessions


def _rtdetr_pixels(case):
    if case == "odd_channels":
        pixels = np.zeros((11, 17, 3), dtype=np.uint8)
        rows, columns = np.indices(pixels.shape[:2])
        pixels[..., 0] = columns * 13
        pixels[..., 1] = rows * 19
        pixels[..., 2] = (rows * 7 + columns * 11) % 256
        return pixels
    if case == "seeded_noise":
        return np.random.default_rng(260801).integers(0, 256, size=(23, 37, 3), dtype=np.uint8)
    if case == "sharp_edges":
        pixels = np.zeros((19, 29, 3), dtype=np.uint8)
        pixels[:, :14, 0] = 255
        pixels[:9, :, 1] = 173
        pixels[::2, ::3, 2] = 251
        return pixels
    if case == "identity":
        rows, columns = np.indices((640, 640))
        return np.stack((columns % 256, rows % 256, (rows ^ columns) % 256), axis=-1).astype(np.uint8)
    raise AssertionError(case)


def test_ort_target_prepares_and_runs_exact_rtdetr_contract_with_labels_clipping_and_metadata(monkeypatch, tmp_path):
    model = tmp_path / "rtdetr.onnx"
    _onnx(model)
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [0.0, 0.0, 0.2, 0.2, 0.9, 1]
    output[0, 1] = [0.5, 0.5, 0.2, 0.2, 0.9, 0]
    output[0, 2] = [0.8, 0.8, 0.1, 0.1, 0.3, 0]
    runtime, sessions = _runtime(output)
    plugin = OnnxRuntimeTargetPlugin()
    monkeypatch.setattr(plugin, "_ort", lambda: runtime)
    value = _contract(_sha(model))

    artifact = plugin.prepare(model, value)
    assert dict(artifact.descriptors)["decoder_abi"] == RTDETR_COMBINED_ABI
    sample = PreparedInput(
        np.zeros((1, 3, 640, 640), dtype=np.float32),
        (100, 50),
        "f" * 64,
        GeometryTrace((100, 50), (640, 640), 6.4, 12.8),
    )
    result = plugin.run(artifact, sample)

    assert len(sessions) == 2
    assert result.classes == [0, 1]
    assert result.names == {0: "cat", 1: "dog"}
    assert result.boxes[0].as_xyxy() == pytest.approx([40, 20, 60, 30])
    assert result.boxes[1].as_xyxy() == pytest.approx([0, 0, 10, 5])
    assert result.image_size == (100, 50)
    assert result.metadata == {
        "decoder_abi": RTDETR_COMBINED_ABI,
        "execution_provider": "CPUExecutionProvider",
        "execution_device": "cpu",
        "input_sha256": "f" * 64,
    }

    unclipped = _contract(_sha(model), clip_boxes=False)
    unclipped_result = plugin.run(plugin.prepare(model, unclipped), sample)
    assert unclipped_result.boxes[1].as_xyxy() == pytest.approx([-10, -5, 10, 5])


def test_ort_target_rejects_wrong_runtime_abi_and_missing_trace_before_session(monkeypatch, tmp_path):
    model = tmp_path / "rtdetr.onnx"
    _onnx(model)
    output = np.zeros((1, 300, 6), dtype=np.float32)
    plugin = OnnxRuntimeTargetPlugin()
    wrong_runtime, _ = _runtime(output, output_shape=(1, 299, 6))
    monkeypatch.setattr(plugin, "_ort", lambda: wrong_runtime)
    with pytest.raises(ValueError, match=r"exact float32 \[1,300,6\]"):
        plugin.prepare(model, _contract(_sha(model)))

    runtime, sessions = _runtime(output)
    monkeypatch.setattr(plugin, "_ort", lambda: runtime)
    artifact = plugin.prepare(model, _contract(_sha(model)))
    before = len(sessions)
    without_trace = PreparedInput(np.zeros((1, 3, 640, 640), dtype=np.float32), (100, 50), "f" * 64)
    with pytest.raises(ValueError, match="geometry trace"):
        plugin.run(artifact, without_trace)
    assert len(sessions) == before


def test_ort_target_rejects_nonfrozen_rtdetr_result_contract_before_session(monkeypatch, tmp_path):
    model = tmp_path / "rtdetr.onnx"
    _onnx(model)
    output = np.zeros((1, 300, 6), dtype=np.float32)
    runtime, sessions = _runtime(output)
    plugin = OnnxRuntimeTargetPlugin()
    monkeypatch.setattr(plugin, "_ort", lambda: runtime)
    with pytest.raises(ValueError, match="frozen NMS-free"):
        plugin.prepare(model, _contract(_sha(model), top_k=299))
    assert sessions == []


def test_rtdetr_contract_preprocess_uses_direct_opencv_resize_with_exact_geometry(monkeypatch):
    from PIL import Image

    pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    image = Image.fromarray(pixels, "RGB")
    calls = []

    def observed_direct_resize(value, target):
        calls.append(target)
        return direct_resize(value, target)

    monkeypatch.setattr("vision.preprocess.direct_resize", observed_direct_resize)
    prepared = preprocess(_contract("a" * 64), image)
    resized, expected_trace = direct_resize(image, (640, 640))
    expected = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

    assert calls == [(640, 640)]
    assert np.array_equal(prepared.tensor, expected)
    assert prepared.trace == expected_trace


@pytest.mark.parametrize("case", ["odd_channels", "seeded_noise", "sharp_edges", "identity"])
def test_rtdetr_contract_preprocess_matches_ultralytics_scale_fill_tensor_and_geometry(case):
    from PIL import Image

    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox

    pixels = _rtdetr_pixels(case)
    prepared = preprocess(_contract("a" * 64), Image.fromarray(pixels, "RGB"))
    vendor_bgr = LetterBox((640, 640), auto=False, scale_fill=True)(image=pixels[..., ::-1])
    expected = np.ascontiguousarray(vendor_bgr[..., ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0

    assert np.array_equal(prepared.tensor, expected)
    height, width = pixels.shape[:2]
    assert prepared.trace == GeometryTrace((width, height), (640, 640), 640 / width, 640 / height)
    original = Box(1.0, 2.0, width - 2.0, height - 1.0)
    model_space = Box(
        original.x1 * prepared.trace.scale_x,
        original.y1 * prepared.trace.scale_y,
        original.x2 * prepared.trace.scale_x,
        original.y2 * prepared.trace.scale_y,
    )
    assert prepared.trace.inverse(model_space).as_xyxy() == pytest.approx(original.as_xyxy())


@pytest.mark.parametrize("clip_boxes", [True, False])
def test_certified_v1_onnx_adapter_uses_embedded_contract_labels_and_clipping(clip_boxes):
    value = _contract("a" * 64, clip_boxes=clip_boxes)
    adapter = OnnxAdapter(manifest_from_contract("rtdetr.onnx", value))
    assert adapter.manifest.decoder.kind == "rtdetr_combined_v1"
    adapter.engaged_providers = ["CPUExecutionProvider"]
    adapter.input_name = "images"
    adapter.output_names = {"output0"}
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [0.0, 0.0, 0.2, 0.2, 0.9, 1]
    result = adapter.postprocess(
        ({"output0": output}, GeometryTrace((100, 50), (640, 640), 6.4, 12.8)),
        (100, 50),
    )
    assert result.names == {0: "cat", 1: "dog"}
    expected = [0, 0, 10, 5] if clip_boxes else [-10, -5, 10, 5]
    assert result.boxes[0].as_xyxy() == pytest.approx(expected)
    assert result.metadata["provider"] == "CPUExecutionProvider"
    assert result.metadata["decoder_abi"] == RTDETR_COMBINED_ABI


def test_legacy_onnx_manifest_dispatches_origin_main_decoder_without_v1_metadata():
    manifest = VisionManifest(
        source="rtdetr.onnx",
        preprocessing={"resize": "resize", "implementation": "opencv_linear_scale_fill"},
        output={"box_format": "cxcywh", "boxes_name": "output0"},
        decoder={"kind": "rtdetr_combined", "boxes_normalized": True, "score_threshold": 0.4},
    )
    adapter = OnnxAdapter(manifest)
    adapter.engaged_providers = ["CPUExecutionProvider"]
    adapter.input_name = "images"
    adapter.output_names = {"output0"}
    output = np.array([[0.5, 0.5, 0.2, 0.2, 0.4, 1], [0.5, 0.5, 0.2, 0.2, 0.5, 2]], dtype=np.float64)
    result = adapter.postprocess(
        ({"output0": output}, GeometryTrace((100, 50), (640, 640), 6.4, 12.8)),
        (100, 50),
    )
    assert result.classes == [2]
    assert result.names == {}
    assert "decoder_abi" not in result.metadata
