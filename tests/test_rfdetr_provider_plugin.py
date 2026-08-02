"""RF-DETR provider tests use a fake vendor module and synthetic ONNX only."""

import hashlib
import sys
import types
from pathlib import Path
from unittest.mock import ANY

import numpy as np
import pytest
from PIL import Image

from vision.buildkit import BuildPlanner
from vision.contract import VisionContract
from vision.decoders import RFDETR_COCO_CATEGORY_IDS, RFDETR_COCO_LABELS, rfdetr_contract_output
from vision.plugins import DEFAULT_REGISTRY, OnnxRuntimeTargetPlugin
from vision.preprocess import preprocess
from vision.providers import RFDetrSourceAdapter


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(digest):
    return VisionContract.model_validate(
        {
            "input": {
                "color": "RGB",
                "layout": "NCHW",
                "dtype": "float32",
                "input_name": "input",
                "shape": [1, 3, 384, 384],
            },
            "preprocessing": [
                {
                    "kind": "resize",
                    "height": 384,
                    "width": 384,
                    "interpolation": "bilinear",
                },
                {"kind": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                {"kind": "transpose", "order": [0, 3, 1, 2]},
            ],
            "raw_output": {
                "semantics": "custom",
                "decoder_plugin": "rfdetr.sigmoid_coco_v1",
                "tensors": [
                    {"name": "dets", "dtype": "float32", "shape": [1, 300, 4]},
                    {"name": "labels", "dtype": "float32", "shape": [1, 300, 91]},
                ],
            },
            "result": {
                "box_format": "cxcywh",
                "coordinate_space": "original_image",
                "nms_owner": "none",
                "nms_iou_threshold": None,
                "score_threshold": 0.30,
                "class_aware_nms": False,
                "clip_boxes": True,
                "top_k": 300,
                "max_detections": 300,
                "suppression": "none",
                "ordering": "score_desc",
                "tie_policy": "stable",
            },
            "labels": list(RFDETR_COCO_LABELS),
            "coco_category_ids": list(RFDETR_COCO_CATEGORY_IDS),
            "tolerances": {"box_iou": 0.9, "score_delta": 0.01},
            "provenance": {"source": "synthetic", "source_sha256": digest},
        }
    )


def _onnx(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["dets"], value=helper.make_tensor("d", TensorProto.FLOAT, [1, 300, 4], [0] * 1200)
            ),
            helper.make_node(
                "Constant", [], ["labels"], value=helper.make_tensor("l", TensorProto.FLOAT, [1, 300, 91], [0] * 27300)
            ),
        ],
        "fake-rfdetr",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 384, 384])],
        [
            helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 300, 4]),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 300, 91]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 11
    onnx.save(model, path)


def _raw():
    boxes, logits = np.zeros((1, 300, 4), np.float32), np.full((1, 300, 91), -20, np.float32)
    boxes[0, 0] = [0.0, 1.0, 2.0, 1.0]  # clipping is observable on a 10x20 image
    logits[0, 0, 1] = 2.0
    logits[0, 1, 3] = 2.0  # same score: query 0 wins the stable query tie-break
    logits[0, 2, 12] = 10.0  # forbidden sparse category hole
    return {"dets": boxes, "labels": logits}


def _fake_vendor(monkeypatch, *, raw=None, mutation=None):
    calls = {"factory": [], "predict": [], "export": []}

    class Model:
        # These are the pinned rfdetr==1.9.0 public call signatures.  Do not
        # use **kwargs here: this fixture must reject invented provider flags.
        def __init__(self, pretrain_weights=None):
            calls["factory"].append({"pretrain_weights": pretrain_weights})

        def predict(self, image, threshold=0.5, shape=None):
            calls["predict"].append({"shape": shape, "threshold": threshold})
            return (
                raw
                if raw is not None
                else types.SimpleNamespace(
                    xyxy=np.array([[0, 10, 10, 20], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32),
                    confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
                    class_id=np.array([1, 18, 90], dtype=np.int64),
                )
            )

        def export(
            self, output_dir=".", format="onnx", shape=None, batch_size=1, opset_version=17, fp16=False, verbose=False
        ):
            kwargs = {
                "output_dir": output_dir,
                "format": format,
                "shape": shape,
                "batch_size": batch_size,
                "opset_version": opset_version,
                "fp16": fp16,
                "verbose": verbose,
            }
            calls["export"].append(kwargs)
            if mutation:
                mutation.write_bytes(b"mutated")
            exported = Path(kwargs["output_dir"]) / "rfdetr-nano.onnx"
            _onnx(exported)
            return str(exported)

    monkeypatch.setitem(sys.modules, "rfdetr", types.SimpleNamespace(RFDETRNano=Model))
    monkeypatch.setattr("vision.providers.metadata.version", lambda package: "1.9.0")
    return calls


def test_rfdetr_registry_is_lazy_and_missing_dependency_fails_closed(monkeypatch, tmp_path):
    assert "rfdetr" in DEFAULT_REGISTRY.sources
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    import vision.providers as providers

    original = providers.importlib.import_module
    monkeypatch.setattr(
        providers.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()) if name == "rfdetr" else original(name),
    )
    with pytest.raises(RuntimeError, match=r"vision-buildkit\[rfdetr\]"):
        RFDetrSourceAdapter().capture(checkpoint, _contract(_sha(checkpoint)))


def test_rfdetr_refuses_any_unvalidated_package_version(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("vision.providers.metadata.version", lambda package: "1.9.1")
    monkeypatch.setitem(sys.modules, "rfdetr", types.SimpleNamespace(RFDETRNano=object))
    with pytest.raises(RuntimeError, match=r"exactly 1\.9\.0"):
        RFDetrSourceAdapter().capture(checkpoint, _contract(_sha(checkpoint)))


def test_rfdetr_source_contract_native_export_and_neutral_lineage(monkeypatch, tmp_path):
    checkpoint, output = tmp_path / "model.pth", tmp_path / "build.onnx"
    checkpoint.write_bytes(b"checkpoint")
    calls = _fake_vendor(monkeypatch)
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    captured = adapter.prepare(checkpoint, contract)
    image = tmp_path / "image.png"
    Image.new("RGB", (10, 20), "white").save(image)
    result = adapter.run(captured, adapter.prepare_input(captured, image, contract))
    assert calls["factory"] and calls["predict"] == [{"shape": (384, 384), "threshold": 0.30}]
    assert result.classes == [0, 16, 79] and result.boxes[0].as_xyxy() == pytest.approx([0, 10, 10, 20])
    assert [result.names[index] for index in result.classes] == ["person", "dog", "toothbrush"]
    materialized = adapter.materialize_onnx(captured, contract, output)
    assert calls["export"] == [
        {
            "output_dir": ANY,
            "format": "onnx",
            "shape": (384, 384),
            "batch_size": 1,
            "opset_version": 17,
            "fp16": False,
            "verbose": False,
        }
    ]
    assert materialized.source_sha256 == _sha(checkpoint) and materialized.lineage_path.is_file()
    assert materialized.material_path.name == "model.onnx"
    assert materialized.lineage_path.name == "lineage.json"
    assert materialized.material_path.parent.name.endswith(".rfdetr-bundle")
    output2 = tmp_path / "build-2.onnx"
    _, neutral, artifact = BuildPlanner().materialize(adapter, OnnxRuntimeTargetPlugin(), checkpoint, contract, output2)
    assert neutral.plugin_id == "rfdetr" and artifact.material_path == neutral.material_path


@pytest.mark.parametrize(
    "bad",
    [
        lambda raw: {"dets": raw["dets"]},
        lambda raw: {"dets": raw["dets"][..., :3], "labels": raw["labels"]},
        lambda raw: {"dets": raw["dets"] * np.nan, "labels": raw["labels"]},
    ],
)
def test_rfdetr_raw_decoder_rejects_wrong_or_nonfinite_abi(bad):
    with pytest.raises(ValueError, match="exactly named|raw ABI|finite"):
        rfdetr_contract_output(bad(_raw()), (10, 20), _contract("0" * 64))


def test_rfdetr_contract_preprocess_matches_tensor_bilinear_without_antialias():
    import torch
    import torch.nn.functional as functional

    pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    image = Image.fromarray(pixels, "RGB")
    actual = preprocess(_contract("0" * 64), image).tensor
    expected = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    expected = functional.interpolate(expected, size=(384, 384), mode="bilinear", align_corners=False, antialias=False)
    expected = (expected - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    assert np.array_equal(actual, expected.numpy())


def test_rfdetr_native_postprocessed_object_fails_closed(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    _fake_vendor(monkeypatch, raw=types.SimpleNamespace(xyxy=[]))
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    captured = adapter.prepare(checkpoint, contract)
    image = tmp_path / "image.png"
    Image.new("RGB", (1, 1)).save(image)
    with pytest.raises(ValueError, match="official Detections ABI"):
        adapter.run(captured, adapter.prepare_input(captured, image, contract))


def test_rfdetr_native_rejects_non_coco_sparse_detection_ids(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    _fake_vendor(
        monkeypatch,
        raw=types.SimpleNamespace(
            xyxy=np.zeros((1, 4), dtype=np.float32),
            confidence=np.ones(1, dtype=np.float32),
            class_id=np.array([0], dtype=np.int64),
        ),
    )
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    captured = adapter.prepare(checkpoint, contract)
    image = tmp_path / "image.png"
    Image.new("RGB", (1, 1)).save(image)
    with pytest.raises(ValueError, match="contradicts"):
        adapter.run(captured, adapter.prepare_input(captured, image, contract))


def test_rfdetr_rehashes_source_after_export(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    _fake_vendor(monkeypatch, mutation=checkpoint)
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    with pytest.raises(ValueError, match="bound"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, contract), contract, tmp_path / "out.onnx")


def test_rfdetr_content_addressed_bundle_reuses_exact_content_and_rejects_conflict(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    _fake_vendor(monkeypatch)
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    captured = adapter.prepare(checkpoint, contract)
    output = tmp_path / "out.onnx"
    first = adapter.materialize_onnx(captured, contract, output)
    second = adapter.materialize_onnx(captured, contract, output)
    assert first.material_path == second.material_path
    assert [entry.name for entry in tmp_path.iterdir() if entry.name.endswith(".rfdetr-bundle")] == [
        first.material_path.parent.name
    ]
    first.material_path.write_bytes(b"conflict")
    with pytest.raises(ValueError, match="conflicts"):
        adapter.materialize_onnx(captured, contract, output)


def test_rfdetr_post_commit_crash_leaves_complete_reusable_bundle(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    _fake_vendor(monkeypatch)
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    captured = adapter.prepare(checkpoint, contract)
    output = tmp_path / "out.onnx"
    import vision.providers as providers

    original_fsync, calls = providers.os.fsync, 0

    def crash_after_commit(fd):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise OSError("post-commit crash")
        return original_fsync(fd)

    monkeypatch.setattr(providers.os, "fsync", crash_after_commit)
    with pytest.raises(OSError, match="post-commit crash"):
        adapter.materialize_onnx(captured, contract, output)
    bundles = list(tmp_path.glob("*.rfdetr-bundle"))
    assert len(bundles) == 1 and {item.name for item in bundles[0].iterdir()} == {"model.onnx", "lineage.json"}
    monkeypatch.setattr(providers.os, "fsync", original_fsync)
    assert adapter.materialize_onnx(captured, contract, output).material_path.parent == bundles[0]


def test_rfdetr_export_rejects_symlink_destination_and_never_publishes_incomplete_bundle(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    _fake_vendor(monkeypatch)
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    captured = adapter.prepare(checkpoint, contract)
    output = tmp_path / "out.onnx"
    output.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="destination"):
        adapter.materialize_onnx(captured, contract, output)

    output.unlink()
    import vision.providers as providers

    original_rename = providers.os.rename

    def crash_before_bundle_commit(source, destination):
        if str(destination).endswith(".rfdetr-bundle"):
            raise OSError("simulated crash")
        return original_rename(source, destination)

    monkeypatch.setattr(providers.os, "rename", crash_before_bundle_commit)
    with pytest.raises(OSError, match="simulated crash"):
        adapter.materialize_onnx(captured, contract, output)
    assert not output.exists() and not list(tmp_path.glob("*.rfdetr-bundle"))


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_rfdetr_publication_stage_rejects_unexpected_entries(monkeypatch, tmp_path, kind):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    _fake_vendor(monkeypatch)
    contract = _contract(_sha(checkpoint))
    adapter = RFDetrSourceAdapter()
    captured = adapter.prepare(checkpoint, contract)
    output = tmp_path / "out.onnx"
    import vision.providers as providers

    original_copy = providers.shutil.copy2

    def copy_with_extra(source, destination, *args, **kwargs):
        copied = original_copy(source, destination, *args, **kwargs)
        extra = Path(destination).parent / "unexpected.txt"
        if kind == "file":
            extra.write_bytes(b"unexpected")
        elif kind == "directory":
            extra.mkdir()
        else:
            extra.symlink_to(checkpoint)
        return copied

    monkeypatch.setattr(providers.shutil, "copy2", copy_with_extra)
    with pytest.raises(ValueError, match="publication staging"):
        adapter.materialize_onnx(captured, contract, output)
    assert not output.exists() and not list(tmp_path.glob("*.rfdetr-bundle"))
