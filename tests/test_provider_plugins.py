"""Provider plugins use local fake modules: no checkpoint downloads or network."""

import hashlib
import json
import shutil
import sys
import types
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from test_buildkit import contract

from vision.buildkit import BuildPlanner
from vision.contract import VisionContract
from vision.decoders import yolo_end2end_contract
from vision.plugins import (
    DEFAULT_REGISTRY,
    MaterializedOnnx,
    OnnxRuntimeTargetPlugin,
    PreparedSource,
    TargetPlugin,
    neutral_onnx_request,
)
from vision.processing import GeometryTrace
from vision.providers import LibreYoloSourceAdapter, UltralyticsSourceAdapter


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_provider(monkeypatch, name, exported, mutate_checkpoint=None, export_calls=None):
    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class Model:
        task = "detect"

        def __init__(self, checkpoint):
            assert checkpoint.endswith((".pt", ".onnx"))

        def predict(self, image, **options):
            if name != "ultralytics":
                raise AssertionError("LibreYOLO must use its callable inference API")
            assert options["rect"] is False
            return [
                types.SimpleNamespace(
                    boxes=types.SimpleNamespace(xyxy=Tensor([[0, 0, 1, 1]]), conf=Tensor([0.8]), cls=Tensor([0])),
                    names={0: "person"},
                )
            ]

        def __call__(self, image, **options):
            if name != "libreyolo":
                raise AssertionError("Ultralytics must use .predict")
            assert options == {"conf": 0.0, "iou": 0.5, "imgsz": 32, "max_det": 10}
            return types.SimpleNamespace(
                boxes=types.SimpleNamespace(xyxy=Tensor([[0, 0, 1, 1]]), conf=Tensor([0.8]), cls=Tensor([0])),
                names={0: "person"},
            )

        def export(self, **options):
            if export_calls is not None:
                export_calls.append(options)
            if name == "ultralytics":
                assert options == {"format": "onnx", "imgsz": 32, "batch": 1, "dynamic": False, "verbose": False}
            else:
                expected = {
                    "format": "onnx",
                    "output_path": str(exported) if options.get("nms") is not True else options["output_path"],
                    "imgsz": 32,
                    "batch": 1,
                    "simplify": False,
                    "dynamic": False,
                    "half": False,
                }
                if options.get("nms") is True:
                    expected |= {"nms": True, "conf": 0.0, "iou": 0.5, "max_det": 10}
                assert options == expected
                if options.get("nms") is True:
                    shutil.copy2(exported, options["output_path"])
            if mutate_checkpoint:
                mutate_checkpoint.write_bytes(b"mutated-during-export")
            return options["output_path"] if options.get("nms") is True else str(exported)

    loader = {"YOLO": Model} if name == "ultralytics" else {"LibreYOLO": Model}
    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**loader))


def _onnx(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["b"],
                value=helper.make_tensor("b", TensorProto.FLOAT, [1, 1, 4], [0, 0, 1, 1]),
            ),
            helper.make_node("Constant", [], ["s"], value=helper.make_tensor("s", TensorProto.FLOAT, [1, 1], [1])),
            helper.make_node("Constant", [], ["c"], value=helper.make_tensor("c", TensorProto.INT64, [1, 1], [0])),
        ],
        "fake",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 4]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("c", TensorProto.INT64, [1, 1]),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def _end_to_end_contract(source_digest):
    value = contract(source_digest).model_dump(mode="json")
    value["raw_output"] = {
        "semantics": "yolo_end2end",
        "tensors": [{"name": "output", "dtype": "float32", "shape": [1, "N", 6]}],
    }
    return VisionContract.model_validate(value)


def _end_to_end_onnx(path, outputs=("output", "raw"), max_det=10, output_width=6, metadata=None, metadata_entries=()):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    tensors = {
        "output": helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, max_det, output_width]),
        "raw": helper.make_tensor_value_info("raw", TensorProto.FLOAT, [1, 84, 21]),
    }
    for name in outputs:
        tensors.setdefault(name, helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, max_det, output_width]))
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["output"],
                value=helper.make_tensor(
                    "output", TensorProto.FLOAT, [1, max_det, output_width], [0] * (max_det * output_width)
                ),
            ),
            helper.make_node(
                "Constant", [], ["raw"], value=helper.make_tensor("raw", TensorProto.FLOAT, [1, 84, 21], [0] * 1764)
            ),
        ],
        "fake-end2end",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [tensors[name] for name in outputs],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    for key, value in (
        {
            "nms": "true",
            "nms_raw_output": "true",
            "nms_conf": "0.0",
            "nms_iou": "0.5",
            "max_det": "10",
            "names": json.dumps({"0": "person"}, separators=(",", ":")),
            "nc": "1",
            "nb_classes": "1",
        }
        | (metadata or {})
    ).items():
        if value is None:
            continue
        item = model.metadata_props.add()
        item.key, item.value = key, value
    for key, value in metadata_entries:
        item = model.metadata_props.add()
        item.key, item.value = key, value
    onnx.save(model, path)


def _image_bytes(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 6), "white").save(path)
    return path, path.read_bytes()


def test_target_base_and_ort_reject_legacy_four_argument_prepare_before_verification(monkeypatch):
    calls = []
    monkeypatch.setattr("vision.plugins._verify_neutral_request", lambda request: calls.append(request))

    class MinimalTarget(TargetPlugin):
        plugin_id, plugin_version = "minimal", "1"

        def build(self, captured, candidate, output=None):
            raise AssertionError("must not build")

        def run(self, artifact, sample):
            raise AssertionError("must not run")

    for target in (MinimalTarget(), OnnxRuntimeTargetPlugin()):
        with pytest.raises(TypeError, match=r"0\.2 migration.*exactly one sealed NeutralOnnxRequest"):
            target.prepare_materialized(object(), object(), object(), object())
    assert calls == []


@pytest.mark.parametrize(
    ("adapter_type", "module"), [(UltralyticsSourceAdapter, "ultralytics"), (LibreYoloSourceAdapter, "libreyolo")]
)
def test_provider_reference_and_neutral_materialization(monkeypatch, tmp_path, adapter_type, module):
    checkpoint, exported, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _onnx(exported)
    if module == "libreyolo":
        _onnx(output)
    _fake_provider(monkeypatch, module, output if module == "libreyolo" else exported)
    value = contract(_sha(checkpoint))
    adapter = adapter_type()
    captured = adapter.prepare(checkpoint, value)
    assert set(captured.__dict__) == {
        "material_path",
        "sha256",
        "plugin_id",
        "plugin_version",
        "contract_digest",
        "descriptors",
    }
    image, data = _image_bytes(tmp_path)
    prepared_input = adapter.prepare_input(captured, image, value)
    assert adapter.run(captured, prepared_input).classes == [0]
    assert adapter.run(captured, adapter.prepare_input(captured, data, value)).metadata["input_sha256"] == _sha(image)
    materialized = adapter.materialize_onnx(captured, value, output)
    assert materialized.source_sha256 == _sha(checkpoint)
    assert materialized.sha256 == _sha(output)
    assert materialized.lineage_path.is_file()

    # The planner attests provider provenance before target invocation; targets
    # receive only an opaque neutral request, never a provider runtime.
    target = OnnxRuntimeTargetPlugin()
    _, neutral, artifact = BuildPlanner().materialize(adapter, target, str(checkpoint), value, str(output))
    assert neutral.material_path == output
    assert artifact.material_path == output
    fabricated = MaterializedOnnx(
        neutral.material_path,
        neutral.sha256,
        neutral.source_sha256,
        neutral.plugin_id,
        neutral.plugin_version,
        neutral.contract_digest,
        neutral.lineage_path,
        neutral.lineage_sha256,
        "self-authored",
    )
    with pytest.raises(ValueError, match="centrally verified"):
        target.prepare_materialized(fabricated)
    assert target.verify_artifact(artifact, value)
    assert not target.verify_artifact(replace(artifact, descriptors=artifact.descriptors[:-1]), value)
    request = neutral_onnx_request(neutral, captured, adapter, value)
    neutral.lineage_path.write_text("{}")
    with pytest.raises(ValueError, match="lineage"):
        target.prepare_materialized(request)


def test_provider_rehashes_before_run_and_export(monkeypatch, tmp_path):
    checkpoint, exported = tmp_path / "detector.pt", tmp_path / "native.onnx"
    checkpoint.write_bytes(b"one")
    _onnx(exported)
    _fake_provider(monkeypatch, "ultralytics", exported)
    adapter = UltralyticsSourceAdapter()
    captured = adapter.prepare(checkpoint, contract(_sha(checkpoint)))
    checkpoint.write_bytes(b"two")
    image, _ = _image_bytes(tmp_path)
    with pytest.raises(ValueError, match="bound"):
        adapter.run(captured, adapter.prepare_input(captured, image, contract(captured.sha256)))
    with pytest.raises(ValueError, match="bound"):
        adapter.materialize_onnx(captured, contract(captured.sha256), tmp_path / "out.onnx")


def test_provider_rehashes_checkpoint_after_export(monkeypatch, tmp_path):
    checkpoint, exported = tmp_path / "detector.pt", tmp_path / "native.onnx"
    checkpoint.write_bytes(b"one")
    _onnx(exported)
    _fake_provider(monkeypatch, "ultralytics", exported, mutate_checkpoint=checkpoint)
    adapter = UltralyticsSourceAdapter()
    value = contract(_sha(checkpoint))
    with pytest.raises(ValueError, match="bound"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, tmp_path / "out.onnx")


def test_libreyolo_rejects_non_onnx_destination_without_exporting_or_mutating_it(monkeypatch, tmp_path):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.bin"
    checkpoint.write_bytes(b"checkpoint")
    native.write_bytes(b"not-used")
    output.write_bytes(b"known-good-destination")
    export_calls = []
    _fake_provider(monkeypatch, "libreyolo", native, export_calls=export_calls)
    value = contract(_sha(checkpoint))
    adapter = LibreYoloSourceAdapter()

    with pytest.raises(ValueError, match="requires a .onnx destination"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, output)
    assert output.read_bytes() == b"known-good-destination"
    assert export_calls == []


def test_libreyolo_rejects_checkpoint_as_onnx_destination_without_exporting(monkeypatch, tmp_path):
    checkpoint, native = tmp_path / "detector.onnx", tmp_path / "native.onnx"
    checkpoint.write_bytes(b"checkpoint")
    native.write_bytes(b"not-used")
    export_calls = []
    _fake_provider(monkeypatch, "libreyolo", native, export_calls=export_calls)
    value = contract(_sha(checkpoint))
    adapter = LibreYoloSourceAdapter()

    with pytest.raises(ValueError, match="must differ from the captured checkpoint"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, checkpoint)
    assert checkpoint.read_bytes() == b"checkpoint"
    assert export_calls == []


def test_libreyolo_yolo_end2end_export_canonicalizes_auxiliary_raw_output(monkeypatch, tmp_path):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _end_to_end_onnx(native)
    _fake_provider(monkeypatch, "libreyolo", native)
    value = _end_to_end_contract(_sha(checkpoint))

    adapter = LibreYoloSourceAdapter()
    materialized = adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, output)

    import onnx

    model = onnx.load(output)
    assert [item.name for item in model.graph.output] == ["output"]
    assert "raw" in {name for node in model.graph.node for name in node.output}
    assert {item.key: item.value for item in model.metadata_props}["nms_raw_output"] == "false"
    assert materialized.sha256 == _sha(output)


def test_yolo_end2end_decoder_returns_complete_contract_names():
    value_data = _end_to_end_contract("0" * 64).model_dump(mode="json")
    value_data["labels"] = ["first", "second"]
    value_data["coco_category_ids"] = [1, 2]
    value = VisionContract.model_validate(value_data)
    result = yolo_end2end_contract(
        np.array([[[0, 0, 2, 2, 0.9, 1]]], dtype=np.float32),
        GeometryTrace((8, 8), (32, 32), 1, 1),
        value,
    )
    assert result.names == {0: "first", 1: "second"}


@pytest.mark.parametrize("class_id", [-1, 2])
def test_yolo_end2end_decoder_rejects_out_of_range_contract_classes(class_id):
    value_data = _end_to_end_contract("0" * 64).model_dump(mode="json")
    value_data["labels"] = ["first", "second"]
    value_data["coco_category_ids"] = [1, 2]
    value = VisionContract.model_validate(value_data)
    with pytest.raises(ValueError, match="integral contract label indices"):
        yolo_end2end_contract(
            np.array([[[0, 0, 2, 2, 0.9, class_id]]], dtype=np.float32),
            GeometryTrace((8, 8), (32, 32), 1, 1),
            value,
        )


@pytest.mark.parametrize(
    "metadata,metadata_entries",
    [
        ({"names": None}, ()),
        ({"nc": "2"}, ()),
        ({"nb_classes": "2"}, ()),
        ({"names": '{"0":"person","0":"person"}'}, ()),
        ({"names": '{"1":"person"}'}, ()),
        ({"names": '{"0":"wrong"}'}, ()),
        ({"names": '{"0":"person","1":"extra"}'}, ()),
        ({}, (("names", '{"0":"person"}'),)),
    ],
)
def test_libreyolo_yolo_end2end_export_rejects_noncanonical_class_metadata(
    monkeypatch, tmp_path, metadata, metadata_entries
):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _end_to_end_onnx(native, metadata=metadata, metadata_entries=metadata_entries)
    output.write_bytes(b"known-good-destination")
    _fake_provider(monkeypatch, "libreyolo", native)
    value = _end_to_end_contract(_sha(checkpoint))

    with pytest.raises(
        ValueError, match="metadata (must contain exactly one|names contradict|nc contradicts|nb_classes contradicts)"
    ):
        LibreYoloSourceAdapter().materialize_onnx(LibreYoloSourceAdapter().prepare(checkpoint, value), value, output)
    assert output.read_bytes() == b"known-good-destination"


def test_libreyolo_yolo_end2end_canonicalization_preserves_public_output_bytes(monkeypatch, tmp_path):
    """Dropping `raw` from the public ABI must not alter the NMS graph result."""
    onnxruntime = pytest.importorskip("onnxruntime")
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _end_to_end_onnx(native)
    before = onnxruntime.InferenceSession(str(native), providers=["CPUExecutionProvider"])
    expected = before.run(["output"], {"x": np.zeros((1, 3, 32, 32), dtype=np.float32)})[0]
    _fake_provider(monkeypatch, "libreyolo", native)
    value = _end_to_end_contract(_sha(checkpoint))

    LibreYoloSourceAdapter().materialize_onnx(LibreYoloSourceAdapter().prepare(checkpoint, value), value, output)

    after = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = after.run(None, {"x": np.zeros((1, 3, 32, 32), dtype=np.float32)})
    assert [item.name for item in after.get_outputs()] == ["output"]
    np.testing.assert_array_equal(actual[0], expected)


@pytest.mark.parametrize(
    "outputs", [("output",), ("output", "raw", "raw"), ("output", "raw", "unexpected"), ("wrong", "raw")]
)
def test_libreyolo_yolo_end2end_export_rejects_unexpected_output_topology(monkeypatch, tmp_path, outputs):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _end_to_end_onnx(native, outputs)
    output.write_bytes(b"known-good-destination")
    _fake_provider(monkeypatch, "libreyolo", native)
    value = _end_to_end_contract(_sha(checkpoint))

    with pytest.raises(ValueError, match="exactly the 'output' and auxiliary 'raw'"):
        LibreYoloSourceAdapter().materialize_onnx(LibreYoloSourceAdapter().prepare(checkpoint, value), value, output)
    assert output.read_bytes() == b"known-good-destination"


def test_libreyolo_yolo_end2end_rehashes_checkpoint_after_export(monkeypatch, tmp_path):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"one")
    _end_to_end_onnx(native)
    output.write_bytes(b"known-good-destination")
    _fake_provider(monkeypatch, "libreyolo", native, mutate_checkpoint=checkpoint)
    value = _end_to_end_contract(_sha(checkpoint))

    adapter = LibreYoloSourceAdapter()
    with pytest.raises(ValueError, match="bound"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, output)
    assert output.read_bytes() == b"known-good-destination"


def test_libreyolo_yolo_end2end_export_rejects_output_not_bound_to_max_det(monkeypatch, tmp_path):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _end_to_end_onnx(native, output_width=5)
    _fake_provider(monkeypatch, "libreyolo", native)
    value = _end_to_end_contract(_sha(checkpoint))

    adapter = LibreYoloSourceAdapter()
    with pytest.raises(ValueError, match=r"shape \[1, max_det, 6\]"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, output)


@pytest.mark.parametrize("metadata", [{"nms": None}, {"nms_raw_output": "false"}])
def test_libreyolo_yolo_end2end_export_rejects_missing_or_false_nms_metadata(monkeypatch, tmp_path, metadata):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _end_to_end_onnx(native, metadata=metadata)
    _fake_provider(monkeypatch, "libreyolo", native)
    value = _end_to_end_contract(_sha(checkpoint))

    adapter = LibreYoloSourceAdapter()
    with pytest.raises(ValueError, match="metadata must prove"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, output)


@pytest.mark.parametrize(
    "result_change",
    [{"nms_owner": "target"}, {"class_aware_nms": False}],
)
def test_libreyolo_yolo_end2end_rejects_contracts_contradicting_provider_nms(monkeypatch, tmp_path, result_change):
    checkpoint, native, output = tmp_path / "detector.pt", tmp_path / "native.onnx", tmp_path / "built.onnx"
    checkpoint.write_bytes(b"checkpoint")
    _end_to_end_onnx(native)
    _fake_provider(monkeypatch, "libreyolo", native)
    value_data = _end_to_end_contract(_sha(checkpoint)).model_dump(mode="json")
    value_data["result"].update(result_change)
    value = VisionContract.model_validate(value_data)

    adapter = LibreYoloSourceAdapter()
    with pytest.raises(ValueError, match="source-owned class-aware NMS"):
        adapter.materialize_onnx(adapter.prepare(checkpoint, value), value, output)


@pytest.mark.parametrize("attack", ["input_name", "input_dtype", "input_shape", "output_shape"])
def test_onnx_graph_contract_rejects_interface_attacks(tmp_path, attack):
    import onnx

    from vision.plugins import _validate_onnx_contract

    path = tmp_path / f"{attack}.onnx"
    _onnx(path)
    model = onnx.load(path)
    if attack == "input_name":
        model.graph.input[0].name = "wrong"
    elif attack == "input_dtype":
        model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    elif attack == "input_shape":
        model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value = 64
    else:
        model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = 5
    onnx.save(model, path)
    with pytest.raises(ValueError, match="interface mismatch|tensor names"):
        _validate_onnx_contract(path, contract(_sha(path)))


def test_provider_registry_capabilities_and_lazy_missing_dependency(monkeypatch, tmp_path):
    assert {"ultralytics", "libreyolo"} <= set(DEFAULT_REGISTRY.sources)
    assert "fp32" in UltralyticsSourceAdapter().capabilities.precisions
    assert "fp16" not in UltralyticsSourceAdapter().capabilities.precisions
    checkpoint = tmp_path / "missing.pt"
    checkpoint.write_bytes(b"x")
    import vision.providers as providers

    original_import = providers.importlib.import_module

    def without_libreyolo(name):
        if name == "libreyolo":
            raise ImportError("simulated missing optional dependency")
        return original_import(name)

    monkeypatch.setattr(providers.importlib, "import_module", without_libreyolo)
    with pytest.raises(RuntimeError, match=r"vision-buildkit\[libreyolo\]"):
        LibreYoloSourceAdapter().capture(checkpoint, contract(_sha(checkpoint)))
    with pytest.raises(TypeError):
        PreparedSource(tmp_path / "x", "0" * 64, "p", "v", "d", runner=object())
