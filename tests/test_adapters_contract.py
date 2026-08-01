import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from vision.adapters.processor import DetectionProcessor
from vision.decoders import rfdetr_output, yolo_end2end
from vision.manifest import VisionManifest
from vision.processing import direct_resize, letterbox, libreyolo_yolo9_letterbox


def test_public_load_auto_selects_checksum_bound_compiled_artifact(monkeypatch, tmp_path):
    import hashlib

    import vision.adapters
    import vision.api as api

    artifact = tmp_path / "model.xml"
    artifact.write_bytes(b"openvino")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = VisionManifest(
        source=str(artifact),
        family="onnx",
        artifacts=[
            {
                "path": str(artifact),
                "backend": "openvino",
                "precision": "fp16",
                "validated": True,
                "checksum": checksum,
                "compatible": {"provider": "CPU", "device": "cpu"},
            }
        ],
    )
    captured = {}

    class Adapter:
        def __init__(self, contract, backend, provider, precision):
            captured.update(contract=contract, backend=backend, provider=provider, precision=precision)

        def load(self, source, device):
            captured.update(source=source, device=device)
            return self

    monkeypatch.setattr(vision.adapters, "BackendAdapter", Adapter)
    assert api.load(str(artifact), device="cpu", manifest=manifest).__class__ is Adapter
    assert captured["backend"] == "openvino"
    assert captured["provider"] == "CPU"
    assert captured["precision"] == "fp16"


def test_public_load_rejects_undeclared_compiled_artifact(tmp_path):
    import vision.api as api

    artifact = tmp_path / "model.engine"
    artifact.write_bytes(b"tensorrt")
    with pytest.raises(ValueError, match="validated manifest artifact"):
        api.load(str(artifact), manifest=VisionManifest(source=str(artifact), family="onnx"))


def test_ultralytics_export_binds_manifest_shape_batch_and_dynamic(tmp_path):
    from vision.adapters.ultralytics import UltralyticsAdapter

    captured = {}

    class Vendor:
        def export(self, **options):
            captured.update(options)
            path = tmp_path / "vendor.onnx"
            path.write_bytes(b"onnx")
            return path

    manifest = VisionManifest(source="model.pt", family="yolo26", input={"shape": [2, 3, 512, 512]})
    adapter = UltralyticsAdapter(manifest)
    adapter.model = Vendor()
    output = tmp_path / "selected.onnx"
    assert adapter.export_onnx(output, dynamic=True) == str(output)
    assert captured == {"format": "onnx", "imgsz": 512, "batch": 2, "dynamic": True, "verbose": False}


def test_strict_manifest_and_cross_contracts():
    with pytest.raises(ValidationError):
        VisionManifest(source="x", unexpected=True)
    with pytest.raises(ValidationError, match="RF-DETR"):
        VisionManifest(source="x", family="rfdetr")
    m = VisionManifest(
        source="x",
        family="rfdetr",
        input={"shape": [1, 3, 384, 384]},
        preprocessing={
            "resize": "resize",
            "implementation": "torchvision_bilinear_antialias_false",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        output={"box_format": "cxcywh"},
        decoder={"kind": "rfdetr", "boxes_normalized": True},
    )
    assert m.preprocessing.resize == "resize"


def test_golden_geometry_letterbox_and_direct_resize():
    image = Image.new("RGB", (1000, 500), (255, 0, 0))
    _, trace = letterbox(image, (640, 640))
    assert (trace.scale_x, trace.scale_y, trace.pad_x, trace.pad_y) == pytest.approx((0.64, 0.64, 0, 160))
    assert trace.inverse(__import__("vision").Box(0, 160, 640, 480)).as_xyxy() == pytest.approx([0, 0, 1000, 500])
    _, direct = direct_resize(image, (384, 384))
    assert direct.inverse(__import__("vision").Box(0, 0, 384, 384)).as_xyxy() == pytest.approx([0, 0, 1000, 500])


def test_textured_odd_padding_letterbox_matches_cv2_vendor_kernel():
    import cv2

    pixels = np.arange(101 * 63 * 3, dtype=np.uint8).reshape(63, 101, 3)
    image = Image.fromarray(pixels, "RGB")
    actual, trace = letterbox(image, (640, 638))
    ratio = min(640 / 101, 638 / 63)
    resized = cv2.resize(pixels, (round(101 * ratio), round(63 * ratio)), interpolation=cv2.INTER_LINEAR)
    dx, dy = (640 - resized.shape[1]) / 2, (638 - resized.shape[0]) / 2
    expected = cv2.copyMakeBorder(
        resized,
        round(dy - 0.1),
        round(dy + 0.1),
        round(dx - 0.1),
        round(dx + 0.1),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    assert np.array_equal(actual, expected)
    assert (trace.pad_x, trace.pad_y) == (round(dx - 0.1), round(dy - 0.1))


def test_libreyolo_yolo9_top_left_letterbox_matches_v140_preprocess_numpy():
    import cv2

    pixels = np.arange(63 * 101 * 3, dtype=np.uint8).reshape(63, 101, 3)
    image = Image.fromarray(pixels, "RGB")
    actual, trace = libreyolo_yolo9_letterbox(image, (640, 640))
    ratio = min(640 / 63, 640 / 101)
    height, width = int(63 * ratio), int(101 * ratio)
    expected = np.full((640, 640, 3), 114, dtype=np.uint8)
    expected[:height, :width] = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(actual, expected)
    assert (trace.scale_x, trace.scale_y, trace.pad_x, trace.pad_y) == (ratio, ratio, 0.0, 0.0)
    assert trace.inverse(__import__("vision").Box(0, 0, width, height)).as_xyxy() == pytest.approx(
        [0, 0, width / ratio, height / ratio]
    )


def test_rfdetr_preprocessing_golden_values():
    manifest = VisionManifest(
        source="x",
        family="rfdetr",
        input={"shape": [1, 3, 384, 384]},
        preprocessing={
            "resize": "resize",
            "implementation": "torchvision_bilinear_antialias_false",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        output={"box_format": "cxcywh"},
        decoder={"kind": "rfdetr", "boxes_normalized": True},
    )
    tensor, trace = DetectionProcessor(manifest)(Image.new("RGB", (100, 50), (255, 0, 0)))
    assert tensor.shape == (1, 3, 384, 384)
    assert tensor[0, :, 0, 0] == pytest.approx([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    assert trace.scale_x == pytest.approx(3.84) and trace.scale_y == pytest.approx(7.68)


def test_rfdetr_tensor_resize_does_not_require_cv2(monkeypatch):
    manifest = VisionManifest(
        source="x",
        family="rfdetr",
        input={"shape": [1, 3, 384, 384]},
        preprocessing={
            "resize": "resize",
            "implementation": "torchvision_bilinear_antialias_false",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        output={"box_format": "cxcywh"},
        decoder={"kind": "rfdetr", "boxes_normalized": True},
    )
    monkeypatch.setattr(
        "vision.adapters.processor.direct_resize", lambda *_: (_ for _ in ()).throw(AssertionError("cv2"))
    )
    tensor, trace = DetectionProcessor(manifest)(Image.new("RGB", (100, 50)))
    assert tensor.shape == (1, 3, 384, 384)
    assert trace.target == (384, 384)


def test_rfdetr_sigmoid_global_query_class_ranking():
    result = rfdetr_output(
        {
            "boxes": np.array([[[0.5, 0.5, 0.2, 0.4], [0.2, 0.2, 0.1, 0.1]]]),
            "logits": np.array([[[0.0, 3.0], [5.0, -2.0]]]),
        },
        (100, 200),
        top_k=2,
        threshold=0.5,
    )
    assert result.classes == [0, 1]  # global 5.0 query/class first, then 3.0
    assert result.boxes[0].as_xyxy() == pytest.approx([15, 30, 25, 50])


def test_onnx_rfdetr_contract_is_not_bypassed_by_family_name():
    with pytest.raises(ValidationError, match="RF-DETR"):
        VisionManifest(source="x.onnx", family="onnx", decoder={"kind": "rfdetr", "boxes_normalized": True})


@pytest.mark.parametrize(
    ("decoder", "output"),
    [
        ({"kind": "yolo_end2end"}, {"boxes_name": "missing"}),
        ({"kind": "yolo_raw"}, {"boxes_name": "missing"}),
        ({"kind": "rtdetr_combined", "boxes_normalized": True}, {"box_format": "cxcywh", "boxes_name": "missing"}),
        (
            {"kind": "rfdetr", "boxes_normalized": True},
            {"box_format": "cxcywh", "boxes_name": "dets", "logits_name": "missing"},
        ),
        ({"kind": "detr"}, {"boxes_name": "missing", "logits_name": "logits"}),
    ],
)
def test_onnx_output_preflight_fails_before_run(decoder, output):
    from vision.adapters.onnx import OnnxAdapter

    preprocessing = {"resize": "resize", "implementation": "opencv_linear_scale_fill"}
    if decoder["kind"] == "rfdetr":
        preprocessing = {
            "resize": "resize",
            "implementation": "torchvision_bilinear_antialias_false",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    input_spec = {"shape": [1, 3, 384, 384]} if decoder["kind"] == "rfdetr" else {}
    manifest = VisionManifest(
        source="x.onnx",
        family="onnx",
        input=input_spec,
        preprocessing=preprocessing if decoder["kind"] != "yolo_end2end" else {},
        output=output,
        decoder=decoder,
    )
    adapter = OnnxAdapter(manifest)
    adapter.output_names = {"output0", "dets", "labels"}
    with pytest.raises(ValueError, match="ONNX outputs"):
        adapter._preflight_outputs()


def test_decoder_threshold_is_strictly_greater_than_vendor_conf():
    _, trace = letterbox(Image.new("RGB", (10, 10)), (10, 10))
    result = yolo_end2end(np.array([[[0, 0, 1, 1, 0.25, 0], [0, 0, 1, 1, 0.2501, 0]]]), trace, 0.25)
    assert len(result.boxes) == 1


def test_rfdetr_missing_checkpoint_fails_before_vendor_load(tmp_path):
    from vision.adapters.rfdetr import RfDetrAdapter

    manifest = VisionManifest(
        source=str(tmp_path / "missing.pth"),
        family="rfdetr",
        input={"shape": [1, 3, 384, 384]},
        preprocessing={
            "resize": "resize",
            "implementation": "torchvision_bilinear_antialias_false",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        output={"box_format": "cxcywh"},
        decoder={"kind": "rfdetr", "boxes_normalized": True},
    )
    with pytest.raises(FileNotFoundError):
        RfDetrAdapter(manifest).load(manifest.source)


@pytest.mark.parametrize(
    ("alias", "variant", "resolution"),
    [
        ("rfdetr-nano", "nano", 384),
        ("rfdetr-small", "small", 512),
        ("rfdetr-medium", "medium", 576),
        ("rfdetr-base", "base", 560),
        ("rfdetr-large", "large", 704),
    ],
)
def test_public_load_builds_explicit_official_rfdetr_variant_contract(monkeypatch, alias, variant, resolution):
    import vision.api as api

    captured = {}

    class Adapter:
        def load(self, source, device):
            captured.update(source=source, device=device)
            return self

    def resolve(source, contract, family):
        captured.update(contract=contract, family=family)
        return Adapter()

    monkeypatch.setattr(api, "resolve_adapter", resolve)
    api.load(alias, device="cpu")
    assert captured["family"] == "rfdetr"
    assert captured["source"] == alias
    assert captured["contract"].variant == variant
    assert captured["contract"].input.shape == [1, 3, resolution, resolution]


@pytest.mark.parametrize(
    ("source", "normalized", "family"),
    [
        ("ultralytics/yolo26m", "yolo26m.pt", "yolo26"),
        ("yolo26n", "yolo26n.pt", "yolo26"),
        ("ultralytics/rtdetr-l", "rtdetr-l.pt", "rtdetr"),
    ],
)
def test_public_load_recognizes_only_explicit_official_native_identifiers(monkeypatch, source, normalized, family):
    import vision.api as api

    captured = {}

    class Adapter:
        def load(self, actual, device):
            captured.update(source=actual, device=device)
            return self

    def resolve(actual, contract, actual_family):
        captured.update(contract=contract, family=actual_family)
        return Adapter()

    monkeypatch.setattr(api, "resolve_adapter", resolve)
    api.load(source, device="cpu")
    expected = (
        VisionManifest(
            source=normalized,
            family=family,
            output={"boxes_name": "output0"},
            decoder={"kind": "yolo_end2end"},
        )
        if family == "yolo26"
        else VisionManifest(
            source=normalized,
            family=family,
            preprocessing={"resize": "resize", "implementation": "opencv_linear_scale_fill"},
            output={"box_format": "cxcywh", "boxes_name": "output0"},
            decoder={"kind": "rtdetr_combined", "boxes_normalized": True},
        )
    )
    assert captured == {
        "source": normalized,
        "device": "cpu",
        "contract": expected,
        "family": family,
    }


def test_ultralytics_onnx_metadata_builds_manifest_for_direct_load(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    import vision.api as api

    path = tmp_path / "yolo26n.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "model",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 300, 6])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for key, value in {
        "author": "Ultralytics",
        "task": "detect",
        "head": "Detect",
        "end2end": "True",
        "imgsz": "[640, 640]",
    }.items():
        item = model.metadata_props.add()
        item.key, item.value = key, value
    onnx.save(model, path)
    contract = api._ultralytics_onnx_manifest(str(path))
    assert contract is not None
    assert contract.family == "onnx"
    assert contract.decoder.kind == "yolo_end2end"
    assert contract.output.boxes_name == "output0"


def test_capsule_load_rebinds_real_verified_onnx_capsule(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from vision.capsule import create_capsule, verify_capsule

    model_path = tmp_path / "tiny.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)
    capsule = tmp_path / "model.vcap"
    original = VisionManifest(
        source="original.pt",
        family="yolo26",
        input={"shape": [1, 3, 8, 8]},
        output={"boxes_name": "output0"},
        artifacts=[{"path": "artifacts/tiny.onnx", "backend": "onnxruntime", "validated": True}],
        decoder={"kind": "yolo_end2end", "score_threshold": 0.25},
    )
    validation = {
        "reference_id": "r",
        "candidate_id": "c",
        "reference_digest": "a",
        "candidate_digest": "b",
        "parity": {},
        "gates": {"parity": True},
        "passed": True,
    }
    root = create_capsule(capsule, original, artifacts=[model_path], validation=validation)
    with pytest.raises(ValueError, match="unvalidated development"):
        verify_capsule(root)
