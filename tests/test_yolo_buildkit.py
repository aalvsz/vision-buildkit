import hashlib

import pytest

from vision.capsule import create_certified_capsule, verify_capsule
from vision.conformance import EvidenceFile, _canonical_predictions, conform
from vision.contract import VisionContract
from vision.plugins import OnnxRuntimeTargetPlugin, PreparedSource, SourceAdapter
from vision.preprocess import preprocess


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(digest):
    return VisionContract.model_validate(
        {
            "input": {
                "input_name": "images",
                "color": "RGB",
                "layout": "NCHW",
                "dtype": "float32",
                "shape": [1, 3, 640, 640],
            },
            "preprocessing": [
                {
                    "kind": "letterbox",
                    "height": 640,
                    "width": 640,
                    "pad_value": 114,
                    "interpolation": "bilinear",
                    "placement": "center",
                    "rounding": "round_half_up",
                    "scale": "min_ratio",
                    "implementation": "ultralytics_letterbox_cv2",
                },
                {"kind": "normalize", "mean": [0, 0, 0], "std": [255, 255, 255]},
            ],
            "raw_output": {
                "semantics": "yolo_end2end",
                "tensors": [{"name": "output0", "dtype": "float32", "shape": [1, 300, 6]}],
            },
            "result": {
                "box_format": "xyxy",
                "coordinate_space": "original_image",
                "nms_owner": "none",
                "nms_iou_threshold": None,
                "score_threshold": 0.5,
                "class_aware_nms": False,
                "clip_boxes": True,
                "top_k": 2,
                "max_detections": 2,
                "suppression": "none",
                "ordering": "score_desc",
                "tie_policy": "stable",
            },
            "labels": ["person"],
            "coco_category_ids": [1],
            "tolerances": {"box_iou": 0.9, "score_delta": 0.01},
            "provenance": {"source": "synthetic", "source_sha256": digest},
        }
    )


def test_yolo_end2end_onnx_preflight_and_odd_letterbox_inverse(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    from PIL import Image

    model = tmp_path / "yolo.onnx"
    rows = [320, 320, 640, 640, 0.8, 0, 0, 0, 10, 10, 0.9, 0, 1, 1, 2, 2, 0.5, 0]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["output0"], value=helper.make_tensor("out", TensorProto.FLOAT, [1, 3, 6], rows)
            )
        ],
        "yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 6])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), model)
    contract = _contract(_sha(model)).model_copy(
        update={
            "raw_output": _contract(_sha(model)).raw_output.model_copy(
                update={
                    "tensors": (_contract(_sha(model)).raw_output.tensors[0].model_copy(update={"shape": (1, 3, 6)}),)
                }
            )
        }
    )
    plugin = OnnxRuntimeTargetPlugin()
    artifact = plugin.prepare(model, contract)
    image = tmp_path / "odd.png"
    Image.new("RGB", (333, 197), (12, 34, 56)).save(image)
    result = plugin.run(artifact, plugin.prepare_input(artifact, image, contract))
    assert result.scores == pytest.approx([0.9, 0.8])
    assert result.classes == [0, 0]
    assert result.image_size == (333, 197)
    assert all(
        0 <= value <= bound
        for box in result.boxes
        for value, bound in zip((box.x1, box.y1, box.x2, box.y2), (333, 197, 333, 197), strict=True)
    )
    prepared_input = plugin.prepare_input(artifact, image, contract)
    assert prepared_input.tensor.min() >= 0
    assert prepared_input.tensor.max() <= 1
    assert prepared_input.tensor[
        :, :, int(prepared_input.trace.pad_y), int(prepared_input.trace.pad_x)
    ].flatten() == pytest.approx([12 / 255, 34 / 255, 56 / 255])
    from vision.adapters.onnx import OnnxAdapter
    from vision.manifest import manifest_from_contract

    packaged = OnnxAdapter(manifest_from_contract(str(model), contract)).load(model, "cpu")
    packaged_result = packaged(image)
    assert packaged_result.scores == pytest.approx([0.9, 0.8])
    assert packaged_result.classes == [0, 0]
    unexpected = tmp_path / "unexpected.onnx"
    attack = onnx.load(str(model))
    attack.graph.node.append(
        helper.make_node(
            "Constant",
            [],
            ["unexpected"],
            value=helper.make_tensor("unexpected", TensorProto.FLOAT, [1, 1, 6], [0, 0, 0, 0, 0, 0]),
        )
    )
    attack.graph.output.append(helper.make_tensor_value_info("unexpected", TensorProto.FLOAT, [1, 1, 6]))
    onnx.save(attack, unexpected)
    with pytest.raises(ValueError, match="output names contradict"):
        OnnxAdapter(manifest_from_contract(str(unexpected), contract)).load(unexpected, "cpu")
    tagged = onnx.load(str(model))
    for key, value in {
        "author": "Ultralytics",
        "task": "detect",
        "head": "Detect",
        "end2end": "true",
        "args": "{'nms': False}",
        "imgsz": "(640, 640)",
        "names": "{0: 'person'}",
    }.items():
        item = tagged.metadata_props.add()
        item.key, item.value = key, value
    onnx.save(tagged, model)
    valid_contract = _contract(_sha(model)).model_copy(update={"raw_output": contract.raw_output})
    plugin.prepare(model, valid_contract)
    item = tagged.metadata_props.add()
    item.key, item.value = "nms", "true"
    onnx.save(tagged, model)
    with pytest.raises(ValueError, match="metadata 'nms' contradicts args"):
        plugin.prepare(model, _contract(_sha(model)).model_copy(update={"raw_output": contract.raw_output}))


COCO80 = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def test_local_yolo26n_metadata_preflight_and_normalization(tmp_path):
    pytest.importorskip("onnxruntime")
    from PIL import Image

    model = __import__("pathlib").Path("yolo26n.onnx")
    if not model.is_file():
        pytest.skip("optional real yolo26n.onnx fixture is not distributed with the repository")
    base = _contract(_sha(model))
    contract = base.model_copy(
        update={
            "raw_output": base.raw_output.model_copy(
                update={"tensors": (base.raw_output.tensors[0].model_copy(update={"shape": (1, 300, 6)}),)}
            ),
            "labels": COCO80,
            "coco_category_ids": tuple(range(80)),
            "result": base.result.model_copy(update={"top_k": 300, "max_detections": 300}),
        }
    )
    plugin = OnnxRuntimeTargetPlugin()
    artifact = plugin.prepare(model, contract)
    image = tmp_path / "red.png"
    Image.new("RGB", (1, 1), (255, 0, 0)).save(image)
    prepared = plugin.prepare_input(artifact, image, contract)
    assert prepared.tensor.shape == (1, 3, 640, 640)
    assert prepared.tensor[:, :, 0, 0].flatten() == pytest.approx([1.0, 0.0, 0.0])
    from vision.manifest import manifest_from_contract

    assert manifest_from_contract(str(model), contract).preprocessing.normalization == "1/255"


def test_yolo_certified_capsule_runtime_parity(tmp_path):
    # Reuse the established file-backed evidence fixture; predictions are
    # overwritten only with bytes generated by this prepared YOLO artifact.
    import json
    from pathlib import Path

    from onnx import TensorProto, helper
    from PIL import Image
    from test_buildkit import files

    from vision import load
    from vision.manifest import VisionManifest

    model = tmp_path / "yolo.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["output0"],
                value=helper.make_tensor("out", TensorProto.FLOAT, [1, 1, 6], [0, 128, 640, 512, 0.9, 0]),
            )
        ],
        "yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 6])],
    )
    import onnx

    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), model)
    base = _contract(_sha(model))
    contract = base.model_copy(
        update={
            "raw_output": base.raw_output.model_copy(
                update={"tensors": (base.raw_output.tensors[0].model_copy(update={"shape": (1, 1, 6)}),)}
            )
        }
    )
    image = tmp_path / "image.png"
    Image.new("RGB", (333, 197), (10, 20, 30)).save(image)
    payload = image.read_bytes()
    target = OnnxRuntimeTargetPlugin()
    artifact = target.prepare(model, contract)

    class Source(SourceAdapter):
        plugin_id, plugin_version, capabilities = "yolo-source", "1", target.capabilities

        def capture(self, source, contract):
            return source

        def prepare(self, source, contract):
            return PreparedSource(
                source,
                _sha(source),
                self.plugin_id,
                self.plugin_version,
                contract.digest,
                (("contract_json", contract.canonical_json()),),
            )

        def prepare_input(self, captured, sample, contract):
            return preprocess(contract, sample)

        def run(self, captured, sample):
            return target.run(
                target.prepare(
                    captured.material_path,
                    VisionContract.model_validate_json(dict(captured.descriptors)["contract_json"]),
                ),
                sample,
            )

    source = Source()
    captured = source.prepare(model, contract)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    golden, runtime, accuracy, bundle = files(
        evidence_root,
        _sha(model),
        _sha(model),
        image_bytes=payload,
        case_bytes=payload,
        source_plugin=source.plugin_id,
        target_plugin=target.plugin_id,
        target_version=target.plugin_version,
        metric=0.0,
    )
    policy = bundle.policy.verify()
    policy.write_text(
        '{"metric_name":"map","maximum_drop":0.0,"golden_parity_lost":0,"golden_parity_added":0,"mean_iou_min":0.95,"min_iou_min":0.85,"max_score_delta":0.12}'
    )
    bundle = bundle.model_copy(update={"policy": EvidenceFile(path=str(policy), sha256=_sha(policy))})
    pred = _canonical_predictions(target, artifact, contract, bundle.dataset_manifest.verify())
    prediction_path = bundle.predictions.verify()
    prediction_path.write_bytes(pred)
    metrics_path = Path(bundle.accuracy_report.path)
    metrics = json.loads(metrics_path.read_text())
    source_prediction_path = Path(bundle.source_predictions.path)
    source_prediction_path.write_bytes(pred)
    from vision.conformance import _prediction_parity_report

    metrics.update(
        {
            "source_predictions_path": source_prediction_path.name,
            "source_predictions_sha256": _sha(source_prediction_path),
            "source_metric_value": 0.0,
            "parity": _prediction_parity_report(source_prediction_path, prediction_path, (1,)),
        }
    )
    metrics_path.write_text(json.dumps(metrics, sort_keys=True, separators=(",", ":")))
    prediction = EvidenceFile(path=str(prediction_path), sha256=_sha(prediction_path))
    record = bundle.runtime_record.verify()
    body = json.loads(record.read_text())
    body.update(
        {
            "backend": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "device": "cpu",
            "runtime_version": target.plugin_version,
        }
    )
    record.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")))
    bundle = bundle.model_copy(
        update={
            "predictions": prediction,
            "source_predictions": EvidenceFile(path=str(source_prediction_path), sha256=_sha(source_prediction_path)),
            "accuracy_report": EvidenceFile(path=str(metrics_path), sha256=_sha(metrics_path)),
            "runtime_record": EvidenceFile(path=str(record), sha256=_sha(record)),
        }
    )
    accuracy = accuracy.model_copy(update={"prediction_digest": prediction.sha256, "report_digest": _sha(metrics_path)})
    runtime = runtime.model_copy(
        update={
            "backend": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "device": "cpu",
            "runtime_version": target.plugin_version,
        }
    )
    report = conform(
        source,
        target,
        captured,
        artifact,
        [payload],
        contract,
        golden=golden,
        policy_digest=bundle.policy.sha256,
        source_digest=_sha(model),
        artifact_digest=_sha(model),
        runtime=runtime,
        accuracy=accuracy,
        evidence=bundle,
    )
    assert report.status == "PASS", report.details
    root = create_certified_capsule(
        tmp_path / "yolo.vcap", VisionManifest(source=str(model)), contract, report, source=model, artifact=model
    )
    expected_capsule_id = json.loads((root / "capsule.json").read_text())["capsule_id"]
    index = verify_capsule(root, expected_capsule_id=expected_capsule_id)
    loaded = load(
        str(root), backend="onnxruntime", provider="CPUExecutionProvider", expected_capsule_id=index.capsule_id
    )
    expected = target.run(artifact, target.prepare_input(artifact, image, contract))
    actual = loaded(image)
    assert (actual.boxes, actual.scores, actual.classes, actual.image_size) == (
        expected.boxes,
        pytest.approx(expected.scores),
        expected.classes,
        expected.image_size,
    )
