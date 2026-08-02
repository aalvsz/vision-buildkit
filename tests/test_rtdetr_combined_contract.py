"""Frozen RT-DETR combined-output ABI coverage."""

from __future__ import annotations

import hashlib

import pytest

from vision.contract import VisionContract
from vision.manifest import VisionManifest, manifest_from_contract, validate_contract_equivalence


def _contract(
    *,
    resize_implementation="opencv_linear_scale_fill",
    resize_interpolation="bilinear",
    **output,
):
    raw = {
        "semantics": "rtdetr_combined",
        "tensors": [{"name": "output0", "dtype": "float32", "shape": [1, 300, 6]}],
    } | output
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
                    "interpolation": resize_interpolation,
                }
                | ({"implementation": resize_implementation} if resize_implementation is not None else {}),
                {"kind": "normalize", "mean": [0, 0, 0], "std": [255, 255, 255]},
                {"kind": "transpose", "order": [0, 3, 1, 2]},
            ],
            "raw_output": raw,
            "result": {
                "box_format": "xyxy",
                "coordinate_space": "original_image",
                "nms_owner": "none",
                "nms_iou_threshold": None,
                "ordering": "score_desc",
                "tie_policy": "class_then_box",
                "score_threshold": 0.4,
                "class_aware_nms": False,
                "clip_boxes": True,
                "top_k": 300,
                "max_detections": 300,
                "suppression": "none",
            },
            "labels": ["person"],
            "coco_category_ids": [1],
            "tolerances": {"box_iou": 0.99, "score_delta": 0.01},
            "provenance": {"source": "rtdetr-l.pt", "source_sha256": "a" * 64},
        }
    )


def test_rtdetr_combined_manifest_round_trip_is_exact_and_truthful():
    contract = _contract()
    manifest = manifest_from_contract("artifact.onnx", contract)
    assert manifest.decoder.kind == "rtdetr_combined_v1"
    assert manifest.preprocessing.implementation == "opencv_linear_scale_fill"
    assert manifest.decoder.boxes_normalized is True
    assert manifest.output.box_format == "cxcywh"
    assert manifest.output.boxes_name == "output0"
    assert VisionManifest.model_validate(manifest.model_dump(mode="json")) == manifest
    validate_contract_equivalence(manifest, contract)


@pytest.mark.parametrize(
    "change",
    [
        {"tensors": [{"name": "combined", "dtype": "float32", "shape": [1, 300, 6]}]},
        {"tensors": [{"name": "output0", "dtype": "float16", "shape": [1, 300, 6]}]},
        {"tensors": [{"name": "output0", "dtype": "float32", "shape": [1, "N", 6]}]},
        {"roles": {"boxes": "output0"}},
    ],
)
def test_rtdetr_combined_rejects_every_nonfrozen_abi(change):
    with pytest.raises(ValueError, match="rtdetr_combined"):
        _contract(**change)


@pytest.mark.parametrize("implementation", [None, "pil", "torchvision_bilinear_antialias_false"])
def test_rtdetr_combined_rejects_non_opencv_resize_implementation(implementation):
    with pytest.raises(ValueError, match="OpenCV INTER_LINEAR scale-fill"):
        _contract(resize_implementation=implementation)


def test_resize_implementation_is_backward_compatible_for_non_rtdetr_contracts():
    contract = _contract(
        resize_implementation=None,
        semantics="custom",
        decoder_plugin="local",
        tensors=[{"name": "output0", "dtype": "float32", "shape": [1, 300, 6]}],
    )
    assert contract.preprocessing[0].implementation == "pil"
    canonical = contract.model_dump(mode="json")
    assert "implementation" not in canonical["preprocessing"][0]
    assert "implementation" not in contract.canonical_json()
    assert contract.digest == hashlib.sha256(contract.canonical_json().encode()).hexdigest()


def test_explicit_nondefault_resize_implementation_is_canonicalized():
    contract = _contract()
    canonical = contract.model_dump(mode="json")
    assert canonical["preprocessing"][0]["implementation"] == "opencv_linear_scale_fill"
    assert contract.digest == hashlib.sha256(contract.canonical_json().encode()).hexdigest()


def test_opencv_scale_fill_rejects_non_linear_interpolation():
    with pytest.raises(ValueError, match="requires bilinear interpolation"):
        _contract(resize_interpolation="bicubic")


def test_generic_custom_output_remains_fail_closed_for_capsule_projection():
    contract = _contract(
        semantics="custom",
        decoder_plugin="local",
        tensors=[{"name": "output0", "dtype": "float32", "shape": [1, 300, 6]}],
    )
    with pytest.raises(ValueError, match="cannot certify raw output semantics"):
        manifest_from_contract("artifact.onnx", contract)


def test_legacy_decoder_kind_cannot_be_misrepresented_as_certified_v1():
    contract = _contract()
    manifest = manifest_from_contract("artifact.onnx", contract)
    with pytest.raises(ValueError, match="legacy rtdetr_combined"):
        VisionManifest.model_validate(
            manifest.model_dump(mode="json") | {"decoder": manifest.decoder.model_dump() | {"kind": "rtdetr_combined"}}
        )
    with pytest.raises(ValueError, match="requires an embedded"):
        VisionManifest(
            source="artifact.onnx",
            preprocessing={"resize": "resize", "implementation": "opencv_linear_scale_fill"},
            output={"box_format": "cxcywh", "boxes_name": "output0"},
            decoder={"kind": "rtdetr_combined_v1", "boxes_normalized": True},
        )


def test_v1_manifest_rejects_a_semantics_only_but_nonexact_contract():
    manifest = manifest_from_contract("artifact.onnx", _contract())
    body = manifest.model_dump(mode="json")
    body["contract"]["raw_output"]["tensors"][0]["shape"] = [1, 299, 6]
    with pytest.raises(ValueError, match="exact valid VisionContract"):
        VisionManifest.model_validate(body)


def test_manifest_equivalence_rejects_resize_implementation_drift():
    contract = _contract()
    body = manifest_from_contract("artifact.onnx", contract).model_dump(mode="json")
    body["preprocessing"]["implementation"] = "pil"
    manifest = VisionManifest.model_validate(body)
    with pytest.raises(ValueError, match="resize implementation"):
        validate_contract_equivalence(manifest, contract)
