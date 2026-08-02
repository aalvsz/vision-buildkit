"""Adversarial coverage for the frozen RT-DETR combined decoder."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from vision.decoders import rtdetr_combined, rtdetr_combined_v1
from vision.processing import GeometryTrace


def _tensor() -> np.ndarray:
    value = np.zeros((1, 300, 6), dtype=np.float32)
    value[0, :, 4] = 0.1
    value[0, 0] = [0.5, 0.5, 0.5, 0.5, 0.9, 2]
    value[0, 1] = [0.5, 0.5, 0.5, 0.5, 0.9, 1]
    value[0, 2] = [0.25, 0.5, 0.25, 0.5, 0.8, 0]
    return value


def _trace() -> GeometryTrace:
    return GeometryTrace(original=(320, 160), target=(640, 640), scale_x=2.0, scale_y=4.0)


def test_exact_decode_orders_deterministically_scales_and_keeps_overlaps():
    result = rtdetr_combined_v1(_tensor(), _trace(), 0.4, class_count=3)
    assert result.classes == [1, 2, 0]
    assert result.scores == pytest.approx([0.9, 0.9, 0.8])
    assert [list(box.as_xyxy()) for box in result.boxes] == [
        [80.0, 40.0, 240.0, 120.0],
        [80.0, 40.0, 240.0, 120.0],
        [40.0, 40.0, 120.0, 120.0],
    ]


def test_frozen_v1_excludes_score_exactly_at_threshold():
    tensor = _tensor()
    tensor[0, 3] = [0.5, 0.5, 0.5, 0.5, 0.4, 0]

    result = rtdetr_combined_v1(tensor, _trace(), 0.4, class_count=3)

    assert result.classes == [1, 2, 0]
    assert result.scores == pytest.approx([0.9, 0.9, 0.8])


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((300, 6), dtype=np.float32),
        np.zeros((1, 299, 6), dtype=np.float32),
        np.zeros((1, 300, 7), dtype=np.float32),
        np.zeros((1, 300, 6), dtype=np.float64),
        [[0.0] * 6] * 300,
    ],
)
def test_rejects_nonfrozen_shape_dtype_and_container(value):
    with pytest.raises(ValueError, match=r"exact np.float32 \[1,300,6\]"):
        rtdetr_combined_v1(value, _trace())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (0, np.nan, "finite"),
        (1, np.inf, "finite"),
        (2, -0.01, r"\[0,1\]"),
        (3, 1.01, r"\[0,1\]"),
        (4, 1.01, r"\[0,1\]"),
        (5, 1.5, "integral"),
        (5, -1.0, "integral"),
        (5, 3.0, "integral"),
    ],
)
def test_rejects_nonfinite_ranges_and_classes(column, value, message):
    tensor = _tensor()
    tensor[0, 0, column] = value
    with pytest.raises(ValueError, match=message):
        rtdetr_combined_v1(tensor, _trace(), class_count=3)


@pytest.mark.parametrize("class_count", [0, -1, 1.5, True])
def test_rejects_invalid_class_domain(class_count):
    with pytest.raises(ValueError, match="positive integer"):
        rtdetr_combined_v1(_tensor(), _trace(), class_count=class_count)


@pytest.mark.parametrize("max_detections", [0, -1, 1.5, True])
def test_rejects_invalid_detection_limit(max_detections):
    with pytest.raises(ValueError, match="max_detections must be a positive integer"):
        rtdetr_combined_v1(_tensor(), _trace(), class_count=3, max_detections=max_detections)


@pytest.mark.parametrize("labels", [["only"], ["a", "b", ""], {0: "a", 1: "b", 2: "c"}])
def test_rejects_nonexact_label_inventory(labels):
    with pytest.raises(ValueError, match="labels must exactly cover"):
        rtdetr_combined_v1(_tensor(), _trace(), class_count=3, labels=labels)


def test_max_detection_limit_and_contract_labels_are_applied_without_nms():
    result = rtdetr_combined_v1(
        _tensor(),
        _trace(),
        0.4,
        class_count=3,
        labels=["zero", "one", "two"],
        max_detections=2,
    )
    assert result.classes == [1, 2]
    assert result.names == {0: "zero", 1: "one", 2: "two"}
    assert result.boxes[0] == result.boxes[1]


def test_legacy_decoder_preserves_origin_main_rank_dtype_threshold_and_names_behavior():
    signature = inspect.signature(rtdetr_combined)
    assert list(signature.parameters) == ["a", "trace", "threshold"]
    assert signature.parameters["threshold"].default == 0.25
    value = np.array(
        [
            [0.5, 0.5, 0.5, 0.5, 0.4, 2, 999],
            [0.25, 0.5, 0.25, 0.5, 0.4000000001, 1, 999],
        ],
        dtype=np.float64,
    )
    result = rtdetr_combined(value, _trace(), 0.4)
    assert result.classes == [1]
    assert result.scores == [pytest.approx(0.4000000001)]
    assert result.names == {}
    assert result.boxes[0].as_xyxy() == pytest.approx([40, 40, 120, 120])

    rank3 = rtdetr_combined(value[None, ...], _trace(), 0.4)
    assert rank3.to_dict() == result.to_dict()
    from_list = rtdetr_combined(value.tolist(), _trace(), 0.4)
    assert from_list.to_dict() == result.to_dict()


@pytest.mark.parametrize("value", [np.zeros(6), np.zeros((2, 5)), np.zeros((1, 2, 5))])
def test_legacy_decoder_preserves_origin_main_rejection_boundary(value):
    with pytest.raises(ValueError, match=r"must be \[N,6\]"):
        rtdetr_combined(value, _trace())
