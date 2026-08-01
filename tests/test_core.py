import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from vision.backends import TensorRTBackend
from vision.capsule import create_capsule, verify_capsule
from vision.decoders import detr, yolo_end2end, yolo_raw
from vision.geometry import letterbox_params, nms, undo_letterbox, xywh_to_xyxy
from vision.manifest import VisionManifest
from vision.optimizer import Candidate, optimize_candidates
from vision.parity import compare, gate_map_drop
from vision.processing import letterbox
from vision.types import Box, DetectionResult


def result(boxes):
    return DetectionResult([Box(*b) for b in boxes], [0.9] * len(boxes), [1] * len(boxes), {1: "person"})


def test_manifest_roundtrip(tmp_path):
    m = VisionManifest(source="yolo26n.pt")
    p = tmp_path / "manifest.yaml"
    m.to_yaml(p)
    assert VisionManifest.from_yaml(p).model_dump() == m.model_dump()


def test_geometry_inverse_and_formats():
    b = Box(10, 20, 100, 200)
    s, px, py = letterbox_params((320, 240), (640, 640))
    q = Box(b.x1 * s + px, b.y1 * s + py, b.x2 * s + px, b.y2 * s + py)
    assert undo_letterbox(q, (320, 240), (640, 640)).x2 == pytest.approx(100)
    assert xywh_to_xyxy(1, 2, 3, 4) == Box(1, 2, 4, 6)
    assert nms([Box(0, 0, 10, 10), Box(1, 1, 10, 10)], [0.9, 0.8], [1, 1]) == [0]


def test_parity_lost_added_and_class_aware():
    a = result([(0, 0, 10, 10)])
    b = result([(30, 30, 40, 40)])
    report = compare(a, b)
    assert report.lost == [0] and report.added == [0] and not report.passed
    b = result([(0, 0, 10, 10)])
    b.classes = [2]
    assert compare(a, b).class_mismatches == [(0, 0)]


def test_capsule_fail_closed_and_tamper(tmp_path):
    m = VisionManifest(source="x.onnx")
    root = create_capsule(tmp_path / "cap", m, validation={"passed": False})
    with pytest.raises(ValueError, match="unvalidated"):
        verify_capsule(root)
    verify_capsule(root, allow_unvalidated=True)
    (root / "benchmarks.json").write_text("{} ")
    with pytest.raises(ValueError, match="checksum"):
        verify_capsule(root, True)


def test_optimizer_rejection_and_selection():
    # A dict saying ``validated=True`` is not auditable dataset evidence.
    # Typed SearchRequest/CandidateEvaluation coverage lives in
    # test_optimization.py; this legacy entry point remains fail-closed.
    with pytest.raises(RuntimeError, match="fail-closed"):
        optimize_candidates([Candidate("missing"), Candidate("fake")], lambda c: {"validated": True})


def test_import_has_no_vendor_runtime_imports():
    code = (
        "import sys; import vision; "
        "print(','.join(x for x in "
        "('torch','onnxruntime','ultralytics','openvino','tensorrt') if x in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == ""


@pytest.mark.parametrize("rel", ["../x", "/tmp/x", "a\\b", ""])
def test_capsule_rejects_unsafe_paths(tmp_path, rel):
    from vision.capsule import _safe

    with pytest.raises(ValueError):
        _safe(tmp_path, rel)


def test_capsule_rejects_added_file(tmp_path):
    root = create_capsule(tmp_path / "cap", VisionManifest(source="x.onnx"))
    (root / "injected.txt").write_text("bad")
    with pytest.raises(ValueError, match="inventory"):
        verify_capsule(root, True)


def test_capsule_rejects_missing_checksum(tmp_path):
    root = create_capsule(tmp_path / "cap", VisionManifest(source="x.onnx"))
    (root / "checksums.json").unlink()
    with pytest.raises(ValueError, match="checksums"):
        verify_capsule(root, True)


@pytest.mark.parametrize("name", ["x.onnx", "nested/model.onnx", "a.engine", "artifact.bin"])
def test_safe_relative_paths_accept_normal_paths(tmp_path, name):
    from vision.capsule import _safe

    assert _safe(tmp_path, name).parent.resolve().is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize("box,area", [(Box(0, 0, 1, 1), 1), (Box(1, 1, 0, 0), 0), (Box(0, 0, 2, 3), 6)])
def test_box_area(box, area):
    assert box.area() == area


@pytest.mark.parametrize(
    "boxes,expected", [([], []), ([Box(0, 0, 1, 1)], [0]), ([Box(0, 0, 1, 1), Box(4, 4, 5, 5)], [0, 1])]
)
def test_nms_basic_cases(boxes, expected):
    assert nms(boxes, [0.9] * len(boxes), [1] * len(boxes)) == expected


def test_letterbox_trace_inverse_non_square():
    arr, t = letterbox(Image.new("RGB", (320, 100)), (640, 640))
    assert arr[0, 0].tolist() == [114] * 3
    assert t.inverse(Box(0, 220, 640, 420)).as_xyxy() == pytest.approx([0, 0, 320, 100])


def test_yolo_end2end_decoder_non_square():
    _, t = letterbox(Image.new("RGB", (320, 100)), (640, 640))
    r = yolo_end2end(np.array([[[0, 220, 640, 420, 0.9, 2]]]), t)
    assert r.boxes[0].x2 == pytest.approx(320) and r.classes == [2]


def test_yolo_raw_transposed_and_class_nms():
    _, t = letterbox(Image.new("RGB", (100, 100)), (100, 100))
    a = np.array([[[50, 50], [50, 50], [20, 20], [20, 20], [0.9, 0.8], [0.1, 0.95]]])
    r = yolo_raw(a, t)
    assert len(r.boxes) == 2 and r.classes == [1, 0]


def test_detr_logits_decoder():
    r = detr([[[0.5, 0.5, 0.2, 0.4]]], logits=[[[0, 3, 0]]], image_size=(100, 100))
    assert r.classes == [1] and r.boxes[0].x1 == pytest.approx(40)


def test_forged_validation_passed_rejected(tmp_path):
    root = create_capsule(tmp_path / "cap", VisionManifest(source="x.onnx"), validation={"passed": True})
    with pytest.raises(ValueError, match="unvalidated development"):
        verify_capsule(root)


def test_trt_command_is_builder_only():
    cmd = TensorRTBackend.build_command("a.onnx", "b.engine", fp16=True, min_shape="images:1x3x32x32")
    assert cmd[0] == "trtexec" and "--fp16" in cmd
    with pytest.raises(ValueError):
        TensorRTBackend.build_command("a", "b", int8=True)


@pytest.mark.parametrize(
    "a,b,d,ok", [(1, 0.9, 0.2, True), (0.9, 0.5, 0.2, False), (1.1, 0.5, 0.2, False), (1, 0.9, -1, False)]
)
def test_map_gate_ranges(a, b, d, ok):
    if not ok and (a > 1 or d < 0):
        with pytest.raises(ValueError):
            gate_map_drop(a, b, d)
    else:
        assert gate_map_drop(a, b, d)["passed"] == ok
