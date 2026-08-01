# ruff: noqa: E501
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from typer.testing import CliRunner

from vision.backends import TensorRTBackend
from vision.cli import app
from vision.export import BuildReport, BuildRequest, __all__, build
from vision.manifest import CalibrationManifest, VisionManifest


def tiny(path: Path, dynamic: bool = False) -> None:
    dimensions = ["batch", 1, 4, 4] if dynamic else [1, 1, 4, 4]
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, dimensions)
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, dimensions)
    weight = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    graph = helper.make_graph([helper.make_node("Conv", ["images", "w"], ["out"])], "tiny", [x], [y], [weight])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def calibration(tmp_path: Path, array: np.ndarray | None = None) -> tuple[Path, CalibrationManifest]:
    array = np.ones((1, 1, 4, 4), dtype=np.float32) if array is None else array
    sample = tmp_path / "sample.npy"
    np.save(sample, array)
    data = {
        "samples": [{"path": str(sample), "sha256": hashlib.sha256(sample.read_bytes()).hexdigest()}],
        "sample_count": 1,
        "preprocessing_digest": "a" * 64,
        "input_name": "images",
        "shape": [1, 1, 4, 4],
        "dtype": str(array.dtype),
    }
    data["digest"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data))
    return path, CalibrationManifest.model_validate(data)


def test_ort_fp32_real(tmp_path):
    source, output = tmp_path / "source.onnx", tmp_path / "output.onnx"
    tiny(source)
    assert build(BuildRequest(source=str(source), output=str(output))).effective_precision == "fp32"


def test_public_export_api_is_minimal():
    assert __all__ == ["BuildRequest", "BuildReport", "build"]
    assert BuildReport and BuildRequest and build


def test_checkpoint_export_is_isolated_from_existing_adjacent_onnx(tmp_path, monkeypatch):
    from vision.export.pipeline import normalize_source

    source = tmp_path / "model.pt"
    adjacent = tmp_path / "model.onnx"
    source.write_bytes(b"checkpoint")
    adjacent.write_bytes(b"keep-me")
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    class FakeAdapter:
        def __init__(self, isolated):
            self.isolated = Path(isolated)

        def export_onnx(self, output, **_kwargs):
            assert self.isolated.parent == build_dir
            generated = self.isolated.with_suffix(".onnx")
            tiny(generated)
            Path(output).write_bytes(generated.read_bytes())
            return output

    monkeypatch.setattr("vision.api.load", lambda isolated, **_kwargs: FakeAdapter(isolated))
    request = BuildRequest(
        source=str(source), output=str(tmp_path / "model.engine"), target="tensorrt", family="yolo26"
    )
    normalized = normalize_source(request, build_dir)
    assert normalized == build_dir / "source.onnx"
    assert adjacent.read_bytes() == b"keep-me"


def test_ort_fp16_real(tmp_path):
    source, output = tmp_path / "source.onnx", tmp_path / "half.onnx"
    tiny(source)
    assert build(BuildRequest(source=str(source), output=str(output), precision="fp16")).effective_precision == "fp16"


def test_ort_int8_qdq_real(tmp_path):
    source, output = tmp_path / "source.onnx", tmp_path / "quant.onnx"
    tiny(source)
    manifest, _ = calibration(tmp_path)
    report = build(
        BuildRequest(source=str(source), output=str(output), precision="int8", calibration_manifest_path=str(manifest))
    )
    assert report.effective_precision == "int8" and "QuantizeLinear" in report.model["nodes"]


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_openvino_cpu_real(tmp_path, precision):
    pytest.importorskip("openvino")
    if precision == "int8":
        pytest.importorskip("nncf")
    source, output = tmp_path / "source.onnx", tmp_path / f"{precision}.xml"
    tiny(source)
    manifest, _ = calibration(tmp_path)
    request = BuildRequest(
        source=str(source),
        output=str(output),
        target="openvino",
        precision=precision,
        calibration_manifest_path=str(manifest) if precision == "int8" else None,
        tolerances={"max_abs": 0.1, "mean_abs": 0.1, "max_rel": 0.2, "min_cosine": 0.99},
    )
    report = build(request)
    assert report.validation["passed"] and output.with_suffix(".bin").is_file()
    assert report.model["inputs"][0]["name"] == "images"
    if precision != "fp32":
        assert report.model["precision_inspection"]


def test_rfdetr_export_smoke_compares_decoded_sets_not_query_order():
    from vision.export.pipeline import Tolerances, _compare_manifest_detections

    manifest = VisionManifest(
        source="model.onnx",
        family="onnx",
        input={"shape": [1, 3, 560, 560]},
        preprocessing={
            "resize": "resize",
            "implementation": "torchvision_bilinear_antialias_false",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        output={"box_format": "cxcywh", "boxes_name": "dets", "logits_name": "labels"},
        decoder={"kind": "rfdetr", "boxes_normalized": True},
    )
    boxes = np.asarray([[[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]], dtype=np.float32)
    logits = np.asarray([[[8.0, -20.0], [-20.0, 7.0]]], dtype=np.float32)
    order = [1, 0]
    result = _compare_manifest_detections(
        manifest,
        ["dets", "labels"],
        [boxes, logits],
        [boxes[:, order], logits[:, order]],
        Tolerances(),
    )
    assert result["passed"] and result["comparison"] == "manifest_decoded_detection_set"


def test_calibration_digest_rejected(tmp_path):
    _, manifest = calibration(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        CalibrationManifest.model_validate(manifest.model_dump() | {"digest": "0" * 64})


def test_calibration_count_rejected(tmp_path):
    _, manifest = calibration(tmp_path)
    data = manifest.model_dump()
    data["sample_count"] = 2
    with pytest.raises(ValueError, match="sample_count"):
        CalibrationManifest.model_validate(data)


def test_calibration_bad_dtype_rejected(tmp_path):
    source, output = tmp_path / "source.onnx", tmp_path / "out.onnx"
    tiny(source)
    manifest, _ = calibration(tmp_path, np.ones((1, 1, 4, 4), dtype=np.float16))
    with pytest.raises(ValueError, match="dtype"):
        build(
            BuildRequest(
                source=str(source), output=str(output), precision="int8", calibration_manifest_path=str(manifest)
            )
        )


def test_calibration_nan_rejected(tmp_path):
    source, output = tmp_path / "source.onnx", tmp_path / "out.onnx"
    tiny(source)
    manifest, _ = calibration(tmp_path, np.full((1, 1, 4, 4), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        build(
            BuildRequest(
                source=str(source), output=str(output), precision="int8", calibration_manifest_path=str(manifest)
            )
        )


@pytest.mark.parametrize(
    "field, value, message", [("sha256", "0" * 64, "digest mismatch"), ("shape", [1, 1, 3, 3], "shape mismatch")]
)
def test_calibration_hash_and_shape_rejected(tmp_path, field, value, message):
    source, output = tmp_path / "source.onnx", tmp_path / "out.onnx"
    tiny(source)
    manifest, _ = calibration(tmp_path)
    data = json.loads(manifest.read_text())
    if field == "sha256":
        data["samples"][0][field] = value
    else:
        data[field] = value
    canonical = {key: item for key, item in data.items() if key != "digest"}
    data["digest"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        build(
            BuildRequest(
                source=str(source), output=str(output), precision="int8", calibration_manifest_path=str(manifest)
            )
        )


def test_source_and_stale_output_rejected(tmp_path):
    source = tmp_path / "source.onnx"
    tiny(source)
    with pytest.raises(ValueError, match="differ"):
        BuildRequest(source=str(source), output=str(source))
    output = tmp_path / "out.onnx"
    output.write_bytes(b"stale")
    with pytest.raises(ValueError, match="stale"):
        BuildRequest(source=str(source), output=str(output))


def test_symlink_rejected(tmp_path):
    source, link = tmp_path / "source.onnx", tmp_path / "link.onnx"
    tiny(source)
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        BuildRequest(source=str(link), output=str(tmp_path / "out.onnx"))


def test_dynamic_mismatch_rejected(tmp_path):
    source = tmp_path / "source.onnx"
    tiny(source)
    with pytest.raises(ValueError, match="dynamic"):
        build(BuildRequest(source=str(source), output=str(tmp_path / "out.onnx"), dynamic=True))


def test_dynamic_symbolic_accepted(tmp_path):
    source = tmp_path / "source.onnx"
    tiny(source, dynamic=True)
    report = build(BuildRequest(source=str(source), output=str(tmp_path / "out.onnx"), dynamic=True))
    assert report.dynamic


def test_openvino_dynamic_cpu_real(tmp_path):
    pytest.importorskip("openvino")
    source, output = tmp_path / "source.onnx", tmp_path / "dynamic.xml"
    tiny(source, dynamic=True)
    report = build(BuildRequest(source=str(source), output=str(output), target="openvino", dynamic=True))
    assert report.dynamic and report.validation["passed"]


def test_openvino_grid_sample_linear_compatibility_rewrite_is_explicit(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from vision.export.openvino import _compatibility_source

    source = tmp_path / "grid.onnx"
    node = helper.make_node("GridSample", ["input", "grid"], ["output"], mode="linear")
    graph = helper.make_graph(
        [node],
        "grid",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 1, 1, 2]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), source)
    rewritten, changes = _compatibility_source(source, tmp_path)
    assert rewritten != source
    assert changes == [
        {
            "node_index": 0,
            "op_type": "GridSample",
            "attribute": "mode",
            "from": "linear",
            "to": "bilinear",
        }
    ]
    assert (
        next(attribute.s for attribute in onnx.load(str(rewritten)).graph.node[0].attribute if attribute.name == "mode")
        == b"bilinear"
    )
    assert (
        next(attribute.s for attribute in onnx.load(str(source)).graph.node[0].attribute if attribute.name == "mode")
        == b"linear"
    )


def test_openvino_fp32_rejects_quantized_source(tmp_path):
    source, quantized, output = tmp_path / "source.onnx", tmp_path / "quantized.onnx", tmp_path / "out.xml"
    tiny(source)
    manifest, _ = calibration(tmp_path)
    build(
        BuildRequest(
            source=str(source), output=str(quantized), precision="int8", calibration_manifest_path=str(manifest)
        )
    )
    with pytest.raises(RuntimeError, match="inspection"):
        build(BuildRequest(source=str(quantized), output=str(output), target="openvino", precision="fp32"))


def test_profile_order_rejected(tmp_path):
    source = tmp_path / "source.onnx"
    tiny(source, dynamic=True)
    with pytest.raises(ValueError, match="min <="):
        BuildRequest(
            source=str(source),
            output=str(tmp_path / "out.onnx"),
            dynamic=True,
            profiles=[{"input_name": "images", "min_shape": [2], "opt_shape": [1], "max_shape": [3]}],
        )


def test_profile_static_dimensions_checked_before_trt(tmp_path):
    source = tmp_path / "source.onnx"
    tiny(source, dynamic=True)
    request = BuildRequest(
        source=str(source),
        output=str(tmp_path / "out.engine"),
        target="tensorrt",
        dynamic=True,
        profiles=[
            {"input_name": "images", "min_shape": [1, 1, 2, 2], "opt_shape": [1, 1, 4, 4], "max_shape": [1, 1, 8, 8]}
        ],
    )
    with pytest.raises(ValueError, match="min profile dimensions"):
        build(request)


def test_atomic_cleanup_on_failure(tmp_path):
    source = tmp_path / "source.onnx"
    tiny(source)
    with pytest.raises(ValueError):
        build(BuildRequest(source=str(source), output=str(tmp_path / "out.onnx"), precision="int8"))
    assert not list(tmp_path.glob(".out.build-*"))


def test_report_commit_failure_rolls_back_artifact(tmp_path, monkeypatch):
    source, output = tmp_path / "source.onnx", tmp_path / "out.onnx"
    tiny(source)
    import vision.export.pipeline as pipeline

    original = pipeline._publish_no_clobber

    def fail_report(src, dst):
        if str(dst).endswith(".build-report.json"):
            raise OSError("report commit failed")
        return original(src, dst)

    monkeypatch.setattr(pipeline, "_publish_no_clobber", fail_report)
    with pytest.raises(OSError, match="report commit"):
        build(BuildRequest(source=str(source), output=str(output)))
    assert not output.exists() and not Path(f"{output}.build-report.json").exists()


def test_concurrent_output_creation_refused(tmp_path, monkeypatch):
    source, output = tmp_path / "source.onnx", tmp_path / "out.onnx"
    tiny(source)
    import vision.export.pipeline as pipeline

    original = pipeline._publish_no_clobber
    calls = 0

    def race(staged, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            output.write_bytes(b"race")
        return original(staged, destination)

    monkeypatch.setattr(pipeline, "_publish_no_clobber", race)
    with pytest.raises(RuntimeError, match="concurrently"):
        build(BuildRequest(source=str(source), output=str(output)))
    assert output.read_bytes() == b"race"


def test_cli_convert_and_help(tmp_path):
    source = tmp_path / "source.onnx"
    tiny(source)
    runner = CliRunner()
    assert runner.invoke(app, ["convert", "--help"]).exit_code == 0
    result = runner.invoke(app, ["convert", str(source), str(tmp_path / "out.onnx")])
    assert result.exit_code == 0 and '"effective_precision": "fp32"' in result.stdout


def test_cli_trt_options_construct_request(tmp_path, monkeypatch):
    source, cache = tmp_path / "source.onnx", tmp_path / "cache"
    tiny(source)
    cache.write_bytes(b"cache")
    monkeypatch.setattr(
        "vision.cli.build",
        lambda request: type(
            "R", (), {"model_dump": lambda self, **_: {"target": request.target, "precision": request.precision}}
        )(),
    )
    result = CliRunner().invoke(
        app,
        [
            "convert",
            str(source),
            str(tmp_path / "out.engine"),
            "--target",
            "tensorrt",
            "--precision",
            "int8",
            "--calibration-cache-path",
            str(cache),
            "--dynamic",
            "--input-name",
            "images",
            "--min-shape",
            "1x1x4x4",
            "--opt-shape",
            "1x1x4x4",
            "--max-shape",
            "2x1x4x4",
            "--trt-timeout-s",
            "12",
        ],
    )
    assert result.exit_code == 0 and '"tensorrt"' in result.stdout


def test_trt_mock_success_and_redaction(tmp_path, monkeypatch):
    source, output = tmp_path / "source.onnx", tmp_path / "engine"
    tiny(source)
    cache = tmp_path / "cache"
    cache.write_bytes(b"cache")
    monkeypatch.setattr(
        TensorRTBackend, "capabilities", classmethod(lambda cls: type("C", (), {"available": True, "reason": None})())
    )

    def fake_run(command, **kwargs):
        if "--version" in command:
            return type("R", (), {"stdout": "trtexec 1"})()
        Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--saveEngine="))).write_bytes(b"engine")
        return type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("vision.backends.subprocess.run", fake_run)
    request = BuildRequest(
        source=str(source),
        output=str(tmp_path / "final.engine"),
        target="tensorrt",
        precision="int8",
        calibration_cache_path=str(cache),
    )
    _, command, _, fingerprint = TensorRTBackend.build_atomic(source, output, request)
    assert any("redacted" in arg for arg in command) and fingerprint["calibration_cache_sha256"]


def test_trt_mock_failure_cleans_partial(tmp_path, monkeypatch):
    source, output = tmp_path / "source.onnx", tmp_path / "engine"
    tiny(source)
    monkeypatch.setattr(
        TensorRTBackend, "capabilities", classmethod(lambda cls: type("C", (), {"available": True, "reason": None})())
    )
    monkeypatch.setattr(
        "vision.backends.subprocess.run",
        lambda *args, **kwargs: type("R", (), {"returncode": 1, "stdout": "", "stderr": "bad"})(),
    )
    request = BuildRequest(source=str(source), output=str(tmp_path / "final.engine"), target="tensorrt")
    with pytest.raises(RuntimeError, match="trtexec failed"):
        TensorRTBackend.build_atomic(source, output, request)
    assert not output.exists()


def test_public_build_trt_mock_commits_engine_and_report(tmp_path, monkeypatch):
    source, output = tmp_path / "source.onnx", tmp_path / "final.engine"
    tiny(source)
    cache = tmp_path / "cache"
    cache.write_bytes(b"cache")
    import vision.backends as backends

    def fake_build(cls, src, staged, request):
        staged.write_bytes(b"engine")
        return (
            staged,
            ["trtexec", "--calib=<redacted-path>"],
            "log",
            {"calibration_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()},
        )

    monkeypatch.setattr(backends.TensorRTBackend, "build_atomic", classmethod(fake_build))
    report = build(
        BuildRequest(
            source=str(source),
            output=str(output),
            target="tensorrt",
            precision="int8",
            calibration_cache_path=str(cache),
        )
    )
    assert output.is_file() and Path(f"{output}.build-report.json").is_file()
    assert report.command[-1] == "--calib=<redacted-path>" and report.target_fingerprint["calibration_cache_sha256"]


def test_recovery_removes_only_journal_matching_orphan(tmp_path):
    import vision.export.pipeline as pipeline

    orphan, foreign, report = (
        tmp_path / "orphan.onnx",
        tmp_path / "foreign.onnx",
        tmp_path / "orphan.onnx.build-report.json",
    )
    orphan.write_bytes(b"owned")
    foreign.write_bytes(b"foreign")
    journal = tmp_path / ".vision-build-pending-test.json"
    journal.write_text(
        json.dumps(
            {
                "report": str(report),
                "artifacts": {str(orphan): hashlib.sha256(b"owned").hexdigest(), str(foreign): "0" * 64},
            }
        )
    )
    pipeline.recover_orphan_builds(tmp_path)
    assert not orphan.exists() and foreign.read_bytes() == b"foreign" and not journal.exists()


def test_subprocess_crash_orphan_is_recovered(tmp_path):
    orphan, report, journal = (
        tmp_path / "crash.onnx",
        tmp_path / "crash.onnx.build-report.json",
        tmp_path / ".vision-build-pending-crash.json",
    )
    code = "import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_bytes(b'owned'); Path(sys.argv[2]).write_text(json.dumps({'report':sys.argv[3],'artifacts':{str(p):hashlib.sha256(b'owned').hexdigest()}}))"
    subprocess.run([sys.executable, "-c", code, str(orphan), str(journal), str(report)], check=True)
    from vision.export.pipeline import recover_orphan_builds

    recover_orphan_builds(tmp_path)
    assert not orphan.exists() and not journal.exists() and not report.exists()
