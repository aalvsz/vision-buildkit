import json
import time
from types import SimpleNamespace

import pytest

from vision.capsule import (
    BenchmarkSnapshot,
    BuildSnapshot,
    CapsuleReleaseSnapshot,
    RuntimeRequest,
    ValidationSnapshot,
    create_development_capsule,
    recover_orphan_capsules,
    select_runtime,
    verify_capsule,
)
from vision.manifest import VisionManifest
from vision.optimization import BenchmarkReport, SearchReport, optimize_search


def test_development_capsule_is_integrity_checked_but_not_default_selectable(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    root = create_development_capsule(tmp_path / "model.vcap", VisionManifest(source=str(source)), artifacts=[source])
    with pytest.raises(ValueError, match="unvalidated"):
        verify_capsule(root)
    index = verify_capsule(root, allow_development=True)
    assert index.mode == "development"
    with pytest.raises(ValueError, match="unvalidated"):
        select_runtime(root, RuntimeRequest(backend="onnxruntime"))


@pytest.mark.parametrize("mutation", ("add", "delete", "symlink"))
def test_capsule_inventory_rejects_mutation(tmp_path, mutation):
    root = create_development_capsule(tmp_path / "model.vcap", VisionManifest(source="placeholder.onnx"))
    if mutation == "add":
        (root / "extra").write_text("x")
    elif mutation == "delete":
        (root / "benchmarks.json").unlink()
    else:
        (root / "artifacts" / "link").symlink_to(root / "benchmarks.json")
    with pytest.raises(ValueError):
        verify_capsule(root, allow_development=True)


def test_capsule_index_is_content_addressed_and_movable(tmp_path):
    manifest = VisionManifest(source="placeholder.onnx")
    left = create_development_capsule(tmp_path / "left.vcap", manifest)
    right = create_development_capsule(tmp_path / "right.vcap", manifest)
    assert (
        verify_capsule(left, allow_development=True).capsule_id
        == verify_capsule(right, allow_development=True).capsule_id
    )
    moved = tmp_path / "moved.vcap"
    left.rename(moved)
    assert verify_capsule(moved, allow_development=True).capsule_id


def test_release_snapshots_are_path_free_and_content_addressed():
    common = "a" * 64
    with pytest.raises(ValueError, match="path-free"):
        CapsuleReleaseSnapshot(
            selected_candidate_id=common,
            search_digest=common,
            dataset_digest=common,
            source_model_digest=common,
            plan={"artifact_path": "/private/model.onnx"},
        )
    release = CapsuleReleaseSnapshot(
        selected_candidate_id=common,
        search_digest=common,
        dataset_digest=common,
        source_model_digest=common,
        plan={"backend": "onnxruntime"},
    )
    validation = ValidationSnapshot(
        report_digest=common,
        policy_digest=common,
        dataset_digest=common,
        candidate_source_sha256=common,
        candidate_manifest_digest=common,
        reference_source_sha256=common,
        publishable=True,
        passed=True,
        evidence_kind="observed",
    )
    benchmark = BenchmarkSnapshot(
        report_digest=common,
        artifact_digest=common,
        fingerprint={"mode": "execution", "hardware_digest": common},
        resolution=(640, 640),
        batch=1,
        dynamic=False,
        profiles=(),
        samples_ms=(1.0,),
        repeats=1,
        p50_ms=1.0,
        p95_ms=1.0,
        mean_ms=1.0,
        throughput_items_per_s=1000.0,
        mode="execution",
    )
    build = BuildSnapshot(
        resolved_digest=common,
        build_digest=common,
        artifact_digest=common,
        manifest_digest=common,
        target_digest=common,
        candidate_id=common,
        backend="onnxruntime",
        precision="fp32",
        device="cpu",
        provider="CPUExecutionProvider",
        resolution=(640, 640),
        batch=1,
        dynamic=False,
        profiles=(),
    )
    assert all(item.digest for item in (release, validation, benchmark, build))
    with pytest.raises(ValueError, match="compile-only"):
        BuildSnapshot(**(build.model_dump() | {"build_only": True}))


def test_recovery_only_removes_matching_dead_owner_stage(tmp_path):
    output = "safe.vcap"
    token = "dead-token"
    lock = tmp_path / f".{output}.lock"
    lock.write_text(
        json.dumps(
            {
                "token": token,
                "pid": 999999999,
                "time": time.time() - 1000,
                "output": output,
                "stage": f".{output}.stage-owned",
            }
        )
    )
    owned = tmp_path / f".{output}.stage-owned"
    owned.mkdir()
    (owned / ".capsule-stage.json").write_text(json.dumps({"token": token, "output": output}))
    foreign = tmp_path / f".{output}.stage-foreign"
    foreign.mkdir()
    (foreign / ".capsule-stage.json").write_text(json.dumps({"token": "foreign", "output": output}))
    removed = recover_orphan_capsules(tmp_path, min_age=0)
    assert str(owned) in removed and not owned.exists()
    assert foreign.exists() and lock.exists()


def test_observed_execution_release_publishes_verifies_and_selects(tmp_path):
    # Reuse the public optimization fixture: it creates committed observed
    # validation evidence and resolved material, then make its runtime harness
    # explicitly execution-mode for this capsule boundary.
    from test_optimization import evaluate, request

    from vision.capsule import publish_capsule

    req, data = request(tmp_path)
    search = optimize_search(req, lambda item: evaluate(tmp_path, req, data, item, 1.0))
    values = []
    for item in search.evaluations:
        if item.benchmark is None:
            values.append(item)
            continue
        fingerprint = item.benchmark.request.fingerprint.model_copy(update={"mode": "execution"})
        benchmark = BenchmarkReport.model_validate(
            item.benchmark.model_dump(exclude={"digest"})
            | {"request": item.benchmark.request.model_dump() | {"fingerprint": fingerprint.model_dump()}}
        )
        values.append(item.model_copy(update={"benchmark": benchmark}))
    release = SearchReport.model_validate(search.model_dump(exclude={"digest"}) | {"evaluations": values})
    root = publish_capsule(tmp_path / "release.vcap", release)
    index = verify_capsule(root)
    selected = select_runtime(root, RuntimeRequest(expected_capsule_id=index.capsule_id))
    assert selected.selected is not None and selected.selected.executable
    with pytest.raises(ValueError, match="unexpected capsule id"):
        select_runtime(root, RuntimeRequest(expected_capsule_id="0" * 64))


@pytest.mark.parametrize(
    "relative", ("original_weights/capsule.json", "artifacts/parts/capsule.json", "artifacts/extra.bin")
)
def test_reindexed_unreferenced_release_material_is_rejected(tmp_path, relative):
    from test_optimization import evaluate, request

    from vision.capsule import CapsuleIndex, _digest, _inventory, _write_json, publish_capsule

    req, data = request(tmp_path)
    search = optimize_search(req, lambda item: evaluate(tmp_path, req, data, item, 1.0))
    values = []
    for item in search.evaluations:
        if item.benchmark is None:
            values.append(item)
            continue
        fp = item.benchmark.request.fingerprint.model_copy(update={"mode": "execution"})
        bench = BenchmarkReport.model_validate(
            item.benchmark.model_dump(exclude={"digest"})
            | {"request": item.benchmark.request.model_dump() | {"fingerprint": fp.model_dump()}}
        )
        values.append(item.model_copy(update={"benchmark": bench}))
    root = publish_capsule(
        tmp_path / "release.vcap",
        SearchReport.model_validate(search.model_dump(exclude={"digest"}) | {"evaluations": values}),
    )
    pinned = verify_capsule(root).capsule_id
    extra = root / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"attacker material")
    index = CapsuleIndex.model_validate_json((root / "capsule.json").read_text())
    files = _inventory(root)
    body = index.model_dump(exclude={"capsule_id"}) | {"files": files}
    forged = CapsuleIndex(capsule_id=_digest(body), **body)
    _write_json(root / "checksums.json", files)
    _write_json(root / "capsule.json", forged.model_dump(mode="json"))
    with pytest.raises(ValueError, match="unexpected capsule id|unreferenced|original weight"):
        verify_capsule(root, expected_capsule_id=pinned)


@pytest.mark.parametrize("attempt", range(2))
def test_publish_preserves_foreign_replacement_lock(tmp_path, attempt):
    from vision.capsule import _publish

    output = tmp_path / f"capsule-{attempt}.vcap"
    lock = tmp_path / f".{output.name}.lock"
    foreign = b'{"token":"foreign","owner":"attacker"}'

    def replace(_stage, _output, _capability):
        lock.unlink()
        lock.write_bytes(foreign)
        raise RuntimeError("stop after replacement")

    with pytest.raises(RuntimeError, match="stop"):
        _publish(output, replace)
    assert lock.read_bytes() == foreign


def test_portable_strips_nested_derived_digest_facts():
    from vision.capsule import _portable, _semantic_digest

    first = {"dataset": {"digest": "a" * 64, "images": [{"sha256": "b" * 64}]}, "nested": {"policy_digest": "c" * 64}}
    second = {"dataset": {"digest": "x" * 64, "images": [{"sha256": "b" * 64}]}, "nested": {"policy_digest": "y" * 64}}
    assert _portable(first) == _portable(second)
    assert _semantic_digest(first) == _semantic_digest(second)


def test_openvino_weights_keep_the_content_addressed_xml_stem():
    from vision.capsule import _part_logical_path

    artifact_digest = "a" * 64
    assert (
        _part_logical_path("model.xml", artifact_digest, "model.bin", 1, "b" * 64) == f"artifacts/{artifact_digest}.bin"
    )
    assert (
        _part_logical_path("model.onnx", artifact_digest, "calibration.bin", 1, "b" * 64)
        == "artifacts/parts/0001-bbbbbbbbbbbbbbbb-calibration.bin"
    )


def test_observed_runtime_evidence_closes_build_stage_only_status():
    from vision.capsule import _execution_closes_build_only

    artifact = "a" * 64
    validation = SimpleNamespace(
        evidence_kind="observed",
        publishable=True,
        passed=True,
        candidate=SimpleNamespace(runtime=SimpleNamespace(backend="tensorrt", source_sha256=artifact)),
    )
    benchmark = SimpleNamespace(
        request=SimpleNamespace(
            fingerprint=SimpleNamespace(mode="execution", backend="tensorrt", artifact_digest=artifact)
        )
    )
    resolved = SimpleNamespace(artifact_digest=artifact)
    record = SimpleNamespace(backend="tensorrt")
    assert _execution_closes_build_only(validation, benchmark, resolved, record)

    benchmark.request.fingerprint.artifact_digest = "b" * 64
    assert not _execution_closes_build_only(validation, benchmark, resolved, record)
