"""Resolved optimizer material is re-opened from exact bytes and paths."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from vision.export import BuildReport
from vision.manifest import VisionManifest
from vision.optimization import (
    CandidatePlan,
    ResolvedArtifactEvidence,
    SearchReport,
    digest,
    expected_build_request_digest,
    load_search_report,
    verify_resolved_material,
)
from vision.validation import ModelTarget


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(root: Path, name: str, family: str = "torch") -> ModelTarget:
    source = root / f"{name}{'.onnx' if family == 'onnx' else '.pt'}"
    source.write_bytes(name.encode())
    manifest_path = root / f"{name}.yaml"
    manifest = VisionManifest(source=str(source), family=family)
    manifest.to_yaml(manifest_path)
    return ModelTarget(
        source=str(source),
        source_sha256=_sha(source),
        manifest_path=str(manifest_path),
        manifest_sha256=_sha(manifest_path),
        manifest_digest=digest(manifest.model_dump(mode="json")),
        device="cpu",
        class_mapping={"0": 1},
    )


def _plan(source: str, *, backend: str = "pytorch") -> CandidatePlan:
    return CandidatePlan(
        backend=backend,
        precision="fp32",
        device="cpu",
        provider="torch" if backend == "pytorch" else "CPUExecutionProvider",
        resolution=(640, 640),
        batch=1,
        source_model_digest=source,
    )


def _resolved(root: Path, plan: CandidatePlan, target: ModelTarget, *, exported: bool) -> ResolvedArtifactEvidence:
    upstream_path = None
    if exported:
        upstream_path = root / "upstream.json"
        upstream = BuildReport(
            source=target.source,
            intermediate=None,
            output=target.source,
            target=plan.backend,
            requested_precision=plan.precision,
            effective_precision=plan.precision,
            source_digest=plan.source_model_digest,
            intermediate_digest=plan.source_model_digest,
            output_digest=target.source_sha256,
            artifact_digests={Path(target.source).name: target.source_sha256},
            dynamic=False,
        ).finalized()
        upstream_path.write_text(json.dumps(upstream.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
    raw = {
        "schema_version": 1,
        "kind": "exported" if exported else "original",
        "candidate_id": plan.candidate_id,
        "plan_digest": plan.candidate_id,
        "source_model_digest": plan.source_model_digest,
        "build_template_digest": "b" * 64,
        "build_request_digest": expected_build_request_digest(plan, "b" * 64),
        "artifact_path": target.source,
        "artifact_digest": target.source_sha256,
        "manifest_path": target.manifest_path,
        "manifest_digest": target.manifest_digest,
        "target_digest": digest(target.model_dump(mode="json")),
        "backend": plan.backend,
        "precision": plan.precision,
        "device": plan.device,
        "provider": plan.provider,
        "resolution": plan.resolution,
        "batch": plan.batch,
        "dynamic": False,
        "profiles": (),
        "upstream_build_report_path": str(upstream_path) if upstream_path else None,
        "upstream_build_report_file_sha256": _sha(upstream_path) if upstream_path else None,
        "upstream_build_report_self_digest": (
            json.loads(upstream_path.read_text())["self_digest"] if upstream_path else None
        ),
    }
    record = raw | {"self_digest": digest(raw)}
    record_path = root / "resolved.json"
    record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return ResolvedArtifactEvidence(
        candidate_id=plan.candidate_id,
        artifact_digest=target.source_sha256,
        build_request_digest=raw["build_request_digest"],
        manifest_digest=target.manifest_digest,
        target_digest=raw["target_digest"],
        build_report_digest=record["self_digest"],
        artifact_path=target.source,
        artifact_sha256=_sha(Path(target.source)),
        manifest_path=target.manifest_path,
        manifest_sha256=_sha(Path(target.manifest_path)),
        build_report_path=str(record_path),
        build_report_sha256=_sha(record_path),
    )


def test_original_target_binding_rejects_decoy_paths_and_missing_bytes(tmp_path):
    target = _target(tmp_path, "reference")
    plan = _plan(target.source_sha256)
    resolved = _resolved(tmp_path, plan, target, exported=False)
    assert verify_resolved_material(resolved, plan, target).kind == "original"

    decoy = _target(tmp_path, "decoy")
    with pytest.raises(ValueError, match="exact target source"):
        verify_resolved_material(resolved, plan, decoy)
    Path(target.source).unlink()
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        verify_resolved_material(resolved, plan, target)


def test_exported_record_reopens_build_and_all_reported_artifacts(tmp_path):
    target = _target(tmp_path, "candidate", "onnx")
    plan = _plan(target.source_sha256, backend="onnxruntime")
    resolved = _resolved(tmp_path, plan, target, exported=True)
    assert verify_resolved_material(resolved, plan, target).kind == "exported"

    record = json.loads(Path(resolved.build_report_path).read_text())
    upstream = Path(record["upstream_build_report_path"])
    payload = json.loads(upstream.read_text())
    payload["requested_precision"] = "fp16"
    upstream.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        verify_resolved_material(resolved, plan, target)


def test_ancestor_symlink_is_never_trusted(tmp_path):
    target = _target(tmp_path, "reference")
    plan = _plan(target.source_sha256)
    resolved = _resolved(tmp_path, plan, target, exported=False)
    link_root = tmp_path / "linked"
    link_root.symlink_to(tmp_path, target_is_directory=True)
    forged = resolved.model_copy(update={"build_report_path": str(link_root / "resolved.json")})
    with pytest.raises(ValueError, match="symlink"):
        verify_resolved_material(forged, plan, target)


def _persisted_search(root: Path):
    """Use the public signed-evidence test fixture without bypassing its evaluator."""
    root.mkdir()
    helpers = runpy.run_path(str(Path(__file__).with_name("test_optimization.py")))
    request, data = helpers["request"](root)
    report = helpers["optimize_search"](
        request,
        lambda item: helpers["evaluate"](root, request, data, item, 2.0 if item.backend == "pytorch" else 1.0),
    )
    return request, report, Path(request.output) / "optimization_report.json"


def _self_digest(record: dict) -> None:
    record["self_digest"] = digest({key: value for key, value in record.items() if key != "self_digest"})


def _coordinated_decoy(raw: dict, entry: dict) -> None:
    """Rewrite each nested signature so only the final acceptance is invalid."""
    resolved = entry["resolved"]
    artifact = Path(resolved["artifact_path"])
    artifact.write_bytes(b"coordinated-decoy")
    artifact_sha = _sha(artifact)

    build_path = Path(resolved["build_report_path"])
    build = json.loads(build_path.read_text())
    upstream_path = Path(build["upstream_build_report_path"])
    upstream = json.loads(upstream_path.read_text())
    upstream["output_digest"] = artifact_sha
    upstream["artifact_digests"][artifact.name] = artifact_sha
    _self_digest(upstream)
    upstream_path.write_text(json.dumps(upstream, sort_keys=True, separators=(",", ":")))
    build["artifact_digest"] = artifact_sha
    build["upstream_build_report_file_sha256"] = _sha(upstream_path)
    build["upstream_build_report_self_digest"] = upstream["self_digest"]
    _self_digest(build)
    build_path.write_text(json.dumps(build, sort_keys=True, separators=(",", ":")))

    resolved["artifact_digest"] = artifact_sha
    resolved["artifact_sha256"] = artifact_sha
    resolved["build_report_digest"] = build["self_digest"]
    resolved["build_report_sha256"] = _sha(build_path)
    assert _sha(artifact) == resolved["artifact_sha256"]
    assert _sha(build_path) == resolved["build_report_sha256"]

    evidence = entry["validation"]
    validation = evidence["report"]
    candidate = validation["candidate"]
    candidate["target"]["source_sha256"] = artifact_sha
    target_digest = digest(candidate["target"])
    resolved["target_digest"] = target_digest
    build["target_digest"] = target_digest
    _self_digest(build)
    build_path.write_text(json.dumps(build, sort_keys=True, separators=(",", ":")))
    resolved["build_report_digest"] = build["self_digest"]
    resolved["build_report_sha256"] = _sha(build_path)
    candidate["runtime"]["source_sha256"] = artifact_sha
    candidate["runtime"]["digest"] = digest(
        {key: value for key, value in candidate["runtime"].items() if key != "digest"}
    )
    validation["policy"]["candidate_runtime"] = candidate["runtime"]
    validation["policy_digest"] = digest(validation["policy"])
    manifest = VisionManifest.from_yaml(candidate["target"]["manifest_path"])
    candidate["input_preprocessing_digest"] = digest(
        {
            "source_sha256": artifact_sha,
            "manifest_digest": candidate["target"]["manifest_digest"],
            "images": [{"id": image["id"], "sha256": image["sha256"]} for image in validation["dataset"]["images"]],
            "preprocessing": manifest.preprocessing.model_dump(mode="json"),
        }
    )
    candidate["digest"] = digest({key: value for key, value in candidate.items() if key != "digest"})

    result_root = Path(evidence["root"]) / "result"
    record_path = result_root / candidate["record_path"]
    record_path.write_text(json.dumps(candidate, sort_keys=True, separators=(",", ":")))
    validation["digest"] = digest({key: value for key, value in validation.items() if key != "digest"})
    validation_bytes = json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()
    (result_root / "validation_report.json").write_bytes(validation_bytes)
    marker = hashlib.sha256(validation_bytes).hexdigest()
    (result_root / "COMMITTED").write_text(marker)
    (Path(evidence["root"]) / "COMMITTED").write_text(marker)

    evidence["resolved"] = resolved
    evidence["expected_policy_digest"] = validation["policy_digest"]
    evidence["binding_digest"] = digest(
        {
            "report_digest": validation["digest"],
            "resolved": resolved,
            "policy_digest": evidence["expected_policy_digest"],
        }
    )
    benchmark = entry["benchmark"]
    benchmark["request"]["resolved"] = resolved
    benchmark["request"]["fingerprint"]["artifact_digest"] = artifact_sha
    # This is a valid benchmark object, but deliberately incomparable with the
    # signed baseline.  It is why an accepted terminal status must be rejected.
    benchmark["request"]["fingerprint"]["workload_digest"] = "c" * 64
    benchmark["digest"] = digest({key: value for key, value in benchmark.items() if key != "digest"})


@pytest.mark.parametrize("selected_none", (False, True))
def test_load_rejects_coordinated_decoy_nonwinner_terminal_acceptance(tmp_path, selected_none):
    request, report, report_path = _persisted_search(tmp_path / ("none" if selected_none else "winner"))
    # Rebase the otherwise-valid control report onto the signed baseline.  The
    # prior winner becomes the deliberately rejected nonwinner we will forge.
    nonwinner = next(item for item in report.evaluations if item.plan.candidate_id == report.selected_candidate_id)
    raw = json.loads(report_path.read_text())
    entry = next(item for item in raw["evaluations"] if digest(item["plan"]) == nonwinner.plan.candidate_id)
    _coordinated_decoy(raw, entry)
    rejected = {"stage": "rejected", "payload": {"reason": "pre-mutation control"}}
    rejected["digest"] = digest(rejected)
    entry["stages"][-1] = rejected
    raw["selected_candidate_id"] = request.baseline.candidate_id
    raw["digest"] = digest({key: value for key, value in raw.items() if key != "digest"})
    report_path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")))
    # Control: every embedded Pydantic object, committed validation result, and
    # source artifact now verifies before the forged terminal state is applied.
    control = SearchReport.model_validate(raw)
    control_nonwinner = next(
        item for item in control.evaluations if item.plan.candidate_id == nonwinner.plan.candidate_id
    )
    assert control_nonwinner.validation.verify(control.request, control_nonwinner.plan)
    accepted = {"stage": "accepted", "payload": {"reason": "forged"}}
    accepted["digest"] = digest(accepted)
    entry["stages"][-1] = accepted
    if selected_none:
        raw["selected_candidate_id"] = None
    raw["digest"] = digest({key: value for key, value in raw.items() if key != "digest"})
    report_path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="terminal accepted candidate is invalid"):
        load_search_report(report_path)
