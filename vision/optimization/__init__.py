"""Auditable, fail-closed optimization search.

This module intentionally does not run an exporter or benchmark by itself.  A
caller supplies an evaluator which produces evidence for each stage.  The
search engine owns validation, comparability, filtering, deterministic ranking
and the durable report; it never turns a boolean such as ``validated=True``
into a quality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..manifest import VisionManifest
from ..validation import (
    ModelTarget,
    RuntimeExecutionIdentity,
    ValidationPolicy,
    ValidationReport,
    _policy_digest,
    runtime_policy_contract,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _trusted_regular(path: str | Path) -> Path:
    item = Path(path)
    if not item.is_absolute() or item.is_symlink() or not item.is_file():
        raise ValueError("material path must be an absolute regular non-symlink file")
    for parent in item.parents:
        if parent.is_symlink():
            raise ValueError("material path has a symlink ancestor")
    return item


def _hex(value: str) -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("digest must be a lowercase SHA-256 hex value")
    return value


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ShapeProfile(Strict):
    input_name: str = Field(min_length=1)
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]

    @model_validator(mode="after")
    def _valid(self):
        if not (len(self.min_shape) == len(self.opt_shape) == len(self.max_shape) >= 1):
            raise ValueError("shape profiles require equally ranked min/opt/max shapes")
        if any(x <= 0 for group in (self.min_shape, self.opt_shape, self.max_shape) for x in group):
            raise ValueError("shape profile dimensions must be positive")
        if any(not lo <= mid <= hi for lo, mid, hi in zip(self.min_shape, self.opt_shape, self.max_shape, strict=True)):
            raise ValueError("shape profile requires min <= opt <= max")
        return self


class CandidatePlan(Strict):
    """Pre-build candidate choice; deliberately contains no generated artifact."""

    backend: str = Field(min_length=1)
    precision: str = Field(pattern="^(fp32|fp16|int8)$")
    device: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    resolution: tuple[int, int]
    batch: int = Field(gt=0)
    dynamic: bool = False
    profiles: tuple[ShapeProfile, ...] = ()
    source_model_digest: str

    @model_validator(mode="after")
    def _valid(self):
        if any(value <= 0 for value in self.resolution):
            raise ValueError("plan resolution dimensions must be positive")
        _hex(self.source_model_digest)
        if self.dynamic != bool(self.profiles):
            raise ValueError("dynamic plan/profile mismatch")
        for profile in self.profiles:
            if (
                len(profile.opt_shape) != 4
                or profile.opt_shape[0] != self.batch
                or profile.opt_shape[-2:] != self.resolution
            ):
                raise ValueError("profile opt shape must bind plan batch and resolution")
        return self

    @property
    def candidate_id(self) -> str:
        return digest(self.model_dump(mode="json"))


class ResolvedBuildRecord(Strict):
    schema_version: int = 1
    kind: Literal["original", "exported"]
    candidate_id: str
    plan_digest: str
    source_model_digest: str
    build_template_digest: str
    build_request_digest: str
    artifact_path: str
    artifact_digest: str
    manifest_path: str
    manifest_digest: str
    target_digest: str
    backend: str
    precision: str
    device: str
    provider: str
    resolution: tuple[int, int]
    batch: int
    dynamic: bool
    profiles: tuple[ShapeProfile, ...] = ()
    upstream_build_report_path: str | None = None
    upstream_build_report_file_sha256: str | None = None
    upstream_build_report_self_digest: str | None = None
    self_digest: str

    @model_validator(mode="after")
    def _valid(self):
        for value in (
            self.candidate_id,
            self.plan_digest,
            self.source_model_digest,
            self.build_template_digest,
            self.build_request_digest,
            self.artifact_digest,
            self.manifest_digest,
            self.target_digest,
            self.self_digest,
        ):
            _hex(value)
        raw = self.model_dump(exclude={"self_digest"}, mode="json")
        if self.self_digest != digest(raw):
            raise ValueError("resolved build record self digest mismatch")
        upstream = (
            self.upstream_build_report_path,
            self.upstream_build_report_file_sha256,
            self.upstream_build_report_self_digest,
        )
        if self.kind == "original" and any(item is not None for item in upstream):
            raise ValueError("original records must not carry upstream build reports")
        if self.kind == "exported" and any(item is None for item in upstream):
            raise ValueError("exported records require upstream build report path/file/self digests")
        for value in upstream[1:]:
            if value is not None:
                _hex(value)
        return self


class ResolvedArtifactEvidence(Strict):
    """Evaluator-produced immutable build result, separate from the plan."""

    candidate_id: str
    artifact_digest: str
    build_request_digest: str
    manifest_digest: str
    target_digest: str
    build_report_digest: str
    artifact_path: str
    artifact_sha256: str
    manifest_path: str
    manifest_sha256: str
    build_report_path: str
    build_report_sha256: str

    @model_validator(mode="after")
    def _valid(self):
        for value in (
            self.candidate_id,
            self.artifact_digest,
            self.build_request_digest,
            self.manifest_digest,
            self.target_digest,
            self.build_report_digest,
            self.artifact_sha256,
            self.manifest_sha256,
            self.build_report_sha256,
        ):
            _hex(value)
        for path, expected in (
            (self.artifact_path, self.artifact_sha256),
            (self.manifest_path, self.manifest_sha256),
            (self.build_report_path, self.build_report_sha256),
        ):
            item = _trusted_regular(path)
            if hashlib.sha256(item.read_bytes()).hexdigest() != expected:
                raise ValueError("resolved artifact material identity mismatch")
        record = ResolvedBuildRecord.model_validate_json(Path(self.build_report_path).read_text())
        if record.self_digest != self.build_report_digest or record.candidate_id != self.candidate_id:
            raise ValueError("resolved build record digest mismatch")
        if (
            Path(record.artifact_path).resolve() != Path(self.artifact_path).resolve()
            or Path(record.manifest_path).resolve() != Path(self.manifest_path).resolve()
        ):
            raise ValueError("resolved build record paths do not match evidence")
        manifest = VisionManifest.from_yaml(self.manifest_path)
        if digest(manifest.model_dump(mode="json")) != self.manifest_digest:
            raise ValueError("resolved manifest canonical digest mismatch")
        return self


def _same_path(left: str | Path, right: str | Path) -> bool:
    """Compare an exact absolute path, never a symlink-resolved alias."""
    return Path(left) == Path(right) and Path(left).is_absolute() and Path(right).is_absolute()


def _assert_target_material(resolved: ResolvedArtifactEvidence, target: ModelTarget) -> None:
    """Bind an evaluator's evidence to the exact signed target, including bytes."""
    if not _same_path(resolved.artifact_path, target.source):
        raise ValueError("resolved artifact path is not the exact target source path")
    if not _same_path(resolved.manifest_path, target.manifest_path):
        raise ValueError("resolved manifest path is not the exact target manifest path")
    if resolved.artifact_digest != resolved.artifact_sha256 or resolved.artifact_digest != target.source_sha256:
        raise ValueError("resolved artifact digest is not the signed target bytes")
    if resolved.manifest_sha256 != target.manifest_sha256 or resolved.manifest_digest != target.manifest_digest:
        raise ValueError("resolved manifest digest is not the signed target bytes")
    if resolved.target_digest != digest(target.model_dump(mode="json")):
        raise ValueError("resolved target digest is not the signed target")


def verify_resolved_material(
    resolved: ResolvedArtifactEvidence,
    plan: CandidatePlan | None = None,
    target: ModelTarget | None = None,
    search: SearchRequest | None = None,
) -> ResolvedBuildRecord:
    """Re-open immutable material on every trust boundary; never trust cached models."""
    ResolvedArtifactEvidence.model_validate(resolved.model_dump(mode="json"))
    report_path = _trusted_regular(resolved.build_report_path)
    record = ResolvedBuildRecord.model_validate_json(report_path.read_text())
    if record.self_digest != resolved.build_report_digest or record.candidate_id != resolved.candidate_id:
        raise ValueError("resolved build record changed after evidence construction")
    if plan is not None and (
        record.plan_digest != plan.candidate_id or record.source_model_digest != plan.source_model_digest
    ):
        raise ValueError("resolved material does not match candidate plan")
    if target is not None:
        _assert_target_material(resolved, target)
    if not (
        _same_path(record.artifact_path, resolved.artifact_path)
        and _same_path(record.manifest_path, resolved.manifest_path)
        and record.artifact_digest == resolved.artifact_digest
        and record.manifest_digest == resolved.manifest_digest
        and record.target_digest == resolved.target_digest
    ):
        raise ValueError("resolved build record does not bind exact artifact and manifest material")
    if search is not None:
        if plan is None or target is None:
            raise ValueError("search-bound material verification requires plan and target anchors")
        if record.build_template_digest != search.build_template_digest:
            raise ValueError("resolved build template is not search anchored")
        if record.build_request_digest != expected_build_request_digest(plan, search.build_template_digest):
            raise ValueError("resolved build request is not derived from search plan")
        is_baseline = plan == search.baseline
        is_original = record.kind == "original"
        if is_original != is_baseline:
            raise ValueError("only the exact baseline plan may use original material")
        if is_original:
            if not (
                plan.backend == "pytorch"
                and plan.precision == "fp32"
                and target == search.reference_target
                and _same_path(resolved.artifact_path, search.reference_target.source)
                and _same_path(resolved.manifest_path, search.reference_target.manifest_path)
            ):
                raise ValueError("original material must be the exact external PyTorch baseline")
        elif record.kind != "exported":
            raise ValueError("non-baseline material must be exported")
    if record.kind == "exported":
        from ..export import BuildReport

        path = _trusted_regular(record.upstream_build_report_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.upstream_build_report_file_sha256:
            raise ValueError("upstream build report material changed")
        upstream = BuildReport.model_validate_json(path.read_text())
        if (
            upstream.self_digest != record.upstream_build_report_self_digest
            or upstream.finalized().self_digest != upstream.self_digest
            or not _same_path(upstream.output, resolved.artifact_path)
            or upstream.output_digest != resolved.artifact_digest
        ):
            raise ValueError("upstream build report does not bind resolved artifact")
        # Re-open every reported output/sidecar rather than treating the build
        # report's inventory as an assertion.  Export reports normally use
        # basenames; absolute paths remain supported only when trusted.
        for name, expected in upstream.artifact_digests.items():
            _hex(expected)
            item = _trusted_regular(Path(name) if Path(name).is_absolute() else path.parent / name)
            if hashlib.sha256(item.read_bytes()).hexdigest() != expected:
                raise ValueError("upstream artifact inventory digest changed")
        if resolved.artifact_digest not in upstream.artifact_digests.values():
            raise ValueError("upstream artifact inventory omits selected artifact bytes")
        if plan is not None and (
            upstream.source_digest != plan.source_model_digest
            or upstream.target != plan.backend
            or upstream.effective_precision != plan.precision
            or upstream.dynamic != plan.dynamic
        ):
            raise ValueError("upstream build report does not match candidate plan")
        if search is not None and (
            not _same_path(upstream.source, search.reference_target.source)
            or upstream.source_digest != search.reference_target.source_sha256
            or upstream.requested_precision != plan.precision
            or upstream.effective_precision != plan.precision
            or upstream.dynamic != plan.dynamic
            or tuple(ShapeProfile.model_validate(profile) for profile in upstream.profiles) != plan.profiles
        ):
            raise ValueError("upstream build report does not match search anchors")
    return record


def expected_build_request_digest(plan: CandidatePlan, build_template_digest: str) -> str:
    _hex(build_template_digest)
    return digest({"plan": plan.model_dump(mode="json"), "build_template_digest": build_template_digest})


class RuntimeFingerprint(Strict):
    hardware_digest: str
    harness_digest: str
    workload_digest: str
    input_digest: str
    artifact_digest: str
    backend: str = Field(min_length=1)
    precision: str = Field(pattern="^(fp32|fp16|int8)$")
    runtime_version: str = Field(min_length=1)
    device: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    threads: int = Field(gt=0)
    mode: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid(self):
        for value in (
            self.hardware_digest,
            self.harness_digest,
            self.workload_digest,
            self.input_digest,
            self.artifact_digest,
        ):
            _hex(value)
        return self

    @property
    def fingerprint_digest(self) -> str:
        return digest(self.model_dump(mode="json"))


class BenchmarkRequest(Strict):
    fingerprint: RuntimeFingerprint
    resolved: ResolvedArtifactEvidence
    resolution: tuple[int, int]
    dynamic: bool
    profiles: tuple[ShapeProfile, ...] = ()
    warmup: int = Field(ge=0)
    repeats: int = Field(gt=0)
    batch: int = Field(gt=0)

    @model_validator(mode="after")
    def _valid(self):
        if any(x <= 0 for x in self.resolution):
            raise ValueError("benchmark resolution dimensions must be positive")
        if self.dynamic != bool(self.profiles):
            raise ValueError("dynamic benchmark requests require profiles; static requests must not carry them")
        return self


class BenchmarkReport(Strict):
    request: BenchmarkRequest
    samples_ms: tuple[float, ...]
    memory_mb: float | None = Field(default=None, ge=0)
    artifact_size_bytes: int = Field(ge=0)
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    mean_ms: float | None = None
    std_ms: float | None = None
    throughput_items_per_s: float | None = None
    digest: str | None = None

    @staticmethod
    def _percentile(samples: tuple[float, ...], percentile: float) -> float:
        ordered = sorted(samples)
        pos = (len(ordered) - 1) * percentile / 100
        lo, hi = int(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    @model_validator(mode="after")
    def _derived(self):
        if len(self.samples_ms) != self.request.repeats:
            raise ValueError("benchmark sample count must equal requested repeats")
        if any(not math.isfinite(x) or x <= 0 for x in self.samples_ms):
            raise ValueError("benchmark samples must be finite positive milliseconds")
        expected = {
            "p50_ms": self._percentile(self.samples_ms, 50),
            "p95_ms": self._percentile(self.samples_ms, 95),
            "p99_ms": self._percentile(self.samples_ms, 99),
            "mean_ms": statistics.fmean(self.samples_ms),
            "std_ms": statistics.pstdev(self.samples_ms),
        }
        expected["throughput_items_per_s"] = self.request.batch * 1000 / expected["mean_ms"]
        for name, value in expected.items():
            recorded = getattr(self, name)
            if recorded is not None and (not math.isfinite(recorded) or abs(recorded - value) > 1e-10):
                raise ValueError(f"{name} must be derived from raw samples")
            object.__setattr__(self, name, value)
        body = self.model_dump(exclude={"digest"}, mode="json")
        expected_digest = digest(body)
        if self.digest is not None and self.digest != expected_digest:
            raise ValueError("benchmark digest mismatch")
        object.__setattr__(self, "digest", expected_digest)
        return self


class ValidationEvidence(Strict):
    report: ValidationReport
    root: str = Field(min_length=1)
    expected_policy_digest: str
    resolved: ResolvedArtifactEvidence
    binding_digest: str

    @model_validator(mode="after")
    def _valid(self):
        _hex(self.expected_policy_digest)
        for value in (self.binding_digest,):
            _hex(value)
        if self.expected_policy_digest != self.report.policy_digest:
            raise ValueError("validation evidence policy digest mismatch")
        if self.report.evidence_kind != "observed" or not self.report.publishable:
            raise ValueError("optimization requires observed publishable validation evidence")
        if self.report.candidate.target.source_sha256 != self.resolved.artifact_digest:
            raise ValueError("validation candidate target is not bound to artifact digest")
        binding = digest(
            {
                "report_digest": self.report.digest,
                "resolved": self.resolved.model_dump(mode="json"),
                "policy_digest": self.expected_policy_digest,
            }
        )
        if self.binding_digest != binding:
            raise ValueError("validation binding digest mismatch")
        return self

    def verify(self, search: SearchRequest | None = None, plan: CandidatePlan | None = None) -> bool:
        """Verify publishable evidence only with its public search/target anchors."""
        if search is None or plan is None:
            raise ValueError("publishable validation evidence requires search and plan anchors")
        verify_resolved_material(self.resolved, plan, self.report.candidate.target, search)
        return self.report.verify(self.root, self.expected_policy_digest)


class Stage(Enum):
    capability = "capability"
    build = "build"
    inspect = "inspect"
    parity = "parity"
    dataset_map = "dataset_map"
    benchmark = "benchmark"
    accepted = "accepted"
    rejected = "rejected"


class StageEvidence(Strict):
    stage: Stage
    payload: dict[str, Any] = Field(default_factory=dict)
    digest: str | None = None

    @model_validator(mode="after")
    def _digest(self):
        expected = digest({"stage": self.stage.value, "payload": self.payload})
        if self.digest is not None and self.digest != expected:
            raise ValueError("stage evidence digest mismatch")
        object.__setattr__(self, "digest", expected)
        return self


class CandidateEvaluation(Strict):
    plan: CandidatePlan
    resolved: ResolvedArtifactEvidence | None = None
    stages: tuple[StageEvidence, ...]
    validation: ValidationEvidence | None = None
    benchmark: BenchmarkReport | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _state(self):
        ordered = [entry.stage for entry in self.stages]
        allowed = [Stage.capability, Stage.build, Stage.inspect, Stage.parity, Stage.dataset_map, Stage.benchmark]
        terminal = ordered[-1:] if ordered[-1:] in ([Stage.accepted], [Stage.rejected]) else []
        progress = ordered[:-1] if terminal else ordered
        if progress != allowed[: len(progress)]:
            raise ValueError("candidate stages must be a consecutive ordered prefix")
        complete = progress == allowed
        if terminal == [Stage.accepted] and not complete:
            raise ValueError("accepted terminal stage requires complete stage sequence")
        if terminal == [Stage.accepted] and self.error is not None:
            raise ValueError("accepted candidate cannot carry an error")
        if (self.validation is not None or self.benchmark is not None) and not complete:
            raise ValueError("validation and benchmark evidence require complete stage sequence")
        if complete and self.error is None and (self.validation is None or self.benchmark is None):
            raise ValueError("completed candidate requires validation and benchmark evidence")
        if (self.validation is not None or self.benchmark is not None) and self.resolved is None:
            raise ValueError("completed candidate requires resolved artifact evidence")
        if self.resolved is not None and self.resolved.candidate_id != self.plan.candidate_id:
            raise ValueError("resolved artifact is bound to a different candidate plan")
        return self


class SearchSpace(Strict):
    backends: tuple[str, ...] = ("pytorch",)
    precisions: tuple[str, ...] = ("fp32",)
    devices: tuple[str, ...] = ("cpu",)
    providers: tuple[str, ...] = ("default",)
    resolutions: tuple[tuple[int, int], ...] = ((640, 640),)
    batches: tuple[int, ...] = (1,)
    dynamics: tuple[bool, ...] = (False,)
    profiles: tuple[ShapeProfile, ...] = ()
    exact_plans: tuple[CandidatePlan, ...] = ()
    max_candidates: int = Field(gt=0, le=10_000)

    @model_validator(mode="after")
    def _valid(self):
        dimensions = (
            self.backends,
            self.precisions,
            self.devices,
            self.providers,
            self.resolutions,
            self.batches,
            self.dynamics,
        )
        if not all(dimensions):
            raise ValueError("search dimensions may not be empty")
        if any(batch <= 0 for batch in self.batches) or any(
            any(value <= 0 for value in item) for item in self.resolutions
        ):
            raise ValueError("search batch and resolution values must be positive")
        if self.profiles and True not in self.dynamics:
            raise ValueError("shape profiles require dynamic search candidates")
        for profile in self.profiles:
            if (
                len(profile.opt_shape) != 4
                or profile.opt_shape[0] not in self.batches
                or profile.opt_shape[-2:] not in self.resolutions
            ):
                raise ValueError("profile opt shape is absent from search batch/resolution matrix")
        if len({plan.candidate_id for plan in self.exact_plans}) != len(self.exact_plans):
            raise ValueError("exact candidate plans must be unique")
        if len(self.exact_plans) > self.max_candidates:
            raise ValueError("exact candidate plans exceed max_candidates")
        return self


class SearchPolicy(Strict):
    max_map_drop: float = Field(ge=0, le=1)
    max_p95_ms: float | None = Field(default=None, gt=0)
    max_memory_mb: float | None = Field(default=None, ge=0)
    max_artifact_size_bytes: int | None = Field(default=None, ge=0)
    normalized_workload_semantics: bool = False
    allow_shape_search_comparison: bool = False


class SearchRequest(Strict):
    baseline: CandidatePlan
    space: SearchSpace
    policy: SearchPolicy
    validation_policy: ValidationPolicy
    dataset_digest: str
    reference_target: ModelTarget
    reference_runtime: RuntimeExecutionIdentity
    reference_source_model_digest: str
    build_template_digest: str
    output: str = Field(min_length=1)

    @model_validator(mode="after")
    def _baseline(self):
        if self.baseline.backend != "pytorch" or self.baseline.precision != "fp32":
            raise ValueError("baseline must be PyTorch fp32")
        for value in (self.dataset_digest, self.reference_source_model_digest, self.build_template_digest):
            _hex(value)
        if self.baseline.source_model_digest != self.reference_source_model_digest:
            raise ValueError("baseline source model digest must match external reference anchor")
        if self.reference_source_model_digest != self.reference_target.source_sha256:
            raise ValueError("reference source model digest must match reference target")
        if self.validation_policy.reference_runtime is not None or self.validation_policy.candidate_runtime is not None:
            raise ValueError("search validation policy template must not carry report-derived runtime contracts")
        if self.reference_runtime.source_sha256 != self.reference_target.source_sha256:
            raise ValueError("reference runtime is not bound to reference target")
        if self.reference_runtime.manifest_digest != self.reference_target.manifest_digest:
            raise ValueError("reference runtime manifest is not bound to reference target")
        runtime_policy_contract(
            self.reference_target,
            backend=self.reference_runtime.backend,
            provider=self.reference_runtime.provider,
            device=self.reference_runtime.device,
            precision=self.reference_runtime.precision,
            input_shape=self.reference_runtime.input_shape,
        )
        return self


class SearchReport(Strict):
    request: SearchRequest
    evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate_id: str | None
    digest: str | None = None

    @model_validator(mode="after")
    def _valid(self):
        ids = [x.plan.candidate_id for x in self.evaluations]
        if len(ids) != len(set(ids)):
            raise ValueError("search report has duplicate candidates")
        if self.selected_candidate_id is not None and self.selected_candidate_id not in ids:
            raise ValueError("selected candidate not in evaluations")
        baseline = next(
            (item for item in self.evaluations if item.plan.candidate_id == self.request.baseline.candidate_id),
            None,
        )
        # A persisted report is an auditable ledger, not a shortlist.  Never
        # quietly ignore a forged/non-reproducible accepted nonwinner: every
        # terminal acceptance must still satisfy the full public evidence path.
        if baseline is not None:
            for item in self.evaluations:
                if item.stages and item.stages[-1].stage is Stage.accepted:
                    accepted_ok, accepted_reason = _accepted(item, baseline, self.request.policy, self.request)
                    if not accepted_ok:
                        raise ValueError(f"terminal accepted candidate is invalid: {accepted_reason}")
        if self.selected_candidate_id is not None:
            selected = next(item for item in self.evaluations if item.plan.candidate_id == self.selected_candidate_id)
            if baseline is None or not selected.stages or selected.stages[-1].stage is not Stage.accepted:
                raise ValueError("selected candidate is not terminal accepted")
            selected_ok, _ = _accepted(selected, baseline, self.request.policy, self.request)
            if not selected_ok:
                raise ValueError("selected candidate evidence is not valid and bound")
            accepted = [
                item
                for item in self.evaluations
                if item.stages
                and item.stages[-1].stage is Stage.accepted
                and _accepted(item, baseline, self.request.policy, self.request)[0]
            ]
            if not accepted or min(accepted, key=_ranking_key).plan.candidate_id != self.selected_candidate_id:
                raise ValueError("selected candidate is not deterministic ranked winner")
        elif baseline is not None:
            accepted = [
                item
                for item in self.evaluations
                if item.stages
                and item.stages[-1].stage is Stage.accepted
                and _accepted(item, baseline, self.request.policy, self.request)[0]
            ]
            if accepted:
                raise ValueError("selected candidate is required when accepted evidence exists")
        body = self.model_dump(exclude={"digest"}, mode="json")
        value = digest(body)
        if self.digest is not None and self.digest != value:
            raise ValueError("search report digest mismatch")
        object.__setattr__(self, "digest", value)
        return self

    def verify(self) -> bool:
        self.__class__.model_validate(self.model_dump(mode="json"))
        return True


@runtime_checkable
class Evaluator(Protocol):
    def __call__(self, candidate: CandidatePlan) -> CandidateEvaluation: ...


def expand_candidates(request: SearchRequest) -> tuple[CandidatePlan, ...]:
    """Build the explicit candidate matrix and reject invalid combinations."""
    import itertools

    source = request.baseline
    candidates: dict[str, CandidatePlan] = {source.candidate_id: source}
    if request.space.exact_plans:
        for candidate in request.space.exact_plans:
            if candidate.source_model_digest != source.source_model_digest:
                raise ValueError("exact candidate source digest differs from baseline")
            candidates[candidate.candidate_id] = candidate
        result = tuple(candidates.values())
        if len(result) > request.space.max_candidates:
            raise ValueError("exact candidate matrix exceeds max_candidates after baseline")
        return result
    for backend, precision, device, provider, resolution, batch, dynamic in itertools.product(
        request.space.backends,
        request.space.precisions,
        request.space.devices,
        request.space.providers,
        request.space.resolutions,
        request.space.batches,
        request.space.dynamics,
    ):
        try:
            candidate = CandidatePlan(
                backend=backend,
                precision=precision,
                device=device,
                provider=provider,
                resolution=resolution,
                batch=batch,
                dynamic=dynamic,
                profiles=request.space.profiles if dynamic else (),
                source_model_digest=source.source_model_digest,
            )
        except ValueError:
            continue
        candidates[candidate.candidate_id] = candidate
    result = tuple(candidates.values())
    if len(result) > request.space.max_candidates:
        raise ValueError("candidate matrix exceeds max_candidates after dedupe")
    return result


def _comparable(reference: BenchmarkReport, candidate: BenchmarkReport, policy: SearchPolicy) -> bool:
    left, right = reference.request.fingerprint, candidate.request.fingerprint
    immutable = (
        left.hardware_digest == right.hardware_digest
        and left.harness_digest == right.harness_digest
        and left.threads == right.threads
        and left.mode == right.mode
    )
    if not immutable:
        return False
    left_request, right_request = reference.request, candidate.request
    same_shape = (
        left_request.batch == right_request.batch
        and left_request.resolution == right_request.resolution
        and left_request.dynamic == right_request.dynamic
        and left_request.profiles == right_request.profiles
    )
    if not same_shape and not policy.allow_shape_search_comparison:
        return False
    same_workload = left.workload_digest == right.workload_digest and left.input_digest == right.input_digest
    return same_workload or policy.normalized_workload_semantics


def _ranking_key(value: CandidateEvaluation) -> tuple[float, float, float, float, int, str]:
    bench = value.benchmark
    if bench is None:  # defensive: only accepted/bound evaluations reach this helper
        raise ValueError("ranked candidate has no benchmark")
    return (
        bench.p50_ms,
        bench.p95_ms,
        -bench.throughput_items_per_s,
        float("inf") if bench.memory_mb is None else bench.memory_mb,
        bench.artifact_size_bytes,
        value.plan.candidate_id,
    )


def _bound(evaluation: CandidateEvaluation, search: SearchRequest | None = None) -> tuple[bool, str]:
    """Verify all evidence names the exact candidate, not just a real report."""
    evidence, bench, plan, resolved = evaluation.validation, evaluation.benchmark, evaluation.plan, evaluation.resolved
    if evidence is None or bench is None or resolved is None:
        return False, "missing completed validation or benchmark evidence"
    if search is None:
        return False, "publishable optimization evidence requires external search anchors"
    try:
        verify_resolved_material(resolved, plan, evidence.report.candidate.target, search)
    except Exception as exc:
        return False, f"resolved material verification failed: {exc}"
    if evidence.resolved != resolved or resolved.candidate_id != plan.candidate_id:
        return False, "validation evidence is bound to a different candidate"
    request = bench.request
    if (
        request.resolved != resolved
        or request.fingerprint.artifact_digest != resolved.artifact_digest
        or request.fingerprint.backend != plan.backend
        or request.fingerprint.precision != plan.precision
        or request.fingerprint.provider != plan.provider
        or request.fingerprint.device != plan.device
        or request.batch != plan.batch
        or request.resolution != plan.resolution
        or request.dynamic != plan.dynamic
        or request.profiles != plan.profiles
    ):
        return False, "benchmark evidence is bound to a different candidate"
    runtime = evidence.report.candidate.runtime
    if (
        runtime.backend != plan.backend
        or runtime.provider != plan.provider
        or runtime.device != plan.device
        or runtime.precision != plan.precision
        or runtime.source_sha256 != resolved.artifact_digest
        or runtime.input_shape[0] != plan.batch
        or runtime.input_shape[-2:] != plan.resolution
    ):
        return False, "validation runtime identity is bound to a different candidate"
    expected_runtime = runtime_policy_contract(
        evidence.report.candidate.target,
        backend=plan.backend,
        provider=plan.provider,
        device=plan.device,
        precision=plan.precision,
        input_shape=(plan.batch, runtime.input_shape[1], *plan.resolution),
    )
    if evidence.report.policy.candidate_runtime != expected_runtime:
        return False, "validation policy runtime contract is not plan-derived"
    record = ResolvedBuildRecord.model_validate_json(Path(resolved.build_report_path).read_text())
    if (
        record.plan_digest != plan.candidate_id
        or record.source_model_digest != plan.source_model_digest
        or record.build_request_digest != resolved.build_request_digest
        or record.artifact_digest != resolved.artifact_digest
        or record.manifest_digest != resolved.manifest_digest
        or record.target_digest != resolved.target_digest
        or (
            record.backend,
            record.precision,
            record.device,
            record.provider,
            record.resolution,
            record.batch,
            record.dynamic,
            record.profiles,
        )
        != (
            plan.backend,
            plan.precision,
            plan.device,
            plan.provider,
            plan.resolution,
            plan.batch,
            plan.dynamic,
            plan.profiles,
        )
    ):
        return False, "resolved build record does not match plan"
    report = evidence.report
    if (
        report.dataset.digest != search.dataset_digest
        or report.reference.target != search.reference_target
        or report.reference.runtime != search.reference_runtime
        or plan.source_model_digest != search.reference_source_model_digest
        or resolved.manifest_digest != report.candidate.target.manifest_digest
        or resolved.target_digest != digest(report.candidate.target.model_dump(mode="json"))
    ):
        return False, "validation evidence does not match external search anchors"
    expected_policy = search.validation_policy.model_copy(
        update={"reference_runtime": search.reference_runtime, "candidate_runtime": expected_runtime}
    )
    expected_policy_digest = _policy_digest(expected_policy)
    if evidence.expected_policy_digest != expected_policy_digest or report.policy_digest != expected_policy_digest:
        return False, "validation policy digest does not match external search anchor"
    if resolved.build_request_digest != expected_build_request_digest(plan, search.build_template_digest):
        return False, "resolved build request digest is not search-derived"
    return True, "bound"


def _accepted(
    evaluation: CandidateEvaluation,
    baseline: CandidateEvaluation,
    policy: SearchPolicy,
    search: SearchRequest | None = None,
) -> tuple[bool, str]:
    if evaluation.error:
        return False, f"evaluator error: {evaluation.error}"
    bound, reason = _bound(evaluation, search)
    if not bound:
        return False, reason
    try:
        evaluation.validation.verify(search, evaluation.plan)
    except Exception as exc:  # evidence boundaries must fail closed
        return False, f"invalid validation evidence: {exc}"
    report = evaluation.validation.report
    if not report.passed or report.map_reference - report.map_candidate > policy.max_map_drop:
        return False, "quality gate rejected candidate"
    if baseline.benchmark is None or not _comparable(baseline.benchmark, evaluation.benchmark, policy):
        return False, "benchmark fingerprint is not comparable with baseline"
    bench = evaluation.benchmark
    if policy.max_p95_ms is not None and bench.p95_ms > policy.max_p95_ms:
        return False, "p95 latency SLO exceeded"
    if policy.max_memory_mb is not None and (bench.memory_mb is None or bench.memory_mb > policy.max_memory_mb):
        return False, "memory SLO exceeded"
    if policy.max_artifact_size_bytes is not None and bench.artifact_size_bytes > policy.max_artifact_size_bytes:
        return False, "artifact size SLO exceeded"
    return True, "accepted"


def _reject(evaluation: CandidateEvaluation, reason: str) -> CandidateEvaluation:
    evidence = StageEvidence(stage=Stage.rejected, payload={"reason": reason})
    return evaluation.model_copy(update={"stages": evaluation.stages + (evidence,)})


def _persist(report: SearchReport) -> None:
    parent = Path(report.request.output)
    parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(report.model_dump(mode="json")).encode()
    nonce = uuid.uuid4().hex
    stage = parent / f".vision-optimization-{nonce}.tmp"
    journal = parent / f".vision-optimization-{nonce}.journal"
    target = parent / "optimization_report.json"
    journal.write_text(_canonical({"target": target.name, "sha256": hashlib.sha256(payload).hexdigest()}))
    with stage.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    # Never overwrite an unrelated existing report: a result is immutable.
    try:
        os.link(stage, target)
    except FileExistsError:
        stage.unlink(missing_ok=True)
        journal.unlink(missing_ok=True)
        raise
    stage.unlink(missing_ok=True)
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    journal.unlink(missing_ok=True)


def recover_orphan_searches(output: str | Path) -> tuple[str, ...]:
    """Delete only our own uncommitted files; completed reports are immutable."""
    parent = Path(output)
    removed: list[str] = []
    for journal in parent.glob(".vision-optimization-*.journal") if parent.is_dir() else ():
        try:
            record = json.loads(journal.read_text())
            if set(record) != {"target", "sha256"} or record["target"] != "optimization_report.json":
                continue
            _hex(record["sha256"])
            target = parent / record["target"]
            prefix = journal.name.removesuffix(".journal")
            stage = parent / f"{prefix}.tmp"
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]:
                # Crash after link() but before cleanup: only our matching journal is removed.
                stage.unlink(missing_ok=True)
                removed.extend([str(stage), str(journal)])
                journal.unlink(missing_ok=True)
            elif not target.exists():
                stage.unlink(missing_ok=True)
                removed.extend([str(stage), str(journal)])
                journal.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError):
            continue
    return tuple(removed)


def load_search_report(path: str | Path) -> SearchReport:
    """Parse a persisted report and reject schema/digest tampering."""
    report = SearchReport.model_validate_json(Path(path).read_text())
    report.verify()
    return report


def optimize_search(request: SearchRequest, evaluator: Evaluator) -> SearchReport:
    """Evaluate a finite matrix and select only proven, comparable candidates."""
    recover_orphan_searches(request.output)
    candidates = expand_candidates(request)
    evaluations: list[CandidateEvaluation] = []
    baseline: CandidateEvaluation | None = None
    for candidate in candidates:
        try:
            value = evaluator(candidate)
            if not isinstance(value, CandidateEvaluation):
                raise TypeError("evaluator must return CandidateEvaluation")
            if value.plan != candidate:
                raise ValueError("evaluator returned evidence for a different candidate")
        except Exception as exc:
            value = CandidateEvaluation(plan=candidate, stages=(), error=f"{type(exc).__name__}: {exc}")
        if candidate.candidate_id == request.baseline.candidate_id:
            baseline = value
        evaluations.append(value)
    if baseline is None:
        raise RuntimeError("baseline was not expanded")
    baseline_ok, baseline_reason = _accepted(baseline, baseline, request.policy, request)
    if not baseline_ok:
        evaluations = [
            _reject(x, baseline_reason if x is baseline else "baseline failed: " + baseline_reason) for x in evaluations
        ]
        report = SearchReport(request=request, evaluations=tuple(evaluations), selected_candidate_id=None)
        _persist(report)
        raise RuntimeError("baseline PyTorch fp32 failed: " + baseline_reason)
    accepted: list[CandidateEvaluation] = []
    finalized: list[CandidateEvaluation] = []
    for value in evaluations:
        ok, reason = _accepted(value, baseline, request.policy, request)
        if ok:
            evidence = StageEvidence(stage=Stage.accepted, payload={"reason": reason})
            value = value.model_copy(update={"stages": value.stages + (evidence,)})
            accepted.append(value)
        else:
            value = _reject(value, reason)
        finalized.append(value)
    # Quality and SLO filtering happened above. This tuple is intentionally stable.
    ranked = sorted(
        accepted,
        key=lambda x: (
            x.benchmark.p50_ms,
            x.benchmark.p95_ms,
            -x.benchmark.throughput_items_per_s,
            float("inf") if x.benchmark.memory_mb is None else x.benchmark.memory_mb,
            x.benchmark.artifact_size_bytes,
            x.plan.candidate_id,
        ),
    )
    selected = ranked[0].plan.candidate_id if ranked else None
    report = SearchReport(request=request, evaluations=tuple(finalized), selected_candidate_id=selected)
    _persist(report)
    return report


__all__ = [
    "BenchmarkReport",
    "BenchmarkRequest",
    "CandidateEvaluation",
    "CandidatePlan",
    "Evaluator",
    "RuntimeFingerprint",
    "ResolvedArtifactEvidence",
    "SearchPolicy",
    "SearchReport",
    "SearchRequest",
    "SearchSpace",
    "ShapeProfile",
    "Stage",
    "StageEvidence",
    "ValidationEvidence",
    "digest",
    "expected_build_request_digest",
    "expand_candidates",
    "load_search_report",
    "optimize_search",
    "recover_orphan_searches",
    "verify_resolved_material",
]
