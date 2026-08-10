"""Strict canonical schema for development experiment evidence reports.

Prevents combining metrics, configuration identity, and gate booleans from
different candidates — the root cause of contradictory OpenVINO/TensorRT
hand-authored JSON summaries.

schema_version=2 is canonical for new evidence. Existing schema_version=1
evidence is NOT migrated or repaired; it must be rerun or
artifact-backed-migrated into this canonical format.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _Strict(BaseModel):
    """Frozen, extra-forbidding base; non-finite floats are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ComparisonBinding(_Strict):
    """Shared comparison binding for a candidate vs. reference.

    The preprocessing, postprocessing, contracts, dataset, hardware, and
    harness fields are single shared values so that raw/e2e or different
    postprocessing baselines cannot be silently presented as matched.
    """

    model_id: str = Field(min_length=1)
    task: Literal["object_detection"] = "object_detection"
    source_artifact_digest: Hex
    reference_artifact_digest: Hex
    candidate_artifact_digest: Hex
    candidate_graph_digest: Hex
    input_contract_digest: Hex
    output_contract_digest: Hex
    preprocessing_digest: Hex
    postprocessing_digest: Hex
    evaluation_dataset_digest: Hex
    calibration_dataset_digest: Hex
    policy_digest: Hex
    validation_report_digest: Hex
    reference_runtime_fingerprint_digest: Hex
    candidate_runtime_fingerprint_digest: Hex
    hardware_digest: Hex
    harness_digest: Hex

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class GatePolicy(_Strict):
    """Acceptance thresholds for parity gates (no hard-coded project values)."""

    max_map_drop: float = Field(ge=0)
    min_coverage: float = Field(ge=0, le=1)
    max_score_delta: float = Field(ge=0)
    max_class_mismatches: int = Field(ge=0)
    high_confidence_threshold: float = Field(ge=0, le=1)
    max_high_confidence_lost: int = Field(ge=0)
    max_high_confidence_added: int = Field(ge=0)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _validate_json_serializable(value: Any) -> None:
    """Reject non-JSON-serializable values and nested NaN/Inf."""
    json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )


def _derive_configuration_id(parameters: dict[str, Any], binding: ComparisonBinding) -> str:
    payload = {"parameters": parameters, "binding": binding.model_dump(mode="json")}
    return _sha256(_canonical_json(payload))


class Configuration(_Strict):
    """A candidate configuration with a human label and derived identity."""

    label: str = Field(min_length=1)
    parameters: dict[str, Any]
    binding: ComparisonBinding
    configuration_id: Hex

    @model_validator(mode="after")
    def _validate_configuration_id(self) -> Configuration:
        _validate_json_serializable(self.parameters)
        derived = _derive_configuration_id(self.parameters, self.binding)
        if self.configuration_id != derived:
            raise ValueError("configuration_id does not match the derived canonical ID")
        return self

    @classmethod
    def with_derived_id(cls, label: str, parameters: dict[str, Any], binding: ComparisonBinding) -> Configuration:
        _validate_json_serializable(parameters)
        return cls(
            label=label,
            parameters=parameters,
            binding=binding,
            configuration_id=_derive_configuration_id(parameters, binding),
        )


class DetectionMetrics(_Strict):
    """Detection parity metrics sufficient for current gates."""

    reference_map: float = Field(ge=0, le=1)
    candidate_map: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    max_score_delta: float = Field(ge=0)
    high_confidence_lost: int = Field(ge=0)
    high_confidence_added: int = Field(ge=0)
    class_mismatches: int = Field(ge=0)
    matched: int = Field(ge=0)
    lost: int = Field(ge=0)
    added: int = Field(ge=0)
    expected_images: int = Field(gt=0)
    reference_evaluated_images: int = Field(ge=0)
    candidate_evaluated_images: int = Field(ge=0)
    inventory_complete: bool
    map_drop: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def _validate_map_drop(self) -> DetectionMetrics:
        expected = self.reference_map - self.candidate_map
        if not math.isclose(self.map_drop, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"map_drop={self.map_drop} does not match derived value {expected}")
        return self

    @model_validator(mode="after")
    def _validate_metric_counts(self) -> DetectionMetrics:
        total = self.matched + self.lost
        expected_coverage = self.matched / total if total > 0 else 1.0
        if not math.isclose(self.coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"coverage={self.coverage} does not match derived value {expected_coverage}")
        if self.high_confidence_lost > self.lost:
            raise ValueError(f"high_confidence_lost={self.high_confidence_lost} exceeds lost={self.lost}")
        if self.high_confidence_added > self.added:
            raise ValueError(f"high_confidence_added={self.high_confidence_added} exceeds added={self.added}")
        if self.class_mismatches > self.matched:
            raise ValueError(f"class_mismatches={self.class_mismatches} exceeds matched={self.matched}")
        expected_inventory = (
            self.reference_evaluated_images == self.expected_images
            and self.candidate_evaluated_images == self.expected_images
        )
        if self.inventory_complete != expected_inventory:
            raise ValueError("inventory_complete does not match evaluated image counts")
        if (
            self.reference_evaluated_images > self.expected_images
            or self.candidate_evaluated_images > self.expected_images
        ):
            raise ValueError("evaluated image counts exceed the expected inventory")
        return self

    @classmethod
    def build(
        cls,
        reference_map: float,
        candidate_map: float,
        coverage: float,
        max_score_delta: float,
        high_confidence_lost: int,
        high_confidence_added: int,
        class_mismatches: int,
        matched: int,
        lost: int,
        added: int,
        expected_images: int = 1,
        reference_evaluated_images: int = 1,
        candidate_evaluated_images: int = 1,
    ) -> DetectionMetrics:
        map_drop = reference_map - candidate_map
        return cls(
            reference_map=reference_map,
            candidate_map=candidate_map,
            coverage=coverage,
            max_score_delta=max_score_delta,
            high_confidence_lost=high_confidence_lost,
            high_confidence_added=high_confidence_added,
            class_mismatches=class_mismatches,
            matched=matched,
            lost=lost,
            added=added,
            expected_images=expected_images,
            reference_evaluated_images=reference_evaluated_images,
            candidate_evaluated_images=candidate_evaluated_images,
            inventory_complete=(
                reference_evaluated_images == expected_images and candidate_evaluated_images == expected_images
            ),
            map_drop=map_drop,
        )


GateKind = Literal[
    "inventory",
    "map_drop",
    "coverage",
    "score_delta",
    "class_mismatches",
    "high_confidence_lost",
    "high_confidence_added",
]
_EXPECTED_GATE_KINDS: frozenset[str] = frozenset(
    {
        "map_drop",
        "inventory",
        "coverage",
        "score_delta",
        "class_mismatches",
        "high_confidence_lost",
        "high_confidence_added",
    }
)


class GateResult(_Strict):
    """Result of a single parity gate."""

    kind: GateKind
    passed: bool


def _compute_gate_results(metrics: DetectionMetrics, policy: GatePolicy) -> tuple[GateResult, ...]:
    return (
        GateResult(kind="inventory", passed=metrics.inventory_complete),
        GateResult(kind="map_drop", passed=metrics.map_drop <= policy.max_map_drop),
        GateResult(kind="coverage", passed=metrics.coverage >= policy.min_coverage),
        GateResult(kind="score_delta", passed=metrics.max_score_delta <= policy.max_score_delta),
        GateResult(
            kind="class_mismatches",
            passed=metrics.class_mismatches <= policy.max_class_mismatches,
        ),
        GateResult(
            kind="high_confidence_lost",
            passed=metrics.high_confidence_lost <= policy.max_high_confidence_lost,
        ),
        GateResult(
            kind="high_confidence_added",
            passed=metrics.high_confidence_added <= policy.max_high_confidence_added,
        ),
    )


class CandidateResult(_Strict):
    """A single candidate's configuration, metrics, and gate results."""

    configuration: Configuration
    metrics: DetectionMetrics
    gate_results: tuple[GateResult, ...]

    @model_validator(mode="after")
    def _validate_gate_kinds(self) -> CandidateResult:
        kinds = [g.kind for g in self.gate_results]
        if len(kinds) != len(_EXPECTED_GATE_KINDS):
            raise ValueError(f"expected {len(_EXPECTED_GATE_KINDS)} gate_results, got {len(kinds)}")
        if set(kinds) != _EXPECTED_GATE_KINDS:
            raise ValueError(f"gate_results must cover exactly {_EXPECTED_GATE_KINDS}, got {set(kinds)}")
        if len(set(kinds)) != len(kinds):
            raise ValueError("duplicate gate kinds in gate_results")
        return self

    @classmethod
    def build(cls, configuration: Configuration, metrics: DetectionMetrics, policy: GatePolicy) -> CandidateResult:
        return cls(
            configuration=configuration,
            metrics=metrics,
            gate_results=_compute_gate_results(metrics, policy),
        )


ReportStatus = Literal["PASS", "FAIL"]


class ExperimentReport(_Strict):
    """Canonical experiment evidence report (schema_version=2)."""

    schema_version: Literal[2] = 2
    evidence_kind: Literal["observed_unsigned_development"] = "observed_unsigned_development"
    policy: GatePolicy
    policy_digest: Hex
    candidates: tuple[CandidateResult, ...] = Field(min_length=1)
    selected_configuration_id: Hex
    status: ReportStatus

    @model_validator(mode="after")
    def _validate_report(self) -> ExperimentReport:
        if self.policy_digest != self.policy.digest:
            raise ValueError("policy_digest does not match the policy canonical digest")

        for candidate in self.candidates:
            derived_id = _derive_configuration_id(
                candidate.configuration.parameters,
                candidate.configuration.binding,
            )
            if candidate.configuration.configuration_id != derived_id:
                raise ValueError(
                    f"configuration_id does not match recomputed ID "
                    f"for candidate {candidate.configuration.configuration_id}"
                )

        config_ids = [c.configuration.configuration_id for c in self.candidates]
        if len(set(config_ids)) != len(config_ids):
            raise ValueError("candidate configuration IDs must be unique")

        selected = [c for c in self.candidates if c.configuration.configuration_id == self.selected_configuration_id]
        if len(selected) != 1:
            raise ValueError("selected_configuration_id must name exactly one candidate")

        for candidate in self.candidates:
            if candidate.configuration.binding.policy_digest != self.policy_digest:
                raise ValueError(
                    f"binding policy_digest does not match report policy_digest "
                    f"for candidate {candidate.configuration.configuration_id}"
                )
            expected = {gate.kind: gate.passed for gate in _compute_gate_results(candidate.metrics, self.policy)}
            for gate in candidate.gate_results:
                if gate.passed != expected[gate.kind]:
                    raise ValueError(
                        f"gate {gate.kind} passed={gate.passed} does not match "
                        f"recomputed passed={expected[gate.kind]} for candidate "
                        f"{candidate.configuration.configuration_id}"
                    )

        selected_candidate = selected[0]
        expected_status: ReportStatus = "PASS" if all(g.passed for g in selected_candidate.gate_results) else "FAIL"
        if self.status != expected_status:
            raise ValueError(
                f"status={self.status} does not match derived status={expected_status} from selected candidate"
            )

        return self

    @property
    def report_digest(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify_validation_binding(self, validation_report: Any) -> bool:
        """Rebuild this report from its committed validation source and compare exactly."""

        binding = self.candidates[0].configuration.binding
        expected = experiment_report_from_validation(
            validation_report,
            candidate_graph_digest=binding.candidate_graph_digest,
            calibration_dataset_digest=binding.calibration_dataset_digest,
        )
        if expected != self:
            raise ValueError("experiment evidence does not match its validation report")
        return True

    @classmethod
    def build(
        cls,
        policy: GatePolicy,
        candidates: Sequence[CandidateResult],
        selected_configuration_id: str,
    ) -> ExperimentReport:
        policy_digest = policy.digest
        config_ids = [c.configuration.configuration_id for c in candidates]
        if selected_configuration_id not in config_ids:
            raise ValueError("selected_configuration_id does not match any candidate")
        selected = next(c for c in candidates if c.configuration.configuration_id == selected_configuration_id)
        status: ReportStatus = "PASS" if all(g.passed for g in selected.gate_results) else "FAIL"
        return cls(
            policy=policy,
            policy_digest=policy_digest,
            candidates=candidates,
            selected_configuration_id=selected_configuration_id,
            status=status,
        )


def load_experiment_report(path: str | Path) -> ExperimentReport:
    """Load and strictly validate an experiment evidence report.

    Rejects symlinks, symlinked parent paths, and non-file paths.
    """
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"symlinks are not accepted: {path}")
    for parent in p.parents:
        if parent.is_symlink():
            raise ValueError(f"symlinked parent paths are not accepted: {parent}")
    if not p.is_file():
        raise ValueError(f"not a regular file: {path}")
    return ExperimentReport.model_validate_json(p.read_text())


NOT_APPLICABLE_DIGEST = hashlib.sha256(b"not-applicable").hexdigest()


def experiment_report_from_validation(
    validation_report: Any,
    *,
    candidate_graph_digest: str | None = None,
    calibration_dataset_digest: str | None = None,
) -> ExperimentReport:
    """Derive canonical experiment evidence from a validated observed report.

    No metric, gate, runtime, policy, dataset, or artifact identity is accepted
    from the caller. Only build-only identities absent from ValidationReport may
    be supplied explicitly, and they remain content-bound in the result.
    """

    validation_report.__class__.model_validate(validation_report.model_dump(mode="json"))
    if validation_report.evidence_kind != "observed":
        raise ValueError("canonical experiment evidence requires observed validation")
    if not validation_report.hardware_digest or not validation_report.harness_digest:
        raise ValueError("validation report lacks hardware or harness identity")

    policy = GatePolicy(
        max_map_drop=validation_report.policy.max_map_drop,
        min_coverage=validation_report.policy.min_coverage,
        max_score_delta=validation_report.policy.score_delta,
        max_class_mismatches=validation_report.policy.max_class_mismatches,
        high_confidence_threshold=validation_report.policy.high_confidence,
        max_high_confidence_lost=0,
        max_high_confidence_added=0,
    )
    reference, candidate = validation_report.reference, validation_report.candidate
    input_contract_digest = _sha256(
        _canonical_json(
            {
                "reference": reference.input_preprocessing_digest,
                "candidate": candidate.input_preprocessing_digest,
                "dataset": validation_report.dataset.preprocessing_digest,
                "allowed": list(validation_report.dataset.allowed_preprocessing_digests),
            }
        )
    )
    output_contract_digest = _sha256(
        _canonical_json(
            {
                "reference": reference.output_postprocessing_digest,
                "candidate": candidate.output_postprocessing_digest,
                "dataset": validation_report.dataset.postprocessing_digest,
                "allowed": list(validation_report.dataset.allowed_postprocessing_digests),
            }
        )
    )
    binding = ComparisonBinding(
        model_id=candidate.target.manifest().family,
        source_artifact_digest=reference.target.source_sha256,
        reference_artifact_digest=reference.target.source_sha256,
        candidate_artifact_digest=candidate.target.source_sha256,
        candidate_graph_digest=candidate_graph_digest
        or validation_report.candidate_graph_digest
        or candidate.target.source_sha256,
        input_contract_digest=input_contract_digest,
        output_contract_digest=output_contract_digest,
        preprocessing_digest=validation_report.dataset.preprocessing_digest,
        postprocessing_digest=validation_report.dataset.postprocessing_digest,
        evaluation_dataset_digest=validation_report.dataset.digest,
        calibration_dataset_digest=calibration_dataset_digest
        or validation_report.calibration_digest
        or NOT_APPLICABLE_DIGEST,
        policy_digest=policy.digest,
        validation_report_digest=validation_report.digest,
        reference_runtime_fingerprint_digest=reference.runtime.digest,
        candidate_runtime_fingerprint_digest=candidate.runtime.digest,
        hardware_digest=validation_report.hardware_digest,
        harness_digest=validation_report.harness_digest,
    )
    configuration = Configuration.with_derived_id(
        label=f"{candidate.runtime.backend}-{candidate.runtime.precision}-{candidate.runtime.device}",
        parameters={
            "backend": candidate.runtime.backend,
            "provider": candidate.runtime.provider,
            "device": candidate.runtime.device,
            "precision": candidate.runtime.precision,
            "input_shape": list(candidate.runtime.input_shape),
            "validation_policy_digest": validation_report.policy_digest,
        },
        binding=binding,
    )
    parity = validation_report.parity
    metrics = DetectionMetrics.build(
        reference_map=validation_report.map_reference,
        candidate_map=validation_report.map_candidate,
        coverage=parity["coverage"],
        max_score_delta=parity["max_score_delta"],
        high_confidence_lost=len(parity["high_conf_lost"]),
        high_confidence_added=len(parity["high_conf_added"]),
        class_mismatches=len(parity["class_mismatches"]),
        matched=parity["matched"],
        lost=len(parity["lost"]),
        added=len(parity["added"]),
        expected_images=validation_report.dataset.expected_image_count,
        reference_evaluated_images=len(reference.evaluated_image_ids),
        candidate_evaluated_images=len(candidate.evaluated_image_ids),
    )
    result = CandidateResult.build(configuration=configuration, metrics=metrics, policy=policy)
    report = ExperimentReport.build(
        policy=policy,
        candidates=(result,),
        selected_configuration_id=configuration.configuration_id,
    )
    expected_status: ReportStatus = "PASS" if validation_report.passed else "FAIL"
    if report.status != expected_status:
        raise ValueError("canonical experiment status disagrees with validation report")
    return report
