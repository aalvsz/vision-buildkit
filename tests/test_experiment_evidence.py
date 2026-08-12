"""Acceptance tests for the canonical experiment-evidence schema."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from vision.cli import app
from vision.experiment_evidence import (
    CandidateResult,
    ComparisonBinding,
    Configuration,
    DetectionMetrics,
    ExperimentReport,
    GatePolicy,
    load_experiment_report,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _policy(**updates: object) -> GatePolicy:
    values = {
        "max_map_drop": 0.01,
        "min_coverage": 0.98,
        "max_score_delta": 0.12,
        "max_class_mismatches": 0,
        "high_confidence_threshold": 0.5,
        "max_high_confidence_lost": 0,
        "max_high_confidence_added": 0,
    }
    values.update(updates)
    return GatePolicy.model_validate(values)


def _binding(*, policy: GatePolicy | None = None, **updates: object) -> ComparisonBinding:
    values = {
        "model_id": "YOLO26n",
        "source_artifact_digest": DIGEST_A,
        "reference_artifact_digest": DIGEST_A,
        "candidate_artifact_digest": DIGEST_A,
        "candidate_graph_digest": DIGEST_A,
        "input_contract_digest": DIGEST_A,
        "output_contract_digest": DIGEST_A,
        "preprocessing_digest": DIGEST_A,
        "postprocessing_digest": DIGEST_A,
        "evaluation_dataset_digest": DIGEST_A,
        "calibration_dataset_digest": DIGEST_A,
        "policy_digest": (policy or _policy()).digest,
        "validation_report_digest": DIGEST_A,
        "reference_runtime_fingerprint_digest": DIGEST_A,
        "candidate_runtime_fingerprint_digest": DIGEST_A,
        "hardware_digest": DIGEST_A,
        "harness_digest": DIGEST_A,
    }
    values.update(updates)
    return ComparisonBinding.model_validate(values)


def _metrics_passing() -> DetectionMetrics:
    return DetectionMetrics.build(
        reference_map=0.317,
        candidate_map=0.317,
        coverage=1.0,
        max_score_delta=0.003,
        high_confidence_lost=0,
        high_confidence_added=0,
        class_mismatches=0,
        matched=57,
        lost=0,
        added=0,
    )


def _metrics_failing() -> DetectionMetrics:
    return DetectionMetrics.build(
        reference_map=0.317,
        candidate_map=0.300,
        coverage=52 / 57,
        max_score_delta=0.168,
        high_confidence_lost=0,
        high_confidence_added=1,
        class_mismatches=0,
        matched=52,
        lost=5,
        added=1,
    )


def _candidate(
    *,
    label: str = "fp32",
    parameters: dict[str, object] | None = None,
    binding: ComparisonBinding | None = None,
    metrics: DetectionMetrics | None = None,
    policy: GatePolicy | None = None,
) -> CandidateResult:
    selected_policy = policy or _policy()
    configuration = Configuration.with_derived_id(
        label=label,
        parameters=parameters or {"precision": "fp32"},
        binding=binding or _binding(policy=selected_policy),
    )
    return CandidateResult.build(
        configuration=configuration,
        metrics=metrics or _metrics_passing(),
        policy=selected_policy,
    )


def _valid_report(*, select_failing: bool = False) -> ExperimentReport:
    policy = _policy()
    passing = _candidate(policy=policy)
    failing = _candidate(
        label="fp16",
        parameters={"precision": "fp16"},
        binding=_binding(
            policy=policy,
            candidate_artifact_digest=DIGEST_B,
            candidate_graph_digest=DIGEST_B,
            candidate_runtime_fingerprint_digest=DIGEST_B,
        ),
        metrics=_metrics_failing(),
        policy=policy,
    )
    selected = failing if select_failing else passing
    return ExperimentReport.build(
        policy=policy,
        candidates=[passing, failing],
        selected_configuration_id=selected.configuration.configuration_id,
    )


class TestConfigurationIdentity:
    def test_parameters_and_binding_change_id_but_label_does_not(self):
        policy = _policy()
        binding = _binding(policy=policy)
        fp32 = Configuration.with_derived_id("a", {"precision": "fp32"}, binding)
        fp16 = Configuration.with_derived_id("a", {"precision": "fp16"}, binding)
        renamed = Configuration.with_derived_id("renamed", {"precision": "fp32"}, binding)
        changed_binding = Configuration.with_derived_id(
            "a",
            {"precision": "fp32"},
            _binding(policy=policy, candidate_artifact_digest=DIGEST_B),
        )

        assert fp32.configuration_id != fp16.configuration_id
        assert fp32.configuration_id == renamed.configuration_id
        assert fp32.configuration_id != changed_binding.configuration_id

    @pytest.mark.parametrize(
        "field",
        [name for name in ComparisonBinding.model_fields if name.endswith("_digest")],
    )
    def test_every_binding_digest_changes_configuration_id(self, field: str):
        policy = _policy()
        base = _binding(policy=policy)
        changed = _binding(policy=policy, **{field: DIGEST_B})
        parameters = {"precision": "int8"}

        base_config = Configuration.with_derived_id("base", parameters, base)
        changed_config = Configuration.with_derived_id("changed", parameters, changed)

        assert base_config.configuration_id != changed_config.configuration_id

    def test_forged_id_rejected(self):
        with pytest.raises(ValidationError, match="configuration_id"):
            Configuration(
                label="forged",
                parameters={"precision": "fp32"},
                binding=_binding(),
                configuration_id="0" * 64,
            )

    @pytest.mark.parametrize(
        "parameters",
        [
            {"nested": [float("nan")]},
            {"nested": {"value": float("inf")}},
            {"unsupported": object()},
        ],
    )
    def test_parameters_must_be_canonical_json(self, parameters: dict[str, object]):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            Configuration.with_derived_id("invalid", parameters, _binding())

    def test_report_rederives_configuration_id_after_parameter_mutation(self):
        policy = _policy()
        candidate = _candidate(policy=policy)
        candidate.configuration.parameters["precision"] = "int8"

        with pytest.raises(ValidationError, match="configuration_id"):
            ExperimentReport.build(
                policy=policy,
                candidates=[candidate],
                selected_configuration_id=candidate.configuration.configuration_id,
            )


class TestMetrics:
    def test_map_drop_is_signed_and_preserves_improvement(self):
        metrics = DetectionMetrics.build(
            reference_map=0.30,
            candidate_map=0.32,
            coverage=1.0,
            max_score_delta=0.0,
            high_confidence_lost=0,
            high_confidence_added=0,
            class_mismatches=0,
            matched=1,
            lost=0,
            added=0,
        )
        assert metrics.map_drop == pytest.approx(-0.02)

    def test_inconsistent_map_drop_rejected(self):
        data = _metrics_passing().model_dump(mode="json")
        data["map_drop"] = 0.5
        with pytest.raises(ValidationError, match="map_drop"):
            DetectionMetrics.model_validate(data)

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"coverage": 0.9}, "coverage"),
            ({"high_confidence_lost": 1}, "high_confidence_lost"),
            ({"high_confidence_added": 1}, "high_confidence_added"),
            ({"class_mismatches": 58}, "class_mismatches"),
            ({"inventory_complete": False}, "inventory_complete"),
            ({"candidate_evaluated_images": 2}, "evaluated image counts"),
        ],
    )
    def test_inconsistent_counts_rejected(self, updates: dict[str, object], message: str):
        data = _metrics_passing().model_dump(mode="json")
        data.update(updates)
        with pytest.raises(ValidationError, match=message):
            DetectionMetrics.model_validate(data)

    def test_zero_reference_detections_have_full_coverage(self):
        metrics = DetectionMetrics.build(
            reference_map=0.0,
            candidate_map=0.0,
            coverage=1.0,
            max_score_delta=0.0,
            high_confidence_lost=0,
            high_confidence_added=0,
            class_mismatches=0,
            matched=0,
            lost=0,
            added=0,
        )
        assert metrics.coverage == 1.0

    @pytest.mark.parametrize("field", ["reference_map", "max_score_delta"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_metrics_rejected(self, field: str, value: float):
        data = _metrics_passing().model_dump(mode="json")
        data[field] = value
        with pytest.raises(ValidationError):
            DetectionMetrics.model_validate(data)


class TestGatesAndReport:
    def test_high_confidence_added_is_a_real_gate(self):
        policy = _policy(max_high_confidence_added=0)
        candidate = _candidate(metrics=_metrics_failing(), policy=policy)
        gates = {gate.kind: gate.passed for gate in candidate.gate_results}
        assert gates["high_confidence_added"] is False

    def test_forged_gate_boolean_rejected(self):
        report = _valid_report(select_failing=True)
        data = report.model_dump(mode="json")
        added_gate = next(
            gate for gate in data["candidates"][1]["gate_results"] if gate["kind"] == "high_confidence_added"
        )
        added_gate["passed"] = True

        with pytest.raises(ValidationError, match="recomputed"):
            ExperimentReport.model_validate(data)

    def test_duplicate_configuration_ids_fail_during_build(self):
        policy = _policy()
        first = _candidate(label="first", policy=policy)
        duplicate = _candidate(label="duplicate", policy=policy)

        with pytest.raises(ValidationError, match="unique"):
            ExperimentReport.build(
                policy=policy,
                candidates=[first, duplicate],
                selected_configuration_id=first.configuration.configuration_id,
            )

    def test_selected_id_must_exist(self):
        policy = _policy()
        with pytest.raises(ValueError, match="selected_configuration_id"):
            ExperimentReport.build(
                policy=policy,
                candidates=[_candidate(policy=policy)],
                selected_configuration_id="d" * 64,
            )

    def test_status_is_derived_from_selected_candidate(self):
        assert _valid_report().status == "PASS"
        assert _valid_report(select_failing=True).status == "FAIL"

    def test_forged_status_and_partial_pass_rejected(self):
        report = _valid_report()
        for invalid in ("FAIL", "PARTIAL_PASS"):
            data = report.model_dump(mode="json")
            data["status"] = invalid
            with pytest.raises(ValidationError):
                ExperimentReport.model_validate(data)

    def test_policy_and_binding_policy_digests_are_checked(self):
        report = _valid_report()
        data = report.model_dump(mode="json")
        data["policy_digest"] = DIGEST_B
        with pytest.raises(ValidationError, match="policy_digest"):
            ExperimentReport.model_validate(data)

        data = report.model_dump(mode="json")
        data["candidates"][0]["configuration"]["binding"]["policy_digest"] = DIGEST_B
        with pytest.raises(ValidationError, match="configuration_id|policy_digest"):
            ExperimentReport.model_validate(data)

    def test_sequences_are_immutable_tuples(self):
        report = _valid_report()
        assert isinstance(report.candidates, tuple)
        assert all(isinstance(candidate.gate_results, tuple) for candidate in report.candidates)
        assert {gate.kind for gate in report.candidates[0].gate_results} == {
            "inventory",
            "map_drop",
            "coverage",
            "score_delta",
            "class_mismatches",
            "high_confidence_lost",
            "high_confidence_added",
        }


class TestStrictLoadingAndCLI:
    def test_valid_report_round_trips(self):
        report = _valid_report()
        restored = ExperimentReport.model_validate_json(report.model_dump_json())
        assert restored == report
        assert restored.report_digest == report.report_digest

    def test_extra_fields_and_schema_v1_rejected(self):
        report_data = _valid_report().model_dump(mode="json")
        report_data["extra"] = True
        with pytest.raises(ValidationError, match="extra"):
            ExperimentReport.model_validate(report_data)
        with pytest.raises(ValidationError):
            ExperimentReport.model_validate({"schema_version": 1})

    def test_direct_symlink_rejected(self, tmp_path: Path):
        target = tmp_path / "report.json"
        target.write_text(_valid_report().model_dump_json())
        link = tmp_path / "link.json"
        os.symlink(target, link)
        with pytest.raises(ValueError, match="symlink"):
            load_experiment_report(link)

    def test_symlinked_parent_rejected(self, tmp_path: Path):
        real_parent = tmp_path / "real"
        real_parent.mkdir()
        report_path = real_parent / "report.json"
        report_path.write_text(_valid_report().model_dump_json())
        linked_parent = tmp_path / "linked"
        os.symlink(real_parent, linked_parent)

        with pytest.raises(ValueError, match="symlinked parent"):
            load_experiment_report(linked_parent / "report.json")

    @pytest.mark.parametrize("kind", ["missing", "directory"])
    def test_non_file_rejected(self, tmp_path: Path, kind: str):
        path = tmp_path / "missing.json" if kind == "missing" else tmp_path
        with pytest.raises(ValueError, match="not a regular file"):
            load_experiment_report(path)

    def test_cli_returns_canonical_identity(self, tmp_path: Path):
        report = _valid_report()
        path = tmp_path / "report.json"
        path.write_text(report.model_dump_json())

        result = CliRunner().invoke(app, ["validate-experiment-evidence", str(path)])

        assert result.exit_code == 0
        assert json.loads(result.stdout) == {
            "policy_digest": report.policy_digest,
            "report_digest": report.report_digest,
            "schema_version": 2,
            "selected_configuration_id": report.selected_configuration_id,
            "status": "PASS",
        }

    def test_cli_rejects_invalid_report(self, tmp_path: Path):
        data = _valid_report().model_dump(mode="json")
        data["status"] = "FAIL"
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(data))

        result = CliRunner().invoke(app, ["validate-experiment-evidence", str(path)])

        assert result.exit_code != 0
