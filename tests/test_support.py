from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vision.cli import app
from vision.support import doctor_report, load_support_matrix


def test_support_matrix_is_evidence_bound():
    matrix = load_support_matrix()
    assert matrix["schema_version"] == 1
    assert len(matrix["entries"]) >= 5
    assert all(entry["status"] in matrix["policy"]["statuses"] for entry in matrix["entries"])


def test_support_cli_emits_json():
    result = CliRunner().invoke(app, ["support"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert any(entry["id"] == "axelera-metis-int8" for entry in payload["entries"])


def test_doctor_is_explicitly_non_claiming():
    report = doctor_report()
    assert report["support_matrix"] == "docs/support-matrix.yml"
    assert "do not constitute" in report["note"]


def test_invalid_support_matrix_rejects_missing_evidence(tmp_path: Path):
    path = tmp_path / "matrix.yml"
    path.write_text(
        "schema_version: 1\nentries:\n  - id: bad\n    status: development_verified\n    evidence: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must bind evidence"):
        load_support_matrix(path)
