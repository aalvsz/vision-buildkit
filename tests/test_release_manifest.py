from __future__ import annotations

import json

import pytest

from scripts.release_manifest import build_manifest, verify_manifest


def test_release_manifest_binds_current_tracked_inventory(tmp_path):
    manifest = build_manifest(allow_dirty=True)
    assert manifest["schema_version"] == 1
    assert manifest["working_tree_clean"] is False
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert verify_manifest(path)["manifest_sha256"] == manifest["manifest_sha256"]


def test_release_manifest_rejects_self_digest_tampering(tmp_path):
    manifest = build_manifest(allow_dirty=True)
    manifest["version"] = "tampered"
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="self-digest"):
        verify_manifest(path)
