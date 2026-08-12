# Release checklist

This repository is source-available research software. A release is a
provenance event, not only a version edit.

## Before tagging

```bash
uv sync --frozen --all-extras
uv run ruff check .
uv run ruff format --check vision tests scripts
uv run python scripts/verify_repository.py
uv run python scripts/release_manifest.py create /tmp/vision-buildkit-release-manifest.json
uv run python scripts/release_manifest.py verify /tmp/vision-buildkit-release-manifest.json
uv run python scripts/dependency_inventory.py --output /tmp/vision-buildkit-dependencies.json
uv run python scripts/verify_repository.py
uv run pytest -q
uv build
```

The CI environment uses the `httpx2` compatibility package required by the
current Starlette test client. A full local run should therefore show no
Starlette/httpx deprecation warning. The remaining TorchScript warning comes
from the optional legacy `TorchAdapter` loader and is tracked separately from
the release gate; it must not be silently replaced because changing the loader
would alter model compatibility.

Then review `git diff --stat`, `git status`, the support matrix, and every
release-bound evidence digest. Generated model binaries, engines, capsules,
calibration data, and private datasets must remain outside normal source
history; publish only their scripts, manifests, receipts, and permitted
content hashes.

## Version and claims

- Update `pyproject.toml` and `CHANGELOG.md` together.
- Do not call a release stable while the support matrix contains only
  development-scoped evidence.
- Do not promote `unsupported`, `build_only`, or `api_only` entries through
  prose, examples, or badges.
- Preserve the research-only license and third-party artifact terms.

Tagging and pushing are intentionally separate from local validation and
require explicit release approval.
