# Contributing

Thanks for helping improve Vision BuildKit.

Vision BuildKit is a provider-neutral build and conformance layer: frameworks
produce models, runtimes execute them, and this project records and checks the
semantic boundary between the two. Please keep this distinction clear in code,
tests, and documentation.

## Before opening a change

- Use an issue to discuss substantial features, new model families, or changes
  to public contracts.
- Keep pull requests focused on one problem.
- Do not commit model weights, TensorRT engines, datasets, credentials, or
  machine-specific benchmark artifacts.
- Clearly distinguish synthetic tests from measurements collected on real
  hardware and datasets.
- Do not describe an adapter, exporter, import test, or successful compilation
  as validated runtime support. Claims need a paired source/candidate run on a
  declared dataset, artifact, hardware, runtime, and harness.
- Keep semantic changes explicit: preprocessing, decoding, NMS ownership,
  coordinate space, labels, and tolerances belong in a versioned Vision
  Contract and need contract tests.

## Development setup

Vision BuildKit supports Python 3.10–3.13 and uses
[uv](https://docs.astral.sh/uv/) for reproducible environments:

```bash
git clone https://github.com/aalvsz/vision-buildkit.git
cd vision-buildkit
uv sync --all-extras
```

Run the required checks:

```bash
uv run ruff check .
uv run ruff format --check vision tests scripts
uv run pytest -q
uv build
```

## Pull requests

Include:

- the problem and intended behavior;
- tests covering the change;
- the model, dataset, hardware, runtime, and harness for performance or
  accuracy claims;
- compatibility or migration notes when a manifest, capsule, or API contract
  changes.

For a new provider or target plugin, include a fail-closed capability boundary
and tests showing that target code consumes neutral material plus a contract,
not provider-specific runtime state.

By intentionally submitting a contribution for inclusion, you agree to the
contribution terms in section 5 of the
[Vision BuildKit Research and Evaluation License 1.0](LICENSE). Those terms let
the Licensor use, commercialize, and relicense the contribution. Please do not
submit code unless you have the right to grant those permissions.

## License reminder

Contributions and forks remain subject to the [license](LICENSE). It is
source-available and permits non-commercial research and evaluation only; it
does not grant permission for commercial, production, hosted, paid, or
internal-business use. It also does not grant rights to third-party models,
weights, datasets, or runtimes. The license text is not legal advice; obtain
counsel for a use case that depends on legal interpretation.
