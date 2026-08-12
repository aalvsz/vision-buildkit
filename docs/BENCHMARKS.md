# Benchmark scopes

The benchmark runner records raw latency samples, derived percentiles,
throughput, workload identity, host identity, runtime identity, and harness
digests. `vision verify-benchmark` recomputes those values and rejects forged or
stale reports.

The repository distinguishes three evidence scopes:

1. **Conformance** — the frozen ten-image COCO subset. This is a fast semantic
   regression gate, not a full accuracy or speed claim.
2. **Quality** — a sealed stratified or full COCO manifest with paired source
   and candidate execution and content-bound accuracy evidence.
3. **Performance** — repeated inference on an explicit workload, with exact
   artifact, runtime, device, and harness identity.

The machine-readable declaration is [`benchmark-plan.yml`](benchmark-plan.yml).
Planned scopes are intentionally not represented as completed evidence. The
current development records remain unsigned historical observations.

For a repeatable local smoke run, use the checked-in runner with explicit
artifact manifests:

```bash
uv run python scripts/run_local_benchmark.py \
  --image path/to/frozen-image.jpg \
  --output-dir autoresearch/local-benchmark-YYMMDD-HHMM \
  pytorch path/to/model.pt path/to/model-manifest.yaml cpu \
  onnx path/to/model.onnx path/to/onnx-manifest.yaml cpu
```

Each report is independently verified before it is written. The resulting
numbers are workload- and host-bound development evidence, not a cross-device
ranking.
