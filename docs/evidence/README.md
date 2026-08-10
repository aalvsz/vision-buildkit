# Experiment Evidence Schema

## Canonical schema (schema_version=2)

All new development experiment evidence must conform to the strict canonical
schema defined in `vision/experiment_evidence.py`. The schema uses frozen
Pydantic models with:

- **Extra fields forbidden** — any unknown key causes rejection.
- **Non-finite floats rejected** — `inf`, `-inf`, and `nan` are not accepted.
- **Lowercase SHA-256 Hex** digests for all identity and artifact fields.
- **Shared comparison binding** — preprocessing, postprocessing, contracts,
  dataset, hardware, and harness are single shared values per candidate, so
  raw/e2e or different postprocessing baselines cannot be silently presented
  as matched.
- **Derived configuration_id** — computed from canonical JSON of parameters
  + binding (excluding the human label). Supplied IDs that do not equal the
  derived ID are rejected.
- **Six recomputed gate booleans** — mAP drop, coverage, maximum score delta,
  class mismatches, high-confidence losses, and high-confidence additions are
  recomputed from the candidate's metrics and report policy. Forged or
  mismatched booleans are rejected.
- **Signed mAP delta** — `map_drop = reference_map - candidate_map`; a negative
  value records a real candidate improvement instead of clamping it to zero.
- **Count consistency** — coverage and high-confidence/class counts must agree
  with the matched, lost, and added detection counts.
- **Derived report status** — `PASS` or `FAIL` only, derived from the
  selected candidate's gates. `PARTIAL_PASS` is not a canonical status.

### Validation

```bash
vision validate-experiment-evidence path/to/report.json
```

Emits a concise JSON object with `schema_version`, `selected_configuration_id`,
`status`, `policy_digest`, and `report_digest`.

## Legacy evidence (schema_version=1)

Existing `schema_version=1` JSON files in this directory are **legacy
development records**. They are:

- **Not accepted** by the new canonical validator (`schema_version=2` required).
- **Not migrated or repaired** — they remain as historical records only.
- **Not relabeled** as schema_version=2.

To produce canonical evidence for an experiment, the experiment must be
rerun (or be artifact-backed-migrated) through the schema_version=2 pipeline
with full artifact digests, binding, metrics, and recomputed gates.

## File index

| File | Schema | Status |
|------|--------|--------|
| `yolo26n-tensorrt-jetson-orin-nano.json` | 1 (legacy) | Historical record |
| `yolo26n-tensorrt-int8-raw-jetson-orin-nano.json` | 1 (legacy) | Historical record |
| `yolo26n-tensorrt-int8-jetson-orin-nano-FAIL.json` | 1 (legacy) | Historical record |
| `yolo26n-openvino-apple-cpu.json` | 1 (legacy) | Historical record |
| `yolo26n-openvino-int8-apple-cpu-FAIL.json` | 1 (legacy) | Historical record |
| `yolo26n-axelera-metis-aipu-int8-FAIL.json` | 1 (legacy) | Historical record |
| `rtdetr-l-onnxruntime-apple-cpu.json` | 1 (legacy) | Historical record |
| `rfdetr-base-onnxruntime-apple-cpu.json` | 1 (legacy) | Historical record |
| `libreyolo9s-onnxruntime-apple-cpu.json` | 1 (legacy) | Historical record |
