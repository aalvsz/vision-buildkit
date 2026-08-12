# Axelera YOLO26n INT8 report

Status: **blocked for parity under Voyager SDK 1.7.0**. The implementation,
host quantization, and CPU-simulator evidence are reproducible; the frozen
quality gate does not pass, so no new AIPU run or production claim is made.

## Baseline and gates

The target is the YOLO26n raw-output graph (`[1, 84, 8400]`) on an Axelera
Metis device. The frozen parity policy is:

| Gate | Requirement |
|---|---:|
| mAP drop | `<= 0.002` |
| Detection coverage | `>= 0.98` |
| Maximum score delta | `<= 0.10` |
| Class mismatches | `0` |

The historical real-device INT8 run deployed successfully and preserved box
localization, but confidence scores were compressed. It achieved mAP
`0.200382`, coverage `0.403509`, and max score delta `0.373610`.

## Work completed

- Implemented a custom Voyager-style integer fake-quantizer with explicit
  `round -> clamp` forward behavior and bounded straight-through gradients.
- Matched the observed terminal-QConv boundaries: asymmetric activation
  qparams, per-output-channel weight qparams, integer bias/accumulator handling,
  per-channel accumulator dequantization, and output QRE.
- Ran the bounded one-epoch CPU QAT pilot over 126 sealed curated pairs.
- Exported a plain opset-17 raw-output ONNX and host-quantized it with the
  unchanged Voyager `per_tensor_min_max` configuration.
- Screened the exact candidate with the frozen CPU quantized-model simulator.

## Candidate result

The candidate passed structural/export and host-quantization checks, but the
frozen simulator returned **FAIL**:

| Metric | Candidate | Gate |
|---|---:|---:|
| mAP50-95 | `0.324403` | pass |
| Coverage | `0.965517` | **fail** (`>= 0.98`) |
| Class mismatches | `1` | **fail** (`0`) |
| Maximum score delta | `0.448225` | **fail** (`<= 0.10`) |

The result is a CPU-simulator screen only. It was not promoted to AIPU
execution because the frozen screen failed.

## SDK capability finding

The live Voyager SDK was inspected on `axe-ubuntu-125` at commit `7e2672f`
(runtime `1.7.0-1`, firmware `1.7.0`). The public `CompilerConfig` exposes
graph-cleaner split nodes only for `MatMul`, `Gemm`, and `Clip`. It does not
expose:

- a terminal-Conv INT8 ignore scope;
- a per-node mixed-precision policy for the six Detect terminal Convs; or
- an output-redefinition/host-FP head hook.

The capability result is recorded in
[`docs/evidence/yolo26n-axelera-sdk-17-capability.json`](evidence/yolo26n-axelera-sdk-17-capability.json),
and the repeatable probe is
[`scripts/inspect_axelera_sdk_capabilities.py`](../scripts/inspect_axelera_sdk_capabilities.py).

## Decision and next action

Do not spend another blind QAT or AIPU window under SDK 1.7. The remaining
problem is compiler expressiveness: training changes the source weights, but
the compiler still quantizes the score head and compresses confidence values.

Re-open this experiment only when a vendor-supported terminal-head exclusion,
output redefinition, or newer SDK is available. Then require both the frozen
and held-out CPU screens to pass before any AIPU execution. The complete
implementation and negative evidence remain in
`autoresearch/axelera-head-qat-pilot-260811/`.

## Validation

- Focused QAT suite: `7 passed`.
- Full repository suite: `618 passed, 2 skipped`.
- Repository integrity verifier: passed.
- Targeted Ruff checks for the new capability probe and QAT runner/tests:
  passed.
- No AIPU process was interrupted or modified by this follow-up.
