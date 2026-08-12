# YOLO26 head-only fake-QAT pilot

This is a bounded local preparation artifact, not an accuracy result. It targets only the six raw Detect terminal Conv2d modules: `model.23.cv2.{0,1,2}.2` and `model.23.cv3.{0,1,2}.2`. All other parameters are frozen.

The fake quantizer is deliberately custom: symmetric narrow-range signed INT8 (`scale=max(abs(x))/127`, round, clamp to `[-127,127]`, dequantize) plus the straight-through estimator. PyTorch's built-in fake-quant operator is unsupported by MPS when `PYTORCH_ENABLE_MPS_FALLBACK=0`; this implementation has no fallback path.

`one_batch_step` combines the ordinary Ultralytics detection loss with a small MSE consistency loss on the pre-Sigmoid `one2many` score logits from a frozen FP teacher. The runner composes this primitive into a bounded one-to-five-epoch CPU experiment; it never silently changes device or enables fallback.

## Current fail-closed training gate

The code seeds Python, NumPy, Torch, and MPS and enables strict deterministic algorithms. On the pinned Torch/MPS stack, ordinary Ultralytics-loss backward then fails at `index_put_with_accumulate_mps`, which has no deterministic MPS implementation. The smoke test proves this failure is surfaced as `FAIL_CLOSED`; it does not enable `PYTORCH_ENABLE_MPS_FALLBACK`, `warn_only`, or a hidden CPU step. Therefore this artifact proves the QAT wrapping/export path, but **does not authorize a deterministic MPS fine-tuning epoch**. An experiment needs an explicitly approved deterministic backend or a PyTorch fix before proceeding.

Validation:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/ander.alvarez/dev/perso/cv-engine/.venv/bin/python -m pytest -q autoresearch/axelera-head-qat-pilot-260811/tests
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/ander.alvarez/dev/perso/cv-engine/.venv/bin/python autoresearch/axelera-head-qat-pilot-260811/prepare_eval_split.py \
  --annotations /Users/ander.alvarez/dev/perso/cv-engine/autoresearch/vision-buildkit-phase18-libreyolo-260731/reference/instances_val2017.full.json \
  --output autoresearch/axelera-head-qat-pilot-260811/eval-split
```

The evaluation script does not download unless `--fetch --official-url URL` is supplied, and verifies the pinned full-annotation SHA-256 before selecting its deterministic ten-image dev and ten-image heldout splits. Neither split may overlap the ten regression IDs or ten previously inspected LibreYOLO IDs.

The supplied COCO128 corpus currently fails the strict image/label-stem gate (two image-only and two label-only stems), so no new training manifest is emitted from it. This is an intentional fail-closed data blocker.

All twelve terminal weight/bias tensors are trainable and are the exact optimizer set. The CPU replay-smoke uses decoded curated image/label pairs and fails on missing or non-finite gradients; the update must mutate target state and never mutate the teacher or frozen graph. Biases are intentionally trainable because the final Conv2d layers have biases and score calibration depends on their offsets.

The smoke and export use 640x640 and prove output `[1,84,8400]`, plain opset 17, ONNX checker, and frozen corpus hashes. It must not claim AIPU evidence without a separately authorized AIPU run.

## Completed CPU-simulator screen

The corrected plain-ONNX head-QAT artifact was quantized with the unchanged
Voyager 1.7 `per_tensor_min_max` configuration and screened through the CPU
quantized-model simulator. The run is recorded in
`artifacts/cpu-screen/receipt.json`; the simulator returned exit code 2 and
the frozen decision was **FAIL**. Candidate mAP50-95 was `0.324355` versus
`0.282699` for the corrected source on the ten-image regression fixture, but
fidelity failed: coverage `0.965517` (required `>=0.98`), one class mismatch
(required `0`), and maximum score delta `0.448224` (required `<=0.10`). The
candidate prediction bytes differ from the prior baseline, so this is a real
negative result rather than a no-op. The candidate is not eligible for AIPU
execution or production promotion. Its 19 unique scores, including nine
exactly equal to `0.5`, confirm that score compression remains the dominant
failure mode.

## Voyager-qparam surrogate (2026-08-12)

The compiler was inspected through its loaded TorchScript quantized model. The
terminal QConv path is fixed affine quantization: asymmetric per-tensor input
qparams, per-output-channel symmetric weights, an integer bias, and a separate
per-tensor output QRE. Those qparams are sealed in
`autoresearch/axelera-head-qat-qparam-surrogate-260812/artifacts/qparams.json`.
The QAT harness now supports `--quantizer voyager`,
`--quantizer voyager-integer`, and `--quantizer voyager-integer-clipped`. The
integer modes emulate the observed deployed
sequence—integer-domain activation and weights, integer bias, accumulator
rounding, and output QRE—rather than merely dequantizing each boundary.

A one-epoch CPU run and a three-epoch CPU run were both exported and passed the
plain-ONNX/topology checks. The one-epoch candidate was quantized and screened;
it still failed with coverage `0.982759`, one class mismatch, and maximum score
delta `0.448222`. The three-epoch candidate failed earlier at Voyager graph
export with the SDK's pinned MatMul constraint (`At least one pinned zero
(zero_point=0) input is needed for inputs to MatMul ops`) because the surrogate
qparams are not valid graph-wide calibration overrides. This establishes the
correct training simulation boundary, but not a passing artifact.

The supported route is therefore **not** to force those six qparams into a
whole-graph ONNX compile. The next implementation must either (a) expose a
Voyager graph-cleaner/model-card output-redefinition or ignore-scope hook that
keeps the score head out of AIPU quantization, or (b) implement a compiler-aware
QAT/export path that preserves required MatMul zero-point constraints outside
the terminal head. The model-local custom-function host-FP prototype is not
supported by SDK 1.7.0 because graph cleaning strips its FunctionProto bodies.

The integer emulator matches the loaded Voyager terminal QConv exactly on
controlled tensors (including bias and accumulator rounding). The custom
`_VoyagerIntegerQuantize` autograd operator now preserves that literal
`round -> clamp` forward while providing identity or bounded saturated-bin
derivatives. Its output boundary also includes the compiler's per-channel
accumulator scale/zero-point before the output QRE; omitting that boundary
produced a misleading multi-million detection loss and is now covered by the
replay gate.

The bounded custom mode (`voyager-integer-clipped`) passes the sealed replay
smoke with finite loss (`detection_loss=35.4570`,
`consistency_loss=0.01214`) and finite gradients on all twelve terminal
weight/bias tensors. This is an implementation/surrogate gate only: no full
epoch, compile, simulator screen, or AIPU execution has been authorized for
this mode yet. The earlier three-epoch candidate still fails at graph export
because the SDK requires pinned zero-point inputs for MatMul.

That replay gate is now superseded by the completed epoch and screen below.
Any coverage/class/score failure is terminal for the candidate; do not tune the
frozen fixture or use post-hoc score calibration as an INT8 artifact claim.

## Custom-operator epoch and screen (2026-08-12)

The bounded custom mode completed one CPU epoch over all 126 sealed curated
pairs. Detection loss decreased from `1.87593` to `1.30799`; the exported
plain raw-output ONNX passed the opset-17, shape, checker, and topology gates
(`196b3b70602d87b8225bbfa719cdcbc7c0cabde5a85a1c2e64cfbcf8f30083ab`). The
candidate was then transplanted into the pinned raw topology and quantized
with the unchanged Voyager min-max compiler configuration. Host quantization
completed successfully; no Metis AIPU execution was attempted.

The candidate-specific CPU simulator screen is recorded under
`autoresearch/qat-voyager-integer-clipped-epoch-260812/screen-results.json` and
is **FAIL**: mAP50-95 `0.324403` versus source `0.282699` passed, but coverage
was `0.965517` (required `>=0.98`), class mismatches were `1` (required `0`),
and maximum score delta was `0.448225` (required `<=0.10`). This confirms the
custom autograd implementation is executable and provenance-correct, but it
does not solve the deployed score-compression gap. The candidate is not
eligible for AIPU or production promotion.

## Follow-up CPU QAT attempts (2026-08-12)

An isolated one-epoch CPU run over the sealed 126-pair curated corpus
(`seed=260812`, `lr=1e-4`, consistency weight `0.01`) was exported as a plain
raw-output ONNX, transplanted into the pinned 368-node raw topology, quantized
with the unchanged Voyager per-tensor min-max configuration, and screened by
the frozen CPU simulator. The source artifact SHA is
`b0d92a6f18944b60a0539983cb80c7461a76b8e18d79da682412725de70aa6ef`; the
quantized model SHA is
`62d72461b252373139c19cedab5c4aad60ee414eb757ecfe1326e6426146f2f7`.

The result was **FAIL**: mAP50-95 `0.319957` vs source `0.282699` and
coverage `0.982759` passed, but one class mismatch and maximum score delta
`0.448143` failed. A second isolated run fake-quantizing the terminal Conv
output during training produced the same failure signature. Neither candidate
was compiled or run on the Metis AIPU. Detailed handoffs remain under the
ignored `autoresearch/qat-epoch-260812/` and
`autoresearch/qat-epoch-output-260812-v2/` directories as local research
records only.
