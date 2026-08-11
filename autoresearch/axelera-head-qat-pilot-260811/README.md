# YOLO26 head-only fake-QAT pilot

This is a bounded local preparation artifact, not an accuracy result. It targets only the six raw Detect terminal Conv2d modules: `model.23.cv2.{0,1,2}.2` and `model.23.cv3.{0,1,2}.2`. All other parameters are frozen.

The fake quantizer is deliberately custom: symmetric narrow-range signed INT8 (`scale=max(abs(x))/127`, round, clamp to `[-127,127]`, dequantize) plus the straight-through estimator. PyTorch's built-in fake-quant operator is unsupported by MPS when `PYTORCH_ENABLE_MPS_FALLBACK=0`; this implementation has no fallback path.

`one_batch_step` combines the ordinary Ultralytics detection loss with a small MSE consistency loss on the pre-Sigmoid `one2many` score logits from a frozen FP teacher. It is intentionally a one-batch primitive; no epoch is started by this pilot.

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

All twelve terminal weight/bias tensors are trainable and are the exact optimizer set. The CPU replay-smoke uses decoded curated image/label pairs and fails unless every one of the twelve target tensors changes. Biases are intentionally trainable because the final Conv2d layers have biases and score calibration depends on their offsets.

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

The next technical path is a new, disjoint QAT pilot whose fake-quantizer is
matched more closely to Voyager's per-tensor min/max behavior, followed by
source-retention and held-out quantization gates. Do not tune the frozen
ten-image fixture or use post-hoc score calibration as an INT8 artifact claim.
