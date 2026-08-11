# YOLO26 head-only fake-QAT pilot

This is a bounded local preparation artifact, not an accuracy result. It targets only the six raw Detect terminal Conv2d modules: `model.23.cv2.{0,1,2}.2` and `model.23.cv3.{0,1,2}.2`. All other parameters are frozen.

The fake quantizer is deliberately custom: symmetric signed INT8 (`scale=max(abs(x))/127`, round, clamp to `[-128,127]`, dequantize) plus the straight-through estimator. PyTorch's built-in fake-quant operator is unsupported by MPS when `PYTORCH_ENABLE_MPS_FALLBACK=0`; this implementation has no fallback path.

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

All twelve terminal weight/bias tensors are trainable and are the exact optimizer set. The synthetic one-batch CPU smoke requires finite nonzero updates only for tensors reached by its single synthetic label; a real run must fail unless every one of the twelve target tensors changes. Biases are intentionally trainable because the final Conv2d layers have biases and score calibration depends on their offsets.

The smoke test uses 64x64 only to bound work. A real experiment must export at 640x640 and prove output `[1,84,8400]`, plain opset 17, ONNX checker, ORT/PyTorch parity, frozen corpus hashes, host-only quantization artifact change, then frozen-harness screening. It must not claim AIPU evidence without a separately authorized AIPU run.
