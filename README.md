# Vision BuildKit

![Vision BuildKit — build, validate, and package the evidence](assets/social-preview.jpg)

[![CI](https://github.com/aalvsz/vision-buildkit/actions/workflows/ci.yml/badge.svg)](https://github.com/aalvsz/vision-buildkit/actions/workflows/ci.yml)
[![License: research only](https://img.shields.io/badge/license-research--only-orange.svg)](LICENSE)

**Train anywhere. Build and check for a specific device.**

Vision BuildKit is an early, source-available toolchain for taking a vision model from a provider, describing its *end-to-end* semantics, producing a target artifact, comparing it against the source model, and packaging the evidence. It is deliberately not a training framework, model zoo, or new inference runtime.

The repository is organized around a strict boundary: planning and API
availability are useful, but only evidence-bound paired execution earns a
development-support claim. See the [support matrix](docs/support-matrix.yml),
[operations boundary](docs/OPERATIONS.md), and [release checklist](docs/RELEASE.md).

Frameworks such as Ultralytics or LibreYOLO train and export models. Runtimes such as ONNX Runtime, TensorRT, and OpenVINO execute graphs. Vision BuildKit lives between them: it records what the model means, plans a compatible build, and refuses to represent an unverified result as a certified one.

> **Research and evaluation only.** This repository is source-available, not OSI-approved open source. Non-commercial study, evaluation, modification, and forks are permitted under the [license](LICENSE); production, internal business, commercial, hosted, and paid use require a separate written agreement.

## The problem it addresses

An exported graph alone does not say whether input pixels were RGB or BGR, which letterbox rule was used, whether boxes refer to model pixels or original image pixels, or where NMS happened. Those omitted details are a common cause of a model that “runs” but does not behave like its source.

Vision BuildKit makes those facts explicit in a **Vision Contract**, then joins source adapters to target plugins:

```text
source framework                         target runtime / device
----------------                         -----------------------
Ultralytics ─┐                         ┌─ ONNX Runtime (current plugin)
LibreYOLO ───┼─ SourceAdapter          ├─ OpenVINO (current CPU plugin)
custom ONNX ─┘       │                 ├─ TensorRT (FP32/FP16 verified on Jetson Orin Nano; INT8 e2e build fails on TRT 10.3.0, raw-output INT8 builds — 4/6 gates pass: mAP 0.334, 0 hc lost/added, max_delta 0.168)
                       ▼                ├─ Axelera Metis AIPU (executed INT8; YOLO26n parity gate FAILED, root cause documented)
                Vision Contract         └─ other target plugins
                       │
                       ▼
        build plan → conformance evidence → Vision Capsule
```

The intended integration model is **N source adapters + M target plugins**, not a bespoke N × M integration. A target plugin receives neutral material and a contract, rather than an object from a specific provider.

## What is verified today

This table is deliberately narrower than the APIs and tests in the repository. “Verified” means a real, paired source/candidate execution in the same harness, not merely an import, unit test, export attempt, or engine build.

| Area | Status | Evidence scope |
|---|---|---|
| Source provider | **Verified** | Ultralytics YOLO26n native PyTorch reference |
| Target | **Verified** | Static FP32 ONNX executed by ONNX Runtime `CPUExecutionProvider` |
| OpenVINO | **Verified in a development scope** | YOLO26n CPU FP32 and FP16 on Apple Silicon; see the exact limitations below |
| Dataset | **Verified** | 10 frozen COCO validation images, paired detections and COCO mAP |
| Machines | **Verified** | Apple Silicon Mac and Intel Axelera host; both CPU runs |
| Axelera-Metis AIPU | **Executed, parity gate FAILED** | YOLO26n compiled and ran INT8 on a Metis AIPU; localization preserved but INT8 score compression dropped mAP from 0.442554 to 0.200382 (root cause documented) |
| Parity | **Verified in that scope** | mAP50–95 unchanged; no lost or added detections in the frozen subset |
| Capsule | **Verified in that scope** | Content-bound capsule created, re-verified, and reloaded |
| LibreYOLO | **LibreYOLO9s verified in a development scope** | LibreYOLO9s FP32 ONNX executed by ONNX Runtime CPU on Apple Silicon, zero mAP drop vs PyTorch reference (threshold 0.25: 76/76 matched, coverage 1.0, mean IoU 0.999999) |
| TensorRT | **Verified in a development scope** | YOLO26n TensorRT FP32 and FP16 on NVIDIA Jetson Orin Nano (CUDA); raw-output INT8 is a bounded partial pass; see the exact limitations below |
| RF-DETR and RT-DETR | **RF-DETR-Base and RT-DETR-L verified in a development scope** | RF-DETR-Base and RT-DETR-L FP32 ONNX both executed by ONNX Runtime CPU on Apple Silicon, zero mAP drop vs PyTorch reference |
| Jetson / accelerator execution | **Verified in a development scope** | YOLO26n TensorRT built and validated on a Jetson Orin Nano unit; Axelera-Metis AIPU executed but failed the INT8 parity gate |
| INT8 (TensorRT and OpenVINO) | Partial (TensorRT) | Calibration interfaces exist. OpenVINO INT8 MIXED + ignored_scope on Sigmoid+Softmax achieves mAP 0.301 (drop 0.014) but max_score_delta 0.155 > 0.12 gate. TensorRT INT8 on the e2e ONNX fails to build (TRT 10.3.0 checkLinks bug with TopK/GatherElements); a raw-output ONNX (end2end=False) + MinMax calibrator + FLOAT OBEY (last 30 Conv) + 500 calib images achieves mAP 0.334 (better than FP32 ref) with 0 hc lost — 4 of 6 gates pass (coverage 0.912 < 0.98 and max_score_delta 0.168 > 0.12 fail). Axelera Metis INT8 remains unsupported: score compression persists after compiler-matched head QAT, and SDK 1.7 exposes no public terminal-head exclusion hook. |

The 10-image subset is a build/parity gate, **not** a full COCO accuracy claim, a speed claim, or a promise that any target hardware will work. Measurements are tied to the exact machine, artifact, runtime, input set, and harness.

The canonical support policy is maintained in [`docs/support-matrix.yml`](docs/support-matrix.yml). Inspect it locally with:

```bash
uv run vision support
uv run vision doctor
```

`vision doctor` reports installed modules and environment information only. It does not upgrade an API, import, compile, or smoke test into a runtime-support claim. The repository's fail-closed integrity check is:

```bash
uv run python scripts/verify_repository.py
```

**OpenVINO — verified in this development scope.** YOLO26n OpenVINO CPU FP32 and FP16 both passed observed validation on Apple Silicon (macOS arm64) over the frozen 10-image COCO subset. Source mAP50–95 was `0.317278` and both candidates achieved `0.318140`; detection coverage was `56/57` (`0.982456`), with no high-confidence losses/additions or class mismatches. This evidence is unsigned, not certified, not a full-COCO result, and not a speedup claim. The content-bound summary is in [`docs/evidence/yolo26n-openvino-apple-cpu.json`](docs/evidence/yolo26n-openvino-apple-cpu.json).

**TensorRT on Jetson — verified in this development scope.** YOLO26n TensorRT FP32 and FP16 both passed observed validation on an NVIDIA Jetson Orin Nano (aarch64, Linux tegra, CUDA 12.6, TensorRT 10.3.0) over the frozen 10-image COCO subset. Source mAP50–95 was `0.317278` and both candidate engines achieved `0.317278` (zero mAP drop); detection coverage was `57/57` (`1.0`), with no lost or added detections and no class mismatches. FP32 min IoU was `0.998986`; FP16 min IoU was `0.986776`. The run was executed in a bubblewrap sandbox with `--unshare-net --clearenv`, Python isolated mode (`-I -B`), and 24 overlay module origins verified under the sealed venv. This evidence is unsigned, self-reviewed (not an independent reviewer pass), not certified, not a full-COCO result, not a speedup claim, and not a cross-hardware claim (single Jetson unit). INT8 TensorRT and Axelera-Metis runtime remain unverified. The content-bound summary is in [`docs/evidence/yolo26n-tensorrt-jetson-orin-nano.json`](docs/evidence/yolo26n-tensorrt-jetson-orin-nano.json).

**RT-DETR-L on ONNX Runtime — verified in this development scope.** RT-DETR-L FP32 ONNX executed by ONNX Runtime `CPUExecutionProvider` passed observed validation on Apple Silicon (macOS arm64) over the frozen 10-image COCO subset. Source mAP50–95 was `0.464263` and the candidate achieved `0.464263` (zero mAP drop); detection coverage was `173/173` (`1.0`), with no lost or added detections and no class mismatches; mean IoU was `0.999832` and max score delta was `0.000011`. This evidence is unsigned, not certified, not a full-COCO result, and not a speedup claim. The content-bound summary is in [`docs/evidence/rtdetr-l-onnxruntime-apple-cpu.json`](docs/evidence/rtdetr-l-onnxruntime-apple-cpu.json).

**RF-DETR-Base on ONNX Runtime — verified in this development scope.** RF-DETR-Base FP32 ONNX executed by ONNX Runtime `CPUExecutionProvider` passed observed validation on Apple Silicon (macOS arm64) over the frozen 10-image COCO subset. Source mAP50–95 was `0.361232` and the candidate achieved `0.361338` (zero mAP drop); detection coverage was `67/67` (`1.0`), with no lost or added detections and no class mismatches; mean IoU was `0.999918` and max score delta was `0.001469`. This evidence is unsigned, not certified, not a full-COCO result, and not a speedup claim. The content-bound summary is in [`docs/evidence/rfdetr-base-onnxruntime-apple-cpu.json`](docs/evidence/rfdetr-base-onnxruntime-apple-cpu.json).

**LibreYOLO9s on ONNX Runtime — verified in this development scope.** LibreYOLO9s FP32 ONNX executed by ONNX Runtime `CPUExecutionProvider` passed observed validation on Apple Silicon (macOS arm64) over the frozen 10-image COCO subset. At the standard score threshold of 0.25, source mAP50–95 was `0.348215` and the candidate achieved `0.348215` (zero mAP drop); detection coverage was `76/76` (`1.0`), with no lost or added detections and no class mismatches; mean IoU was `0.999999` and max score delta was `0.000004`. This evidence is unsigned, not certified, not a full-COCO result, and not a speedup claim. The content-bound summary is in [`docs/evidence/libreyolo9s-onnxruntime-apple-cpu.json`](docs/evidence/libreyolo9s-onnxruntime-apple-cpu.json).

**OpenVINO INT8 on YOLO26n — NOT verified (negative result, narrowed).** A thorough NNCF configuration sweep was conducted across 14 configurations: PERFORMANCE preset with 10 and 128 calibration images, MIXED preset with 128 images, MIXED with ignored_scope (Sigmoid/Softmax/TopK, Sigmoid+Softmax+Mul, model.23 head), MIXED + SmoothQuant (α=0.3, 0.5, 0.7), MIXED + overflow_fix=enable, quantize_outputs=True, and quantize_with_accuracy_control (iterative quantizer removal). The best configuration is MIXED + ignored_scope on all 88 Sigmoid + 2 Softmax operations, which achieves mAP `0.300698` (drop `0.014233`), coverage `0.964286`, **0 high-confidence lost/added**, and max_score_delta `0.155029` — a 36% reduction from the plain MIXED max_delta of `0.244714`. The key insight is that Sigmoid operations in the YOLO26n confidence path are the primary source of INT8 quantization error — keeping them in FP32 via ignored_scope reduces max_score_delta by 36%. However, the remaining `0.155` exceeds the `0.12` gate, with 27 detections having deltas > 0.05 (the largest: img=458054 cat=70, 0.410→0.255, delta 0.155). SmoothQuant alpha tuning produced identical outputs (no effect on this graph), overflow_fix degraded results, and quantize_with_accuracy_control proved computationally infeasible (1000+ validation calls in 30+ minutes without converging). Root cause: INT8 activation quantization in Conv layers feeding the Sigmoid ops systematically shifts confidence scores by 0.06–0.15. The detail-bound summary is in [`docs/evidence/yolo26n-openvino-int8-apple-cpu-FAIL.json`](docs/evidence/yolo26n-openvino-int8-apple-cpu-FAIL.json).

**Axelera Metis AIPU INT8 on YOLO26n — NOT verified (negative result).** YOLO26n was compiled (`deploy.py --mode QUANTCOMPILE`) and executed on a real Axelera Metis AIPU (metis-0:3:0, hw_gen=omega, 4 AI cores, M.2 board, runtime 1.7.0-1, firmware v1.7.0) using the default `per_tensor_min_max` quantization scheme and 200 representative COCO val2017 calibration images. Bounding-box localization was preserved (mean IoU 0.960547 on matched detections, all >= 0.93, no false positives, no class mismatches), but roughly half of the FP32 reference detections had their confidence scores compressed below the 0.25 parity threshold, dropping mAP from `0.442554` (FP32 reference) to `0.200382` (INT8 AIPU) at threshold 0.25 and coverage from `57/57` (`1.0`) to `23/57` (`0.403509`). Root cause: INT8 quantization of the YOLO26n end-to-end confidence head compresses the score distribution. Alternative schemes were tried and did not help: `hybrid_per_tensor_per_channel` and `per_tensor_histogram` both failed compilation with a MatMul zero-point constraint; SmoothQuant (coefficient 0.5) compiled and ran but produced effectively identical scores. A compiler-matched custom integer QAT head was then trained for one CPU epoch, quantized, and screened; it still failed coverage (`0.965517`), class parity (1 mismatch), and maximum score drift (`0.448225`). A direct public-schema audit of Voyager SDK 1.7 confirms that graph cleaning supports only MatMul, Gemm, and Clip split nodes; it exposes no terminal-Conv ignore scope or output-redefinition hook. The next valid path is therefore a vendor-supported head-exclusion feature or a newer SDK, not another blind QAT sweep. The Metis AIPU is an INT8 accelerator by design and no native FP32 AIPU execution path is available. See [`docs/evidence/yolo26n-axelera-metis-aipu-int8-FAIL.json`](docs/evidence/yolo26n-axelera-metis-aipu-int8-FAIL.json), [`docs/evidence/yolo26n-axelera-sdk-17-capability.json`](docs/evidence/yolo26n-axelera-sdk-17-capability.json), and the bounded QAT handoff under `autoresearch/`.

**TensorRT INT8 on YOLO26n (end-to-end ONNX) — NOT verified (build failure).** TensorRT INT8 engine build failed on Jetson Orin Nano (TensorRT 10.3.0) for the YOLO26n end-to-end ONNX graph. The error is an internal assertion failure — `checkSanity.cpp::checkLinks:218: Assertion item.second != nullptr failed. region should have been removed from Graph::regions` — that occurs during INT8 calibration graph optimization, before any calibration data is consumed. Six build strategies were attempted: (1) `trtexec --int8 --calib`, (2) Python API with `IInt8EntropyCalibrator2` and 200 COCO calibration images, (3) `CALIBRATE_BEFORE_FUSION` quantization flag, (4) `builder_optimization_level=0`, (5) `trtexec --int8 --stronglyTyped`, and (6) `trtexec --int8` without calibration cache (hangs indefinitely). Strategy 5 (`--stronglyTyped`) is the only one that builds, but it produces an effectively FP32 engine (all I/O `DataType.FLOAT`, no actual INT8 layers) because the ONNX model has no explicit Q/DQ (QuantizeDequantize) nodes to trigger quantization. Root cause: the end-to-end ONNX graph contains a `TopK` + `GatherElements` + `Gather` pattern (from the baked-in NMS in `model.23` head) that triggers the TensorRT 10.3.0 INT8 calibration graph optimization bug. The FP32 and FP16 engines (from attempt45) build successfully because they do not use the INT8 calibration graph path. The detail-bound summary is in [`docs/evidence/yolo26n-tensorrt-int8-jetson-orin-nano-FAIL.json`](docs/evidence/yolo26n-tensorrt-int8-jetson-orin-nano-FAIL.json).

**TensorRT INT8 on YOLO26n (raw-output ONNX) — Partial pass (4 of 6 gates).** A raw-output YOLO26n ONNX was exported with `end2end=False` on the Ultralytics `Detect` head. This removes the TensorRT 10.3.0 INT8 calibration failure caused by the e2e TopK/GatherElements path. The best of 20 configurations — MinMax + OBEY with the last 30 head Conv layers in FP32 + 500 calibration images — achieves mAP `0.333989`, with 0 high-confidence detections lost and 0 class mismatches, but coverage `0.912281` and max_score_delta `0.167859` still fail. The same five threshold-boundary detections were lost across all configurations; more calibration data did not help. The raw-output result is a bounded partial pass, not a production INT8 artifact. Compiler-aware QAT remains exploratory and has not produced a passing frozen gate. The detail-bound summary is in [`docs/evidence/yolo26n-tensorrt-int8-raw-jetson-orin-nano.json`](docs/evidence/yolo26n-tensorrt-int8-raw-jetson-orin-nano.json).

## Install

Python 3.10–3.13 is supported. Development setup uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/aalvsz/vision-buildkit.git
cd vision-buildkit
uv sync --extra yolo --extra onnx --extra validation
```

Add the OpenVINO target and INT8 calibration dependencies when needed:

```bash
uv sync --extra yolo --extra openvino --extra validation
```

The distribution is `vision-buildkit`; the Python module and CLI are `vision`. Provider and backend dependencies are optional and imported only when needed.

Version 0.2.2 adds the provider-neutral OpenVINO CPU target for static FP32,
FP16, and calibrated INT8 builds. It atomically publishes and re-verifies the
IR XML/BIN pair plus its receipt, while preserving the Vision Contract across
YOLO26, RF-DETR, and RT-DETR decoder ABIs. Only YOLO26n FP32/FP16 currently has
accepted real-model OpenVINO development evidence; INT8 and the DETR families
remain unverified on real models.

Upgrading from 0.1.x, 0.2.0, or 0.2.1? Read [MIGRATING.md](MIGRATING.md) before updating
custom plugins, manifests, or capsules. Version 0.2.0 sealed the
provider-neutral target boundary; 0.2.1 added exact resize semantics; 0.2.2
adds the sealed multi-file OpenVINO target.

## A small workflow

### 1. Describe the behavior, not just tensors

The schema is strict and versioned. This abbreviated YAML declares the input and result semantics needed to reproduce a detector safely:

```yaml
schema_version: 1
task: detection
input:
  semantic_type: image
  color: RGB
  layout: NCHW
  dtype: float32
  input_name: images
  shape: [1, 3, 640, 640]
preprocessing:
  - kind: letterbox
    height: 640
    width: 640
    pad_value: 114
    interpolation: bilinear
    implementation: ultralytics_letterbox_cv2
  - kind: normalize
    mean: [0.0, 0.0, 0.0]
    std: [255.0, 255.0, 255.0]
  - kind: transpose
    order: [0, 3, 1, 2]
result:
  box_format: xyxy
  coordinate_space: original_image
  nms_owner: source
  nms_iou_threshold: 0.7
  score_threshold: 0.25
  class_aware_nms: true
  clip_boxes: true
  max_detections: 300
  suppression: greedy_iou
  ordering: score_desc
  tie_policy: stable
```

The excerpt is illustrative rather than a complete file: a valid contract also pins output tensor roles, class mappings, tolerances, source provenance, and decoding semantics. Validate that complete file before planning a build:

```bash
uv run vision contract validate contract.yaml
```

### 2. Run a provider model

```python
import vision

model = vision.load("ultralytics/yolo26n", device="auto")
result = model("image.jpg")

for box, score, class_id in zip(result.boxes, result.scores, result.classes, strict=True):
    print(class_id, round(score, 3), box.as_xyxy())
```

`DetectionResult` normalizes the public result surface to `boxes`, `scores`, `classes`, `names`, original image size, and runtime metadata. It does not prove that two providers are semantically equivalent; that is the contract and conformance workflow's job.

### 3. Inspect adapters and targets, then plan

```bash
uv run vision plugins list

uv run vision build ultralytics onnxruntime \
  --contract contract.yaml \
  --source-digest "$(shasum -a 256 yolo26n.pt | cut -d' ' -f1)" \
  --precisions fp32 \
  --resolutions 640x640 \
  --batches 1 \
  --dynamics false \
  --nms-owners source
```

`vision build` creates a deterministic candidate plan; it does **not** claim that a target was compiled or validated. The provider-neutral materialization bridge is available programmatically:

```python
from vision import BuildPlanner, VisionContract
from vision.plugins import DEFAULT_REGISTRY

contract = VisionContract.from_path("contract.yaml")
source = DEFAULT_REGISTRY.sources["ultralytics"]
target = DEFAULT_REGISTRY.targets["onnxruntime"]

prepared, exported, artifact = BuildPlanner().materialize(
    source, target, "yolo26n.pt", contract, "artifacts/model.onnx"
)
```

This workflow is intentionally fail-closed: source bytes, contract, exported ONNX, lineage, and target artifact are checked at each boundary.

For OpenVINO, configure the destination and precision explicitly. The target
receives the same neutral ONNX request; it never receives the Ultralytics or
LibreYOLO provider object:

```python
from vision import BuildPlanner, OpenVinoTargetPlugin
from vision.plugins import DEFAULT_REGISTRY

source = DEFAULT_REGISTRY.sources["ultralytics"]
target = OpenVinoTargetPlugin("artifacts/model.xml", precision="fp16")

captured, exported, artifact = BuildPlanner().materialize(
    source, target, "yolo26n.pt", contract, "build/neutral.onnx"
)
assert target.verify_artifact(artifact, contract)
```

An OpenVINO artifact consists of `model.xml`, `model.bin`, and a bound receipt.
FP32/FP16 require no calibration. INT8 requires an immutable
`CalibrationManifest`; availability in the API is not a real-model INT8
verification claim.

`manual-onnx` can capture an existing ONNX file for planning and neutral materialization. It deliberately has no generic reference runner: a parity run still needs a source/runtime adapter that can execute the declared semantics.

### 4. Validate a candidate

For a release gate, compare a source and candidate over a content-bound dataset manifest. Validation records matched boxes, IoU, score drift, class mismatches, lost/added detections, and COCO mAP where labels are available. A candidate that exceeds the requested limit should not be published as a validated capsule.

```bash
COCO80_CLASS_MAP='{"0":1,"1":2,"2":3,"3":4,"4":5,"5":6,"6":7,"7":8,"8":9,"9":10,"10":11,"11":13,"12":14,"13":15,"14":16,"15":17,"16":18,"17":19,"18":20,"19":21,"20":22,"21":23,"22":24,"23":25,"24":27,"25":28,"26":31,"27":32,"28":33,"29":34,"30":35,"31":36,"32":37,"33":38,"34":39,"35":40,"36":41,"37":42,"38":43,"39":44,"40":46,"41":47,"42":48,"43":49,"44":50,"45":51,"46":52,"47":53,"48":54,"49":55,"50":56,"51":57,"52":58,"53":59,"54":60,"55":61,"56":62,"57":63,"58":64,"59":65,"60":67,"61":70,"62":72,"63":73,"64":74,"65":75,"66":76,"67":77,"68":78,"69":79,"70":80,"71":81,"72":82,"73":84,"74":85,"75":86,"76":87,"77":88,"78":89,"79":90}'

uv run vision validate-dataset \
  --dataset frozen-coco-subset.json \
  --output-dir evidence/run-001 \
  --max-map-drop 0.2 \
  --reference-source yolo26n.pt \
  --reference-manifest source-manifest.yaml \
  --candidate-source artifacts/model.onnx \
  --candidate-manifest candidate-manifest.yaml \
  --reference-class-mapping "$COCO80_CLASS_MAP" \
  --candidate-class-mapping "$COCO80_CLASS_MAP"
```

The CLI accepts mappings as inline JSON, not filenames. The example's map is the COCO-80 model-class-to-COCO-category mapping and is valid only when the dataset manifest declares exactly those categories. For another dataset or label order, pass a one-to-one inline JSON mapping that covers exactly its declared categories.

Every observed validation now atomically publishes both
`result/validation_report.json` and the derived canonical
`result/experiment_report.json`, with separate commit markers. The canonical
report recomputes seven gates and binds the dataset, artifact and graph
digests, actual runtime identities, host, exact validation harness, and source
validation report. Observed INT8 validation additionally requires the
candidate's adjacent content-bound build report and calibration digest; a bare
precision label is rejected.

## Vision Capsule

A capsule is the portable result of a build, not the product by itself. A **certified** capsule has this exact top-level layout; the checksum-indexed material filenames are derived from their SHA-256 digests:

```text
model.vcap/
├── capsule.json              # content-addressed index, mode, and runtime axes
├── checksums.json            # inventory of every indexed file
├── manifest.yaml             # runnable artifact projection of the contract
├── vision-contract.json      # canonical semantic Vision Contract
├── certification-report.json # PASS report bound to material and evidence
├── preprocessing.json
├── postprocessing.json
├── validation_report.json
├── benchmarks.json
├── original_weights/
│   └── <source-sha256>.<suffix>
├── artifacts/
│   └── <artifact-sha256>.<suffix>
└── evidence/
    └── <content-bound evidence files>
```

`manifest.yaml` points to the packaged runnable artifact; `vision-contract.json` is the semantic source of truth. Certified evidence includes the declared cases, inputs, predictions, source predictions, dataset and policy records, accuracy report, runtime record, and environment record. Development and release capsules have different exact layouts. Release capsules contain path-free snapshots plus the canonical `evidence/experiment_report.json`; its exact bytes and source-validation digest are bound into `capsule.json`, and publication rejects missing, failed, or mismatched experiment evidence. Only certified capsules contain `vision-contract.json` and `certification-report.json`. Release capsules produced before this evidence binding must be rebuilt before the current verifier will accept them.

Every certified capsule verification requires an expected capsule ID from a trusted registry or explicit out-of-band pin. The ID is an authenticity anchor; capsule checksums establish only internal integrity and cannot authenticate a coherently rewritten capsule. A future signature service could provide an alternative anchor.

```bash
uv run vision validate model.vcap --expected-capsule-id CAPSULE_SHA256
uv run vision certify model.vcap --expected-capsule-id CAPSULE_SHA256
```

`vision certify` does **not** create a certification. It verifies an already packaged capsule and returns `self_attested_conformant: true` only when the capsule is already in certified mode with a PASS certification report. Creating a certified capsule is currently a Python API for a previously created, file-backed PASS report:

```python
from vision.capsule import create_certified_capsule

path = create_certified_capsule(
    "model.vcap", manifest, contract, report, source="source.pt", artifact="model.onnx"
)
```

The API rechecks the report, contract, source, artifact, and evidence before publication. Neither it nor `vision certify` is an independent laboratory certification, a signature service, or a guarantee of fitness for a particular device or use case.

## Non-goals

This MVP does not try to replace any of the following:

- training, fine-tuning, model hosting, annotation, or dataset management;
- a C++ runtime, custom CUDA kernels, or a new compute IR;
- a visual workflow product or distributed serving platform;
- universal support for all model repositories, tasks, hardware, or SDKs.

Detection is the current task focus. Segmentation, pose, mobile/web targets, and custom accelerators remain outside the verified scope.

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check vision tests scripts
uv run pytest -q
uv build
```

Please see [CONTRIBUTING.md](CONTRIBUTING.md). In particular, keep claims about accuracy, latency, hardware compatibility, and certification tied to a reproducible artifact and harness.

## License and legal limits

This project uses the [Vision BuildKit Research and Evaluation License 1.0](LICENSE). It permits non-commercial research, education, benchmarking, evaluation, modification, forks, and qualifying redistribution under the same terms. It does **not** grant a commercial, production, hosted-service, internal-business, or patent license. Third-party providers, weights, datasets, runtimes, and dependencies remain subject to their own terms.

The license is intended to reserve rights, but it cannot guarantee that nobody will copy an idea, file a patent, or misuse the repository. Enforceability and the treatment of patent restrictions vary by jurisdiction and facts. This README and license are informational, not legal advice; obtain qualified legal counsel before relying on them or publishing sensitive intellectual property.

For permissions outside the license, contact the author through [GitHub](https://github.com/aalvsz).
