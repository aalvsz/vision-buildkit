# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting for this repository:

1. Open the repository's **Security** tab.
2. Select **Report a vulnerability**.
3. Include affected versions, impact, reproduction steps, and any suggested
   mitigation.

You should receive an initial acknowledgement within seven days. No fixed
remediation timeline is promised; timing depends on severity and the scope of
the affected runtime.

## Supported versions

Until the project reaches a stable release, security fixes apply to the latest
commit on `main`. Older commits and generated model artifacts are not
independently supported.

## Model artifacts

Model checkpoints, ONNX graphs, TensorRT engines, OpenVINO artifacts, manifests,
and Vision Capsules should be treated as untrusted input unless their expected
content digest comes from a trusted source.
