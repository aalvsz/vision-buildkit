#!/usr/bin/env python3
"""Create an auditable local-function rewrite of YOLO26n's six final head Convs.

This is an offline ONNX structural prototype.  It neither quantizes nor compiles
the model.  Each selected 1x1 Conv is replaced by a model-local ONNX function
whose body is the same FP32 Conv over the original activation, weight, and bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import FunctionProto, ModelProto, NodeProto, helper

SOURCE_SHA256 = "db42840c9ae5f1490c3d748f0bc72253d1509ed2c0f5d20b3f3847102596e9a9"
CUSTOM_DOMAIN = "custom_host_onnx_functions"
OUTPUT_SIGNATURE = ("output0", (1, 84, 8400))
TARGET_NODE_NAMES = (
    "/model.23/cv2.0/cv2.0.2/Conv",
    "/model.23/cv3.0/cv3.0.2/Conv",
    "/model.23/cv2.1/cv2.1.2/Conv",
    "/model.23/cv3.1/cv3.1.2/Conv",
    "/model.23/cv2.2/cv2.2.2/Conv",
    "/model.23/cv3.2/cv3.2.2/Conv",
)
CONV_ATTRIBUTE_NAMES = ("dilations", "group", "kernel_shape", "pads", "strides")
EXPECTED_1X1_CONV_ATTRIBUTES = {
    "dilations": (1, 1),
    "group": 1,
    "kernel_shape": (1, 1),
    "pads": (0, 0, 0, 0),
    "strides": (1, 1),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def output_signature(model: ModelProto) -> tuple[str, tuple[int, ...]]:
    if len(model.graph.output) != 1:
        raise AssertionError(f"expected one graph output, found {len(model.graph.output)}")
    output = model.graph.output[0]
    dimensions = tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
    return output.name, dimensions


def conv_attributes(node: NodeProto) -> dict[str, int | tuple[int, ...]]:
    """Return the five Conv attributes with scalar values preserved as scalars."""
    attributes: dict[str, int | tuple[int, ...]] = {}
    for attribute in node.attribute:
        if attribute.name not in CONV_ATTRIBUTE_NAMES:
            raise AssertionError(f"unexpected Conv attribute {attribute.name!r} on {node.name}")
        value = helper.get_attribute_value(attribute)
        attributes[attribute.name] = tuple(value) if isinstance(value, list) else value
    if set(attributes) != set(CONV_ATTRIBUTE_NAMES):
        raise AssertionError(f"{node.name} does not declare all required Conv attributes: {attributes}")
    return attributes


def verify_target_nodes(model: ModelProto) -> list[NodeProto]:
    named = {node.name: node for node in model.graph.node}
    if set(TARGET_NODE_NAMES) - set(named):
        missing = sorted(set(TARGET_NODE_NAMES) - set(named))
        raise AssertionError(f"exact terminal Conv identities missing: {missing}")
    selected = [named[name] for name in TARGET_NODE_NAMES]
    initializers = {initializer.name for initializer in model.graph.initializer}
    for node in selected:
        if node.op_type != "Conv" or node.domain:
            raise AssertionError(f"{node.name} is not a standard-domain Conv")
        if len(node.input) != 3 or len(node.output) != 1:
            raise AssertionError(f"{node.name} does not have activation, weight, bias, and one output")
        if node.input[1] not in initializers or node.input[2] not in initializers:
            raise AssertionError(f"{node.name} does not retain weight/bias initializer identity")
        if conv_attributes(node) != EXPECTED_1X1_CONV_ATTRIBUTES:
            raise AssertionError(f"{node.name} is not the expected FP32 1x1 Conv: {conv_attributes(node)}")
    return selected


def make_function(source_conv: NodeProto, function_name: str) -> FunctionProto:
    """Create one local function with the source Conv's exact attributes."""
    body = helper.make_node("Conv", ["activation", "weight", "bias"], ["output"], name="fp32_1x1_conv")
    body.attribute.extend(source_conv.attribute)
    if conv_attributes(body) != conv_attributes(source_conv):
        raise AssertionError(f"failed to copy exact Conv attributes for {source_conv.name}")
    return helper.make_function(
        CUSTOM_DOMAIN,
        function_name,
        ["activation", "weight", "bias"],
        ["output"],
        [body],
        [helper.make_operatorsetid("", 17)],
        doc_string="Equivalent FP32 1x1 Conv body for one YOLO26n raw detection-head terminal Conv.",
    )


def rewrite(source: ModelProto) -> tuple[ModelProto, list[dict[str, Any]]]:
    if output_signature(source) != OUTPUT_SIGNATURE:
        raise AssertionError(f"source output signature changed: {output_signature(source)}")
    selected = verify_target_nodes(source)
    selected_by_name = {node.name: node for node in selected}
    rewritten = ModelProto()
    rewritten.CopyFrom(source)
    del rewritten.functions[:]
    function_names = {name: f"HostFPDetectHeadConv{index}" for index, name in enumerate(TARGET_NODE_NAMES)}
    rewritten.functions.extend(
        make_function(selected_by_name[name], function_names[name]) for name in TARGET_NODE_NAMES
    )
    if not any(opset.domain == CUSTOM_DOMAIN for opset in rewritten.opset_import):
        rewritten.opset_import.extend([helper.make_operatorsetid(CUSTOM_DOMAIN, 1)])

    replacements: list[NodeProto] = []
    identities: list[dict[str, Any]] = []
    for node in rewritten.graph.node:
        original = selected_by_name.get(node.name)
        if original is None:
            replacement = NodeProto()
            replacement.CopyFrom(node)
            replacements.append(replacement)
            continue
        replacement = helper.make_node(
            function_names[original.name],
            list(original.input),
            list(original.output),
            name=original.name,
            domain=CUSTOM_DOMAIN,
        )
        replacements.append(replacement)
        identities.append(
            {
                "node": original.name,
                "function": function_names[original.name],
                "activation": original.input[0],
                "weight_initializer": original.input[1],
                "bias_initializer": original.input[2],
                "output": original.output[0],
                "conv_attributes": conv_attributes(original),
            }
        )
    del rewritten.graph.node[:]
    rewritten.graph.node.extend(replacements)
    return rewritten, identities


def verify_rewrite(source: ModelProto, rewritten: ModelProto, identities: list[dict[str, Any]]) -> None:
    if output_signature(rewritten) != OUTPUT_SIGNATURE:
        raise AssertionError(f"rewritten output signature changed: {output_signature(rewritten)}")
    if len(identities) != len(TARGET_NODE_NAMES):
        raise AssertionError("did not rewrite exactly six terminal Conv nodes")
    source_nodes = {node.name: node for node in source.graph.node}
    rewritten_nodes = {node.name: node for node in rewritten.graph.node}
    for identity in identities:
        before = source_nodes[identity["node"]]
        after = rewritten_nodes[identity["node"]]
        if after.domain != CUSTOM_DOMAIN or after.op_type != identity["function"]:
            raise AssertionError(f"{identity['node']} was not replaced by its local function")
        if list(after.input) != list(before.input) or list(after.output) != list(before.output):
            raise AssertionError(f"{identity['node']} changed tensor or initializer identity")
    functions = {(function.domain, function.name): function for function in rewritten.functions}
    for identity in identities:
        before = source_nodes[identity["node"]]
        function = functions.get((CUSTOM_DOMAIN, identity["function"]))
        if function is None:
            raise AssertionError(f"missing local function {identity['function']}")
        if list(function.input) != ["activation", "weight", "bias"] or list(function.output) != ["output"]:
            raise AssertionError(f"unexpected local-function signature for {identity['function']}")
        if len(function.node) != 1 or function.node[0].op_type != "Conv" or function.node[0].domain:
            raise AssertionError(f"unexpected local-function body for {identity['function']}")
        if list(function.node[0].input) != ["activation", "weight", "bias"]:
            raise AssertionError(f"local function {identity['function']} does not preserve Conv inputs")
        if conv_attributes(function.node[0]) != conv_attributes(before):
            raise AssertionError(f"local function {identity['function']} does not preserve exact Conv attributes")
    qdq = {"QuantizeLinear", "DequantizeLinear"}
    if any(node.op_type in qdq for node in rewritten.graph.node):
        raise AssertionError("rewrite introduced Q/DQ nodes")
    onnx.checker.check_model(rewritten)


def try_ort_parity(source_path: Path, output_path: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        return {"status": "BLOCKED", "blocker": "onnxruntime is unavailable in the executing Python environment"}
    try:
        source_session = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
        rewritten_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    except Exception as error:
        return {
            "status": "BLOCKED",
            "blocker": (
                "onnxruntime cannot execute model-local custom-domain functions: "
                f"{type(error).__name__}: {error}"
            ),
        }
    input_meta = source_session.get_inputs()[0]
    rng = np.random.default_rng(260811)
    input_array = rng.standard_normal((1, 3, 640, 640), dtype=np.float32)
    source_output = source_session.run(None, {input_meta.name: input_array})
    rewritten_output = rewritten_session.run(None, {input_meta.name: input_array})
    maximum_absolute_difference = max(
        float(np.max(np.abs(left - right))) for left, right in zip(source_output, rewritten_output, strict=True)
    )
    if maximum_absolute_difference != 0.0:
        raise AssertionError(f"ORT parity failure: max abs diff {maximum_absolute_difference}")
    return {"status": "PASS", "onnxruntime_version": ort.__version__, "max_abs_diff": maximum_absolute_difference}


def default_source() -> Path:
    return Path(__file__).resolve().parents[3] / "yolo26n-raw.onnx"


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--output", type=Path, default=here / "yolo26n-raw-host-fp-detect-head.onnx")
    parser.add_argument("--proof", type=Path, default=here / "rewrite-proof.json")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise SystemExit("refusing to overwrite the source model")
    if sha256_file(args.source) != SOURCE_SHA256:
        raise SystemExit("source SHA-256 does not match the pinned raw YOLO26n graph")
    source = onnx.load(args.source, load_external_data=False)
    rewritten, identities = rewrite(source)
    verify_rewrite(source, rewritten, identities)
    first_bytes = rewritten.SerializeToString(deterministic=True)
    second_rewrite, _ = rewrite(source)
    second_bytes = second_rewrite.SerializeToString(deterministic=True)
    if first_bytes != second_bytes:
        raise AssertionError("rewrite is not byte-deterministic")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(first_bytes)
    persisted = onnx.load(args.output, load_external_data=False)
    verify_rewrite(source, persisted, identities)
    parity = try_ort_parity(args.source, args.output)
    proof = {
        "schema_version": 1,
        "source": {
            "path": str(args.source.resolve()),
            "sha256": SOURCE_SHA256,
            "output_signature": list(OUTPUT_SIGNATURE),
        },
        "rewritten": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "output_signature": list(output_signature(persisted)),
        },
        "custom_domain": CUSTOM_DOMAIN,
        "rewritten_terminal_convs": identities,
        "onnx_checker": "PASS",
        "deterministic_serialization": "PASS",
        "qdq_introduced": False,
        "ort_parity": parity,
    }
    args.proof.parent.mkdir(parents=True, exist_ok=True)
    args.proof.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps(proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
