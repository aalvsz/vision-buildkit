from pathlib import Path

import numpy as np
import pytest


def test_project_metadata_advertises_python_310_support():
    metadata = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert 'requires-python = ">=3.10,<3.14"' in metadata
    assert '"Programming Language :: Python :: 3.10"' in metadata
    assert 'rfdetr = ["rfdetr[onnx]==1.9.0"]' in metadata


def test_onnx_model_created_by_installed_onnx_executes_in_ort(tmp_path):
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    model_path = tmp_path / "identity.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)

    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert np.array_equal(session.run(None, {"input": np.array([3.0], dtype=np.float32)})[0], np.array([3.0]))
