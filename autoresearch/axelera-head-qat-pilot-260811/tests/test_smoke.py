import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import head_qat_pilot as pilot
from head_qat_pilot import (
    TARGETS,
    VoyagerIntegerSTEQATConv2d,
    assert_only_targets_changed,
    assert_state_only_targets_changed,
    build_dataset_manifest,
    changed_parameter_names,
    export_plain_raw_onnx,
    freeze_except_targets,
    get_target_modules,
    install_head_fake_qat,
    load_voyager_qparams,
    one_batch_step,
    parameter_hashes,
    seed_everything,
    state_dict_hashes,
    strip_head_fake_qat,
)

ROOT = Path("/Users/ander.alvarez/dev/perso/cv-engine")
WEIGHTS = ROOT / "yolo26n.pt"


def test_stock_checkpoint_and_head_qat_mps_smoke(tmp_path):
    from ultralytics import YOLO

    before_file = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    seed_everything()
    model = YOLO(WEIGHTS).model.to("mps")
    install_head_fake_qat(model)
    trainable = freeze_except_targets(model)
    assert {n.rsplit(".", 1)[0] for n in trainable} == set(TARGETS)
    before = parameter_hashes(model)
    model.train()
    out = model(torch.zeros((1, 3, 640, 640), device="mps"))
    score = out["one2many"]["scores"]
    loss = score.square().mean()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad, alpha=-1e-5)
    changed = changed_parameter_names(before, model)
    assert changed
    assert_only_targets_changed(changed)
    assert all(get_target_modules(model)[n].fake_quant_calls.item() > 0 for n in TARGETS)
    strip_head_fake_qat(model)
    output = tmp_path / "pilot.onnx"
    sample = torch.zeros((1, 3, 640, 640), device="mps")
    export_plain_raw_onnx(model, output, sample)
    import onnx
    import onnxruntime as ort

    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    assert [(x.domain, x.version) for x in graph.opset_import] == [("", 17)]
    assert (
        graph.graph.output[0].name == "output0"
        and graph.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    )
    assert len(graph.functions) == 0
    assert [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim] == [1, 84, 8400]
    assert not any(n.op_type in {"QuantizeLinear", "DequantizeLinear", "QLinearConv"} for n in graph.graph.node)
    model.cpu().eval()
    model.model[-1].end2end = False
    with torch.inference_mode():
        expected = model(torch.zeros((1, 3, 640, 640)))
        expected = (expected[0] if isinstance(expected, tuple) else expected).cpu().numpy()
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
        None, {"images": torch.zeros((1, 3, 640, 640)).numpy()}
    )[0]
    assert abs(expected - actual).max() < 2e-3
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == before_file


def test_dataset_manifest(tmp_path):
    root = Path("/Users/ander.alvarez/dev/datasets/coco128-qat-260811/coco128")
    with pytest.raises(RuntimeError, match="stem mismatch"):
        build_dataset_manifest(
            root,
            Path(__file__).parents[1] / "calibration-names.snapshot.txt",
            ROOT / "autoresearch/vision-buildkit-phase17-axelera-260731/run/portable-dataset",
            tmp_path / "dataset.json",
        )


def test_ordinary_ultralytics_loss_fails_closed_on_strict_deterministic_mps():
    from ultralytics import YOLO

    seed_everything()
    student = YOLO(WEIGHTS).model.to("mps")
    teacher = copy.deepcopy(student).to("mps")
    install_head_fake_qat(student)
    freeze_except_targets(student)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=1e-5)
    batch = {
        "img": torch.zeros((1, 3, 640, 640), device="mps"),
        "batch_idx": torch.tensor([0], device="mps"),
        "cls": torch.tensor([[0.0]], device="mps"),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device="mps"),
    }
    with pytest.raises(RuntimeError, match="FAIL_CLOSED"):
        one_batch_step(student, teacher, batch, optimizer)


def _cpu_one_batch():
    from ultralytics import YOLO

    seed_everything()
    student = YOLO(WEIGHTS).model.cpu()
    teacher = copy.deepcopy(student).cpu()
    teacher_before = state_dict_hashes(teacher)
    install_head_fake_qat(student)
    freeze_except_targets(student)
    before = state_dict_hashes(student)
    optimizer = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=1e-5)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in student.parameters() if p.requires_grad
    }
    batch = {
        "img": torch.zeros((1, 3, 640, 640)),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    values = one_batch_step(student, teacher, batch, optimizer)
    changed = assert_state_only_targets_changed(before, student)
    assert state_dict_hashes(teacher) == teacher_before and not teacher.training
    assert all(
        torch.isfinite(p).all() and p.grad is not None and torch.isfinite(p.grad).all()
        for p in student.parameters()
        if p.requires_grad
    )
    return values, changed, parameter_hashes(student)


def test_two_seeded_cpu_one_batches_are_reproducible_and_target_only():
    first = _cpu_one_batch()
    second = _cpu_one_batch()
    assert first[0] == second[0] and first[2] == second[2]
    assert any(name.endswith((".weight", ".bias")) for name in first[1])


def test_voyager_integer_surrogate_preserves_integer_forward_and_gradient():
    source = torch.nn.Conv2d(2, 1, 1, bias=True)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[[[1.0]], [[-1.0]]]]))
        source.bias.fill_(0.5)
    qparams = {
        "input_scale": [1.0],
        "input_zero_point": [0.0],
        "weight_scale": [1.0],
        "weight_zero_point": [0.0],
        "output_qre_scale": [1.0],
        "output_qre_zero_point": [0.0],
    }
    module = VoyagerIntegerSTEQATConv2d(source, qparams)
    value = torch.tensor([[[[2.0]], [[1.0]]]], requires_grad=True)
    output = module(value)
    assert output.item() == 1.0
    output.backward()
    assert module.weight.grad is not None
    assert torch.isfinite(module.weight.grad).all()


def test_custom_voyager_integer_operator_has_exact_forward_and_bounded_backward():
    value = torch.tensor([-200.0, -127.4, 0.4, 200.0], requires_grad=True)
    output = pilot._affine_quant_integer_bounded_ste(value, torch.tensor([1.0]), torch.tensor([0.0]))
    assert output.tolist() == [-128.0, -127.0, 0.0, 127.0]
    output.sum().backward()
    assert value.grad is not None
    assert torch.allclose(value.grad, torch.tensor([0.05, 1.0, 1.0, 0.05]))


def test_voyager_qparams_include_accumulator_boundary():
    qparams = load_voyager_qparams()
    assert set(qparams) == set(TARGETS)
    for row in qparams.values():
        assert len(row["output_scale"]) == len(row["weight_scale"])
        assert len(row["output_zero_point"]) == len(row["weight_scale"])
        assert len(row["output_qre_scale"]) == 1
        assert len(row["output_qre_zero_point"]) == 1
