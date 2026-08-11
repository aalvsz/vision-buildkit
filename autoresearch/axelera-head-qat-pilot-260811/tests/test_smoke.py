import hashlib
from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from head_qat_pilot import *

ROOT=Path('/Users/ander.alvarez/dev/perso/cv-engine')
WEIGHTS=ROOT/'yolo26n.pt'

def test_stock_checkpoint_and_head_qat_mps_smoke(tmp_path):
    from ultralytics import YOLO
    before_file=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    seed_everything(); model=YOLO(WEIGHTS).model.to('mps'); teacher=copy.deepcopy(model).to('mps')
    install_head_fake_qat(model); trainable=freeze_except_targets(model)
    assert {n.rsplit('.',1)[0] for n in trainable} == set(TARGETS)
    before=parameter_hashes(model)
    model.train(); out=model(torch.zeros((1,3,640,640),device='mps'))
    score=out['one2many']['scores']; loss=score.square().mean(); loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None: p.add_(p.grad, alpha=-1e-5)
    changed=changed_parameter_names(before,model); assert changed; assert_only_targets_changed(changed)
    assert all(get_target_modules(model)[n].fake_quant_calls.item()>0 for n in TARGETS)
    strip_head_fake_qat(model)
    output=tmp_path/'pilot.onnx'; sample=torch.zeros((1,3,640,640),device='mps'); export_plain_raw_onnx(model,output,sample)
    import onnx, onnxruntime as ort
    graph=onnx.load(output); onnx.checker.check_model(graph)
    assert [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim] == [1,84,8400]
    assert not any(n.op_type in {'QuantizeLinear','DequantizeLinear','QLinearConv'} for n in graph.graph.node)
    model.cpu().eval(); model.model[-1].end2end=False
    with torch.inference_mode():
        expected=model(torch.zeros((1,3,640,640)))
        expected=(expected[0] if isinstance(expected, tuple) else expected).cpu().numpy()
    actual=ort.InferenceSession(str(output),providers=['CPUExecutionProvider']).run(None,{'images':torch.zeros((1,3,640,640)).numpy()})[0]
    assert abs(expected-actual).max()<2e-3
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==before_file

def test_dataset_manifest(tmp_path):
    root=Path('/Users/ander.alvarez/dev/datasets/coco128-qat-260811/coco128')
    result=build_dataset_manifest(root,Path('/tmp/cv_engine_axelera_calibration_names.txt'),ROOT/'autoresearch/vision-buildkit-phase17-axelera-260731/run/portable-dataset',tmp_path/'dataset.json')
    assert result['file_count']==256 and result['overlap']==[]

def test_ordinary_ultralytics_loss_fails_closed_on_strict_deterministic_mps():
    from ultralytics import YOLO
    seed_everything(); student=YOLO(WEIGHTS).model.to('mps'); teacher=copy.deepcopy(student).to('mps')
    install_head_fake_qat(student); freeze_except_targets(student)
    for parameter in teacher.parameters(): parameter.requires_grad=False
    optimizer=torch.optim.SGD([p for p in student.parameters() if p.requires_grad],lr=1e-5)
    batch={'img':torch.zeros((1,3,640,640),device='mps'),'batch_idx':torch.tensor([0],device='mps'),
           'cls':torch.tensor([[0.]],device='mps'),'bboxes':torch.tensor([[0.5,0.5,0.2,0.2]],device='mps')}
    with pytest.raises(RuntimeError,match='FAIL_CLOSED'):
        one_batch_step(student,teacher,batch,optimizer)
