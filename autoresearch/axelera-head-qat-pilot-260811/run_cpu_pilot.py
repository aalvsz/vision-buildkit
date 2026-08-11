#!/usr/bin/env python3
"""Bounded deterministic CPU QAT pilot; intentionally refuses every non-CPU mode."""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from head_qat_pilot import (TARGETS, assert_state_only_targets_changed, export_plain_raw_onnx,
    freeze_except_targets, install_head_fake_qat, load_pinned_yolo, parameter_hashes, seed_everything, sha256_file, state_dict_hashes, one_batch_step)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--device',required=True,choices=['cpu']); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--max-batches',type=int,required=True); p.add_argument('--epochs',type=int,required=True); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset-manifest',type=Path,required=True); p.add_argument('--dataset-manifest-sha256',required=True); p.add_argument('--fresh-output',type=Path,required=True); p.add_argument('--lr',type=float,required=True); p.add_argument('--consistency-weight',type=float,required=True); p.add_argument('--replay-smoke',action='store_true')
    a=p.parse_args()
    if a.epochs != 1 or (not a.replay_smoke and a.max_batches != 126) or (a.replay_smoke and a.max_batches != 1) or a.fresh_output.exists() or not (a.lr>0 and a.consistency_weight>0): raise SystemExit('FAIL_CLOSED: invalid bounded numeric arguments')
    if sha256_file(a.dataset_manifest) != a.dataset_manifest_sha256: raise SystemExit('FAIL_CLOSED: curated manifest SHA mismatch')
    manifest=json.loads(a.dataset_manifest.read_text());
    if manifest.get('status') != 'accepted_curated_pairs' or manifest.get('cardinality') != 126 or len(manifest.get('records',[])) != 252: raise SystemExit('FAIL_CLOSED: malformed curated manifest')
    included=set(manifest.get('included_stems',[])); record_stems={x['stem'] for x in manifest['records']}
    if len(included)!=126 or included != record_stems or not all(key in manifest for key in ('source_inventory','source_inventory_sha256','calibration_binding','regression_binding','eval_binding')): raise SystemExit('FAIL_CLOSED: incomplete provenance binding')
    if hashlib.sha256(json.dumps(manifest['source_inventory'],sort_keys=True).encode()).hexdigest()!=manifest['source_inventory_sha256']: raise SystemExit('FAIL_CLOSED: source inventory digest')
    eval_binding=manifest['eval_binding']
    if not eval_binding.get('files') or not eval_binding.get('root'): raise SystemExit('FAIL_CLOSED: missing eval binding')
    for record in eval_binding['files']:
        path=Path(eval_binding['root'])/record['path']
        if path.is_symlink() or not path.is_file() or sha256_file(path)!=record['sha256']: raise SystemExit('FAIL_CLOSED: sealed eval binding')
    root=a.dataset_manifest.parent
    expected_paths=set()
    for record in manifest.get('records',[]):
        path=root/record['kind']/f"{record['stem']}.{'jpg' if record['kind']=='image' else 'txt'}"
        expected_paths.add(path)
        if path.is_symlink() or not path.is_file() or sha256_file(path)!=record['sha256']: raise SystemExit(f'FAIL_CLOSED: curated file hash {path}')
    actual_paths=set((root/'image').glob('*.jpg'))|set((root/'label').glob('*.txt'))
    if actual_paths != expected_paths: raise SystemExit('FAIL_CLOSED: curated corpus extras or omissions')
    seed_everything(a.seed); a.fresh_output.mkdir(parents=True)
    student=load_pinned_yolo(a.checkpoint,'cpu'); teacher=copy.deepcopy(student).cpu(); teacher_before=state_dict_hashes(teacher)
    install_head_fake_qat(student); trainable=freeze_except_targets(student); before=state_dict_hashes(student)
    optimizer=torch.optim.SGD([x for x in student.parameters() if x.requires_grad],lr=a.lr)
    if {id(x) for g in optimizer.param_groups for x in g['params']} != {id(x) for x in student.parameters() if x.requires_grad}: raise SystemExit('FAIL_CLOSED: optimizer membership')
    # Deterministic, worker-free curated batch. Choose pairs covering small,
    # medium and large normalized boxes, then fill in sorted stem order.
    labels=sorted((root/'label').glob('*.txt'))
    candidates=[]
    for label in labels:
        rows=[line.split() for line in label.read_text().splitlines() if line.strip()]
        for row in rows:
            if len(row)!=5: raise SystemExit('FAIL_CLOSED: malformed YOLO row')
            values=[float(x) for x in row]
            if not all(np.isfinite(values)) or not (0<=values[0]<80) or not all(0<=x<=1 for x in values[1:]) or values[3]<=0 or values[4]<=0: raise SystemExit('FAIL_CLOSED: invalid YOLO row')
        if rows: candidates.append((label.stem, rows))
    bins=[None,None,None]
    for stem, rows in candidates:
        area=float(rows[0][3])*float(rows[0][4]); index=0 if area<.02 else 1 if area<.15 else 2
        if bins[index] is None: bins[index]=(stem,rows)
    coverage=[x for x in bins if x is not None]
    for item in candidates:
        if item not in coverage: coverage.append(item)
        if len(coverage)>=min(16,len(candidates)): break
    # Normal mode is a true epoch iterator: sorted pairs, each exactly once.
    # Replay smoke intentionally uses the fixed multi-scale coverage batch.
    selected_batches=[coverage] if a.replay_smoke else [[item] for item in candidates]
    if not a.replay_smoke and a.max_batches > len(selected_batches): raise SystemExit('FAIL_CLOSED: max-batches exceeds 126-pair epoch')
    def make_batch(selected):
        images=[]; batch_idx=[]; classes=[]; boxes=[]
        for index,(stem,rows) in enumerate(selected):
            array=np.asarray(Image.open(root/'image'/f'{stem}.jpg').convert('RGB').resize((640,640)),dtype=np.float32).transpose(2,0,1)/255
            images.append(torch.from_numpy(array))
            for row in rows:
                batch_idx.append(index); classes.append([float(row[0])]); boxes.append([float(x) for x in row[1:5]])
        return {'img':torch.stack(images),'batch_idx':torch.tensor(batch_idx),'cls':torch.tensor(classes),'bboxes':torch.tensor(boxes)}
    losses=[]
    seen=[]
    for selected in selected_batches[:a.max_batches]:
        seen.extend(stem for stem,_ in selected); losses.append(one_batch_step(student,teacher,make_batch(selected),optimizer,a.consistency_weight))
    if not a.replay_smoke and (seen != sorted(included) or len(seen)!=126 or len(set(seen))!=126): raise SystemExit('FAIL_CLOSED: incomplete epoch iterator')
    changed=assert_state_only_targets_changed(before,student); expected={f'{t}.{x}' for t in TARGETS for x in ('weight','bias')}
    if not expected <= changed or state_dict_hashes(teacher) != teacher_before: raise SystemExit('FAIL_CLOSED: incomplete target mutation or teacher drift')
    from head_qat_pilot import strip_head_fake_qat
    strip_head_fake_qat(student); onnx_path=a.fresh_output/'trained-plain.onnx'; export_plain_raw_onnx(student,onnx_path,torch.zeros((1,3,640,640)))
    import onnx, onnxruntime as ort
    graph=onnx.load(onnx_path); onnx.checker.check_model(graph)
    if [(x.domain,x.version) for x in graph.opset_import] != [('',17)] or graph.graph.output[0].name!='output0' or len(graph.functions) or any(n.op_type in {'QuantizeLinear','DequantizeLinear','QLinearConv'} for n in graph.graph.node): raise SystemExit('FAIL_CLOSED: ONNX gate')
    (a.fresh_output/'result.json').write_text(json.dumps({'losses':losses,'ordered_stems':seen,'epoch_cardinality':len(candidates),'target_hashes':{k:v for k,v in parameter_hashes(student).items() if k in expected},'onnx_sha256':sha256_file(onnx_path),'manifest_sha256':sha256_file(a.dataset_manifest)},sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
