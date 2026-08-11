#!/usr/bin/env python3
"""Bounded deterministic CPU QAT pilot; intentionally refuses every non-CPU mode."""
from __future__ import annotations
import argparse, copy, json, sys
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
    p.add_argument('--dataset-manifest',type=Path,required=True); p.add_argument('--fresh-output',type=Path,required=True); p.add_argument('--lr',type=float,required=True); p.add_argument('--consistency-weight',type=float,required=True)
    a=p.parse_args()
    if a.epochs != 1 or a.max_batches < 1 or a.fresh_output.exists(): raise SystemExit('FAIL_CLOSED: epochs=1, max-batches>=1, and absent fresh output required')
    manifest=json.loads(a.dataset_manifest.read_text());
    if manifest.get('status') != 'accepted_curated_pairs': raise SystemExit('FAIL_CLOSED: immutable curated manifest required')
    seed_everything(a.seed); a.fresh_output.mkdir(parents=True)
    student=load_pinned_yolo(a.checkpoint,'cpu'); teacher=copy.deepcopy(student).cpu(); teacher_before=state_dict_hashes(teacher)
    install_head_fake_qat(student); trainable=freeze_except_targets(student); before=state_dict_hashes(student)
    optimizer=torch.optim.SGD([x for x in student.parameters() if x.requires_grad],lr=a.lr)
    if {id(x) for g in optimizer.param_groups for x in g['params']} != {id(x) for x in student.parameters() if x.requires_grad}: raise SystemExit('FAIL_CLOSED: optimizer membership')
    # Deterministic, worker-free curated batch. Choose pairs covering small,
    # medium and large normalized boxes, then fill in sorted stem order.
    root=a.dataset_manifest.parent; labels=sorted((root/'label').glob('*.txt'))
    candidates=[]
    for label in labels:
        rows=[line.split() for line in label.read_text().splitlines() if line.strip()]
        if rows: candidates.append((label.stem, rows))
    bins=[None,None,None]
    for stem, rows in candidates:
        area=float(rows[0][3])*float(rows[0][4]); index=0 if area<.02 else 1 if area<.15 else 2
        if bins[index] is None: bins[index]=(stem,rows)
    selected=[x for x in bins if x is not None]
    for item in candidates:
        if item not in selected: selected.append(item)
        if len(selected)>=min(16,len(candidates)): break
    images=[]; batch_idx=[]; classes=[]; boxes=[]
    for index,(stem,rows) in enumerate(selected):
        array=np.asarray(Image.open(root/'image'/f'{stem}.jpg').convert('RGB').resize((640,640)),dtype=np.float32).transpose(2,0,1)/255
        images.append(torch.from_numpy(array))
        for row in rows:
            batch_idx.append(index); classes.append([float(row[0])]); boxes.append([float(x) for x in row[1:5]])
    batch={'img':torch.stack(images),'batch_idx':torch.tensor(batch_idx),'cls':torch.tensor(classes),'bboxes':torch.tensor(boxes)}
    losses=[]
    for _ in range(a.max_batches): losses.append(one_batch_step(student,teacher,batch,optimizer,a.consistency_weight))
    changed=assert_state_only_targets_changed(before,student); expected={f'{t}.{x}' for t in TARGETS for x in ('weight','bias')}
    if not expected <= changed or state_dict_hashes(teacher) != teacher_before: raise SystemExit('FAIL_CLOSED: incomplete target mutation or teacher drift')
    from head_qat_pilot import strip_head_fake_qat
    strip_head_fake_qat(student); onnx=a.fresh_output/'trained-plain.onnx'; export_plain_raw_onnx(student,onnx,torch.zeros((1,3,640,640)))
    (a.fresh_output/'result.json').write_text(json.dumps({'losses':losses,'target_hashes':{k:v for k,v in parameter_hashes(student).items() if k in expected},'onnx_sha256':sha256_file(onnx),'manifest_sha256':sha256_file(a.dataset_manifest)},sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
