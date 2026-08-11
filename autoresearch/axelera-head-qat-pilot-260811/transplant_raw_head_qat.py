#!/usr/bin/env python3
"""Fail-closed transplant of twelve trained Detect-head tensors into frozen raw ONNX."""
from __future__ import annotations
import argparse, copy, hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

TARGETS=tuple([f'model.23.cv2.{i}.2' for i in range(3)]+[f'model.23.cv3.{i}.2' for i in range(3)])
RAW_SHA='db42840c9ae5f1490c3d748f0bc72253d1509ed2c0f5d20b3f3847102596e9a9'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--trained-state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if sha(a.raw)!=RAW_SHA or a.output.exists(): raise SystemExit('FAIL_CLOSED: raw pin or fresh output')
 import torch
 state=torch.load(a.trained_state,map_location='cpu',weights_only=False)
 state=state.get('model',state).state_dict() if hasattr(state.get('model',state),'state_dict') else state.get('state_dict',state)
 needed={f'{target}.{field}' for target in TARGETS for field in ('weight','bias')}
 if set(state)<needed: raise SystemExit('FAIL_CLOSED: trained state lacks exact twelve tensors')
 model=onnx.load(a.raw); before=copy.deepcopy(model)
 initializers={x.name:x for x in model.graph.initializer}; replaced=[]
 for target in TARGETS:
  for field in ('weight','bias'):
   name=f'{target}.{field}'
   if name not in initializers: raise SystemExit(f'FAIL_CLOSED: raw initializer missing {name}')
   initializers[name].CopyFrom(numpy_helper.from_array(state[name].detach().cpu().numpy(),name)); replaced.append(name)
 onnx.checker.check_model(model)
 if [(x.domain,x.version) for x in model.opset_import] != [('',17)] or len(model.graph.output)!=1 or model.graph.output[0].name!='output0' or len(model.functions) or any(n.op_type in {'QuantizeLinear','DequantizeLinear','QLinearConv'} for n in model.graph.node): raise SystemExit('FAIL_CLOSED: graph contract')
 # Graph topology/IO/non-target tensors must be bit-identical to frozen raw.
 for old,new in zip(before.graph.node,model.graph.node):
  if old.SerializeToString()!=new.SerializeToString(): raise SystemExit('FAIL_CLOSED: topology mutation')
 old_init={x.name:x.SerializeToString() for x in before.graph.initializer}
 for item in model.graph.initializer:
  if item.name not in replaced and old_init[item.name]!=item.SerializeToString(): raise SystemExit(f'FAIL_CLOSED: non-target initializer {item.name}')
 onnx.save(model,a.output); print(sha(a.output))
if __name__=='__main__':main()
