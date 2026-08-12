#!/usr/bin/env python3
"""Read-only, fail-closed extractor for Voyager terminal-head qparams."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np

TARGETS=[f'model.23.cv2.{i}.2' for i in range(3)]+[f'model.23.cv3.{i}.2' for i in range(3)]
ENCODED={f'model.23.cv{branch}.{i}.2': f'model_DOT_23/cv{branch}_DOT_{i}/cv{branch}_DOT_{i}_DOT_2/Conv.0' for branch in (2,3) for i in range(3)}
EXPECTED_MANIFEST_SHA256='bdb83c0ba59e480793b48ea09512023218209e56b6ff79ee7d5528186c787949'
EXPECTED_QUANTIZED_MODEL_SHA256='d3f55bc8db137b687a049f87c3be0905ba297b3d91f938365d43e0b9faedb4fe'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.manifest=a.manifest.resolve();a.output=a.output.resolve()
 if a.output.exists() or not a.manifest.is_file() or sha(a.manifest)!=EXPECTED_MANIFEST_SHA256: raise SystemExit('FAIL_CLOSED: sealed manifest receipt')
 payload=json.loads(a.manifest.read_text()); model_path=(a.manifest.parent/payload['quantized_model_file']).resolve() if 'quantized_model_file' in payload else None
 if model_path is None or not model_path.is_file() or sha(model_path)!=EXPECTED_QUANTIZED_MODEL_SHA256: raise SystemExit('FAIL_CLOSED: sealed quantized model receipt')
 from axelera.compiler.quantized_model import AxeleraQuantizedModel
 from axelera.types.manifest import Manifest
 os.chdir(a.manifest.parent)
 model=AxeleraQuantizedModel.load(Manifest.from_json(a.manifest.read_text()))
 modules=dict(model.core_model.named_modules())
 rows=[]
 for name in TARGETS:
  encoded=ENCODED[name]
  module=modules.get(encoded)
  if module is None: raise SystemExit(f'FAIL_CLOSED: missing terminal module {name}')
  fields={key:getattr(module,key,None) for key in ('input_scale','input_zero_point','weight_scale','weight_zero_point','output_scale','output_zero_point')}
  if any(value is None for value in fields.values()): raise SystemExit(f'FAIL_CLOSED: missing qparam field {name}')
  for key,value in fields.items():
   values=np.asarray(value).reshape(-1) if hasattr(value,'__array__') else [value]
   if not all(math.isfinite(float(x)) for x in values) or ('scale' in key and not all(float(x)>0 for x in values)): raise SystemExit(f'FAIL_CLOSED: invalid qparam {name}:{key}')
  branch=name.split('.')[2][-1]; index=name.split('.')[3]
  qre_name=f'model_dot_23_cv{branch}_dot_{index}_cv{branch}_dot_{index}_dot_2_conv_to_output_qre'
  if qre_name not in modules: raise SystemExit(f'FAIL_CLOSED: expected exact output qre for {name}')
  qre_module=modules[qre_name]; qre_fields={key:getattr(qre_module,key,None) for key in ('scale','zero_point','output_scale','output_zero_point')}
  direct=(qre_fields['scale'],qre_fields['zero_point']); output=(qre_fields['output_scale'],qre_fields['output_zero_point'])
  direct_present=tuple(value is not None for value in direct); output_present=tuple(value is not None for value in output)
  if direct_present not in {(False,False),(True,True)} or output_present not in {(False,False),(True,True)} or (all(direct_present) == all(output_present)):
   raise SystemExit(f'FAIL_CLOSED: ambiguous or incomplete QRE affine pair {name}')
  for key,value in qre_fields.items():
   if value is not None:
    values=np.asarray(value).reshape(-1) if hasattr(value,'__array__') else [value]
    if not all(math.isfinite(float(x)) for x in values) or ('scale' in key and not all(float(x)>0 for x in values)): raise SystemExit(f'FAIL_CLOSED: invalid QRE qparam {name}:{key}')
  rows.append({'name':name,'encoded_module':encoded,'output_qre_module':qre_name,'output_qre_qparams':{key:(value.tolist() if hasattr(value,'tolist') else value) for key,value in qre_fields.items()},'qparams':{key:(value.tolist() if hasattr(value,'tolist') else value) for key,value in fields.items()}})
 with a.output.open('x') as handle:
  handle.write(json.dumps({'manifest_sha256':sha(a.manifest),'targets':rows},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
