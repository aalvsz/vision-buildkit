#!/usr/bin/env python3
"""Read-only, fail-closed extractor for Voyager terminal-head qparams."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

TARGETS=[f'model.23.cv2.{i}.2' for i in range(3)]+[f'model.23.cv3.{i}.2' for i in range(3)]
ENCODED={f'model.23.cv{branch}.{i}.2': f'model_DOT_23/cv{branch}_DOT_{i}/cv{branch}_DOT_{i}_DOT_2/Conv.0' for branch in (2,3) for i in range(3)}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if len(a.manifest_sha256)!=64 or a.output.exists() or not a.manifest.is_file() or sha(a.manifest)!=a.manifest_sha256: raise SystemExit('FAIL_CLOSED: exact manifest receipt or fresh output')
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
  branch=name.split('.')[2][-1]; index=name.split('.')[3]
  qre_prefix=f'model_dot_23_cv{branch}_dot_{index}_'
  qre=[key for key in modules if key.lower().startswith(qre_prefix) and 'conv_to_output_qre' in key.lower()]
  if not qre: raise SystemExit(f'FAIL_CLOSED: missing output qre for {name}')
  rows.append({'name':name,'encoded_module':encoded,'output_qre_modules':qre,'qparams':{key:(value.tolist() if hasattr(value,'tolist') else value) for key,value in fields.items()}})
 a.output.write_text(json.dumps({'manifest_sha256':sha(a.manifest),'targets':rows},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
