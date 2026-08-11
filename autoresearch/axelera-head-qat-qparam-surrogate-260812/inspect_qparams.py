#!/usr/bin/env python3
"""Read-only, fail-closed extractor for Voyager terminal-head qparams."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

TARGETS=[f'model.23.cv2.{i}.2' for i in range(3)]+[f'model.23.cv3.{i}.2' for i in range(3)]
EXPECTED_MANIFEST_SHA256_PREFIX='bdb83c0b'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists() or not a.manifest.is_file() or not sha(a.manifest).startswith(EXPECTED_MANIFEST_SHA256_PREFIX): raise SystemExit('FAIL_CLOSED: manifest receipt or fresh output')
 from axelera.compiler.quantized_model import AxeleraQuantizedModel
 model=AxeleraQuantizedModel.from_manifest(a.manifest)
 rows=[]
 for name in TARGETS:
  module=dict(model.core_model.named_modules()).get(name)
  if module is None: raise SystemExit(f'FAIL_CLOSED: missing terminal module {name}')
  fields={key:getattr(module,key,None) for key in ('input_scale','input_zero_point','weight_scale','weight_zero_point','output_scale','output_zero_point')}
  rows.append({'name':name,'qparams':{key:(value.tolist() if hasattr(value,'tolist') else value) for key,value in fields.items()}})
 a.output.write_text(json.dumps({'manifest_sha256':sha(a.manifest),'targets':rows},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
