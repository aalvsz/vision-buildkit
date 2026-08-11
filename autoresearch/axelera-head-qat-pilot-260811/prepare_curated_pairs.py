#!/usr/bin/env python3
"""Copy-on-write exact matched-pair corpus derivation; source remains untouched."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
EXCLUDED={'000000000250':'image_without_label','000000000508':'image_without_label','000000000656':'label_without_image','000000000659':'label_without_image'}
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--regression',type=Path,required=True);p.add_argument('--eval-dir',type=Path,required=True);a=p.parse_args()
 if a.output.exists(): raise SystemExit('fresh output required')
 im=a.source/'images/train2017';la=a.source/'labels/train2017'; I={x.stem:x for x in im.glob('*.jpg')};L={x.stem:x for x in la.glob('*.txt')}; common=sorted(I.keys()&L.keys())
 if len(common)!=126 or set(I)^set(L)!=set(EXCLUDED): raise SystemExit('unexpected corpus cardinality')
 cal={x.strip() for x in a.calibration.read_text().splitlines() if x.strip()}; reg={x.name for x in a.regression.glob('*.jpg')}; ev={x.name for x in a.eval_dir.glob('*.jpg')}
 if any(I[s].name in cal|reg|ev for s in common): raise SystemExit('forbidden overlap')
 records=[]
 for stem in common:
  for src,kind in ((I[stem],'image'),(L[stem],'label')):
   dst=a.output/kind/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);records.append({'stem':stem,'kind':kind,'sha256':h(dst)})
 inventory=[]
 for path in sorted([*im.glob('*.jpg'),*la.glob('*.txt')]): inventory.append({'path':str(path.relative_to(a.source)),'sha256':h(path)})
 manifest={'status':'accepted_curated_pairs','source_root':str(a.source),'source_inventory_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),'source_inventory':inventory,'rule':'sorted exact image-label stem intersection','cardinality':len(common),'included_stems':common,'excluded':EXCLUDED,'records':records,'calibration_binding':{'path':str(a.calibration),'sha256':h(a.calibration),'names':sorted(cal)},'regression_binding':{'root':str(a.regression),'files':[{ 'name':x.name,'sha256':h(x)} for x in sorted(a.regression.glob('*.jpg'))]},'eval_binding':{'root':str(a.eval_dir),'files':[{ 'name':x.name,'sha256':h(x)} for x in sorted(a.eval_dir.glob('*.jpg'))]}}
 (a.output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
