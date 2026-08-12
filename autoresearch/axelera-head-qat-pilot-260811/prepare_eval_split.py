#!/usr/bin/env python3
"""Create deterministic non-overlapping COCO evaluation ID splits; never downloads by default."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path

SOURCE_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
EXCLUDED = {6818,37777,87038,174482,252219,331352,397133,403385,458054,480985,48153,81061,141821,170955,188296,256868,365642,441247,498807,529528}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ensure_annotations(path: Path, url: str | None, fetch: bool) -> None:
    if not path.exists():
        if not (fetch and url): raise RuntimeError("annotations missing; pass --fetch --official-url explicitly")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    if digest(path) != SOURCE_SHA256: raise RuntimeError("annotation SHA-256 mismatch")

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--annotations',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--download-root',type=Path); p.add_argument('--official-url'); p.add_argument('--fetch',action='store_true'); p.add_argument('--training-manifest',type=Path); p.add_argument('--calibration-snapshot',type=Path); p.add_argument('--regression-dir',type=Path)
    args=p.parse_args();
    if not args.fetch or not args.official_url or args.output.exists(): raise RuntimeError('sealed artifact requires --fetch --official-url and a fresh output root')
    if not (args.training_manifest and args.calibration_snapshot and args.regression_dir): raise RuntimeError('training/calibration/regression bindings required')
    annotations=args.annotations if args.download_root is None else args.download_root / args.annotations.name
    ensure_annotations(annotations,args.official_url,args.fetch)
    source=json.loads(annotations.read_text()); eligible=sorted(i['id'] for i in source['images'] if i['id'] not in EXCLUDED and any(a['image_id']==i['id'] for a in source['annotations']))
    ranked=sorted(eligible,key=lambda i:hashlib.sha256(f'cv-engine-qat-eval-260811:{i}'.encode()).hexdigest())
    selected={'dev':ranked[:10],'heldout':ranked[10:20]}; args.output.mkdir(parents=True,exist_ok=True)
    selected_names={split:{f'{i:012d}.jpg' for i in ids} for split,ids in selected.items()}
    if selected_names['dev'] & selected_names['heldout']: raise RuntimeError('split overlap')
    forbidden=set()
    if args.calibration_snapshot: forbidden|={x.strip() for x in args.calibration_snapshot.read_text().splitlines() if x.strip()}
    if args.regression_dir: forbidden|={x.name for x in args.regression_dir.glob('*.jpg')}
    if args.training_manifest:
        train=json.loads(args.training_manifest.read_text())
        forbidden|={f"{stem}.jpg" for stem in train.get('included_stems',[])}
    if forbidden & (selected_names['dev']|selected_names['heldout']): raise RuntimeError('forbidden evaluation overlap')
    for split, ids in selected.items():
        subset={k:[x for x in source[k] if (x.get('image_id',x.get('id')) in ids)] for k in ('images','annotations')}; subset['categories']=source['categories']
        (args.output/f'{split}.instances.json').write_text(json.dumps(subset,indent=2,sort_keys=True)+'\n')
        annotation=args.output/f'{split}.instances.json'; image_records=[]
        image_dir=args.output/split/'images';image_dir.mkdir(parents=True,exist_ok=True)
        for ident in ids:
            target=image_dir/f'{ident:012d}.jpg'; urllib.request.urlretrieve(args.official_url.rstrip('/')+'/'+target.name,target); image_records.append({'name':target.name,'sha256':digest(target)})
        if len(image_records)!=10: raise RuntimeError('sealed split requires ten images')
        regression_inventory=[{'name':x.name,'sha256':digest(x)} for x in sorted(args.regression_dir.glob('*.jpg'))]
        (args.output/f'{split}.manifest.json').write_text(json.dumps({'source_sha256':SOURCE_SHA256,'excluded_ids':sorted(EXCLUDED),'ids':ids,'annotation_sha256':digest(annotation),'image_count':len(image_records),'images':image_records,'training_manifest_sha256':digest(args.training_manifest),'calibration_snapshot_sha256':digest(args.calibration_snapshot),'regression_inventory_sha256':hashlib.sha256(json.dumps(regression_inventory,sort_keys=True).encode()).hexdigest(),'fetch_url':args.official_url,'generator_commit':'cce91a04bbf40bf6765770edc41ee3dddde58aff'},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
