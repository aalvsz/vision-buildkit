#!/usr/bin/env python3
"""Create deterministic non-overlapping COCO evaluation ID splits; never downloads by default."""
from __future__ import annotations
import argparse, hashlib, json, shutil, urllib.request
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
    p.add_argument('--download-root',type=Path); p.add_argument('--official-url'); p.add_argument('--fetch',action='store_true')
    args=p.parse_args(); annotations=args.annotations if args.download_root is None else args.download_root / args.annotations.name
    ensure_annotations(annotations,args.official_url,args.fetch)
    source=json.loads(annotations.read_text()); eligible=sorted(i['id'] for i in source['images'] if i['id'] not in EXCLUDED and any(a['image_id']==i['id'] for a in source['annotations']))
    ranked=sorted(eligible,key=lambda i:hashlib.sha256(f'cv-engine-qat-eval-260811:{i}'.encode()).hexdigest())
    selected={'dev':ranked[:10],'heldout':ranked[10:20]}; args.output.mkdir(parents=True,exist_ok=True)
    for split, ids in selected.items():
        subset={k:[x for x in source[k] if (x.get('image_id',x.get('id')) in ids)] for k in ('images','annotations')}; subset['categories']=source['categories']
        (args.output/f'{split}.instances.json').write_text(json.dumps(subset,indent=2,sort_keys=True)+'\n')
        (args.output/f'{split}.manifest.json').write_text(json.dumps({'source_sha256':SOURCE_SHA256,'excluded_ids':sorted(EXCLUDED),'ids':ids},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
