"""Create a content-bound COCO subset manifest from official annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from vision.manifest import VisionManifest
from vision.validation import DatasetManifest, _digest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    annotation_output = args.output / "instances_val2017-subset.json"
    manifest_output = args.output / "dataset-manifest.json"
    mapping_output = args.output / "class-mapping.json"
    if any(path.exists() for path in (annotation_output, manifest_output, mapping_output)):
        raise FileExistsError("subset outputs already exist")

    source = json.loads(args.annotations.read_text())
    available = []
    for item in source["images"]:
        path = (args.images / item["file_name"]).absolute()
        if path.is_file():
            available.append((item, path))
        if len(available) == args.count:
            break
    if len(available) != args.count:
        raise ValueError(f"found {len(available)} of {args.count} requested image files")
    selected_ids = {item["id"] for item, _ in available}
    annotations = [item for item in source["annotations"] if item["image_id"] in selected_ids]
    subset = {
        "info": source.get("info", {}),
        "licenses": source.get("licenses", []),
        "images": [item for item, _ in available],
        "annotations": annotations,
        "categories": source["categories"],
    }
    annotation_output.write_text(json.dumps(subset, sort_keys=True, separators=(",", ":")))

    entries = []
    for item, path in available:
        with Image.open(path) as image:
            if image.size != (item["width"], item["height"]):
                raise ValueError(f"image dimensions differ from COCO metadata: {path}")
        entries.append(
            {
                "id": item["id"],
                "path": str(path),
                "sha256": sha256(path),
                "width": item["width"],
                "height": item["height"],
            }
        )
    category_ids = [int(item["id"]) for item in source["categories"]]
    class_mapping = {str(index): category_id for index, category_id in enumerate(category_ids)}
    model = VisionManifest.from_yaml(args.model_manifest)
    body = {
        "format": "coco",
        "split": f"val2017-first-{args.count}",
        "annotations_path": str(annotation_output.absolute()),
        "annotations_sha256": sha256(annotation_output),
        "images": entries,
        "categories": category_ids,
        "preprocessing_digest": _digest(
            {
                "input": model.input.model_dump(mode="json"),
                "preprocessing": model.preprocessing.model_dump(mode="json"),
            }
        ),
        "postprocessing_digest": _digest(
            {
                "output": model.output.model_dump(mode="json"),
                "decoder": model.decoder.model_dump(mode="json"),
                "postprocessing": model.postprocessing,
                "class_mapping": class_mapping,
            }
        ),
        "expected_image_count": len(entries),
        "expected_annotation_count": len(annotations),
        "cocoeval_iou_type": "bbox",
        "cocoeval_version": "pycocotools",
    }
    manifest = DatasetManifest.model_validate(body | {"digest": _digest(body)})
    manifest_output.write_text(manifest.model_dump_json(indent=2))
    mapping_output.write_text(json.dumps(class_mapping, sort_keys=True, indent=2))
    print(manifest_output)


if __name__ == "__main__":
    main()
