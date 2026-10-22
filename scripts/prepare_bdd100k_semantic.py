"""Prepare a bounded BDD100K semantic-segmentation subset for SegFormer.

Expected official BDD100K layout::

    bdd100k/images/10k/{train,val}/*.jpg
    bdd100k/labels/sem_seg/masks/{train,val}/*.png

BDD100K semantic IDs already match Cityscapes' 19-class IDs. Images and masks
are resized together to the pretrained model input shape; labels outside 0..18
remain the ignore value 255.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/external/bdd100k"
DEFAULT_OUTPUT = ROOT / "data/training/bdd100k_semantic"
SIZE = (1024, 512)  # width, height; matches the pretrained SegFormer processor
IGNORE = 255
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def resize_pair(image_path: Path, mask_path: Path, out_image: Path, out_mask: Path) -> bool:
    # Normalize EXIF orientation before comparing against annotation geometry.
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size:
        print(f"Skipping mismatched image/mask pair: {image_path.name}: {image.size} vs {mask.size}", flush=True)
        return False
    # Match the pretrained image processor's target dimensions. Nearest-neighbor
    # resizing keeps semantic class IDs intact.
    image = image.resize(SIZE, Image.Resampling.BILINEAR)
    mask = mask.resize(SIZE, Image.Resampling.NEAREST)
    labels = np.asarray(mask, dtype=np.uint8).copy()
    labels[~np.isin(labels, np.arange(len(CLASS_NAMES)))] = IGNORE
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_image, quality=90)
    Image.fromarray(labels, mode="L").save(out_mask, optimize=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Root containing bdd100k/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-limit", type=int, default=1500, help="Maximum train pairs; 0 uses all")
    parser.add_argument("--val-limit", type=int, default=300, help="Maximum validation pairs; 0 uses all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error("sample limits must be non-negative")

    source = args.source
    if (source / "bdd100k").is_dir():
        source = source / "bdd100k"
    rng = random.Random(args.seed)
    counts: dict[str, int] = {}
    skipped_pairs: dict[str, list[str]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        image_dir = source / "images" / "10k" / split
        mask_dir = source / "labels" / "sem_seg" / "masks" / split
        if not image_dir.is_dir() or not mask_dir.is_dir():
            parser.error(f"Expected BDD100K image/mask folders under {source}; missing split '{split}'")
        image_paths = sorted(image_dir.glob("*.jpg"))
        pairs = [(img, mask_dir / f"{img.stem}.png") for img in image_paths]
        pairs = [(img, mask) for img, mask in pairs if mask.is_file()]
        if not pairs:
            parser.error(f"No paired BDD100K {split} images and semantic masks found")
        rng.shuffle(pairs)
        if limit:
            pairs = pairs[:limit]
        counts[split] = 0
        skipped_pairs[split] = []
        class_counts[split] = {str(i): 0 for i in range(len(CLASS_NAMES))}
        for image_path, mask_path in pairs:
            out_image = args.output / "images" / split / f"{image_path.stem}.jpg"
            out_mask = args.output / "masks" / split / f"{image_path.stem}.png"
            if not resize_pair(image_path, mask_path, out_image, out_mask):
                skipped_pairs[split].append(image_path.name)
                continue
            labels = np.asarray(Image.open(out_mask), dtype=np.uint8)
            present, frequencies = np.unique(labels[labels != IGNORE], return_counts=True)
            for class_id, frequency in zip(present.tolist(), frequencies.tolist()):
                class_counts[split][str(class_id)] += int(frequency)
            counts[split] += 1
        print(f"Prepared {counts[split]} BDD100K {split} image/mask pairs", flush=True)

    manifest = {
        "dataset": "BDD100K semantic segmentation",
        "source": "https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst",
        "source_root": str(source),
        "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "road_id": 0,
        "sidewalk_id": 1,
        "vegetation_id": 8,
        "terrain_id": 9,
        "ignore_index": IGNORE,
        "image_size": list(SIZE),
        "images": counts,
        "skipped_size_mismatches": skipped_pairs,
        "class_pixel_counts": class_counts,
        "seed": args.seed,
        "license_note": "Use only after accepting the official BDD100K dataset license; do not redistribute source images or masks.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"images": counts, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
