"""Prepare MetaPKLot parking-space polygons for Ultralytics YOLO segmentation.

The generated split trains on PKLot and validates on CNRPark-EXT, keeping the
two source datasets separate to measure cross-source generalization.
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "external" / "MetaPKLot-Dataset-main"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "metapklot_yolo"
SOURCES = {
    "PKLot": ("train",),
    "CNRPark-EXT": ("val",),
}


def read_archive(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:xz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".json")]
        if len(members) != 1:
            raise ValueError(f"Expected one JSON file in {path}, found {len(members)}")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError(f"Could not read JSON from {path}")
        return json.load(stream)


def image_link(source: Path, relative: str, destination: Path) -> None:
    source_image = (source / relative).resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Image referenced by annotation is missing: {source_image}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source_image)
    except OSError:
        # Cross-volume links and some Windows configurations do not support hard links.
        import shutil
        shutil.copy2(source_image, destination)


def yolo_polygon(points: list[float], width: int, height: int) -> str | None:
    if len(points) < 6 or len(points) % 2:
        return None
    normalized = []
    for x, y in zip(points[::2], points[1::2]):
        normalized.extend((min(1.0, max(0.0, float(x) / width)), min(1.0, max(0.0, float(y) / height))))
    return " ".join(f"{value:.6f}" for value in normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Extracted MetaPKLot repository")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated dataset folder")
    parser.add_argument("--max-images-per-source", type=int, default=0, help="Optional deterministic cap per annotation archive for a small pilot; 0 uses all images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.max_images_per_source < 0:
        parser.error("--max-images-per-source must be zero or greater")
    if not source.is_dir():
        parser.error(f"MetaPKLot source directory not found: {source}")

    image_records: dict[str, dict[str, Any]] = {}
    annotation_records: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0}

    for dataset, (split,) in SOURCES.items():
        archive_dir = source / "annotations" / "original" / "spots" / dataset
        archives = sorted(archive_dir.glob("*.tar.xz"))
        if not archives:
            raise FileNotFoundError(f"No spot annotations found for {dataset} under {archive_dir}")
        for archive_path in archives:
            payload = read_archive(archive_path)
            images = payload.get("images", [])
            if args.max_images_per_source and len(images) > args.max_images_per_source:
                images = sorted(rng.sample(images, args.max_images_per_source), key=lambda item: item["id"])
            selected_ids = {image["id"] for image in images}
            for image in images:
                relative = image["file_name"].replace("\\", "/")
                image_records[relative] = {**image, "split": split}
            by_id: dict[int, list[str]] = {image_id: [] for image_id in selected_ids}
            for annotation in payload.get("annotations", []):
                image_id = annotation.get("image_id")
                if image_id not in selected_ids:
                    continue
                category = int(annotation.get("category_id", -1))
                segmentation = annotation.get("segmentation") or []
                for polygon in segmentation:
                    # Store raw polygon until image dimensions are available below.
                    if len(polygon) >= 6:
                        by_id[image_id].append(f"{category} " + " ".join(map(str, polygon)))
            for image in images:
                relative = image["file_name"].replace("\\", "/")
                annotation_records[relative] = by_id[image["id"]]

    for relative, image in image_records.items():
        split = image["split"]
        image_dest = output / "images" / split / relative
        label_dest = output / "labels" / split / Path(relative).with_suffix(".txt")
        image_link(source, relative, image_dest)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        width, height = int(image["width"]), int(image["height"])
        lines = []
        for raw in annotation_records[relative]:
            category_text, *coordinates = raw.split()
            polygon_text = yolo_polygon([float(value) for value in coordinates], width, height)
            if polygon_text:
                lines.append(f"{category_text} {polygon_text}")
        label_dest.write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1

    data_yaml = output / "dataset.yaml"
    data_yaml.write_text(
        "# MetaPKLot parking-space instance segmentation\n"
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: empty\n  1: occupied\n",
        encoding="utf-8",
    )
    summary = {
        "dataset": "MetaPKLot",
        "source": str(source),
        "split_policy": "PKLot train; CNRPark-EXT validation (source-held-out)",
        "class_names": ["empty", "occupied"],
        "images": counts,
        "max_images_per_source": args.max_images_per_source or None,
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Dataset config: {data_yaml}")


if __name__ == "__main__":
    main()
