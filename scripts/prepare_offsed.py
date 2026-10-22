"""Fetch only OFFSED-annotated images and convert labels for SegFormer.

OFFSED has 203 pixel-labeled frames. The helper reads the remote ZIP directory
and extracts only images paired with masks, using HTTP byte ranges.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import random
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "data/external/OFFSED/devkit/devkit_offsed/SegmentationClass"
DEFAULT_OUTPUT = ROOT / "data/training/offsed_semantic"
IMAGE_ARCHIVE = "https://www.dfki.uni-kl.de/~neigel/datasets/OPEDD_OFFSED/left.zip"
LABEL_RGB = {
    "road": (128, 64, 128),            # OFFSED drivable pavement
    "sidewalk": (244, 35, 232),        # OFFSED non-drivable pavement proxy
    "grass": (24, 226, 31),
}
IGNORE = 255
SIZE = (512, 512)
DOWNLOAD_WORKERS = 8


class HTTPRangeReader(io.RawIOBase):
    """Seekable range reader so zipfile can access selected entries only."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.cache_start, self.cache = 0, b""
        self.block_size = 4 * 1024 * 1024

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError("Invalid seek mode")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        if size <= 0 or self.position >= self.size:
            return b""
        remaining = min(size, self.size - self.position)
        chunks = []
        while remaining:
            start = self.position
            if not (start >= self.cache_start and start < self.cache_start + len(self.cache)):
                block_start = (start // self.block_size) * self.block_size
                block_end = min(self.size - 1, block_start + self.block_size - 1)
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={block_start}-{block_end}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    self.cache = response.read()
                    content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {block_start}-"):
                    raise OSError(f"Server did not honor range request: {content_range}")
                self.cache_start = block_start
            offset = start - self.cache_start
            count = min(remaining, len(self.cache) - offset)
            if count <= 0:
                break
            chunks.append(self.cache[offset:offset + count])
            self.position += count
            remaining -= count
        return b"".join(chunks)


def letterbox(image: Image.Image, fill: int | tuple[int, int, int]) -> Image.Image:
    return ImageOps.pad(image, SIZE, method=Image.Resampling.BILINEAR, color=fill, centering=(0.5, 0.5))


def fetch_zip_entry(url: str, archive_size: int, info: zipfile.ZipInfo) -> bytes:
    """Fetch one ZIP member with one range request; called concurrently."""
    name_length = len(info.filename.encode("utf-8"))
    end = min(archive_size - 1, info.header_offset + 30 + name_length + 65535 + info.compress_size - 1)
    request = urllib.request.Request(url, headers={"Range": f"bytes={info.header_offset}-{end}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        blob = response.read()
        content_range = response.headers.get("Content-Range", "")
    if not content_range.startswith(f"bytes {info.header_offset}-"):
        raise OSError(f"Server did not honor range request for {info.filename}: {content_range}")
    if len(blob) < 30:
        raise OSError(f"Truncated local ZIP header for {info.filename}")
    fields = struct.unpack("<IHHHHHIIIHH", blob[:30])
    signature, name_length, extra_length = fields[0], fields[-2], fields[-1]
    if signature != 0x04034B50:
        raise OSError(f"Invalid local ZIP header for {info.filename}")
    start = 30 + name_length + extra_length
    compressed = blob[start:start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise OSError(f"Incomplete image data for {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        raise OSError(f"Unsupported ZIP compression method {info.compress_type}")
    if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise OSError(f"Size or CRC mismatch for {info.filename}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of annotated image pairs; 0 uses all")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.05 <= args.val_fraction <= 0.5:
        parser.error("--val-fraction must be from 0.05 to 0.5")
    annotations = args.annotations.resolve()
    if not annotations.is_dir():
        parser.error(f"OFFSED SegmentationClass folder not found: {annotations}")
    mask_paths = sorted(annotations.glob("*.png"), key=lambda path: int(path.stem))
    if args.limit:
        if args.limit < 2:
            parser.error("--limit must be zero or at least 2")
        mask_paths = mask_paths[:args.limit]
    if not mask_paths:
        parser.error("No OFFSED segmentation masks found")

    request = urllib.request.Request(IMAGE_ARCHIVE, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_size = int(response.headers["Content-Length"])
    with zipfile.ZipFile(HTTPRangeReader(IMAGE_ARCHIVE, archive_size)) as archive:
        available = {info.filename for info in archive.infolist()}
        entry_order = {info.filename: index for index, info in enumerate(archive.infolist())}
        pairs = [(path, f"left/{path.stem}.png") for path in mask_paths]
        missing = [name for _, name in pairs if name not in available]
        if missing:
            raise FileNotFoundError(f"OFFSED image missing from OPEDD archive: {missing[:5]}")
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        val_count = max(1, round(len(pairs) * args.val_fraction))
        split_for_id = {mask.stem: ("val" if index < val_count else "train") for index, (mask, _) in enumerate(pairs)}
        pairs.sort(key=lambda pair: entry_order[pair[1]])
        class_ids = {name: index for index, name in enumerate(LABEL_RGB)}
        counts = {"train": 0, "val": 0}

        # The archive contains 1,018 images, but only the 203 named above have
        # semantic masks. Fetch those entries concurrently; do not download left.zip.
        with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            for start in range(0, len(pairs), DOWNLOAD_WORKERS):
                batch_pairs = pairs[start:start + DOWNLOAD_WORKERS]
                infos = [archive.getinfo(image_name) for _, image_name in batch_pairs]
                image_bytes = list(pool.map(lambda info: fetch_zip_entry(IMAGE_ARCHIVE, archive_size, info), infos))
                # RGB palette is exact in published masks; all unrelated classes are ignored.
                for (mask_path, _), raw_image in zip(batch_pairs, image_bytes):
                    split = split_for_id[mask_path.stem]
                    image = Image.open(io.BytesIO(raw_image)).convert("RGB")
                    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
                    if image.size != (mask_rgb.shape[1], mask_rgb.shape[0]):
                        image = image.resize((mask_rgb.shape[1], mask_rgb.shape[0]), Image.Resampling.BILINEAR)
                    target = np.full(mask_rgb.shape[:2], IGNORE, dtype=np.uint8)
                    for name, rgb in LABEL_RGB.items():
                        target[np.all(mask_rgb == rgb, axis=2)] = class_ids[name]
                    image_small = letterbox(image, (0, 0, 0))
                    target_image = Image.fromarray(target, mode="L")
                    target_small = ImageOps.pad(target_image, SIZE, method=Image.Resampling.NEAREST, color=IGNORE, centering=(0.5, 0.5))
                    out_image = args.output / "images" / split / f"{mask_path.stem}.jpg"
                    out_mask = args.output / "masks" / split / f"{mask_path.stem}.png"
                    out_image.parent.mkdir(parents=True, exist_ok=True)
                    out_mask.parent.mkdir(parents=True, exist_ok=True)
                    image_small.save(out_image, quality=88, optimize=True)
                    target_small.save(out_mask, optimize=True)
                    counts[split] += 1
                    if sum(counts.values()) % 20 == 0:
                        print(f"Prepared {sum(counts.values())}/{len(pairs)} OFFSED image/mask pairs", flush=True)

    metadata = {
        "dataset": "OFFSED/OPEDD",
        "source": "https://www.dfki.uni-kl.de/~neigel/offsed.html",
        "task": "semantic surface adaptation; non-drivable pavement is a sidewalk-like proxy, not a true campus sidewalk label",
        "classes": class_ids,
        "ignore_index": IGNORE,
        "image_size": list(SIZE),
        "images": counts,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "license_note": "Original dataset terms were not stated on the download page; retain attribution and verify terms before redistribution or commercial use.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
