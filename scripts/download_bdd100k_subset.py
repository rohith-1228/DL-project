"""Download only a reproducible BDD100K semantic image/mask subset.

The source is a Hugging Face mirror that identifies itself as an unmodified
redistribution of BDD100K files. The original dataset terms apply. Use only for
educational, research, or not-for-profit work unless separately licensed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/external/bdd100k"
REPO = "dronefreak/BDD100K-Semantic-Segmentation"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE_ROOT = f"https://huggingface.co/datasets/{REPO}/resolve/main"
LICENSE_NOTICE = """BDD100K DATA LICENSE NOTICE

Copyright ©2018. The Regents of the University of California (Regents). All Rights Reserved.

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 1/1/2021

Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement; and permission to use, copy, modify and distribute this software for commercial purposes (such rights not subject to transfer) to BDD and BAIR Commons members and their affiliates, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions.

Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Source: https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst
Mirror: https://huggingface.co/datasets/dronefreak/BDD100K-Semantic-Segmentation
"""


def get_json(url: str):
    request = Request(url, headers={"User-Agent": "campus-parking-footpath-training/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read()), response.headers.get("Link", "")


def list_images(split: str) -> list[dict]:
    # expand=false avoids the Hub's recursive expansion response-size/limit
    # error while still returning a cursor for the next page.
    url = f"{API_ROOT}/data/images_10k/10k/{split}?recursive=true&expand=false&limit=1000"
    files: list[dict] = []
    while url:
        entries, link = get_json(url)
        files.extend(item for item in entries if item.get("type") == "file" and item.get("path", "").lower().endswith(".jpg"))
        match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        url = match.group(1) if match else ""
    if not files:
        raise RuntimeError(f"The mirror returned no {split} image files")
    return files


def download_one(task: tuple[str, Path, str]) -> dict:
    remote_path, local_path, expected_sha = task
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.is_file():
        actual = hashlib.sha256(local_path.read_bytes()).hexdigest()
        if not expected_sha or actual == expected_sha:
            return {"path": str(local_path), "status": "cached"}
        local_path.unlink()
    url = f"{RESOLVE_ROOT}/{quote(remote_path, safe='/')}?download=true"
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "campus-parking-footpath-training/1.0"})
            digest = hashlib.sha256()
            with urlopen(request, timeout=120) as response, local_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            actual = digest.hexdigest()
            if expected_sha and actual != expected_sha:
                local_path.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 mismatch for {remote_path}")
            return {"path": str(local_path), "status": "downloaded"}
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            local_path.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Could not download {remote_path}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.train_limit, args.val_limit, args.workers) < 1:
        parser.error("train-limit, val-limit, and workers must be positive")

    rng = random.Random(args.seed)
    tasks: list[tuple[str, Path, str]] = []
    selected: dict[str, list[str]] = {}
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        candidates = list_images(split)
        rng.shuffle(candidates)
        chosen = candidates[:min(limit, len(candidates))]
        selected[split] = []
        for image in chosen:
            image_remote = image["path"]
            image_stem = Path(image_remote).stem
            match = re.search(r"/shard_\d+/", image_remote)
            if not match:
                raise RuntimeError(f"Unexpected mirror path layout: {image_remote}")
            shard_path = match.group(0)
            mask_remote = f"data/seg_maps/labels/{split}{shard_path}{image_stem}_train_id.png"
            image_parts = Path(image_remote).parts
            out_image = args.output / "bdd100k" / "images" / "10k" / split / image_parts[-1]
            out_mask = args.output / "bdd100k" / "labels" / "sem_seg" / "masks" / split / f"{image_stem}.png"
            tasks.append((image_remote, out_image, image.get("lfs", {}).get("oid", "")))
            # The mirror provides the class-id PNG separately from the JPEG.
            tasks.append((mask_remote, out_mask, ""))
            selected[split].append(image_stem)
        print(f"Selected {len(chosen)} {split} images from {len(candidates)} labeled candidates", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "BDD100K_DATA_LICENSE.txt").write_text(LICENSE_NOTICE, encoding="utf-8")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                future.result()
            except Exception as exc:
                failures.append(str(exc))
            if index % 100 == 0 or index == len(futures):
                print(f"Downloaded or checked {index}/{len(futures)} files", flush=True)
    if failures:
        raise RuntimeError("Subset download incomplete:\n" + "\n".join(failures[:10]))

    manifest = {
        "dataset": "BDD100K semantic segmentation",
        "mirror": f"https://huggingface.co/datasets/{REPO}",
        "official_source": "https://bdd-data.berkeley.edu/",
        "selection_seed": args.seed,
        "selected_images": {split: len(names) for split, names in selected.items()},
        "image_ids": selected,
        "license_notice": "BDD100K_DATA_LICENSE.txt",
        "usage_limit": "Educational, research, and not-for-profit use; commercial rights require BDD/BAIR membership or separate licensing.",
    }
    (args.output / "bdd100k_subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_images": manifest["selected_images"]}, indent=2))


if __name__ == "__main__":
    main()
