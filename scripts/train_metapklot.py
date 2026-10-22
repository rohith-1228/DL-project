"""Fine-tune YOLO segmentation on a prepared MetaPKLot dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/metapklot_yolo/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="Use cpu or a CUDA device such as 0")
    parser.add_argument("--name", default="metapklot_seg")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("epochs, image size, and batch must be positive")
    data_yaml, weights = args.data.resolve(), args.weights.resolve()
    if not data_yaml.is_file():
        parser.error(f"Dataset config not found: {data_yaml}. Prepare it with scripts/prepare_metapklot.py first.")
    if not weights.is_file():
        parser.error(f"Segmentation starting weights not found: {weights}")

    model = YOLO(str(weights))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(ROOT / "training_runs" / "metapklot"),
        name=args.name,
        patience=min(10, args.epochs),
        seed=42,
        plots=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished without best weights at {best}")
    destination = ROOT / "models" / "metapklot_parking_spaces.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"Fine-tuned weights saved to: {destination}")
    print(f"Training and validation reports: {results.save_dir}")


if __name__ == "__main__":
    main()
