"""Fine-tune the 19-class Cityscapes SegFormer using prepared BDD100K masks."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
DEFAULT_DATA = ROOT / "data/training/bdd100k_semantic"
DEFAULT_OUTPUT = ROOT / "models/bdd100k_sidewalk_segformer"
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
IGNORE_INDEX = 255


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, split: str, mean: list[float], std: list[float]):
        self.root, self.split = root, split
        self.images = sorted((root / "images" / split).glob("*.jpg"))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        pixels = torch.from_numpy(pixels.transpose(2, 0, 1))
        pixels = (pixels - self.mean) / self.std
        # Keep the split explicit so validation samples never read train labels.
        mask_path = self.root / "masks" / self.split / f"{path.stem}.png"
        labels = torch.from_numpy(np.asarray(Image.open(mask_path), dtype=np.uint8).copy()).long()
        return pixels, labels


def evaluate(model, loader, device, label="validation"):
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for batch_index, (pixels, labels) in enumerate(loader, start=1):
            pixels, labels = pixels.to(device), labels.to(device)
            logits = F.interpolate(model(pixel_values=pixels).logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(dim=1)
            valid = labels != IGNORE_INDEX
            indices = labels[valid] * len(CLASS_NAMES) + predictions[valid]
            confusion += torch.bincount(indices.cpu(), minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"{label}: {batch_index}/{len(loader)} batches", flush=True)
    tp = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - tp
    iou = torch.where(union > 0, tp / union, torch.nan)
    class_iou = {name: (float(iou[i]) if torch.isfinite(iou[i]) else None) for i, name in enumerate(CLASS_NAMES)}
    return confusion.tolist(), class_iou, float(torch.nanmean(iou)), class_iou["sidewalk"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained", default=PRETRAINED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=500, help="Deterministic CPU-friendly subset; 0 uses all prepared training images")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.train_limit < 0:
        parser.error("epochs/batch size must be positive and train limit non-negative")
    if not (args.data / "images/train").is_dir() or not (args.data / "images/val").is_dir():
        parser.error("Prepared BDD100K data not found. Run scripts/prepare_bdd100k_semantic.py first.")

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.pretrained)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained,
        num_labels=len(CLASS_NAMES),
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
    ).to(device)
    train_data = SegmentationDataset(args.data, "train", processor.image_mean, processor.image_std)
    val_data = SegmentationDataset(args.data, "val", processor.image_mean, processor.image_std)
    if not len(train_data) or not len(val_data):
        parser.error("Both BDD100K train and validation splits must contain images")
    if args.train_limit and len(train_data) > args.train_limit:
        train_indices = random.Random(args.seed).sample(range(len(train_data)), args.train_limit)
        train_data = Subset(train_data, train_indices)
    # Freeze the large image encoder and fine-tune the segmentation head. This
    # keeps the adaptation practical on CPU while preserving pretrained features.
    for parameter in model.segformer.parameters():
        parameter.requires_grad = False
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=max(args.batch_size, 8), shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    _, _, baseline_mean_iou, baseline_sidewalk_iou = evaluate(model, val_loader, device, "baseline validation")
    history: list[dict] = []
    best_state, best_score, best_epoch = None, -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        steps = 0
        for batch_index, (pixels, labels) in enumerate(train_loader, start=1):
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = F.interpolate(model(pixel_values=pixels).logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            valid = labels != IGNORE_INDEX
            if not valid.any():
                continue
            loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum") / valid.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; refusing to save this checkpoint")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            steps += 1
            if batch_index % 25 == 0 or batch_index == len(train_loader):
                print(f"epoch {epoch + 1}: trained {batch_index}/{len(train_loader)} batches", flush=True)
        confusion, class_iou, mean_iou, sidewalk_iou = evaluate(model, val_loader, device, f"epoch {epoch + 1} validation")
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_total / max(1, steps),
            "val_mean_iou": mean_iou,
            "val_sidewalk_iou": sidewalk_iou,
            "val_class_iou": class_iou,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        # Select for the sidewalk target rather than a mean dominated by classes
        # irrelevant to this app's surface decision.
        if sidewalk_iou is not None and sidewalk_iou > best_score:
            best_score = sidewalk_iou
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    args.output.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        confusion, class_iou, mean_iou, sidewalk_iou = evaluate(model, val_loader, device)
    else:
        confusion, class_iou, mean_iou, sidewalk_iou = [], {}, float("nan"), None
    improved = sidewalk_iou is not None and baseline_sidewalk_iou is not None and sidewalk_iou > baseline_sidewalk_iou
    if improved:
        model.save_pretrained(args.output)
        processor.save_pretrained(args.output)
    summary = {
        "task": "BDD100K broad-scene semantic adaptation for road, sidewalk, vegetation, and terrain",
        "initial_checkpoint": args.pretrained,
        "labels": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "epochs": args.epochs,
        "train_limit": args.train_limit,
        "frozen_encoder": True,
        "seed": args.seed,
        "best_epoch_by_sidewalk_iou": best_epoch,
        "baseline_sidewalk_iou": baseline_sidewalk_iou,
        "baseline_mean_iou": baseline_mean_iou,
        "history": history,
        "final_confusion_matrix": confusion,
        "final_class_iou": class_iou,
        "final_mean_iou": mean_iou,
        "final_sidewalk_iou": sidewalk_iou,
        "checkpoint_saved": improved,
        "limitation": "A separate campus-camera holdout is still required; BDD100K street scenes do not establish accuracy on every campus driveway or private walkway.",
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if improved:
        print(f"Saved validated BDD100K checkpoint to {args.output}")
    else:
        print("Fine-tuning did not improve held-out BDD100K sidewalk IoU; no model checkpoint was saved.")


if __name__ == "__main__":
    main()
