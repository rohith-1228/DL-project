"""Fine-tune Cityscapes SegFormer on the labeled OFFSED surface subset."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
DEFAULT_DATA = ROOT / "data/training/offsed_semantic"
DEFAULT_OUTPUT = ROOT / "models/offsed_surface_segformer"
CLASS_NAMES = ["road", "sidewalk_proxy", "grass"]
IGNORE_INDEX = 255


class SurfaceDataset(Dataset):
    def __init__(self, root: Path, split: str, mean: list[float], std: list[float]):
        self.root, self.split, self.mean, self.std = root, split, mean, std
        self.images = sorted((root / "images" / split).glob("*.jpg"))
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = (image - self.mean_tensor) / self.std_tensor
        label_path = self.root / "masks" / self.split / f"{image_path.stem}.png"
        labels = torch.from_numpy(np.asarray(Image.open(label_path), dtype=np.uint8).copy()).long()
        return image, labels


def evaluate(model, loader, device):
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            logits = model(pixel_values=pixels).logits
            logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = logits.argmax(dim=1)
            valid = labels != IGNORE_INDEX
            indices = labels[valid] * len(CLASS_NAMES) + predicted[valid]
            confusion += torch.bincount(indices.cpu(), minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    true_positive = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    return confusion.tolist(), {name: (float(iou[i]) if torch.isfinite(iou[i]) else None) for i, name in enumerate(CLASS_NAMES)}, float(torch.nanmean(iou))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained", default=PRETRAINED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if not (args.data / "images/train").is_dir() or not (args.data / "images/val").is_dir():
        parser.error("Prepared OFFSED dataset not found. Run scripts/prepare_offsed.py first.")

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.pretrained)
    # The prepared image/mask pairs are already letterboxed to 512x512. Preserve
    # that geometry so labels line up and app inference uses the same preprocessing.
    processor.do_resize = False
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained,
        num_labels=len(CLASS_NAMES),
        id2label={i: label for i, label in enumerate(CLASS_NAMES)},
        label2id={label: i for i, label in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    ).to(device)
    model.train()
    train_data = SurfaceDataset(args.data, "train", processor.image_mean, processor.image_std)
    val_data = SurfaceDataset(args.data, "val", processor.image_mean, processor.image_std)
    if not len(train_data) or not len(val_data):
        parser.error("Both OFFSED train and validation splits must contain images")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    best_state = None
    best_score = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for pixels, labels in train_loader:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixels).logits
            logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            # Some wide OFFSED frames contain no labeled surface pixels after
            # letterboxing. Cross entropy's mean reduction is NaN for an
            # all-ignore batch, so normalize the summed loss by valid pixels.
            valid_pixels = labels != IGNORE_INDEX
            if not valid_pixels.any():
                continue
            loss = torch.nn.functional.cross_entropy(
                logits, labels, ignore_index=IGNORE_INDEX, reduction="sum"
            ) / valid_pixels.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; refusing to save this checkpoint")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
        confusion, class_iou, mean_iou = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, len(train_loader)), "val_miou": mean_iou, "val_class_iou": class_iou}
        history.append(row)
        print(json.dumps(row), flush=True)
        if mean_iou > best_score:
            best_score = mean_iou
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    confusion, class_iou, mean_iou = evaluate(model, val_loader, device)
    summary = {
        "initial_checkpoint": args.pretrained,
        "task": "OFFSED semantic surface segmentation",
        "labels": {"0": "drivable_pavement/road", "1": "non_drivable_pavement_proxy", "2": "grass"},
        "caveat": "Non-drivable pavement is only a sidewalk-like proxy; OFFSED is an off-road dataset, not labeled campus sidewalks.",
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "history": history,
        "final_confusion_matrix": confusion,
        "final_class_iou": class_iou,
        "final_mean_iou": mean_iou,
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
