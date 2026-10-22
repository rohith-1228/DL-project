"""Evaluate OFFSED surface segmentation on held-out frames and inspect vehicle contact overlap."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/training/offsed_semantic"
WEIGHTS = ROOT / "models/offsed_surface_segformer"
VEHICLE_WEIGHTS = ROOT / "yolov8n.pt"


def make_overlay(image: Image.Image, predicted: np.ndarray, detections: list[dict], title: str, valid_mask: np.ndarray | None = None) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    color = np.zeros_like(base)
    color[predicted == 2] = (36, 210, 76)       # grass
    color[predicted == 1] = (40, 190, 230)      # non-drivable pavement proxy
    active = np.isin(predicted, (1, 2))
    if valid_mask is not None:
        active &= valid_mask
    base[active] = (0.56 * base[active] + 0.44 * color[active]).astype(np.uint8)
    result = Image.fromarray(base)
    draw = ImageDraw.Draw(result)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        outline = (255, 90, 50) if det["status"] == "Review required" else (70, 245, 125)
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=3)
        gx, gy = det["ground_point"]
        draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), fill=(255, 230, 30), outline=(0, 0, 0), width=1)
        draw.text((x1, max(2, y1 - 15)), f"{det['class']} {det['status']}", fill=outline)
    draw.rectangle((0, 0, result.width, 24), fill=(20, 24, 31))
    draw.text((7, 5), title, fill=(255, 255, 255))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--vehicle-weights", type=Path, default=VEHICLE_WEIGHTS)
    parser.add_argument("--output", type=Path, default=ROOT / "training_runs/offsed_surface/evaluation")
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates the entire held-out split")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.weights.is_dir() or not args.vehicle_weights.is_file():
        parser.error("Trained OFFSED weights or existing vehicle weights are missing")

    processor = AutoImageProcessor.from_pretrained(args.weights, local_files_only=True)
    surface_model = SegformerForSemanticSegmentation.from_pretrained(args.weights, local_files_only=True).to(args.device).eval()
    vehicle_model = YOLO(str(args.vehicle_weights))
    image_paths = sorted((args.data / "images/val").glob("*.jpg"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        parser.error("No held-out OFFSED images found")

    args.output.mkdir(parents=True, exist_ok=True)
    records, vehicle_panels, empty_vehicle_panels = [], [], []
    confusion = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(args.device)
            logits = surface_model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
            predicted = probabilities.argmax(axis=0)
            mask_path = args.data / "masks" / "val" / f"{image_path.stem}.png"
            truth = None
            if mask_path.is_file():
                truth = np.asarray(Image.open(mask_path), dtype=np.uint8)
                valid = truth != 255
                encoded = truth[valid].astype(np.int64) * 3 + predicted[valid].astype(np.int64)
                confusion += np.bincount(encoded, minlength=9).reshape(3, 3)
            vehicle_result = vehicle_model.predict(source=np.asarray(image), classes=[2, 3, 5, 7], conf=0.2, verbose=False)[0]
            detections = []
            if vehicle_result.boxes is not None:
                for box in vehicle_result.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]
                    gx, gy = (x1 + x2) // 2, y2
                    y1p, y2p = max(0, gy - 20), min(image.height, gy + 15)
                    x1p, x2p = max(0, gx - 40), min(image.width, gx + 40)
                    class_region = predicted[y1p:y2p, x1p:x2p]
                    conf_region = probabilities[:, y1p:y2p, x1p:x2p]
                    grass_ratio = float(np.mean(class_region == 2)) if class_region.size else 0.0
                    confidence_region = np.max(conf_region, axis=0) if conf_region.size else np.array([])
                    surface_confidence = float(np.mean(confidence_region)) if confidence_region.size else 0.0
                    if grass_ratio >= 0.15 or surface_confidence < 0.60:
                        status = "Review required"
                    else:
                        status = "No predicted grass overlap"
                    class_id = int(box.cls[0].cpu())
                    detections.append({
                        "class": vehicle_model.names[class_id],
                        "confidence": round(float(box.conf[0].cpu()), 3),
                        "bbox": [x1, y1, x2, y2],
                        "ground_point": [gx, gy],
                        "grass_overlap_ratio": round(grass_ratio, 4),
                        "surface_confidence": round(surface_confidence, 4),
                        "status": status,
                    })
            records.append({"image": image_path.name, "vehicle_count": len(detections), "detections": detections})
            title = f"{image_path.stem}: {len(detections)} vehicle(s)"
            if detections:
                title += " | " + "; ".join(f"{d['class']} {d['grass_overlap_ratio']:.0%} grass → {d['status']}" for d in detections)
            panel = make_overlay(image, predicted, detections, title, truth != 255 if truth is not None else None)
            if detections and len(vehicle_panels) < 6:
                vehicle_panels.append(panel)
            elif not detections and len(empty_vehicle_panels) < 2:
                empty_vehicle_panels.append(panel)

    (args.output / "heldout_vehicle_overlap.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (args.output / "heldout_vehicle_overlap.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "class", "confidence", "grass_overlap_ratio", "surface_confidence", "status", "bbox", "ground_point"])
        writer.writeheader()
        for record in records:
            for detection in record["detections"]:
                writer.writerow({"image": record["image"], **detection})

    panels = vehicle_panels + empty_vehicle_panels
    columns, cell = 2, (512, 540)
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), (238, 240, 243))
    for index, panel in enumerate(panels):
        panel.thumbnail((cell[0], cell[1] - 18))
        x, y = (index % columns) * cell[0], (index // columns) * cell[1]
        sheet.paste(panel, (x, y))
    sheet.save(args.output / "heldout_segmentation_and_overlap.png")
    vehicle_total = sum(record["vehicle_count"] for record in records)
    tp = np.diag(confusion).astype(float)
    unions = confusion.sum(0) + confusion.sum(1) - tp
    iou = [float(tp[i] / unions[i]) if unions[i] else None for i in range(3)]
    evaluation = {
        "images_evaluated": len(records),
        "vehicle_detections": vehicle_total,
        "segmentation_confusion_matrix": confusion.tolist(),
        "class_iou": {"road": iou[0], "sidewalk_proxy": iou[1], "grass": iou[2]},
        "mean_iou": float(np.mean([value for value in iou if value is not None])) if any(value is not None for value in iou) else None,
        "visualization": str(args.output / 'heldout_segmentation_and_overlap.png'),
        "overlap_results": str(args.output / 'heldout_vehicle_overlap.csv'),
    }
    (args.output / "heldout_metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
