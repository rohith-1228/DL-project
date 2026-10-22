"""Export saved human prediction feedback and source images for curation."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/parking_system.db"
DEFAULT_OUTPUT = ROOT / "data/training/prediction_feedback"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"Feedback database not found: {args.db}")

    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM prediction_feedback ORDER BY updated_at, id").fetchall()
    except sqlite3.OperationalError as exc:
        parser.error(f"Feedback table is unavailable. Start the app once first. Details: {exc}")
    finally:
        conn.close()

    project_root = ROOT
    exported_images: set[str] = set()
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            stored_path = project_root / Path(row["image_path"])
            if not stored_path.is_file():
                parser.error(f"A feedback image is missing: {stored_path}")
            image_name = f"{row['image_hash']}.jpg"
            destination = image_dir / image_name
            if image_name not in exported_images:
                shutil.copy2(stored_path, destination)
                exported_images.add(image_name)
            record = {
                "image": f"images/{image_name}",
                "analysis_id": row["analysis_id"],
                "vehicle_id": row["vehicle_id"],
                "category_key": row["category_key"],
                "predicted_status": row["predicted_status"],
                "feedback_vote": row["feedback_vote"],
                "corrected_label": row["corrected_label"],
                "bbox": json.loads(row["bbox_json"]),
                "context": json.loads(row["context_json"]),
                "model_name": row["model_name"],
                "updated_at": row["updated_at"],
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "feedback_records": len(rows),
        "unique_images": len(exported_images),
        "manifest": str(manifest_path),
        "note": "Category-level human feedback only. Grass segmentation fine-tuning still requires pixel masks.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
