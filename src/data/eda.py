from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.features.engineering import build_feature_summary, collect_feature_records

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FEATURE_STORE_DIR = REPORTS_DIR / "feature_store"
EDA_JSON = REPORTS_DIR / "eda_report.json"
EDA_MD = REPORTS_DIR / "eda_report.md"
FEATURE_BASELINE_FILE = FEATURE_STORE_DIR / "feature_baseline.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _scan_dataset(dataset_dir: Path) -> dict:
    splits: dict[str, dict] = {}
    for split_dir in sorted(dataset_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        class_counts: dict[str, int] = {}
        widths: list[int] = []
        heights: list[int] = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            count = 0
            for img_path in class_dir.rglob("*"):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                count += 1
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                    widths.append(w)
                    heights.append(h)
                except Exception:
                    pass
            class_counts[class_dir.name] = count

        total = sum(class_counts.values())
        splits[split_dir.name] = {
            "total_images": total,
            "class_counts": class_counts,
            "class_balance": {k: round(v / total, 4) if total else 0.0 for k, v in class_counts.items()},
            "avg_width": round(sum(widths) / len(widths), 1) if widths else None,
            "avg_height": round(sum(heights) / len(heights), 1) if heights else None,
        }
    return splits


def run_eda() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {"processed_dir": str(PROCESSED_DIR), "datasets": {}}

    if PROCESSED_DIR.exists():
        for dataset_dir in sorted(PROCESSED_DIR.iterdir()):
            if dataset_dir.is_dir():
                report["datasets"][dataset_dir.name] = _scan_dataset(dataset_dir)

    with open(EDA_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"EDA JSON written to {EDA_JSON}")

    # Markdown report
    lines: list[str] = ["# EDA Report\n"]
    for dataset_name, splits in report["datasets"].items():
        lines.append(f"## {dataset_name}\n")
        for split_name, stats in splits.items():
            lines.append(f"### {split_name}")
            lines.append(f"- Total images: {stats['total_images']}")
            lines.append(f"- Class counts: {stats['class_counts']}")
            lines.append(f"- Class balance: {stats['class_balance']}")
            if stats["avg_width"]:
                lines.append(f"- Avg size: {stats['avg_width']} x {stats['avg_height']} px")
            lines.append("")

    with open(EDA_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"EDA Markdown written to {EDA_MD}")

    # Feature baseline
    records, skipped = collect_feature_records(PROCESSED_DIR)
    feature_summary = build_feature_summary(records)
    with open(FEATURE_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_summary, f, indent=2)
    print(f"Feature baseline written to {FEATURE_BASELINE_FILE} ({len(records)} records, {len(skipped)} skipped)")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_eda())
