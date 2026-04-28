from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "preprocessing_report.json"
IMAGE_SIZE = (224, 224)
OUTLIER_ASPECT_RATIO = 2.5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def create_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _iter_image_paths() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return [path for path in RAW_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def _safe_save_dir(img_path: Path) -> Path:
    rel_path = img_path.parent.relative_to(RAW_DIR)
    save_dir = PROCESSED_DIR / rel_path
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _copy_brain_tumor_structure() -> list[tuple[str, str]]:
    brain_dir = PROCESSED_DIR / "brain_tumor"
    renames = [("Training", "train"), ("Testing", "test"), ("Validation", "val")]
    applied: list[tuple[str, str]] = []

    if not brain_dir.exists():
        return applied

    for source_name, target_name in renames:
        source = brain_dir / source_name
        target = brain_dir / target_name
        if not source.exists() or source == target:
            continue

        if target.exists():
            for child in source.iterdir():
                destination = target / child.name
                if destination.exists():
                    continue
                shutil.move(str(child), str(destination))
            shutil.rmtree(source)
        else:
            shutil.move(str(source), str(target))

        applied.append((source_name, target_name))

    return applied


def process_images() -> dict:
    image_paths = _iter_image_paths()

    report = {
        "raw_dir": str(RAW_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "image_size": list(IMAGE_SIZE),
        "processed_count": 0,
        "skipped_count": 0,
        "corrupt_count": 0,
        "outlier_count": 0,
        "class_counts": {},
        "skipped_files": [],
        "outlier_files": [],
        "standardized_brain_tumor_dirs": [],
    }

    for img_path in tqdm(image_paths, desc="Preprocessing images", unit="img"):
        try:
            with Image.open(img_path) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size
                aspect_ratio = max(width / height if height else 0.0, height / width if width else 0.0)
                if aspect_ratio > OUTLIER_ASPECT_RATIO:
                    report["outlier_count"] += 1
                    report["outlier_files"].append(str(img_path))

                transformed = rgb_image.resize(IMAGE_SIZE)

            save_dir = _safe_save_dir(img_path)
            save_path = save_dir / img_path.name
            transformed.save(save_path)
            report["processed_count"] += 1

            class_name = img_path.parent.name
            report["class_counts"][class_name] = report["class_counts"].get(class_name, 0) + 1
        except UnidentifiedImageError as exc:
            report["skipped_count"] += 1
            report["corrupt_count"] += 1
            report["skipped_files"].append(f"{img_path}: {exc}")
            print(f"Skipping {img_path}: {exc}")
        except Exception as exc:
            report["skipped_count"] += 1
            report["skipped_files"].append(f"{img_path}: {exc}")
            print(f"Skipping {img_path}: {exc}")

    report["standardized_brain_tumor_dirs"] = [
        {"source": source, "target": target}
        for source, target in _copy_brain_tumor_structure()
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(
        f"Processed {report['processed_count']} images; "
        f"skipped {report['skipped_count']} files"
    )
    print(f"Preprocessing report written to {REPORT_FILE}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean, transform, and standardize raw image data.")
    parser.parse_args()
    create_dirs()
    process_images()
    print("Preprocessing completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())