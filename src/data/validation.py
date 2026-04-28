from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "data_validation_report.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_DATASETS = ["chest_xray", "brain_tumor"]


def _count_images(directory: Path) -> tuple[int, int]:
    valid = 0
    corrupt = 0
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except Exception:
            corrupt += 1
    return valid, corrupt


def _class_distribution(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not split_dir.exists():
        return counts
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(
            1 for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_dir.name] = n
    return counts


def _validate_dataset(dataset_dir: Path) -> dict:
    result: dict = {
        "path": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "splits": {},
        "total_valid": 0,
        "total_corrupt": 0,
        "issues": [],
    }

    if not dataset_dir.exists():
        result["issues"].append(f"Dataset directory missing: {dataset_dir}")
        return result

    for split_name in ("train", "val", "test", "Training", "Testing", "Validation"):
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            continue
        valid, corrupt = _count_images(split_dir)
        dist = _class_distribution(split_dir)
        result["splits"][split_name] = {
            "valid_images": valid,
            "corrupt_images": corrupt,
            "class_distribution": dist,
        }
        result["total_valid"] += valid
        result["total_corrupt"] += corrupt
        if corrupt > 0:
            result["issues"].append(f"{split_name}: {corrupt} corrupt image(s) found")

    if result["total_valid"] == 0:
        result["issues"].append("No valid images found in any split")

    return result


def run_validation() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "raw_dir": str(RAW_DIR),
        "datasets": {},
        "overall_valid": True,
        "summary": {},
    }

    total_valid = 0
    total_corrupt = 0
    all_issues: list[str] = []

    for dataset_name in EXPECTED_DATASETS:
        dataset_dir = RAW_DIR / dataset_name
        result = _validate_dataset(dataset_dir)
        report["datasets"][dataset_name] = result
        total_valid += result["total_valid"]
        total_corrupt += result["total_corrupt"]
        all_issues.extend(result["issues"])

    report["summary"] = {
        "total_valid_images": total_valid,
        "total_corrupt_images": total_corrupt,
        "total_issues": len(all_issues),
        "issues": all_issues,
    }
    report["overall_valid"] = len(all_issues) == 0

    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    status = "PASSED" if report["overall_valid"] else "WARNING"
    print(f"Data validation {status}: {total_valid} valid, {total_corrupt} corrupt, {len(all_issues)} issue(s)")
    print(f"Report written to {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_validation())
