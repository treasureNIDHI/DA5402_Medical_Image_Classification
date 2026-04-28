from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_DIR = Path("data/processed")
FEATURE_STORE_DIR = Path("reports/feature_store")
FEATURE_MANIFEST_FILE = FEATURE_STORE_DIR / "feature_manifest.jsonl"
FEATURE_SPEC_FILE = FEATURE_STORE_DIR / "feature_spec.json"
FEATURE_IMPACT_FILE = FEATURE_STORE_DIR / "feature_impact.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NUMERIC_FEATURES = [
    "width",
    "height",
    "aspect_ratio",
    "file_size_kb",
    "mean_intensity",
    "std_intensity",
    "contrast",
    "edge_energy",
    "red_mean",
    "green_mean",
    "blue_mean",
]
FEATURE_VERSION = "1.0.0"


@dataclass
class FeatureRecord:
    dataset: str
    split: str
    class_name: str
    image_path: str
    width: int
    height: int
    aspect_ratio: float
    file_size_kb: float
    mean_intensity: float
    std_intensity: float
    contrast: float
    edge_energy: float
    red_mean: float
    green_mean: float
    blue_mean: float


def _iter_image_paths(root: Path) -> Iterable[tuple[str, str, str, Path]]:
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for split_dir in sorted(dataset_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for image_path in sorted(class_dir.rglob("*")):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        yield dataset_dir.name, split_dir.name, class_dir.name, image_path


def _extract_features(dataset: str, split: str, class_name: str, image_path: Path) -> FeatureRecord:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0

    grayscale = pixels.mean(axis=2)
    if grayscale.size > 1:
        gradients = []
        if grayscale.shape[0] > 1:
            gradients.append(np.abs(np.diff(grayscale, axis=0)).mean())
        if grayscale.shape[1] > 1:
            gradients.append(np.abs(np.diff(grayscale, axis=1)).mean())
        edge_energy = float(np.mean(gradients)) if gradients else 0.0
    else:
        edge_energy = 0.0

    per_channel_mean = pixels.mean(axis=(0, 1))
    per_channel_std = pixels.std(axis=(0, 1))
    luminance_std = float(grayscale.std())

    return FeatureRecord(
        dataset=dataset,
        split=split,
        class_name=class_name,
        image_path=str(image_path),
        width=width,
        height=height,
        aspect_ratio=float(width / height) if height else 0.0,
        file_size_kb=float(image_path.stat().st_size / 1024.0),
        mean_intensity=float(grayscale.mean()),
        std_intensity=luminance_std,
        contrast=luminance_std,
        edge_energy=edge_energy,
        red_mean=float(per_channel_mean[0]),
        green_mean=float(per_channel_mean[1]),
        blue_mean=float(per_channel_mean[2]),
    )


def collect_feature_records(processed_dir: Path = PROCESSED_DIR) -> tuple[list[FeatureRecord], list[str]]:
    records: list[FeatureRecord] = []
    skipped: list[str] = []

    if not processed_dir.exists():
        return records, skipped

    for dataset, split, class_name, image_path in _iter_image_paths(processed_dir):
        try:
            records.append(_extract_features(dataset, split, class_name, image_path))
        except Exception as exc:
            skipped.append(f"{image_path}: {exc}")

    return records, skipped


def _numeric_summary(values: list[float]) -> dict:
    if not values:
        return {
            "mean": None,
            "variance": None,
            "min": None,
            "max": None,
            "percentiles": {},
            "distribution": {"bins": [], "counts": []},
        }

    array = np.asarray(values, dtype=np.float64)
    bins = min(10, max(1, len(array)))
    counts, bin_edges = np.histogram(array, bins=bins)
    return {
        "mean": float(array.mean()),
        "variance": float(array.var()),
        "min": float(array.min()),
        "max": float(array.max()),
        "percentiles": {
            "p25": float(np.percentile(array, 25)),
            "p50": float(np.percentile(array, 50)),
            "p75": float(np.percentile(array, 75)),
        },
        "distribution": {
            "bins": [float(edge) for edge in bin_edges.tolist()],
            "counts": [int(count) for count in counts.tolist()],
        },
    }


def build_feature_spec() -> dict:
    return {
        "version": FEATURE_VERSION,
        "inputs": ["processed image files under data/processed"],
        "features": {
            "width": "Pixel width after preprocessing",
            "height": "Pixel height after preprocessing",
            "aspect_ratio": "Width divided by height",
            "file_size_kb": "File size in kilobytes",
            "mean_intensity": "Mean grayscale intensity",
            "std_intensity": "Standard deviation of grayscale intensity",
            "contrast": "Alias for grayscale intensity spread",
            "edge_energy": "Average absolute grayscale gradient",
            "red_mean": "Mean red channel value",
            "green_mean": "Mean green channel value",
            "blue_mean": "Mean blue channel value",
        },
    }


def build_feature_summary(records: list[FeatureRecord]) -> dict:
    summary = {
        "version": FEATURE_VERSION,
        "total_images": len(records),
        "datasets": {},
    }

    grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
    for record in records:
        grouped[record.dataset].append(record)

    for dataset, dataset_records in grouped.items():
        dataset_summary = {
            "image_count": len(dataset_records),
            "class_counts": dict(Counter(record.class_name for record in dataset_records)),
            "splits": {},
            "feature_statistics": {},
        }

        split_grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
        for record in dataset_records:
            split_grouped[record.split].append(record)

        for split_name, split_records in split_grouped.items():
            dataset_summary["splits"][split_name] = {
                "image_count": len(split_records),
                "class_counts": dict(Counter(record.class_name for record in split_records)),
            }

        for feature_name in NUMERIC_FEATURES:
            dataset_summary["feature_statistics"][feature_name] = _numeric_summary(
                [float(getattr(record, feature_name)) for record in dataset_records]
            )

        summary["datasets"][dataset] = dataset_summary

    return summary


def _records_to_matrix(records: list[FeatureRecord]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = NUMERIC_FEATURES
    matrix = np.asarray(
        [[float(getattr(record, feature_name)) for feature_name in feature_names] for record in records],
        dtype=np.float64,
    )
    labels = np.asarray([record.class_name for record in records], dtype=object)
    return matrix, labels, feature_names


def _top_importances(feature_names: list[str], values: np.ndarray, limit: int = 5) -> list[dict]:
    ranking = sorted(
        zip(feature_names, values.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature_name, "importance": float(score)}
        for feature_name, score in ranking[:limit]
    ]


def build_feature_impact_report(records: list[FeatureRecord]) -> dict:
    datasets: dict[str, list[FeatureRecord]] = defaultdict(list)
    for record in records:
        datasets[record.dataset].append(record)

    dataset_reports = []
    for dataset, dataset_records in datasets.items():
        train_records = [record for record in dataset_records if record.split == "train"]
        test_records = [record for record in dataset_records if record.split == "test"]

        if len(train_records) < 2 or len(test_records) < 1:
            dataset_reports.append(
                {
                    "dataset": dataset,
                    "status": "insufficient_data",
                    "train_samples": len(train_records),
                    "test_samples": len(test_records),
                }
            )
            continue

        train_matrix, train_labels, feature_names = _records_to_matrix(train_records)
        test_matrix, test_labels, _ = _records_to_matrix(test_records)

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(train_matrix, train_labels)
        dummy_predictions = dummy.predict(test_matrix)

        surrogate = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )
        surrogate.fit(train_matrix, train_labels)
        surrogate_predictions = surrogate.predict(test_matrix)

        permutation = permutation_importance(
            surrogate,
            test_matrix,
            test_labels,
            n_repeats=5,
            random_state=42,
            scoring="f1_macro",
        )

        dummy_accuracy = accuracy_score(test_labels, dummy_predictions)
        dummy_f1 = f1_score(test_labels, dummy_predictions, average="macro")
        surrogate_accuracy = accuracy_score(test_labels, surrogate_predictions)
        surrogate_f1 = f1_score(test_labels, surrogate_predictions, average="macro")

        dataset_reports.append(
            {
                "dataset": dataset,
                "status": "healthy",
                "train_samples": len(train_records),
                "test_samples": len(test_records),
                "dummy_baseline": {
                    "accuracy": float(dummy_accuracy),
                    "f1": float(dummy_f1),
                },
                "surrogate_model": {
                    "accuracy": float(surrogate_accuracy),
                    "f1": float(surrogate_f1),
                },
                "impact": {
                    "accuracy_gain": float(surrogate_accuracy - dummy_accuracy),
                    "f1_gain": float(surrogate_f1 - dummy_f1),
                },
                "top_feature_importance": _top_importances(feature_names, surrogate.feature_importances_),
                "top_permutation_importance": _top_importances(feature_names, permutation.importances_mean),
            }
        )

    overall_status = "healthy" if all(report["status"] == "healthy" for report in dataset_reports) else "needs_attention"
    return {
        "version": FEATURE_VERSION,
        "overall_status": overall_status,
        "feature_names": NUMERIC_FEATURES,
        "datasets": dataset_reports,
    }


def _ensure_feature_store_dir() -> None:
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)


def write_feature_artifacts(records: list[FeatureRecord]) -> dict:
    _ensure_feature_store_dir()

    with open(FEATURE_MANIFEST_FILE, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")

    spec = build_feature_spec()
    with open(FEATURE_SPEC_FILE, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)

    impact = build_feature_impact_report(records)
    with open(FEATURE_IMPACT_FILE, "w", encoding="utf-8") as handle:
        json.dump(impact, handle, indent=2)

    return {
        "spec_file": str(FEATURE_SPEC_FILE),
        "manifest_file": str(FEATURE_MANIFEST_FILE),
        "impact_file": str(FEATURE_IMPACT_FILE),
        "record_count": len(records),
        "status": impact["overall_status"],
    }


def run_feature_engineering(processed_dir: Path = PROCESSED_DIR) -> int:
    records, skipped = collect_feature_records(processed_dir)
    artifacts = write_feature_artifacts(records)
    report = {
        "processed_dir": str(processed_dir),
        "feature_store_dir": str(FEATURE_STORE_DIR),
        "artifacts": artifacts,
        "skipped_files": skipped,
        "skipped_count": len(skipped),
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build versioned image features and feature importance artifacts.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()
    return run_feature_engineering(Path(args.processed_dir))


if __name__ == "__main__":
    raise SystemExit(main())
