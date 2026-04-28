import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torchvision import models

from src.features.engineering import build_feature_summary, collect_feature_records
from src.training.dataset import get_dataloaders
from src.utils.reproducibility import log_reproducibility_context, set_global_seed

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
REPORTS_DIR = Path("reports")
BASELINE_FILE = REPORTS_DIR / "monitoring_baseline.json"
REPORT_FILE = REPORTS_DIR / "monitoring_report.json"
LOG_FILE = REPORTS_DIR / "monitoring.log"
FEATURE_BASELINE_FILE = REPORTS_DIR / "feature_store" / "feature_baseline.json"

DRIFT_L1_THRESHOLD = 0.15
ACCURACY_DROP_THRESHOLD = 0.03
FEATURE_MEAN_SHIFT_THRESHOLD = 0.05
FEATURE_VARIANCE_SHIFT_THRESHOLD = 0.25
FEATURE_DISTRIBUTION_L1_THRESHOLD = 0.35


def configure_logging() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _label_distribution(split_dir: Path) -> dict[str, float]:
    counts = {}
    total = 0
    if not split_dir.exists():
        return {}

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(1 for p in class_dir.iterdir() if p.is_file())
        counts[class_dir.name] = n
        total += n

    if total == 0:
        return {key: 0.0 for key in counts}

    return {key: value / total for key, value in counts.items()}


def _l1_distribution_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def evaluate_model(model_path: str, data_dir: str) -> tuple[float, float]:
    _, _, test_loader = get_dataloaders(data_dir, seed=SEED)

    model = models.resnet50(weights=None)

    # 🔥 load checkpoint first
    state_dict = torch.load(model_path, map_location=DEVICE)

    # 🔥 dynamically infer number of classes
    num_classes = state_dict["fc.weight"].shape[0]

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.numpy())

    return accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average="macro")

def check_dataset(dataset_name: str, model_path: str, data_dir: str) -> dict:
    train_split = Path(data_dir) / "train"
    test_split = Path(data_dir) / "test"

    train_dist = _label_distribution(train_split)
    test_dist = _label_distribution(test_split)
    drift_l1 = _l1_distribution_distance(train_dist, test_dist)

    accuracy, f1 = evaluate_model(model_path, data_dir)

    return {
        "dataset": dataset_name,
        "model_path": model_path,
        "data_dir": data_dir,
        "accuracy": accuracy,
        "f1": f1,
        "drift_l1": drift_l1,
        "train_distribution": train_dist,
        "test_distribution": test_dist,
    }


def create_or_load_baseline(current_report: dict) -> dict:
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)

    baseline = {
        "created_from_report_at": current_report["timestamp"],
        "models": current_report["models"],
        "thresholds": {
            "drift_l1": DRIFT_L1_THRESHOLD,
            "accuracy_drop": ACCURACY_DROP_THRESHOLD,
        },
    }

    with open(BASELINE_FILE, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2)

    return baseline


def compare_against_baseline(report: dict, baseline: dict) -> dict:
    baseline_models = {m["dataset"]: m for m in baseline.get("models", [])}
    checks = []

    for model in report["models"]:
        ref = baseline_models.get(model["dataset"], {})
        baseline_acc = ref.get("accuracy", model["accuracy"])

        acc_drop = baseline_acc - model["accuracy"]
        drift_alert = model["drift_l1"] > DRIFT_L1_THRESHOLD
        perf_alert = acc_drop > ACCURACY_DROP_THRESHOLD

        status = "healthy"
        if drift_alert and perf_alert:
            status = "drift_and_performance_drop"
        elif drift_alert:
            status = "drift_alert"
        elif perf_alert:
            status = "performance_alert"

        checks.append(
            {
                "dataset": model["dataset"],
                "baseline_accuracy": baseline_acc,
                "current_accuracy": model["accuracy"],
                "accuracy_drop": acc_drop,
                "drift_l1": model["drift_l1"],
                "status": status,
            }
        )

    overall = "healthy"
    if any(c["status"] != "healthy" for c in checks):
        overall = "alert"

    return {"overall_status": overall, "checks": checks}


def _normalise_distribution(distribution: dict) -> list[float]:
    counts = distribution.get("counts", [])
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def _compare_feature_baseline(current_summary: dict, baseline_summary: dict) -> dict:
    current_datasets = current_summary.get("datasets", {})
    baseline_datasets = baseline_summary.get("datasets", {})
    dataset_checks = []

    for dataset_name, current_dataset in current_datasets.items():
        baseline_dataset = baseline_datasets.get(dataset_name)
        if baseline_dataset is None:
            dataset_checks.append(
                {
                    "dataset": dataset_name,
                    "status": "missing_baseline",
                    "feature_checks": [],
                }
            )
            continue

        feature_checks = []
        for feature_name, current_stats in current_dataset.get("feature_statistics", {}).items():
            baseline_stats = baseline_dataset.get("feature_statistics", {}).get(feature_name, {})
            current_mean = current_stats.get("mean")
            baseline_mean = baseline_stats.get("mean")
            current_variance = current_stats.get("variance")
            baseline_variance = baseline_stats.get("variance")

            mean_shift = abs((current_mean or 0.0) - (baseline_mean or 0.0)) if current_mean is not None and baseline_mean is not None else None
            variance_shift = None
            if current_variance is not None and baseline_variance is not None:
                denominator = abs(baseline_variance) if baseline_variance else 1.0
                variance_shift = abs(current_variance - baseline_variance) / denominator

            current_distribution = _normalise_distribution(current_stats.get("distribution", {}))
            baseline_distribution = _normalise_distribution(baseline_stats.get("distribution", {}))
            shared_length = min(len(current_distribution), len(baseline_distribution))
            distribution_l1 = sum(
                abs(current_distribution[index] - baseline_distribution[index])
                for index in range(shared_length)
            )

            status = "healthy"
            if (
                (mean_shift is not None and mean_shift > FEATURE_MEAN_SHIFT_THRESHOLD)
                or (variance_shift is not None and variance_shift > FEATURE_VARIANCE_SHIFT_THRESHOLD)
                or distribution_l1 > FEATURE_DISTRIBUTION_L1_THRESHOLD
            ):
                status = "drift_alert"

            feature_checks.append(
                {
                    "feature": feature_name,
                    "mean_shift": mean_shift,
                    "variance_shift": variance_shift,
                    "distribution_l1": distribution_l1,
                    "status": status,
                }
            )

        dataset_status = "healthy"
        if any(check["status"] != "healthy" for check in feature_checks):
            dataset_status = "drift_alert"

        dataset_checks.append(
            {
                "dataset": dataset_name,
                "status": dataset_status,
                "feature_checks": feature_checks,
            }
        )

    overall = "healthy"
    if any(check["status"] != "healthy" for check in dataset_checks):
        overall = "alert"

    return {"overall_status": overall, "checks": dataset_checks}


def _required_paths_exist() -> bool:
    required = [
        Path("models/pneumonia/pneumonia_resnet50.pt"),
        Path("models/brain_tumor/brain_resnet50.pt"),
        Path("data/processed/chest_xray/test"),
        Path("data/splits/brain_tumor/test"),
    ]
    return all(path.exists() for path in required)


def run_monitoring(allow_missing_data: bool = False) -> int:
    configure_logging()
    set_global_seed(SEED)

    if not _required_paths_exist():
        message = "Monitoring skipped: required model/data artifacts are missing."
        logging.warning(message)
        print(message)
        return 0 if allow_missing_data else 1

    mlflow.set_experiment("model-monitoring")

    with mlflow.start_run(run_name="monitoring"):
        log_reproducibility_context(SEED, {"stage": "monitoring"})

        report = {
            "timestamp": mlflow.active_run().info.start_time,
            "device": DEVICE,
            "models": [
                check_dataset(
                    dataset_name="pneumonia",
                    model_path="models/pneumonia/pneumonia_resnet50.pt",
                    data_dir="data/processed/chest_xray",
                ),
                check_dataset(
                    dataset_name="brain_tumor",
                    model_path="models/brain_tumor/brain_resnet50.pt",
                    data_dir="data/splits/brain_tumor",
                ),
            ],
        }

        feature_records, feature_skipped = collect_feature_records(Path("data/processed"))
        current_feature_summary = build_feature_summary(feature_records)

        if FEATURE_BASELINE_FILE.exists():
            with open(FEATURE_BASELINE_FILE, "r", encoding="utf-8") as handle:
                feature_baseline = json.load(handle)
        else:
            FEATURE_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            feature_baseline = current_feature_summary
            with open(FEATURE_BASELINE_FILE, "w", encoding="utf-8") as handle:
                json.dump(feature_baseline, handle, indent=2)

        feature_comparison = _compare_feature_baseline(current_feature_summary, feature_baseline)
        report["feature_summary"] = current_feature_summary
        report["feature_comparison"] = feature_comparison
        report["feature_skipped_count"] = len(feature_skipped)

        baseline = create_or_load_baseline(report)
        comparison = compare_against_baseline(report, baseline)
        report["comparison"] = comparison

        with open(REPORT_FILE, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        for model in report["models"]:
            prefix = model["dataset"]
            mlflow.log_metric(f"{prefix}_accuracy", model["accuracy"])
            mlflow.log_metric(f"{prefix}_f1", model["f1"])
            mlflow.log_metric(f"{prefix}_drift_l1", model["drift_l1"])

        mlflow.log_param("drift_l1_threshold", DRIFT_L1_THRESHOLD)
        mlflow.log_param("accuracy_drop_threshold", ACCURACY_DROP_THRESHOLD)
        mlflow.log_param("feature_mean_shift_threshold", FEATURE_MEAN_SHIFT_THRESHOLD)
        mlflow.log_param("feature_variance_shift_threshold", FEATURE_VARIANCE_SHIFT_THRESHOLD)
        mlflow.log_param("feature_distribution_l1_threshold", FEATURE_DISTRIBUTION_L1_THRESHOLD)
        mlflow.set_tag("overall_status", comparison["overall_status"])
        mlflow.set_tag("feature_overall_status", feature_comparison["overall_status"])

        mlflow.log_artifact(str(REPORT_FILE), artifact_path="monitoring")
        mlflow.log_artifact(str(BASELINE_FILE), artifact_path="monitoring")
        mlflow.log_artifact(str(FEATURE_BASELINE_FILE), artifact_path="monitoring")

    logging.info("Monitoring report generated: %s", REPORT_FILE)
    print(f"Monitoring report generated at {REPORT_FILE}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor model performance and data drift.")
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Return success when model/data artifacts are missing.",
    )
    args = parser.parse_args()

    return run_monitoring(allow_missing_data=args.allow_missing_data)


if __name__ == "__main__":
    raise SystemExit(main())
