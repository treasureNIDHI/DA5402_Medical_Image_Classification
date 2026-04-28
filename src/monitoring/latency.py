from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from PIL import Image

REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "latency_report.json"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_WARMUP_RUNS = 5
DEFAULT_MEASURED_RUNS = 20
DEFAULT_THRESHOLD_MS = 200.0
REQUIRED_MODELS = {
    "pneumonia": Path("models/pneumonia/pneumonia_resnet50.pt"),
    "brain": Path("models/brain_tumor/brain_resnet50.pt"),
}


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _create_dummy_image() -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    handle.close()
    image = Image.new("RGB", DEFAULT_IMAGE_SIZE, color=(128, 128, 128))
    image.save(handle.name)
    return Path(handle.name)


def _build_sample_image(image_path: str | None) -> tuple[Path, bool]:
    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        return path, False
    return _create_dummy_image(), True


def _check_models_exist() -> list[str]:
    missing = []
    for name, path in REQUIRED_MODELS.items():
        if not path.exists():
            missing.append(name)
    return missing


def _load_predictor():
    from src.inference.predict import predict  # local import to avoid eager model loading on missing artifacts

    return predict


def benchmark_model(model_type: str, image_path: Path, runs: int, warmup_runs: int) -> dict:
    predict = _load_predictor()

    for _ in range(warmup_runs):
        predict(str(image_path), model_type)

    latencies_ms = []
    last_prediction = None
    for _ in range(runs):
        start = time.perf_counter()
        last_prediction = predict(str(image_path), model_type)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    return {
        "model_type": model_type,
        "runs": runs,
        "warmup_runs": warmup_runs,
        "latencies_ms": latencies_ms,
        "mean_latency_ms": statistics.mean(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "p95_latency_ms": statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms),
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "last_prediction": last_prediction,
    }


def run_latency_benchmark(model_type: str, image_path: str | None, runs: int, warmup_runs: int, threshold_ms: float) -> int:
    _ensure_reports_dir()
    sample_image, created_temp_image = _build_sample_image(image_path)

    missing_models = _check_models_exist()
    if missing_models:
        report = {
            "status": "skipped",
            "reason": "missing_model_artifacts",
            "missing_models": missing_models,
            "threshold_ms": threshold_ms,
        }
        with open(REPORT_FILE, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Latency benchmark skipped. Missing models: {missing_models}")
        return 0

    try:
        if model_type == "all":
            benchmark_results = [
                benchmark_model("pneumonia", sample_image, runs, warmup_runs),
                benchmark_model("brain", sample_image, runs, warmup_runs),
            ]
        elif model_type in {"pneumonia", "brain"}:
            benchmark_results = [benchmark_model(model_type, sample_image, runs, warmup_runs)]
        else:
            raise ValueError("model_type must be one of: pneumonia, brain, all")

        overall_status = "healthy"
        if any(result["median_latency_ms"] > threshold_ms for result in benchmark_results):
            overall_status = "needs_attention"

        report = {
            "status": overall_status,
            "threshold_ms": threshold_ms,
            "image_path": str(sample_image),
            "results": benchmark_results,
        }

        with open(REPORT_FILE, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        print(f"Latency benchmark written to {REPORT_FILE}")
        for result in benchmark_results:
            print(
                f"{result['model_type']}: median={result['median_latency_ms']:.2f}ms, "
                f"p95={result['p95_latency_ms']:.2f}ms, mean={result['mean_latency_ms']:.2f}ms"
            )

        return 0
    finally:
        if created_temp_image and sample_image.exists():
            sample_image.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark inference latency for the deployed models.")
    parser.add_argument("--model-type", choices=["pneumonia", "brain", "all"], default="all")
    parser.add_argument("--image", default=None, help="Optional image path to benchmark with.")
    parser.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--threshold-ms", type=float, default=DEFAULT_THRESHOLD_MS)
    args = parser.parse_args()

    return run_latency_benchmark(
        model_type=args.model_type,
        image_path=args.image,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        threshold_ms=args.threshold_ms,
    )


if __name__ == "__main__":
    raise SystemExit(main())
