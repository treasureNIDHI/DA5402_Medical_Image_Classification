from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
import torch.nn as nn
from torchvision import models

REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "health_check_report.json"
MODEL_PATHS = {
    "pneumonia": Path("models/pneumonia/pneumonia_resnet50.pt"),
    "brain_tumor": Path("models/brain_tumor/brain_resnet50.pt"),
}
LATENCY_THRESHOLD_MS = 200.0
MEMORY_THRESHOLD_PCT = 90.0
DISK_THRESHOLD_PCT = 90.0
CPU_THRESHOLD_PCT = 95.0
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _check_system_resources() -> dict:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    cpu_pct = psutil.cpu_percent(interval=1)
    return {
        "cpu_percent": cpu_pct,
        "cpu_ok": cpu_pct < CPU_THRESHOLD_PCT,
        "memory_percent": mem.percent,
        "memory_available_gb": round(mem.available / 1e9, 2),
        "memory_ok": mem.percent < MEMORY_THRESHOLD_PCT,
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / 1e9, 2),
        "disk_ok": disk.percent < DISK_THRESHOLD_PCT,
    }


def _check_model(name: str, model_path: Path) -> dict:
    result: dict = {
        "name": name,
        "path": str(model_path),
        "exists": model_path.exists(),
        "loadable": False,
        "inference_ok": False,
        "latency_ms": None,
        "model_size_mb": None,
        "num_classes": None,
        "issues": [],
    }

    if not model_path.exists():
        result["issues"].append(f"Model file not found: {model_path}")
        return result

    result["model_size_mb"] = round(model_path.stat().st_size / 1e6, 2)

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        num_classes = state_dict["fc.weight"].shape[0]
        result["num_classes"] = num_classes

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        result["loadable"] = True
    except Exception as exc:
        result["issues"].append(f"Failed to load model: {exc}")
        return result

    try:
        dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
        # warmup
        with torch.no_grad():
            model(dummy)
        # timed run
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
        latency_ms = (time.perf_counter() - t0) / 10 * 1000
        result["latency_ms"] = round(latency_ms, 2)
        result["inference_ok"] = True
        if latency_ms > LATENCY_THRESHOLD_MS:
            result["issues"].append(
                f"Latency {latency_ms:.1f}ms exceeds threshold {LATENCY_THRESHOLD_MS}ms"
            )
    except Exception as exc:
        result["issues"].append(f"Inference failed: {exc}")

    return result


def _check_pipeline_artifacts() -> dict:
    required = [
        Path("reports/data_validation_report.json"),
        Path("reports/preprocessing_report.json"),
        Path("reports/eda_report.json"),
        Path("reports/evaluation.txt"),
        Path("reports/monitoring_report.json"),
        Path("reports/model_optimization_report.json"),
        Path("reports/feature_store/feature_baseline.json"),
        Path("dvc.lock"),
        Path("dvc.yaml"),
    ]
    present = [str(p) for p in required if p.exists()]
    missing = [str(p) for p in required if not p.exists()]
    return {
        "total_required": len(required),
        "present": len(present),
        "missing_count": len(missing),
        "missing": missing,
        "ok": len(missing) == 0,
    }


def run_health_check() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "overall_status": "healthy",
        "system": {},
        "models": {},
        "pipeline_artifacts": {},
        "thresholds": {
            "latency_ms": LATENCY_THRESHOLD_MS,
            "memory_pct": MEMORY_THRESHOLD_PCT,
            "disk_pct": DISK_THRESHOLD_PCT,
            "cpu_pct": CPU_THRESHOLD_PCT,
        },
        "issues": [],
    }

    # System resources
    system = _check_system_resources()
    report["system"] = system
    for key in ("cpu_ok", "memory_ok", "disk_ok"):
        if not system[key]:
            report["issues"].append(f"System resource issue: {key} = False")

    # Models
    for name, path in MODEL_PATHS.items():
        check = _check_model(name, path)
        report["models"][name] = check
        report["issues"].extend(check["issues"])

    # Pipeline artifacts
    artifacts = _check_pipeline_artifacts()
    report["pipeline_artifacts"] = artifacts
    if not artifacts["ok"]:
        report["issues"].append(f"Missing pipeline artifacts: {artifacts['missing']}")

    # Overall status
    report["overall_status"] = "healthy" if not report["issues"] else "degraded"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    status = report["overall_status"].upper()
    print(f"Health check {status}: {len(report['issues'])} issue(s)")
    for issue in report["issues"]:
        print(f"  ⚠ {issue}")
    print(f"Report written to {REPORT_FILE}")
    return 0 if report["overall_status"] == "healthy" else 1


if __name__ == "__main__":
    raise SystemExit(run_health_check())
