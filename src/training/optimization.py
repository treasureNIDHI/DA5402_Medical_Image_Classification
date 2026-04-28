"""Model optimization utilities for resource-constrained environments."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from pathlib import Path

REPORTS_DIR = Path("reports")
OPTIMIZATION_REPORT_FILE = REPORTS_DIR / "model_optimization_report.json"


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_model_size(model: nn.Module) -> dict:
    """Calculate model size in MB and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Save to temp file to get actual disk size
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        disk_size_mb = Path(f.name).stat().st_size / (1024**2)
        Path(f.name).unlink()
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "disk_size_mb": disk_size_mb,
        "total_params_millions": total_params / 1e6,
    }


def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to reduce model size (CPU-only operation)."""
    # qnnpack is the only backend available on macOS/ARM; fbgemm on x86 Linux
    backend = "qnnpack" if torch.backends.quantized.supported_engines and "qnnpack" in torch.backends.quantized.supported_engines else "fbgemm"
    torch.backends.quantized.engine = backend
    model_cpu = model.to("cpu")
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def prune_model_structured(model: nn.Module, pruning_amount: float = 0.3) -> nn.Module:
    """Apply structured pruning to model layers."""
    import torch.nn.utils.prune as prune
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=pruning_amount, n=2, dim=0)
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=pruning_amount, n=2, dim=0)
    
    # Make pruning permanent
    for module in model.modules():
        try:
            prune.remove(module, 'weight')
        except ValueError:
            pass
    
    return model


def prune_model_unstructured(model: nn.Module, pruning_amount: float = 0.2) -> nn.Module:
    """Apply unstructured pruning to model weights."""
    import torch.nn.utils.prune as prune
    
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    if parameters_to_prune:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount,
        )
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    
    return model


def optimize_model(model: nn.Module, optimization_type: str = "quantize", **kwargs) -> tuple[nn.Module, dict]:
    """
    Apply optimization technique to model.
    
    Args:
        model: PyTorch model to optimize
        optimization_type: One of 'quantize', 'prune_structured', 'prune_unstructured'
        **kwargs: Additional arguments for optimization
    
    Returns:
        Tuple of (optimized_model, optimization_report)
    """
    _ensure_reports_dir()
    
    original_size = calculate_model_size(model)
    
    if optimization_type == "quantize":
        optimized_model = quantize_model_dynamic(model)
        technique = "Dynamic Quantization (INT8, CPU)"
    elif optimization_type == "prune_structured":
        pruning_amount = kwargs.get("pruning_amount", 0.3)
        optimized_model = prune_model_structured(model, pruning_amount)
        technique = f"Structured Pruning ({pruning_amount*100:.0f}%)"
    elif optimization_type == "prune_unstructured":
        pruning_amount = kwargs.get("pruning_amount", 0.2)
        optimized_model = prune_model_unstructured(model, pruning_amount)
        technique = f"Unstructured Pruning ({pruning_amount*100:.0f}%)"
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    optimized_size = calculate_model_size(optimized_model)
    
    size_reduction = (original_size["disk_size_mb"] - optimized_size["disk_size_mb"]) / original_size["disk_size_mb"] * 100
    param_reduction = (original_size["total_parameters"] - optimized_size["total_parameters"]) / original_size["total_parameters"] * 100
    
    report = {
        "technique": technique,
        "original": original_size,
        "optimized": optimized_size,
        "reduction": {
            "disk_size_percent": size_reduction,
            "parameters_percent": param_reduction,
            "disk_size_mb_saved": original_size["disk_size_mb"] - optimized_size["disk_size_mb"],
        },
    }
    
    return optimized_model, report


def benchmark_inference_speed(model: nn.Module, input_shape: tuple = (1, 3, 224, 224), device: str = "cpu", num_runs: int = 100) -> dict:
    """Benchmark model inference speed."""
    import time
    
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    import statistics
    return {
        "device": device,
        "input_shape": input_shape,
        "num_runs": num_runs,
        "mean_latency_ms": statistics.mean(times),
        "median_latency_ms": statistics.median(times),
        "min_latency_ms": min(times),
        "max_latency_ms": max(times),
        "stdev_latency_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def run_optimization_analysis(model_path: str, device: str = "cpu") -> int:
    """Run comprehensive model optimization analysis."""
    _ensure_reports_dir()

    from torchvision import models

    # Load model with dynamic num_classes (same pattern as train/evaluate)
    model = models.resnet50(weights=None)
    try:
        state_dict = torch.load(model_path, map_location=device)
        num_classes = state_dict["fc.weight"].shape[0]
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Note: Could not load model weights, analyzing architecture only: {e}")
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    model.to(device)
    
    report = {
        "model": "resnet50",
        "device": device,
        "techniques": {},
        "inference_benchmarks": {},
    }
    
    # Baseline
    print("📊 Analyzing baseline model...")
    baseline_size = calculate_model_size(model)
    baseline_inference = benchmark_inference_speed(model, device=device)
    
    report["baseline"] = {
        "size": baseline_size,
        "inference": baseline_inference,
    }
    
    import copy

    # Dynamic Quantization (always CPU — quantize_dynamic is CPU-only)
    print("🔧 Applying dynamic quantization...")
    model_quantized, opt_report_quant = optimize_model(copy.deepcopy(model).to("cpu"), "quantize")
    inference_quantized = benchmark_inference_speed(model_quantized, device="cpu")
    opt_report_quant["inference"] = inference_quantized
    report["techniques"]["dynamic_quantization"] = opt_report_quant

    # Structured Pruning
    print("✂️  Applying structured pruning...")
    model_pruned_struct, opt_report_struct = optimize_model(copy.deepcopy(model).to("cpu"), "prune_structured", pruning_amount=0.3)
    inference_pruned_struct = benchmark_inference_speed(model_pruned_struct, device="cpu")
    opt_report_struct["inference"] = inference_pruned_struct
    report["techniques"]["structured_pruning"] = opt_report_struct

    # Unstructured Pruning
    print("✂️  Applying unstructured pruning...")
    model_pruned_unstruct, opt_report_unstruct = optimize_model(copy.deepcopy(model).to("cpu"), "prune_unstructured", pruning_amount=0.2)
    inference_pruned_unstruct = benchmark_inference_speed(model_pruned_unstruct, device="cpu")
    opt_report_unstruct["inference"] = inference_pruned_unstruct
    report["techniques"]["unstructured_pruning"] = opt_report_unstruct
    
    # Add recommendations
    report["recommendations"] = {
        "for_mobile": "Use dynamic quantization for mobile deployment (best size reduction with minimal latency increase)",
        "for_embedded": "Use structured pruning + quantization for memory-constrained devices",
        "for_cpu": "Unstructured pruning provides good speed-up on CPU",
        "for_inference": "Quantization has fastest inference, best for latency-sensitive apps",
    }
    
    with open(OPTIMIZATION_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Model optimization analysis written to {OPTIMIZATION_REPORT_FILE}")
    print(f"\n📈 Key Findings:")
    print(f"  • Baseline size: {baseline_size['disk_size_mb']:.2f} MB")
    print(f"  • Quantized size: {opt_report_quant['optimized']['disk_size_mb']:.2f} MB ({opt_report_quant['reduction']['disk_size_percent']:.1f}% reduction)")
    print(f"  • Baseline latency: {baseline_inference['mean_latency_ms']:.2f} ms")
    print(f"  • Quantized latency: {inference_quantized['mean_latency_ms']:.2f} ms")
    
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze and optimize models for resource-constrained environments.")
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=default_device)
    args = parser.parse_args()

    model_paths = [
        "models/pneumonia/pneumonia_resnet50.pt",
        "models/brain_tumor/brain_resnet50.pt",
    ]

    for path in model_paths:
        if Path(path).exists():
            print(f"\n--- Optimizing {path} ---")
            run_optimization_analysis(path, args.device)
        else:
            print(f"Skipping {path} (not found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
