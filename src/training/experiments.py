"""Model experimentation and hyperparameter tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPORTS_DIR = Path("reports")
EXPERIMENTS_FILE = REPORTS_DIR / "model_experiments.json"


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    name: str
    architecture: str
    pretrained: bool
    num_classes: int
    learning_rate: float
    optimizer: str
    batch_size: int
    epochs: int
    weight_decay: float = 0.0
    dropout: float = 0.0
    augmentation: bool = False
    scheduler: str = "none"


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    experiment_id: str
    config: dict
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_f1: float
    val_f1: float
    test_f1: float
    best_epoch: int
    training_time_seconds: float
    model_size_mb: float
    inference_latency_ms: float
    notes: str


# Predefined model configurations for comparison
MODEL_CONFIGS = {
    "resnet50_baseline": ModelConfig(
        name="ResNet-50 Baseline",
        architecture="resnet50",
        pretrained=True,
        num_classes=2,
        learning_rate=0.0001,
        optimizer="adam",
        batch_size=32,
        epochs=5,
    ),
    "resnet50_aggressive": ModelConfig(
        name="ResNet-50 Aggressive",
        architecture="resnet50",
        pretrained=True,
        num_classes=2,
        learning_rate=0.001,
        optimizer="adam",
        batch_size=16,
        epochs=10,
        weight_decay=0.0001,
    ),
    "resnet34_lightweight": ModelConfig(
        name="ResNet-34 Lightweight",
        architecture="resnet34",
        pretrained=True,
        num_classes=2,
        learning_rate=0.0001,
        optimizer="sgd",
        batch_size=32,
        epochs=5,
        weight_decay=0.0001,
    ),
    "mobilenet_edge": ModelConfig(
        name="MobileNet Edge",
        architecture="mobilenet_v2",
        pretrained=True,
        num_classes=2,
        learning_rate=0.0001,
        optimizer="adam",
        batch_size=64,
        epochs=5,
    ),
    "efficientnet_balanced": ModelConfig(
        name="EfficientNet B0 Balanced",
        architecture="efficientnet_b0",
        pretrained=True,
        num_classes=2,
        learning_rate=0.00005,
        optimizer="adam",
        batch_size=32,
        epochs=10,
        weight_decay=0.0001,
    ),
}


class ExperimentTracker:
    """Track and compare model experiments."""
    
    def __init__(self, experiments_file: Path = EXPERIMENTS_FILE):
        self.experiments_file = experiments_file
        self.experiments: dict[str, ExperimentResult] = self._load_experiments()
    
    def _load_experiments(self) -> dict[str, ExperimentResult]:
        """Load existing experiments from file."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                data = json.load(f)
                return data.get("experiments", {})
        return {}
    
    def add_experiment(self, result: ExperimentResult) -> None:
        """Add new experiment result."""
        self.experiments[result.experiment_id] = asdict(result)
        self._save_experiments()
    
    def _save_experiments(self) -> None:
        """Save experiments to file."""
        self.experiments_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.experiments_file, 'w') as f:
            json.dump(
                {
                    "experiments": self.experiments,
                    "best_by_accuracy": self._get_best_by_metric("test_accuracy"),
                    "best_by_f1": self._get_best_by_metric("test_f1"),
                    "fastest_inference": self._get_best_by_metric("inference_latency_ms", ascending=True),
                    "smallest_model": self._get_best_by_metric("model_size_mb", ascending=True),
                },
                f,
                indent=2
            )
    
    def _get_best_by_metric(self, metric: str, ascending: bool = False) -> dict[str, Any] | None:
        """Get best experiment by metric."""
        if not self.experiments:
            return None
        
        best_exp_id = None
        best_value = float('inf') if ascending else float('-inf')
        
        for exp_id, exp_data in self.experiments.items():
            value = exp_data.get(metric)
            if value is None:
                continue
            
            if ascending:
                if value < best_value:
                    best_value = value
                    best_exp_id = exp_id
            else:
                if value > best_value:
                    best_value = value
                    best_exp_id = exp_id
        
        if best_exp_id:
            return {"experiment_id": best_exp_id, "value": best_value}
        return None
    
    def get_comparison_report(self) -> str:
        """Generate comparison report."""
        if not self.experiments:
            return "No experiments tracked yet."
        
        lines = ["=" * 100]
        lines.append("MODEL EXPERIMENT COMPARISON REPORT")
        lines.append("=" * 100)
        lines.append("")
        
        # Sort by test accuracy
        sorted_exps = sorted(
            self.experiments.items(),
            key=lambda x: x[1].get("test_accuracy", 0),
            reverse=True
        )
        
        lines.append(f"{'Experiment ID':<25} {'Model':<20} {'Test Acc':<10} {'Test F1':<10} {'Latency':<10} {'Size':<10}")
        lines.append("-" * 100)
        
        for exp_id, exp_data in sorted_exps:
            config = exp_data.get("config", {})
            model_name = config.get("architecture", "unknown")[:20]
            test_acc = exp_data.get("test_accuracy", 0)
            test_f1 = exp_data.get("test_f1", 0)
            latency = exp_data.get("inference_latency_ms", 0)
            size = exp_data.get("model_size_mb", 0)
            
            lines.append(
                f"{exp_id:<25} {model_name:<20} {test_acc:<10.4f} {test_f1:<10.4f} {latency:<10.2f} {size:<10.2f}"
            )
        
        lines.append("")
        lines.append("=" * 100)
        
        return "\n".join(lines)


def initialize_experiment_configs() -> None:
    """Initialize default experiment configurations."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    config_file = REPORTS_DIR / "model_configs.json"
    configs_data = {
        "available_configs": {
            name: asdict(config)
            for name, config in MODEL_CONFIGS.items()
        },
        "recommendations": {
            "for_accuracy": {
                "config": "resnet50_aggressive",
                "reason": "Higher learning rate and more epochs for convergence",
                "expected_accuracy": "0.95+",
            },
            "for_speed": {
                "config": "mobilenet_edge",
                "reason": "Lightweight architecture optimized for inference",
                "expected_latency": "<50ms",
            },
            "for_production": {
                "config": "resnet50_baseline",
                "reason": "Good balance of accuracy, speed, and model size",
                "expected_accuracy": "0.92+",
                "expected_latency": "<100ms",
            },
        },
    }
    
    with open(config_file, 'w') as f:
        json.dump(configs_data, f, indent=2)
    
    print(f"✓ Model configurations initialized at {config_file}")


def main() -> int:
    initialize_experiment_configs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
