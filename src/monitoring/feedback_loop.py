"""
Feedback Loop System for Ground Truth Label Collection

Implements mechanism to:
1. Collect ground truth labels as they become available
2. Store predictions with timestamps for comparison
3. Calculate real-world performance metrics
4. Detect performance degradation over time
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path("reports/feedback_loop")
PREDICTIONS_FILE = FEEDBACK_DIR / "predictions.jsonl"
GROUND_TRUTH_FILE = FEEDBACK_DIR / "ground_truth.jsonl"
PERFORMANCE_FILE = FEEDBACK_DIR / "performance_metrics.json"

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Prediction:
    """Recorded prediction from model."""
    timestamp: str
    image_id: str
    model_type: str
    prediction: str
    confidence: float
    latency_ms: float


@dataclass
class GroundTruth:
    """Ground truth label provided later."""
    timestamp: str
    image_id: str
    ground_truth_label: str
    verified_by: str
    notes: Optional[str] = None


@dataclass
class PerformanceMetric:
    """Performance metrics for a time period."""
    period_start: str
    period_end: str
    total_predictions: int
    matched_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    model_type: str


class FeedbackLoopManager:
    """Manages collection and analysis of predictions and ground truth."""
    
    def __init__(self):
        self.predictions_file = PREDICTIONS_FILE
        self.ground_truth_file = GROUND_TRUTH_FILE
        self.performance_file = PERFORMANCE_FILE
        
    def record_prediction(
        self,
        image_id: str,
        model_type: str,
        prediction: str,
        confidence: float,
        latency_ms: float
    ) -> None:
        """Record a model prediction."""
        pred = Prediction(
            timestamp=datetime.now().isoformat(),
            image_id=image_id,
            model_type=model_type,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms
        )
        
        # Append to JSONL file
        with open(self.predictions_file, "a") as f:
            f.write(json.dumps(asdict(pred)) + "\n")
        
        logger.info(f"Recorded prediction for image {image_id}")
    
    def record_ground_truth(
        self,
        image_id: str,
        ground_truth_label: str,
        verified_by: str,
        notes: Optional[str] = None
    ) -> None:
        """Record ground truth label for an image."""
        gt = GroundTruth(
            timestamp=datetime.now().isoformat(),
            image_id=image_id,
            ground_truth_label=ground_truth_label,
            verified_by=verified_by,
            notes=notes
        )
        
        # Append to JSONL file
        with open(self.ground_truth_file, "a") as f:
            f.write(json.dumps(asdict(gt)) + "\n")
        
        logger.info(f"Recorded ground truth for image {image_id}: {ground_truth_label}")
    
    def load_predictions(self) -> List[Dict]:
        """Load all recorded predictions."""
        predictions = []
        if not self.predictions_file.exists():
            return predictions
        
        with open(self.predictions_file) as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        return predictions
    
    def load_ground_truths(self) -> List[Dict]:
        """Load all ground truth labels."""
        truths = []
        if not self.ground_truth_file.exists():
            return truths
        
        with open(self.ground_truth_file) as f:
            for line in f:
                if line.strip():
                    truths.append(json.loads(line))
        
        return truths
    
    def match_predictions_to_ground_truth(self) -> List[Dict]:
        """Match predictions with ground truth labels."""
        predictions = self.load_predictions()
        truths = self.load_ground_truths()
        
        # Create lookup for ground truths
        truth_dict = {t["image_id"]: t for t in truths}
        
        matches = []
        for pred in predictions:
            if pred["image_id"] in truth_dict:
                truth = truth_dict[pred["image_id"]]
                match = {
                    "image_id": pred["image_id"],
                    "prediction": pred["prediction"],
                    "ground_truth": truth["ground_truth_label"],
                    "confidence": pred["confidence"],
                    "correct": pred["prediction"] == truth["ground_truth_label"],
                    "prediction_time": pred["timestamp"],
                    "truth_time": truth["timestamp"],
                    "latency_to_feedback": self._calculate_latency(
                        pred["timestamp"], 
                        truth["timestamp"]
                    ),
                    "model_type": pred["model_type"]
                }
                matches.append(match)
        
        return matches
    
    def _calculate_latency(self, pred_time: str, truth_time: str) -> float:
        """Calculate time between prediction and ground truth."""
        pred_dt = datetime.fromisoformat(pred_time)
        truth_dt = datetime.fromisoformat(truth_time)
        return (truth_dt - pred_dt).total_seconds() / 3600  # Hours
    
    def calculate_performance_metrics(
        self,
        model_type: Optional[str] = None,
        days: int = 1
    ) -> Dict:
        """Calculate performance metrics for recent predictions."""
        matches = self.match_predictions_to_ground_truth()
        
        # Filter by model type if specified
        if model_type:
            matches = [m for m in matches if m["model_type"] == model_type]
        
        # Filter by recency
        cutoff_time = datetime.now() - timedelta(days=days)
        matches = [
            m for m in matches 
            if datetime.fromisoformat(m["prediction_time"]) > cutoff_time
        ]
        
        if not matches:
            return {
                "total_predictions": 0,
                "matched_predictions": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "model_type": model_type or "all",
                "period_days": days
            }
        
        # Calculate metrics
        correct = sum(1 for m in matches if m["correct"])
        accuracy = correct / len(matches) if matches else 0
        
        # Per-class metrics for multi-class
        unique_labels = set(m["ground_truth"] for m in matches)
        
        precision_list = []
        recall_list = []
        
        for label in unique_labels:
            tp = sum(1 for m in matches if m["correct"] and m["prediction"] == label)
            fp = sum(1 for m in matches if not m["correct"] and m["prediction"] == label)
            fn = sum(1 for m in matches if not m["correct"] and m["ground_truth"] == label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
        
        macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0
        macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0
        
        f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) \
            if (macro_precision + macro_recall) > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.load_predictions()),
            "matched_predictions": len(matches),
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": f1,
            "model_type": model_type or "all",
            "period_days": days,
            "average_confidence": sum(m["confidence"] for m in matches) / len(matches),
            "average_latency_hours": sum(m["latency_to_feedback"] for m in matches) / len(matches)
        }
    
    def detect_performance_degradation(
        self,
        threshold: float = 0.05,
        baseline_days: int = 7,
        current_days: int = 1
    ) -> Dict:
        """Detect if current performance has degraded from baseline."""
        baseline_metrics = self.calculate_performance_metrics(days=baseline_days)
        current_metrics = self.calculate_performance_metrics(days=current_days)
        
        if baseline_metrics["accuracy"] == 0 or current_metrics["accuracy"] == 0:
            return {
                "degradation_detected": False,
                "reason": "Insufficient data",
                "baseline_accuracy": baseline_metrics["accuracy"],
                "current_accuracy": current_metrics["accuracy"]
            }
        
        accuracy_drop = baseline_metrics["accuracy"] - current_metrics["accuracy"]
        degraded = accuracy_drop > threshold
        
        return {
            "degradation_detected": degraded,
            "accuracy_drop": accuracy_drop,
            "threshold": threshold,
            "baseline_accuracy": baseline_metrics["accuracy"],
            "current_accuracy": current_metrics["accuracy"],
            "baseline_period_days": baseline_days,
            "current_period_days": current_days,
            "requires_retraining": degraded,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_performance_report(self) -> None:
        """Save comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "last_7_days": self.calculate_performance_metrics(days=7),
            "last_24_hours": self.calculate_performance_metrics(days=1),
            "pneumonia_model": self.calculate_performance_metrics("pneumonia", days=7),
            "brain_tumor_model": self.calculate_performance_metrics("brain", days=7),
            "degradation_check": self.detect_performance_degradation(),
            "total_predictions": len(self.load_predictions()),
            "total_ground_truths": len(self.load_ground_truths()),
            "matched_records": len(self.match_predictions_to_ground_truth())
        }
        
        with open(self.performance_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {self.performance_file}")
        return report
    
    def get_unmapped_predictions(self) -> List[Dict]:
        """Get predictions without ground truth labels."""
        predictions = self.load_predictions()
        truths = self.load_ground_truths()
        
        truth_ids = {t["image_id"] for t in truths}
        unmapped = [p for p in predictions if p["image_id"] not in truth_ids]
        
        return unmapped
    
    def export_for_labeling(self, output_file: str = "reports/predictions_to_label.jsonl") -> None:
        """Export unmapped predictions for manual labeling."""
        unmapped = self.get_unmapped_predictions()
        
        with open(output_file, "w") as f:
            for pred in unmapped:
                f.write(json.dumps(pred) + "\n")
        
        logger.info(f"Exported {len(unmapped)} predictions to {output_file}")


# Global instance
_feedback_manager: Optional[FeedbackLoopManager] = None


def get_feedback_manager() -> FeedbackLoopManager:
    """Get or create feedback loop manager."""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = FeedbackLoopManager()
    return _feedback_manager


def record_prediction_callback(
    image_id: str,
    model_type: str,
    prediction: str,
    confidence: float,
    latency_ms: float
) -> None:
    """Callback to record predictions from API."""
    manager = get_feedback_manager()
    manager.record_prediction(image_id, model_type, prediction, confidence, latency_ms)


def record_ground_truth_callback(
    image_id: str,
    ground_truth_label: str,
    verified_by: str,
    notes: Optional[str] = None
) -> None:
    """Callback to record ground truth labels."""
    manager = get_feedback_manager()
    manager.record_ground_truth(image_id, ground_truth_label, verified_by, notes)


if __name__ == "__main__":
    # Test the feedback loop
    manager = get_feedback_manager()
    
    # Save performance report
    report = manager.save_performance_report()
    print(json.dumps(report, indent=2, default=str))
