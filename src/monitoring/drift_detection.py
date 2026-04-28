"""
Advanced Data Drift Detection System

Implements multiple drift detection methods:
1. Kolmogorov-Smirnov test (statistical)
2. Jensen-Shannon divergence (distribution)
3. Prediction entropy (model confidence)
4. Feature statistics monitoring
5. Anomaly detection on features
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DRIFT_REPORTS_DIR = Path("reports/drift_detection")
DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
KS_THRESHOLD = 0.15  # Kolmogorov-Smirnov
JS_THRESHOLD = 0.25  # Jensen-Shannon divergence
ENTROPY_THRESHOLD = 0.7  # Prediction entropy


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    timestamp: str
    drift_type: str
    metric_name: str
    current_value: float
    threshold: float
    exceeded: bool
    severity: str  # low, medium, high, critical


class AdvancedDriftDetector:
    """Advanced drift detection with multiple methods."""
    
    def __init__(self, baseline_file: str = "reports/feature_store/feature_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline = self._load_baseline()
        self.drift_history = []
        
    def _load_baseline(self) -> Dict:
        """Load baseline statistics."""
        if not self.baseline_file.exists():
            logger.warning(f"Baseline file not found: {self.baseline_file}")
            return {}
        
        with open(self.baseline_file) as f:
            return json.load(f)
    
    def ks_test_detection(
        self,
        feature_name: str,
        current_distribution: List[float]
    ) -> Dict:
        """Kolmogorov-Smirnov test for univariate drift."""
        if feature_name not in self.baseline:
            return {"status": "unknown", "reason": "No baseline"}
        
        baseline_values = self.baseline[feature_name].get("values", [])
        if not baseline_values or not current_distribution:
            return {"status": "insufficient_data"}
        
        # KS test
        statistic, p_value = stats.ks_2samp(baseline_values, current_distribution)
        
        drifted = statistic > KS_THRESHOLD
        
        return {
            "test": "kolmogorov_smirnov",
            "feature": feature_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": KS_THRESHOLD,
            "drifted": drifted,
            "severity": "high" if statistic > 0.25 else "medium" if drifted else "low"
        }
    
    def jensen_shannon_divergence(
        self,
        feature_name: str,
        current_distribution: np.ndarray,
        bins: int = 20
    ) -> Dict:
        """Jensen-Shannon divergence for distribution comparison."""
        if feature_name not in self.baseline:
            return {"status": "unknown"}
        
        baseline_values = np.array(self.baseline[feature_name].get("values", []))
        current_values = np.array(current_distribution)
        
        if len(baseline_values) == 0 or len(current_values) == 0:
            return {"status": "insufficient_data"}
        
        # Create histograms
        baseline_min = min(baseline_values.min(), current_values.min())
        baseline_max = max(baseline_values.max(), current_values.max())
        
        baseline_hist, _ = np.histogram(baseline_values, bins=bins, range=(baseline_min, baseline_max))
        current_hist, _ = np.histogram(current_values, bins=bins, range=(baseline_min, baseline_max))
        
        # Normalize
        baseline_hist = baseline_hist / baseline_hist.sum() + 1e-10
        current_hist = current_hist / current_hist.sum() + 1e-10
        
        # JS divergence
        js_div = jensenshannon(baseline_hist, current_hist)
        
        drifted = js_div > JS_THRESHOLD
        
        return {
            "test": "jensen_shannon",
            "feature": feature_name,
            "divergence": float(js_div),
            "threshold": JS_THRESHOLD,
            "drifted": drifted,
            "severity": "critical" if js_div > 0.4 else "high" if drifted else "low"
        }
    
    def wasserstein_distance(
        self,
        feature_name: str,
        current_distribution: List[float]
    ) -> Dict:
        """Wasserstein distance (earth mover's distance)."""
        if feature_name not in self.baseline:
            return {"status": "unknown"}
        
        baseline_values = np.array(self.baseline[feature_name].get("values", []))
        current_values = np.array(current_distribution)
        
        if len(baseline_values) == 0 or len(current_values) == 0:
            return {"status": "insufficient_data"}
        
        # Wasserstein distance
        distance = stats.wasserstein_distance(baseline_values, current_values)
        
        # Normalize by baseline range
        baseline_range = baseline_values.max() - baseline_values.min()
        if baseline_range > 0:
            normalized_distance = distance / baseline_range
        else:
            normalized_distance = distance
        
        threshold = 0.15  # 15% of baseline range
        drifted = normalized_distance > threshold
        
        return {
            "test": "wasserstein_distance",
            "feature": feature_name,
            "distance": float(distance),
            "normalized_distance": float(normalized_distance),
            "threshold": threshold,
            "drifted": drifted,
            "severity": "high" if normalized_distance > 0.25 else "medium" if drifted else "low"
        }
    
    def prediction_entropy_drift(
        self,
        predictions: List[Dict]
    ) -> Dict:
        """Detect drift via prediction entropy (model uncertainty)."""
        if not predictions:
            return {"status": "insufficient_data"}
        
        confidences = [p["confidence"] for p in predictions]
        
        # Calculate entropy: higher entropy = more uncertainty
        confidence_array = np.array(confidences)
        
        # Expected entropy (baseline)
        baseline_entropy = self.baseline.get("prediction_entropy", 0.5)
        
        # Current entropy
        current_entropy = float(-np.mean(
            confidence_array * np.log(confidence_array + 1e-10) +
            (1 - confidence_array) * np.log(1 - confidence_array + 1e-10)
        ))
        
        entropy_increase = current_entropy - baseline_entropy
        drifted = entropy_increase > ENTROPY_THRESHOLD
        
        return {
            "test": "prediction_entropy",
            "baseline_entropy": float(baseline_entropy),
            "current_entropy": float(current_entropy),
            "entropy_increase": float(entropy_increase),
            "threshold": ENTROPY_THRESHOLD,
            "drifted": drifted,
            "severity": "high" if entropy_increase > 0.15 else "medium" if drifted else "low",
            "interpretation": "Model becoming more uncertain"
        }
    
    def feature_statistics_drift(
        self,
        features: Dict[str, List[float]],
        threshold_pct: float = 0.1  # 10% change
    ) -> Dict:
        """Monitor feature statistics for drift."""
        results = []
        
        for feature_name, values in features.items():
            if feature_name not in self.baseline:
                continue
            
            baseline_stat = self.baseline[feature_name]
            baseline_mean = baseline_stat.get("mean", 0)
            baseline_std = baseline_stat.get("std", 1)
            
            values_array = np.array(values)
            current_mean = np.mean(values_array)
            current_std = np.std(values_array)
            
            # Percentage change
            mean_pct_change = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-10)
            std_pct_change = abs(current_std - baseline_std) / (abs(baseline_std) + 1e-10)
            
            drifted_mean = mean_pct_change > threshold_pct
            drifted_std = std_pct_change > threshold_pct
            
            results.append({
                "feature": feature_name,
                "baseline_mean": float(baseline_mean),
                "current_mean": float(current_mean),
                "mean_pct_change": float(mean_pct_change),
                "mean_drifted": drifted_mean,
                "baseline_std": float(baseline_std),
                "current_std": float(current_std),
                "std_pct_change": float(std_pct_change),
                "std_drifted": drifted_std,
                "threshold_pct": threshold_pct,
                "severity": "high" if (drifted_mean or drifted_std) else "low"
            })
        
        return {
            "test": "feature_statistics",
            "results": results,
            "total_drifted": sum(1 for r in results if r["mean_drifted"] or r["std_drifted"]),
            "total_features": len(results)
        }
    
    def detect_multivariate_drift(
        self,
        current_features: np.ndarray,
        method: str = "mahalanobis"
    ) -> Dict:
        """Detect multivariate drift using Mahalanobis distance."""
        if not self.baseline:
            return {"status": "no_baseline"}
        
        baseline_data = np.array([
            self.baseline[k].get("values", []) 
            for k in self.baseline if k.startswith("feature_")
        ])
        
        if baseline_data.size == 0:
            return {"status": "insufficient_baseline_data"}
        
        # Calculate covariance
        cov_matrix = np.cov(baseline_data)
        
        # Handle singular matrix
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_matrix)
        
        baseline_mean = np.mean(baseline_data, axis=1)
        
        # Mahalanobis distance for each point
        distances = []
        for point in current_features:
            diff = point - baseline_mean
            distance = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
            distances.append(distance)
        
        mean_distance = np.mean(distances)
        threshold = np.mean(distances) + 2 * np.std(distances)
        
        return {
            "test": "mahalanobis_distance",
            "mean_distance": float(mean_distance),
            "threshold": float(threshold),
            "drifted": mean_distance > threshold,
            "max_distance": float(max(distances)),
            "min_distance": float(min(distances))
        }
    
    def run_full_drift_detection(
        self,
        features: Dict[str, List[float]],
        predictions: List[Dict]
    ) -> Dict:
        """Run comprehensive drift detection."""
        timestamp = datetime.now().isoformat()
        
        results = {
            "timestamp": timestamp,
            "tests": [],
            "overall_drift": False,
            "severity_levels": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "alerts": []
        }
        
        # Test 1: Prediction entropy
        entropy_test = self.prediction_entropy_drift(predictions)
        results["tests"].append(entropy_test)
        if entropy_test.get("drifted"):
            results["overall_drift"] = True
            results["alerts"].append(DriftAlert(
                timestamp=timestamp,
                drift_type="prediction_entropy",
                metric_name="prediction_entropy",
                current_value=entropy_test.get("current_entropy", 0),
                threshold=entropy_test.get("threshold", 0),
                exceeded=True,
                severity=entropy_test.get("severity", "medium")
            ))
        
        # Test 2: Feature statistics
        feature_test = self.feature_statistics_drift(features)
        results["tests"].append(feature_test)
        if feature_test.get("total_drifted", 0) > 0:
            results["overall_drift"] = True
            for feature_result in feature_test.get("results", []):
                if feature_result.get("mean_drifted"):
                    results["alerts"].append(DriftAlert(
                        timestamp=timestamp,
                        drift_type="feature_statistics",
                        metric_name=f"{feature_result['feature']}_mean",
                        current_value=feature_result.get("current_mean", 0),
                        threshold=feature_result.get("baseline_mean", 0),
                        exceeded=True,
                        severity=feature_result.get("severity", "low")
                    ))
        
        # Test 3: KS test for each feature
        for feature_name, values in features.items():
            ks_test = self.ks_test_detection(feature_name, values)
            if ks_test.get("drifted"):
                results["tests"].append(ks_test)
                results["overall_drift"] = True
        
        # Count severity levels
        for alert in results["alerts"]:
            severity = alert.severity if isinstance(alert, dict) else alert.get("severity", "low")
            if severity in results["severity_levels"]:
                results["severity_levels"][severity] += 1
        
        # Convert alerts to dicts
        results["alerts"] = [asdict(a) if isinstance(a, DriftAlert) else a for a in results["alerts"]]
        
        return results
    
    def save_drift_report(self, drift_results: Dict) -> None:
        """Save drift detection report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = DRIFT_REPORTS_DIR / f"drift_report_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump(drift_results, f, indent=2, default=str)
        
        logger.info(f"Drift report saved to {report_file}")


# Global instance
_drift_detector: Optional[AdvancedDriftDetector] = None


def get_drift_detector() -> AdvancedDriftDetector:
    """Get or create drift detector."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = AdvancedDriftDetector()
    return _drift_detector


if __name__ == "__main__":
    # Test drift detection
    detector = get_drift_detector()
    
    # Example usage
    test_features = {
        "width": np.random.normal(224, 10, 100).tolist(),
        "height": np.random.normal(224, 10, 100).tolist(),
        "mean_intensity": np.random.normal(128, 30, 100).tolist()
    }
    
    test_predictions = [
        {"confidence": 0.95},
        {"confidence": 0.87},
        {"confidence": 0.92}
    ]
    
    results = detector.run_full_drift_detection(test_features, test_predictions)
    print(json.dumps(results, indent=2, default=str))
