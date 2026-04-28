"""
Automated Model Retraining Pipeline

Implements automatic retraining when:
1. Performance degrades below threshold
2. Data drift is detected
3. On a scheduled basis (periodic)
4. On-demand retraining triggered by monitoring
"""

import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRAINING_LOG_DIR = Path("reports/retraining")
RETRAINING_CONFIG_FILE = RETRAINING_LOG_DIR / "retraining_config.json"
RETRAINING_HISTORY_FILE = RETRAINING_LOG_DIR / "retraining_history.jsonl"

RETRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)


class RetrainingTrigger:
    """Triggers for automated retraining."""
    
    def __init__(self):
        self.performance_threshold = 0.90  # Accuracy threshold
        self.drift_threshold = 0.35  # Data drift threshold
        self.retraining_interval_days = 7  # Periodic retraining
        
    def should_retrain_performance_degradation(
        self,
        current_accuracy: float
    ) -> Tuple[bool, str]:
        """Check if performance has degraded."""
        if current_accuracy < self.performance_threshold:
            return True, f"Accuracy {current_accuracy:.2%} below threshold {self.performance_threshold:.2%}"
        return False, "Performance acceptable"
    
    def should_retrain_data_drift(
        self,
        drift_detected: bool,
        drift_score: float = 0.0
    ) -> Tuple[bool, str]:
        """Check if significant data drift detected."""
        if drift_detected and drift_score > self.drift_threshold:
            return True, f"Data drift score {drift_score:.3f} exceeds threshold {self.drift_threshold}"
        return False, "No significant drift detected"
    
    def should_retrain_periodic(
        self,
        last_retraining_date: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Check if periodic retraining is due."""
        if last_retraining_date is None:
            return True, "No previous retraining found"
        
        last_date = datetime.fromisoformat(last_retraining_date)
        days_since = (datetime.now() - last_date).days
        
        if days_since >= self.retraining_interval_days:
            return True, f"Periodic retraining due ({days_since} days since last)"
        
        return False, f"Next periodic retraining in {self.retraining_interval_days - days_since} days"
    
    def get_retraining_triggers(
        self,
        current_accuracy: Optional[float] = None,
        drift_detected: bool = False,
        drift_score: float = 0.0,
        last_retraining_date: Optional[str] = None
    ) -> Dict:
        """Get comprehensive retraining trigger status."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "should_retrain": False,
            "triggers": [],
            "reasons": []
        }
        
        # Performance check
        if current_accuracy is not None:
            should_retrain, reason = self.should_retrain_performance_degradation(current_accuracy)
            results["triggers"].append({
                "trigger": "performance_degradation",
                "active": should_retrain,
                "reason": reason,
                "metric": current_accuracy
            })
            if should_retrain:
                results["should_retrain"] = True
                results["reasons"].append(reason)
        
        # Drift check
        drift_should_retrain, drift_reason = self.should_retrain_data_drift(
            drift_detected, drift_score
        )
        results["triggers"].append({
            "trigger": "data_drift",
            "active": drift_should_retrain,
            "reason": drift_reason,
            "metric": drift_score
        })
        if drift_should_retrain:
            results["should_retrain"] = True
            results["reasons"].append(drift_reason)
        
        # Periodic check
        periodic_should_retrain, periodic_reason = self.should_retrain_periodic(
            last_retraining_date
        )
        results["triggers"].append({
            "trigger": "periodic",
            "active": periodic_should_retrain,
            "reason": periodic_reason
        })
        if periodic_should_retrain:
            results["should_retrain"] = True
            results["reasons"].append(periodic_reason)
        
        return results


class AutomatedRetrainingPipeline:
    """Manages automated model retraining."""
    
    def __init__(self):
        self.trigger = RetrainingTrigger()
        self.retraining_dir = RETRAINING_LOG_DIR
        
    def evaluate_retraining_need(
        self,
        performance_metrics: Dict,
        drift_status: Dict
    ) -> Dict:
        """Evaluate if retraining is needed."""
        current_accuracy = performance_metrics.get("accuracy", 0)
        drift_detected = drift_status.get("overall_drift", False)
        drift_severity = drift_status.get("severity_levels", {}).get("critical", 0) > 0
        
        last_retraining = self._get_last_retraining_date()
        
        trigger_status = self.trigger.get_retraining_triggers(
            current_accuracy=current_accuracy,
            drift_detected=drift_detected or drift_severity,
            drift_score=drift_status.get("drift_score", 0),
            last_retraining_date=last_retraining
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "retraining_needed": trigger_status["should_retrain"],
            "reasons": trigger_status["reasons"],
            "trigger_details": trigger_status["triggers"],
            "performance_metrics": performance_metrics,
            "drift_status": drift_status
        }
    
    def trigger_retraining(
        self,
        model_type: str = "both"  # "pneumonia", "brain", or "both"
    ) -> Dict:
        """Trigger retraining pipeline."""
        logger.info(f"Starting automated retraining for {model_type}")
        
        retraining_result = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "status": "started",
            "attempts": []
        }
        
        try:
            # Trigger DVC pipeline
            if model_type in ["pneumonia", "both"]:
                result = self._run_training_stage("train_pneumonia")
                retraining_result["attempts"].append({
                    "model": "pneumonia",
                    "stage": "train_pneumonia",
                    "success": result["success"],
                    "message": result.get("message", "")
                })
            
            if model_type in ["brain", "both"]:
                result = self._run_training_stage("train_brain")
                retraining_result["attempts"].append({
                    "model": "brain_tumor",
                    "stage": "train_brain",
                    "success": result["success"],
                    "message": result.get("message", "")
                })
            
            # Run evaluation
            eval_result = self._run_training_stage("evaluate")
            retraining_result["attempts"].append({
                "model": "both",
                "stage": "evaluate",
                "success": eval_result["success"],
                "message": eval_result.get("message", "")
            })
            
            # Update retraining history
            all_success = all(a.get("success", False) for a in retraining_result["attempts"])
            retraining_result["status"] = "completed" if all_success else "failed"
            
            self._log_retraining_history(retraining_result)
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            retraining_result["status"] = "error"
            retraining_result["error"] = str(e)
        
        return retraining_result
    
    def _run_training_stage(self, stage_name: str) -> Dict:
        """Run a DVC pipeline stage."""
        try:
            logger.info(f"Running DVC stage: {stage_name}")
            
            result = subprocess.run(
                [sys.executable, "-m", "dvc", "repro", stage_name],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "returncode": result.returncode,
                "message": result.stderr if not success else result.stdout,
                "timestamp": datetime.now().isoformat()
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Stage {stage_name} timed out")
            return {
                "success": False,
                "message": "Training stage timed out (>1 hour)",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error running stage {stage_name}: {e}")
            return {
                "success": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_last_retraining_date(self) -> Optional[str]:
        """Get timestamp of last retraining."""
        if not RETRAINING_HISTORY_FILE.exists():
            return None
        
        with open(RETRAINING_HISTORY_FILE) as f:
            lines = f.readlines()
            if lines:
                last_entry = json.loads(lines[-1])
                return last_entry.get("timestamp")
        
        return None
    
    def _log_retraining_history(self, retraining_result: Dict) -> None:
        """Log retraining event to history."""
        with open(RETRAINING_HISTORY_FILE, "a") as f:
            f.write(json.dumps(retraining_result) + "\n")
        
        logger.info(f"Retraining logged to {RETRAINING_HISTORY_FILE}")
    
    def get_retraining_schedule(self) -> Dict:
        """Get recommended retraining schedule."""
        return {
            "periodic_interval_days": self.trigger.retraining_interval_days,
            "performance_threshold": self.trigger.performance_threshold,
            "drift_threshold": self.trigger.drift_threshold,
            "recommendations": [
                "Set up cron job: `0 2 * * 0` for weekly retraining (2 AM Sunday)",
                "Monitor drift reports daily",
                "Set up alerts for performance degradation",
                "Keep baseline metrics updated",
                "Archive old models for historical analysis"
            ]
        }
    
    def save_retraining_config(self) -> None:
        """Save retraining configuration."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "performance_threshold": self.trigger.performance_threshold,
            "drift_threshold": self.trigger.drift_threshold,
            "retraining_interval_days": self.trigger.retraining_interval_days,
            "schedule": self.get_retraining_schedule()
        }
        
        with open(RETRAINING_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Retraining config saved to {RETRAINING_CONFIG_FILE}")


# Global instance
_retraining_pipeline: Optional[AutomatedRetrainingPipeline] = None


def get_retraining_pipeline() -> AutomatedRetrainingPipeline:
    """Get or create retraining pipeline."""
    global _retraining_pipeline
    if _retraining_pipeline is None:
        _retraining_pipeline = AutomatedRetrainingPipeline()
    return _retraining_pipeline


def check_and_retrain(
    performance_metrics: Dict,
    drift_status: Dict,
    dry_run: bool = False
) -> Dict:
    """Check if retraining needed and trigger if so."""
    pipeline = get_retraining_pipeline()
    
    evaluation = pipeline.evaluate_retraining_need(performance_metrics, drift_status)
    
    if evaluation["retraining_needed"] and not dry_run:
        logger.info("Retraining criteria met - triggering pipeline")
        retraining_result = pipeline.trigger_retraining()
        evaluation["retraining_triggered"] = True
        evaluation["retraining_result"] = retraining_result
    else:
        evaluation["retraining_triggered"] = False
    
    return evaluation


if __name__ == "__main__":
    # Test retraining pipeline
    pipeline = get_retraining_pipeline()
    
    # Example metrics
    test_performance = {"accuracy": 0.88, "f1": 0.85}
    test_drift = {
        "overall_drift": True,
        "severity_levels": {"critical": 0, "high": 1, "medium": 0, "low": 0},
        "drift_score": 0.36
    }
    
    # Evaluate
    evaluation = pipeline.evaluate_retraining_need(test_performance, test_drift)
    print(json.dumps(evaluation, indent=2, default=str))
