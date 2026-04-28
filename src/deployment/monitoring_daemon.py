"""
Monitoring Daemon for Production Deployment

Runs continuously to monitor:
1. Model performance (accuracy drift, output distribution shift)
2. Infrastructure health (resource usage, error rates)
3. Data quality (label distribution drift, outliers)
4. Alert on anomalies
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
MONITORING_INTERVAL = 300  # Run monitoring every 5 minutes
REPORTS_DIR.mkdir(exist_ok=True)


class ProductionMonitor:
    """Monitors production model deployment."""
    
    def __init__(self):
        self.reports_dir = REPORTS_DIR
        self.last_check = None
        self.anomalies = []
        
    def load_previous_report(self) -> Optional[Dict]:
        """Load the last monitoring report."""
        report_file = self.reports_dir / "monitoring_report.json"
        if report_file.exists():
            try:
                with open(report_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load previous report: {e}")
        return None
    
    def check_model_performance(self) -> Dict:
        """Check if model performance is degrading."""
        logger.info("Checking model performance...")
        
        try:
            # Run model evaluation
            result = subprocess.run(
                [sys.executable, "-m", "src.monitoring.monitor", "--allow-missing-data"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("Model evaluation successful")
                return {"status": "healthy", "message": "Model performing normally"}
            else:
                logger.warning(f"Model evaluation failed: {result.stderr}")
                return {"status": "degraded", "message": result.stderr}
        except subprocess.TimeoutExpired:
            logger.error("Model evaluation timeout")
            return {"status": "error", "message": "Monitoring timeout"}
        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_latency(self) -> Dict:
        """Check inference latency metrics."""
        logger.info("Checking inference latency...")
        
        metrics_file = self.reports_dir / "latency_report.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    latency_data = json.load(f)
                
                # Check against threshold (200ms)
                median_latency = latency_data.get("median_latency_ms", 0)
                p95_latency = latency_data.get("p95_latency_ms", 0)
                
                status = "healthy"
                if median_latency > 200:
                    status = "degraded"
                if p95_latency > 500:
                    status = "critical"
                
                return {
                    "status": status,
                    "median_latency_ms": median_latency,
                    "p95_latency_ms": p95_latency,
                    "threshold_ms": 200
                }
            except Exception as e:
                logger.error(f"Error reading latency report: {e}")
                return {"status": "error", "message": str(e)}
        
        return {"status": "unknown", "message": "No latency report found"}
    
    def check_disk_usage(self) -> Dict:
        """Check if disk space is running low."""
        import shutil
        
        logger.info("Checking disk usage...")
        
        total, used, free = shutil.disk_usage("/")
        percent_used = (used / total) * 100
        
        status = "healthy"
        if percent_used > 80:
            status = "warning"
        if percent_used > 90:
            status = "critical"
        
        return {
            "status": status,
            "percent_used": percent_used,
            "free_gb": free / (1024**3),
            "total_gb": total / (1024**3),
            "threshold_percent": 80
        }
    
    def check_models_available(self) -> Dict:
        """Verify all required models are present."""
        logger.info("Checking model availability...")
        
        models = {
            "pneumonia": Path("models/pneumonia/pneumonia_resnet50.pt"),
            "brain_tumor": Path("models/brain_tumor/brain_resnet50.pt")
        }
        
        all_available = True
        status_dict = {}
        
        for name, path in models.items():
            available = path.exists()
            status_dict[name] = available
            if not available:
                all_available = False
                logger.warning(f"Model not found: {path}")
        
        return {
            "status": "healthy" if all_available else "critical",
            "models": status_dict,
            "all_available": all_available
        }
    
    def run_full_check(self) -> Dict:
        """Run comprehensive health check."""
        logger.info("Starting production monitoring check...")
        
        checks = {
            "timestamp": datetime.now().isoformat(),
            "model_performance": self.check_model_performance(),
            "latency": self.check_latency(),
            "disk_usage": self.check_disk_usage(),
            "models_available": self.check_models_available()
        }
        
        # Determine overall status
        statuses = [check.get("status") for check in checks.values() if isinstance(check, dict)]
        
        if "critical" in statuses:
            overall_status = "critical"
        elif "error" in statuses:
            overall_status = "error"
        elif "degraded" in statuses or "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        checks["overall_status"] = overall_status
        
        return checks
    
    def save_report(self, report: Dict) -> None:
        """Save monitoring report."""
        report_file = self.reports_dir / "production_monitoring_report.json"
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def alert_on_anomalies(self, report: Dict) -> None:
        """Alert if any critical issues detected."""
        overall_status = report.get("overall_status", "unknown")
        
        if overall_status == "critical":
            logger.error(f"🚨 CRITICAL ALERT: {report}")
            # In production, send to alerting system (PagerDuty, Slack, etc.)
        elif overall_status == "error":
            logger.error(f"⚠️ ERROR ALERT: {report}")
        elif overall_status == "warning":
            logger.warning(f"⚠️ WARNING: {report}")
    
    def run_loop(self, interval: int = MONITORING_INTERVAL) -> None:
        """Run monitoring loop."""
        logger.info(f"Starting monitoring loop with {interval}s interval")
        
        try:
            while True:
                try:
                    report = self.run_full_check()
                    self.save_report(report)
                    self.alert_on_anomalies(report)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Monitoring loop interrupted")


def main():
    """Entry point for monitoring daemon."""
    monitor = ProductionMonitor()
    monitor.run_loop()


if __name__ == "__main__":
    main()
