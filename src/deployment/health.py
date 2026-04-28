"""
Model Serving Infrastructure with Health Checks and Metrics

This module provides:
1. Enhanced health checks with model validation
2. Readiness probe metrics (model loaded, cache ready)
3. Model version tracking
4. Performance metrics collection
5. Graceful shutdown support
"""

import os
import json
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torchvision import models

# Global state tracking
MODEL_LOAD_TIME = None
INFERENCE_TIMES = []
HEALTH_CHECK_INTERVAL = 60  # seconds
LAST_HEALTH_CHECK = None
MODEL_VERSIONS = {}
CURRENT_MODELS = {}


class HealthCheckManager:
    """Manages comprehensive health checks and metrics collection."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.slow_request_count = 0
        self.latency_threshold = 0.5  # 500ms threshold
        
    def get_model_version(self, model_path: str) -> Dict:
        """Extract model metadata and version info."""
        try:
            path = Path(model_path)
            stat = path.stat()
            
            return {
                "path": model_path,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "device": self.device,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def check_model_loadable(self, model_path: str, num_classes: int) -> bool:
        """Verify model can be loaded without errors."""
        try:
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
            # Try to load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            
            return True
        except Exception as e:
            print(f"Model load check failed: {e}")
            return False
    
    def check_system_resources(self) -> Dict:
        """Check CPU, memory, and disk availability."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        else:
            gpu_memory = None
            gpu_usage = None
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_available_mb": disk.free / (1024 * 1024),
            "gpu_memory_allocated": gpu_memory,
            "gpu_usage_percent": gpu_usage * 100 if gpu_usage else None,
            "threshold_status": {
                "cpu_ok": cpu_percent < 90,
                "memory_ok": memory.percent < 85,
                "disk_ok": disk.percent < 90,
                "gpu_ok": gpu_usage < 0.9 if gpu_usage else True
            }
        }
    
    def record_inference(self, inference_time: float, success: bool = True):
        """Track inference metrics for performance monitoring."""
        self.request_count += 1
        
        if not success:
            self.error_count += 1
        
        if inference_time > self.latency_threshold:
            self.slow_request_count += 1
        
        INFERENCE_TIMES.append({
            "timestamp": datetime.now().isoformat(),
            "latency_seconds": inference_time,
            "success": success
        })
        
        # Keep only last 1000 records
        if len(INFERENCE_TIMES) > 1000:
            INFERENCE_TIMES.pop(0)
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from recent inferences."""
        if not INFERENCE_TIMES:
            return {
                "total_requests": 0,
                "average_latency": None,
                "p95_latency": None,
                "p99_latency": None,
                "error_rate": 0.0,
                "slow_request_rate": 0.0
            }
        
        latencies = [r["latency_seconds"] for r in INFERENCE_TIMES if r["success"]]
        latencies.sort()
        
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        slow_rate = self.slow_request_count / self.request_count if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.request_count - self.error_count,
            "failed_requests": self.error_count,
            "average_latency": sum(latencies) / len(latencies) if latencies else None,
            "median_latency": latencies[len(latencies) // 2] if latencies else None,
            "p95_latency": latencies[int(len(latencies) * 0.95)] if latencies else None,
            "p99_latency": latencies[int(len(latencies) * 0.99)] if latencies else None,
            "min_latency": min(latencies) if latencies else None,
            "max_latency": max(latencies) if latencies else None,
            "error_rate": error_rate,
            "slow_request_rate": slow_rate,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_liveness_status(self) -> Dict:
        """Liveness probe: Is the service running? Basic checks."""
        return {
            "status": "alive",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "process_available": True
        }
    
    def get_readiness_status(self) -> Dict:
        """Readiness probe: Is the service ready to accept traffic?"""
        resources = self.check_system_resources()
        metrics = self.get_performance_metrics()
        
        # Readiness checks
        resources_ok = all(resources["threshold_status"].values())
        models_loaded = all([
            Path("models/pneumonia/pneumonia_resnet50.pt").exists(),
            Path("models/brain_tumor/brain_resnet50.pt").exists()
        ])
        recent_success = self.error_count < self.request_count * 0.05 if self.request_count > 0 else True
        
        ready = resources_ok and models_loaded and recent_success
        
        return {
            "status": "ready" if ready else "not_ready",
            "ready": ready,
            "models_loaded": models_loaded,
            "resources_available": resources_ok,
            "recent_requests_healthy": recent_success,
            "resource_status": resources["threshold_status"],
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_startup_status(self) -> Dict:
        """Startup probe: Is the service initializing?"""
        startup_timeout = 120  # 2 minutes
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        models_exist = all([
            Path("models/pneumonia/pneumonia_resnet50.pt").exists(),
            Path("models/brain_tumor/brain_resnet50.pt").exists()
        ])
        
        # Try loading models
        pneumonia_loadable = self.check_model_loadable(
            "models/pneumonia/pneumonia_resnet50.pt", 
            num_classes=2
        )
        brain_loadable = self.check_model_loadable(
            "models/brain_tumor/brain_resnet50.pt",
            num_classes=4
        )
        
        ready = models_exist and pneumonia_loadable and brain_loadable
        
        return {
            "status": "ready" if ready else "initializing",
            "ready": ready,
            "elapsed_seconds": elapsed,
            "startup_timeout_seconds": startup_timeout,
            "timed_out": elapsed > startup_timeout,
            "models_exist": models_exist,
            "models_loadable": {
                "pneumonia": pneumonia_loadable,
                "brain": brain_loadable
            },
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
_health_check_manager: Optional[HealthCheckManager] = None


def get_health_manager() -> HealthCheckManager:
    """Get or create the singleton health check manager."""
    global _health_check_manager
    if _health_check_manager is None:
        _health_check_manager = HealthCheckManager()
    return _health_check_manager


# Export functions for use in API
def get_liveness() -> Dict:
    """REST endpoint for liveness probe."""
    manager = get_health_manager()
    return manager.get_liveness_status()


def get_readiness() -> Dict:
    """REST endpoint for readiness probe."""
    manager = get_health_manager()
    return manager.get_readiness_status()


def get_startup() -> Dict:
    """REST endpoint for startup probe."""
    manager = get_health_manager()
    return manager.get_startup_status()


def get_metrics() -> Dict:
    """REST endpoint for metrics and performance data."""
    manager = get_health_manager()
    return {
        "performance": manager.get_performance_metrics(),
        "resources": manager.check_system_resources(),
        "timestamp": datetime.now().isoformat()
    }


def record_inference_time(latency: float, success: bool = True):
    """Record inference time for metrics collection."""
    manager = get_health_manager()
    manager.record_inference(latency, success)


def save_health_check_report():
    """Save detailed health report for debugging."""
    manager = get_health_manager()
    report = {
        "timestamp": datetime.now().isoformat(),
        "liveness": manager.get_liveness_status(),
        "readiness": manager.get_readiness_status(),
        "startup": manager.get_startup_status(),
        "metrics": manager.get_performance_metrics(),
        "resources": manager.check_system_resources()
    }
    
    Path("reports").mkdir(exist_ok=True)
    with open("reports/health_check_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    return report
