"""
Model Registry and Versioning System

Manages model lifecycle:
1. Version tracking
2. Model metadata
3. Performance history
4. Rollback capability
5. Model promotion (dev → prod)
6. MLflow integration for centralized registry
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGISTRY_DIR = Path("models/registry")
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
MODEL_ARTIFACTS_DIR = REGISTRY_DIR / "artifacts"

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelMetadata:
    """Model metadata and versioning."""
    model_id: str
    model_type: str
    version: str
    created_at: str
    architecture: str
    accuracy: float
    f1_score: float
    latency_ms: float
    model_size_mb: float
    training_samples: int
    validation_samples: int
    test_samples: int
    git_commit: str
    mlflow_run_id: str
    status: str  # "draft", "candidate", "production", "archived"
    deployment_date: Optional[str] = None
    rollback_reason: Optional[str] = None


class ModelRegistry:
    """Manages model versioning and lifecycle."""
    
    def __init__(self):
        self.registry_file = REGISTRY_FILE
        self.artifacts_dir = MODEL_ARTIFACTS_DIR
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {"models": [], "deployments": [], "last_updated": None}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)
        logger.info(f"Registry saved to {self.registry_file}")
    
    def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        architecture: str,
        metrics: Dict,
        training_info: Dict,
        git_commit: str,
        mlflow_run_id: str,
        model_path: Path,
        status: str = "draft"
    ) -> Dict:
        """Register a new model."""
        logger.info(f"Registering model {model_id} v{version}")
        
        # Archive model
        archive_path = self.artifacts_dir / f"{model_id}_{version}"
        archive_path.mkdir(exist_ok=True)
        
        if model_path.exists():
            shutil.copy(model_path, archive_path / model_path.name)
            logger.info(f"Model archived to {archive_path}")
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            created_at=datetime.now().isoformat(),
            architecture=architecture,
            accuracy=metrics.get("accuracy", 0),
            f1_score=metrics.get("f1_score", 0),
            latency_ms=metrics.get("latency_ms", 0),
            model_size_mb=metrics.get("size_mb", 0),
            training_samples=training_info.get("training_samples", 0),
            validation_samples=training_info.get("validation_samples", 0),
            test_samples=training_info.get("test_samples", 0),
            git_commit=git_commit,
            mlflow_run_id=mlflow_run_id,
            status=status
        )
        
        # Add to registry
        self.registry["models"].append(asdict(metadata))
        self._save_registry()
        
        # Register in MLflow if run_id is provided
        if mlflow_run_id:
            self._register_in_mlflow(model_id, version, mlflow_run_id, metadata)
        
        return asdict(metadata)
    
    def _register_in_mlflow(
        self,
        model_id: str,
        version: str,
        run_id: str,
        metadata: ModelMetadata
    ) -> None:
        """Register model in MLflow model registry."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Create or get registered model
            try:
                client.create_registered_model(model_id)
                logger.info(f"Created MLflow registered model: {model_id}")
            except mlflow.exceptions.MlflowException:
                logger.info(f"MLflow registered model {model_id} already exists")
            
            # Get model URI from run
            model_uri = f"runs:/{run_id}/model"
            
            # Create model version
            version_info = client.create_model_version(
                name=model_id,
                source=model_uri,
                run_id=run_id,
                description=f"{model_id} v{version} - {metadata.architecture}"
            )
            
            logger.info(
                f"Registered model in MLflow: {model_id} "
                f"v{version_info.version}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to register in MLflow: {e}")
    
    def promote_model(
        self,
        model_id: str,
        version: str,
        target_status: str = "production"
    ) -> Dict:
        """Promote model to production."""
        logger.info(f"Promoting {model_id} v{version} to {target_status}")
        
        # Find model
        model = None
        for m in self.registry["models"]:
            if m["model_id"] == model_id and m["version"] == version:
                model = m
                break
        
        if not model:
            raise ValueError(f"Model {model_id} v{version} not found")
        
        # If promoting to production, demote current production
        if target_status == "production":
            for m in self.registry["models"]:
                if m["model_id"] == model_id and m["status"] == "production":
                    m["status"] = "archived"
                    logger.info(f"Archived previous model {m['model_id']} v{m['version']}")
        
        # Promote model
        model["status"] = target_status
        model["deployment_date"] = datetime.now().isoformat()
        
        # Log deployment
        self.registry["deployments"].append({
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "version": version,
            "action": "promoted",
            "status": target_status
        })
        
        # Update MLflow stage if registered
        self._update_mlflow_stage(model_id, version, target_status)
        
        self._save_registry()
        
        return model
    
    def _update_mlflow_stage(
        self,
        model_id: str,
        version: str,
        target_status: str
    ) -> None:
        """Update model stage in MLflow."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Map our status to MLflow stages
            stage_mapping = {
                "draft": "None",
                "candidate": "Staging",
                "production": "Production",
                "archived": "Archived"
            }
            
            target_stage = stage_mapping.get(target_status, "None")
            
            # Transition model version
            client.transition_model_version_stage(
                name=model_id,
                version=version,
                stage=target_stage
            )
            
            logger.info(
                f"MLflow model {model_id} v{version} "
                f"transitioned to {target_stage}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to update MLflow stage: {e}")
    
    def get_production_models(self) -> List[Dict]:
        """Get currently deployed production models."""
        return [
            m for m in self.registry["models"]
            if m["status"] == "production"
        ]
    
    def get_model_history(self, model_id: str) -> List[Dict]:
        """Get all versions of a model."""
        return [
            m for m in self.registry["models"]
            if m["model_id"] == model_id
        ]
    
    def compare_models(
        self,
        model_id_1: str,
        version_1: str,
        model_id_2: str,
        version_2: str
    ) -> Dict:
        """Compare two models."""
        model1 = None
        model2 = None
        
        for m in self.registry["models"]:
            if m["model_id"] == model_id_1 and m["version"] == version_1:
                model1 = m
            if m["model_id"] == model_id_2 and m["version"] == version_2:
                model2 = m
        
        if not model1 or not model2:
            raise ValueError("One or both models not found")
        
        return {
            "model_1": model1,
            "model_2": model2,
            "comparison": {
                "accuracy_diff": model1["accuracy"] - model2["accuracy"],
                "f1_diff": model1["f1_score"] - model2["f1_score"],
                "latency_diff_ms": model1["latency_ms"] - model2["latency_ms"],
                "size_diff_mb": model1["model_size_mb"] - model2["model_size_mb"],
                "better_model": model_id_1 if model1["accuracy"] > model2["accuracy"] else model_id_2
            }
        }
    
    def rollback_model(
        self,
        model_id: str,
        reason: str
    ) -> Dict:
        """Rollback to previous version."""
        logger.warning(f"Rolling back {model_id}: {reason}")
        
        # Get all versions sorted by date
        versions = sorted(
            [m for m in self.registry["models"] if m["model_id"] == model_id],
            key=lambda x: x["created_at"],
            reverse=True
        )
        
        if len(versions) < 2:
            raise ValueError("No previous version available for rollback")
        
        # Current (problematic) version
        current = versions[0]
        
        # Previous (working) version
        previous = versions[1]
        
        # Update statuses
        current["status"] = "archived"
        current["rollback_reason"] = reason
        previous["status"] = "production"
        previous["deployment_date"] = datetime.now().isoformat()
        
        # Log rollback
        self.registry["deployments"].append({
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "version": previous["version"],
            "action": "rollback",
            "reason": reason,
            "from_version": current["version"]
        })
        
        self._save_registry()
        
        return {
            "status": "rolled_back",
            "rolled_back_to": previous["version"],
            "reason": reason
        }
    
    def get_model_registry_report(self) -> Dict:
        """Generate comprehensive registry report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(self.registry["models"]),
            "production_models": len(self.get_production_models()),
            "by_type": {},
            "by_status": {},
            "deployment_history": self.registry["deployments"][-10:],  # Last 10
            "models": self.registry["models"]
        }
        
        # Group by type
        for model in self.registry["models"]:
            model_type = model["model_type"]
            if model_type not in report["by_type"]:
                report["by_type"][model_type] = []
            report["by_type"][model_type].append({
                "id": model["model_id"],
                "version": model["version"],
                "accuracy": model["accuracy"],
                "status": model["status"]
            })
        
        # Group by status
        for model in self.registry["models"]:
            status = model["status"]
            if status not in report["by_status"]:
                report["by_status"][status] = 0
            report["by_status"][status] += 1
        
        return report
    
    def save_registry_report(self) -> None:
        """Save registry report."""
        report = self.get_model_registry_report()
        
        report_file = REGISTRY_DIR / "registry_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Registry report saved to {report_file}")


# Global instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


if __name__ == "__main__":
    # Test registry
    registry = get_model_registry()
    
    # Example: Register a model
    metadata = registry.register_model(
        model_id="pneumonia_resnet50",
        model_type="pneumonia",
        version="1.0.0",
        architecture="ResNet50",
        metrics={"accuracy": 0.94, "f1_score": 0.93, "latency_ms": 85, "size_mb": 102},
        training_info={"training_samples": 3000, "validation_samples": 500, "test_samples": 500},
        git_commit="abc1234",
        mlflow_run_id="run_123",
        model_path=Path("models/pneumonia/pneumonia_resnet50.pt"),
        status="production"
    )
    
    print(json.dumps(metadata, indent=2, default=str))
    
    # Save report
    registry.save_registry_report()
