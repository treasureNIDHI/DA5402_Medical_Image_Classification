"""
Model Serving Module for dedicated inference service

Provides model caching, batch inference, and performance optimization
"""

import asyncio
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

# Model cache
MODEL_CACHE = {}
CACHE_LOCK = asyncio.Lock()


class ModelServer:
    """Dedicated model serving with caching and batch processing."""
    
    def __init__(self):
        self.device = DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.models_dir = MODELS_DIR
        self.load_times = {}
        self.inference_times = []
        self.batch_size = 32
        
        logger.info(f"Model Server initialized on device: {DEVICE}")
    
    def load_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Load or retrieve cached model."""
        if model_name in MODEL_CACHE:
            logger.info(f"Using cached model: {model_name}")
            return MODEL_CACHE[model_name]
        
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        try:
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
            # Determine model path
            if model_name == "pneumonia":
                model_path = self.models_dir / "pneumonia" / "pneumonia_resnet50.pt"
                num_classes = 2
            elif model_name == "brain":
                model_path = self.models_dir / "brain_tumor" / "brain_resnet50.pt"
                num_classes = 4
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            
            # Cache model
            MODEL_CACHE[model_name] = model
            load_time = time.time() - start_time
            self.load_times[model_name] = load_time
            
            logger.info(f"Model {model_name} loaded in {load_time:.3f}s")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def preprocess_batch(self, image_paths: List[str]) -> torch.Tensor:
        """Preprocess batch of images."""
        tensors = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)
            tensors.append(tensor)
        
        batch = torch.stack(tensors)
        return batch.to(self.device)
    
    def predict_single(self, image_path: str, model_name: str, classes: List[str]) -> Dict:
        """Single image inference."""
        start_time = time.time()
        
        try:
            # Load model
            num_classes = len(classes)
            model = self.load_model(model_name, num_classes)
            
            # Preprocess
            image_tensor = self.preprocess_image(image_path)
            
            # Inference
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return {
                "prediction": classes[pred.item()],
                "confidence": float(confidence.item()),
                "inference_time": inference_time,
                "device": str(self.device)
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def predict_batch(self, image_paths: List[str], model_name: str, classes: List[str]) -> List[Dict]:
        """Batch image inference."""
        start_time = time.time()
        results = []
        
        try:
            # Load model
            num_classes = len(classes)
            model = self.load_model(model_name, num_classes)
            
            # Process in batches
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i+self.batch_size]
                batch_tensor = self.preprocess_batch(batch_paths)
                
                # Inference
                with torch.no_grad():
                    outputs = model(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                
                # Parse results
                for j, path in enumerate(batch_paths):
                    confidence, pred = torch.max(probs[j], 0)
                    results.append({
                        "path": path,
                        "prediction": classes[pred.item()],
                        "confidence": float(confidence.item())
                    })
            
            total_time = time.time() - start_time
            self.inference_times.append(total_time / len(image_paths))  # Average per image
            
            return results
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {
                "total_inferences": 0,
                "average_latency": None
            }
        
        times = sorted(self.inference_times)
        return {
            "total_inferences": len(times),
            "average_latency": sum(times) / len(times),
            "median_latency": times[len(times) // 2],
            "p95_latency": times[int(len(times) * 0.95)],
            "min_latency": min(times),
            "max_latency": max(times),
            "load_times": self.load_times
        }
    
    def health_check(self) -> Dict:
        """Check server health."""
        try:
            # Verify models can load
            pneumonia_model = self.load_model("pneumonia", 2)
            brain_model = self.load_model("brain", 4)
            
            return {
                "status": "healthy",
                "device": str(self.device),
                "models_available": ["pneumonia", "brain"],
                "cache_size": len(MODEL_CACHE),
                "performance": self.get_performance_stats()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global server instance
_server: Optional[ModelServer] = None


def get_server() -> ModelServer:
    """Get or create server instance."""
    global _server
    if _server is None:
        _server = ModelServer()
    return _server


# FastAPI-style endpoint handlers
def health():
    """Health check endpoint."""
    server = get_server()
    return server.health_check()


def ready():
    """Readiness endpoint."""
    health_status = health()
    return {
        "ready": health_status["status"] == "healthy",
        "details": health_status
    }


if __name__ == "__main__":
    # Test the server
    import sys
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    
    app = FastAPI(title="Model Server")
    server = get_server()
    
    @app.get("/health")
    def health_endpoint():
        return server.health_check()
    
    @app.get("/ready")
    def ready_endpoint():
        health_status = server.health_check()
        return {
            "ready": health_status["status"] == "healthy",
            "details": health_status
        }
    
    @app.post("/predict")
    async def predict_endpoint(file_path: str, model_type: str):
        """Predict on single image."""
        try:
            classes = ["NORMAL", "PNEUMONIA"] if model_type == "pneumonia" else ["glioma", "meningioma", "notumor", "pituitary"]
            result = server.predict_single(file_path, model_type, classes)
            return result
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    
    @app.post("/predict/batch")
    async def predict_batch_endpoint(image_paths: List[str], model_type: str):
        """Predict on batch of images."""
        try:
            classes = ["NORMAL", "PNEUMONIA"] if model_type == "pneumonia" else ["glioma", "meningioma", "notumor", "pituitary"]
            results = server.predict_batch(image_paths, model_type, classes)
            return {"predictions": results}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
