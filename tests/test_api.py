"""Unit tests for FastAPI endpoints."""
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def client():
    with patch("src.inference.predict.pneumonia_model", None), \
         patch("src.inference.predict.brain_model", None):
        from src.api.app import app
        return TestClient(app)


def _make_image_bytes(mode="RGB", size=(224, 224), color=(128, 128, 128)) -> bytes:
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


class TestHealthEndpoints:

    def test_liveness_returns_200(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_liveness_status_alive(self, client):
        r = client.get("/healthz")
        assert r.json()["status"] == "alive"

    def test_readiness_returns_200(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200

    def test_readiness_has_models_loaded_field(self, client):
        r = client.get("/readyz")
        assert "models_loaded" in r.json()

    def test_health_combined_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_combined_has_status(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] in ("healthy", "unhealthy")

    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_message(self, client):
        assert "message" in client.get("/").json()


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type_prometheus(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_contains_http_requests(self, client):
        r = client.get("/metrics")
        assert "http_requests_total" in r.text or "python_" in r.text


class TestPredictEndpoint:

    def test_predict_missing_file_returns_422(self, client):
        r = client.post("/predict", data={"model_type": "pneumonia"})
        assert r.status_code == 422

    def test_predict_missing_model_type_returns_422(self, client):
        img_bytes = _make_image_bytes()
        r = client.post("/predict",
                        files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 422

    def test_predict_invalid_model_type_returns_500(self, client):
        img_bytes = _make_image_bytes()
        r = client.post("/predict",
                        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
                        data={"model_type": "invalid_model"})
        assert r.status_code in (422, 500)

    def test_predict_response_has_required_fields(self, client):
        with patch("src.api.app.predict") as mock_predict:
            mock_predict.return_value = {"prediction": "NORMAL", "confidence": 0.95}
            img_bytes = _make_image_bytes(color=(200, 200, 200))
            r = client.post("/predict",
                            files={"file": ("xray.jpg", img_bytes, "image/jpeg")},
                            data={"model_type": "pneumonia"})
            if r.status_code == 200:
                data = r.json()
                assert "prediction" in data
                assert "confidence" in data
                assert "model_type" in data

    def test_predict_confidence_in_valid_range(self, client):
        with patch("src.api.app.predict") as mock_predict:
            mock_predict.return_value = {"prediction": "PNEUMONIA", "confidence": 0.87}
            img_bytes = _make_image_bytes(color=(200, 200, 200))
            r = client.post("/predict",
                            files={"file": ("xray.jpg", img_bytes, "image/jpeg")},
                            data={"model_type": "pneumonia"})
            if r.status_code == 200:
                confidence = r.json()["confidence"]
                assert 0.0 <= confidence <= 1.0
