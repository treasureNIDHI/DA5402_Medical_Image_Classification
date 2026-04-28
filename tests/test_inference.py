"""Unit tests for inference and model loading logic."""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_dummy_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _save_dummy_checkpoint(path: Path, num_classes: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = _make_dummy_model(num_classes)
    torch.save(model.state_dict(), path)


class TestModelLoading:

    def test_load_model_returns_model_and_num_classes(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        _save_dummy_checkpoint(ckpt, num_classes=2)

        from src.inference.predict import load_model
        model, num_classes = load_model(ckpt)
        assert num_classes == 2
        assert isinstance(model, nn.Module)

    def test_load_model_4_class(self, tmp_path):
        ckpt = tmp_path / "brain.pt"
        _save_dummy_checkpoint(ckpt, num_classes=4)

        from src.inference.predict import load_model
        model, num_classes = load_model(ckpt)
        assert num_classes == 4

    def test_load_model_eval_mode(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        _save_dummy_checkpoint(ckpt, num_classes=2)

        from src.inference.predict import load_model
        model, _ = load_model(ckpt)
        assert not model.training

    def test_load_model_missing_file_raises(self):
        from src.inference.predict import load_model
        with pytest.raises(Exception):
            load_model(Path("/nonexistent/model.pt"))


class TestModalityDetection:

    def test_bright_image_classified_as_pneumonia(self):
        bright_img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        from src.inference.predict import detect_modality
        assert detect_modality(bright_img) == "pneumonia"

    def test_dark_image_classified_as_brain(self):
        dark_img = Image.new("RGB", (224, 224), color=(30, 30, 30))
        from src.inference.predict import detect_modality
        assert detect_modality(dark_img) == "brain"

    def test_mid_gray_image_returns_valid_modality(self):
        mid_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        from src.inference.predict import detect_modality
        result = detect_modality(mid_img)
        assert result in ("pneumonia", "brain")


class TestPredictFunction:

    def test_predict_with_loaded_models(self, tmp_path):
        pneu_ckpt = tmp_path / "models" / "pneumonia" / "pneumonia_resnet50.pt"
        brain_ckpt = tmp_path / "models" / "brain_tumor" / "brain_resnet50.pt"
        _save_dummy_checkpoint(pneu_ckpt, num_classes=2)
        _save_dummy_checkpoint(brain_ckpt, num_classes=4)

        img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        img_path = tmp_path / "test.jpeg"
        img.save(img_path)

        from src.inference.predict import load_model
        pneu_model, _ = load_model(pneu_ckpt)
        brain_model, _ = load_model(brain_ckpt)

        with patch("src.inference.predict.pneumonia_model", pneu_model), \
             patch("src.inference.predict.brain_model", brain_model), \
             patch("src.inference.predict.BRAIN_CLASSES", ["glioma", "meningioma", "notumor", "pituitary"]):
            from src.inference.predict import predict
            result = predict(str(img_path), "pneumonia")

        assert "prediction" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_none_model_raises_runtime_error(self, tmp_path):
        img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        img_path = tmp_path / "test.jpeg"
        img.save(img_path)

        with patch("src.inference.predict.pneumonia_model", None), \
             patch("src.inference.predict.detect_modality", return_value="pneumonia"):
            from src.inference.predict import predict
            with pytest.raises(RuntimeError, match="not loaded"):
                predict(str(img_path), "pneumonia")


class TestHealthCheck:

    def test_health_check_runs_without_models(self, tmp_path):
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "reports").mkdir()

        from src.monitoring.health_check import run_health_check
        result = run_health_check()
        assert result in (0, 1)

        import json
        report = json.loads((tmp_path / "reports" / "health_check_report.json").read_text())
        assert "overall_status" in report
        assert "system" in report
        assert "models" in report
        os.chdir(orig)

    def test_health_check_report_has_system_resources(self, tmp_path):
        import os, json
        orig = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "reports").mkdir()

        from src.monitoring.health_check import run_health_check
        run_health_check()
        report = json.loads((tmp_path / "reports" / "health_check_report.json").read_text())
        assert "cpu_percent" in report["system"]
        assert "memory_percent" in report["system"]
        os.chdir(orig)
