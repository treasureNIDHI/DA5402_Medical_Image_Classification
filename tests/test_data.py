"""Unit tests for data pipeline modules."""
import json
import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image


class TestValidation:

    def test_validation_runs_on_empty_dirs(self, tmp_path):
        """Validation should not crash when data/raw is empty."""
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "data" / "raw" / "chest_xray").mkdir(parents=True)
        (tmp_path / "data" / "raw" / "brain_tumor").mkdir(parents=True)
        (tmp_path / "reports").mkdir()

        from src.data.validation import run_validation
        result = run_validation()
        assert result == 0

        report_file = tmp_path / "reports" / "data_validation_report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert "summary" in data
        assert "overall_valid" in data
        os.chdir(orig)

    def test_validation_report_structure(self, tmp_path):
        """Report must have required top-level keys."""
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "data" / "raw" / "chest_xray").mkdir(parents=True)
        (tmp_path / "data" / "raw" / "brain_tumor").mkdir(parents=True)
        (tmp_path / "reports").mkdir()

        from src.data.validation import run_validation
        run_validation()
        data = json.loads((tmp_path / "reports" / "data_validation_report.json").read_text())
        for key in ("raw_dir", "datasets", "overall_valid", "summary"):
            assert key in data, f"Missing key: {key}"
        os.chdir(orig)

    def test_validation_detects_valid_images(self, tmp_path):
        """Validation should count valid JPEG images correctly."""
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        xray_train = tmp_path / "data" / "raw" / "chest_xray" / "train" / "NORMAL"
        xray_train.mkdir(parents=True)
        (tmp_path / "data" / "raw" / "brain_tumor").mkdir(parents=True)
        (tmp_path / "reports").mkdir()

        # Create 3 valid images
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(100, 100, 100))
            img.save(xray_train / f"img_{i}.jpeg")

        from src.data.validation import run_validation
        run_validation()
        data = json.loads((tmp_path / "reports" / "data_validation_report.json").read_text())
        assert data["summary"]["total_valid_images"] == 3
        os.chdir(orig)


class TestPreprocessing:

    def test_preprocessing_creates_output_dir(self, tmp_path):
        """Preprocessing must create data/processed/."""
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        xray = tmp_path / "data" / "raw" / "chest_xray" / "train" / "NORMAL"
        xray.mkdir(parents=True)
        img = Image.new("RGB", (512, 512), color=(150, 150, 150))
        img.save(xray / "sample.jpeg")
        (tmp_path / "reports").mkdir()

        from src.data.preprocessing import process_images, create_dirs
        create_dirs()
        report = process_images()

        assert (tmp_path / "data" / "processed").exists()
        assert report["processed_count"] == 1
        os.chdir(orig)

    def test_preprocessing_resizes_to_224(self, tmp_path):
        """All output images must be exactly 224×224."""
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        xray = tmp_path / "data" / "raw" / "chest_xray" / "train" / "NORMAL"
        xray.mkdir(parents=True)
        img = Image.new("RGB", (512, 400), color=(120, 120, 120))
        img.save(xray / "sample.jpeg")
        (tmp_path / "reports").mkdir()

        from src.data.preprocessing import process_images, create_dirs
        create_dirs()
        process_images()

        out = tmp_path / "data" / "processed" / "chest_xray" / "train" / "NORMAL" / "sample.jpeg"
        assert out.exists()
        with Image.open(out) as img_out:
            assert img_out.size == (224, 224)
        os.chdir(orig)

    def test_preprocessing_skips_corrupt_files(self, tmp_path):
        """Corrupt files should be counted and skipped, not crash."""
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        xray = tmp_path / "data" / "raw" / "chest_xray" / "train" / "NORMAL"
        xray.mkdir(parents=True)
        (xray / "corrupt.jpeg").write_bytes(b"not an image")
        (tmp_path / "reports").mkdir()

        from src.data.preprocessing import process_images, create_dirs
        create_dirs()
        report = process_images()
        assert report["corrupt_count"] >= 1
        os.chdir(orig)


class TestEDA:

    def test_eda_generates_json_report(self, tmp_path):
        """EDA must produce eda_report.json."""
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        split = tmp_path / "data" / "processed" / "chest_xray" / "train" / "NORMAL"
        split.mkdir(parents=True)
        img = Image.new("RGB", (224, 224))
        img.save(split / "img.jpeg")
        (tmp_path / "reports" / "feature_store").mkdir(parents=True)

        from src.data.eda import run_eda
        run_eda()

        assert (tmp_path / "reports" / "eda_report.json").exists()
        assert (tmp_path / "reports" / "eda_report.md").exists()
        os.chdir(orig)
