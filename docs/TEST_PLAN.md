# Test Plan & Test Report

## 1. Acceptance Criteria

| Criterion | Threshold | Status |
|---|---|---|
| All unit tests pass | 34/34 (100%) | ✅ PASS |
| API health endpoints return 200 | All probes live | ✅ PASS |
| Invalid input returns 422 (not 500) | ValueError → 422 | ✅ PASS |
| Models load from checkpoint | Both .pt files loadable | ✅ PASS |
| Preprocessing resizes to 224×224 | Exact pixel size | ✅ PASS |
| Corrupt images are skipped without crash | Non-zero corrupt count | ✅ PASS |
| Modality detection separates X-ray from MRI | Heuristic on intensity | ✅ PASS |
| Health check completes without models | Returns report, exit 0 or 1 | ✅ PASS |
| Docker image builds and runs | `/healthz` returns 200 | ✅ PASS |
| DVC pipeline runs end-to-end | All 13 stages exit 0 | ✅ PASS |
| Inference latency < 200ms (after warmup) | ~70ms in container | ✅ PASS |
| Model size ≤ 100 MB | 90 MB each | ✅ PASS |

---

## 2. Test Suites

### Suite 1 — API Tests (`tests/test_api.py`)
Tests the FastAPI server endpoints using `TestClient` with mocked models.

| # | Test Case | Expected | Result |
|---|---|---|---|
| 1 | GET /healthz returns 200 | 200 OK | ✅ PASS |
| 2 | GET /healthz status = "alive" | `{"status": "alive"}` | ✅ PASS |
| 3 | GET /readyz returns 200 | 200 OK | ✅ PASS |
| 4 | GET /readyz has models_loaded field | Field present | ✅ PASS |
| 5 | GET /health returns 200 | 200 OK | ✅ PASS |
| 6 | GET /health has status field | `healthy` or `unhealthy` | ✅ PASS |
| 7 | GET / returns 200 | 200 OK | ✅ PASS |
| 8 | GET / has message field | `{"message": ...}` | ✅ PASS |
| 9 | GET /metrics returns 200 | 200 OK | ✅ PASS |
| 10 | GET /metrics content-type is text/plain | Prometheus format | ✅ PASS |
| 11 | GET /metrics contains prometheus metrics | `http_requests_total` present | ✅ PASS |
| 12 | POST /predict without file returns 422 | 422 Unprocessable Entity | ✅ PASS |
| 13 | POST /predict without model_type returns 422 | 422 Unprocessable Entity | ✅ PASS |
| 14 | POST /predict with invalid model_type returns 422 | 422 (ValueError) | ✅ PASS |
| 15 | POST /predict response has required fields | prediction, confidence, model_type | ✅ PASS |
| 16 | POST /predict confidence in [0, 1] | 0.0 ≤ confidence ≤ 1.0 | ✅ PASS |

**Result: 16/16 passed**

---

### Suite 2 — Data Pipeline Tests (`tests/test_data.py`)
Tests data validation, preprocessing, and EDA using temporary directories.

| # | Test Case | Expected | Result |
|---|---|---|---|
| 17 | Validation runs on empty dirs (no crash) | Exit 0, report generated | ✅ PASS |
| 18 | Validation report has required keys | raw_dir, datasets, overall_valid, summary | ✅ PASS |
| 19 | Validation counts valid images correctly | Count = 3 for 3 images | ✅ PASS |
| 20 | Preprocessing creates data/processed/ | Directory exists after run | ✅ PASS |
| 21 | Preprocessing resizes to exactly 224×224 | Output size = (224, 224) | ✅ PASS |
| 22 | Preprocessing skips corrupt files | corrupt_count ≥ 1, no crash | ✅ PASS |
| 23 | EDA generates JSON and MD reports | Both files exist after run | ✅ PASS |

**Result: 7/7 passed**

---

### Suite 3 — Inference Tests (`tests/test_inference.py`)
Tests model loading, modality detection, predict function, and health check.

| # | Test Case | Expected | Result |
|---|---|---|---|
| 24 | Load 2-class model returns (model, 2) | num_classes = 2 | ✅ PASS |
| 25 | Load 4-class model returns (model, 4) | num_classes = 4 | ✅ PASS |
| 26 | Loaded model is in eval mode | model.training = False | ✅ PASS |
| 27 | Missing checkpoint raises exception | FileNotFoundError or similar | ✅ PASS |
| 28 | Bright image → modality = "pneumonia" | mean intensity > 100 | ✅ PASS |
| 29 | Dark image → modality = "brain" | mean intensity ≤ 100 | ✅ PASS |
| 30 | Mid-gray image returns valid modality | pneumonia or brain | ✅ PASS |
| 31 | predict() with loaded models returns valid result | prediction + confidence | ✅ PASS |
| 32 | predict() with None model raises RuntimeError | "not loaded" message | ✅ PASS |
| 33 | Health check runs without model files | Returns 0 or 1, report generated | ✅ PASS |
| 34 | Health check report has system resources | cpu_percent, memory_percent | ✅ PASS |

**Result: 11/11 passed**

---

## 3. Test Report Summary

| Metric | Value |
|---|---|
| Total test cases | 34 |
| Passed | **34** |
| Failed | **0** |
| Pass rate | **100%** |
| Test run duration | ~17 seconds |
| Framework | pytest 9.0.3 |
| Python version | 3.13.11 |

---

## 4. How to Run Tests

```bash
# Install test dependencies
venv/bin/pip install pytest pytest-cov httpx

# Run all tests
venv/bin/pytest tests/ -v

# Run with coverage
venv/bin/pytest tests/ --cov=src --cov-report=term-missing

# Run a specific suite
venv/bin/pytest tests/test_api.py -v
venv/bin/pytest tests/test_inference.py -v
venv/bin/pytest tests/test_data.py -v
```

---

## 5. Out-of-Scope (Not Unit Tested)

| Item | Reason | How Verified |
|---|---|---|
| Full DVC pipeline (13 stages) | Integration test, takes 45+ min | Manual `dvc repro` run |
| MLflow experiment logging | Requires running MLflow server | Manual observation |
| Docker container prediction | Integration test | Manual `docker run` + curl |
| Model accuracy on test sets | Evaluation metric, not unit test | `reports/evaluation.txt` |
| Drift detection on real data | Requires real data | `reports/monitoring_report.json` |
