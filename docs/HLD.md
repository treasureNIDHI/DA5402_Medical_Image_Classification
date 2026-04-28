# High-Level Design (HLD) — Medical Image Classification MLOps

## 1. Problem Statement

Automated screening tools for two high-burden clinical tasks:
1. **Pneumonia detection** from chest X-rays — binary classification (NORMAL / PNEUMONIA)
2. **Brain tumor classification** from MRI — 4-class (glioma / meningioma / notumor / pituitary)

The goal is not just model accuracy but a **reproducible, auditable, production-deployable MLOps system** with monitoring and continuous operation capability.

---

## 2. Design Goals

| Goal | Approach |
|---|---|
| Reproducibility | DVC (data + model versioning) + MLflow (experiment tracking) + fixed seeds |
| Low latency inference | Pre-loaded models in memory, CPU-optimized PyTorch, < 200ms target |
| Loose coupling | Frontend ↔ Backend communicate only via REST API |
| Portability | All dependencies containerized; models baked into Docker image |
| Observability | Prometheus metrics on every request; Grafana dashboards; drift monitoring |
| Auditability | All runs logged to MLflow; git history; DVC checksums |

---

## 3. System Components

### 3.1 Data Pipeline (Offline)

```
Kaggle Datasets
    ↓ ingestion.py       — initialize directory structure
    ↓ validation.py      — integrity + class balance checks
    ↓ preprocessing.py   — resize to 224×224, convert to RGB
    ↓ engineering.py     — extract 11 image features → feature store
    ↓ eda.py             — class distribution analysis → reports
    ↓ split.py           — stratified train/val/test split
```

Orchestrated by DVC. All intermediate outputs are MD5-hashed and tracked in `dvc.lock`.

### 3.2 Training Pipeline (Offline)

```
Processed data
    ↓ train.py / train_brain.py   — fine-tune ResNet-50, log to MLflow
    ↓ optimization.py              — quantization + pruning analysis
    ↓ experiments.py               — log 5 model configurations
    ↓ evaluate.py                  — test set evaluation
    ↓ monitor.py                   — establish performance baseline
```

### 3.3 Inference Service (Online)

```
Client Request (image + model_type)
    → FastAPI /predict
    → Modality detection (mean intensity heuristic)
    → ResNet-50 forward pass (pre-loaded model)
    → Confidence thresholding (0.7)
    → JSON response {prediction, confidence, inference_time}
```

### 3.4 Monitoring Pipeline (Periodic)

```
monitor.py
    → Load test data distributions
    → Compute L1 drift distance vs baseline
    → Run model evaluation on test set
    → Write monitoring_report.json
    → Log to MLflow model-monitoring experiment
```

---

## 4. Technology Choices

| Layer | Technology | Reason |
|---|---|---|
| ML Framework | PyTorch + torchvision | Industry standard; pretrained ResNet weights available |
| Model Architecture | ResNet-50 (ImageNet pretrained) | Best accuracy/speed trade-off for ~5K medical images |
| API Framework | FastAPI | Async, auto-docs (Swagger), Pydantic validation, fast |
| Pipeline Orchestration | DVC | Git-native, caching, DAG visualization, reproducible |
| Experiment Tracking | MLflow | Unified tracking, model registry, artifact storage |
| Containerization | Docker + Docker Compose | Reproducible environments; inference image is self-contained |
| Metrics | Prometheus + Grafana | Industry standard; NRT dashboards; alert rules |
| Scheduled Runs | Apache Airflow | DAG scheduler; retry logic; web UI for monitoring |
| CI/CD | GitHub Actions | Syntax validation, Docker build test on every push |
| Frontend | HTML/CSS/JS + Nginx | Zero framework dependencies; lightweight |

---

## 5. Data Flow

```
┌─────────────┐     HTTP POST /predict      ┌─────────────────────┐
│   Browser   │ ─────────────────────────── │    FastAPI :8001     │
│  (Nginx:80) │ ◄────────────────────────── │  (Docker Container)  │
└─────────────┘     JSON response           └──────────┬──────────┘
                                                        │ loads at startup
                                                        ▼
                                              ┌──────────────────┐
                                              │  models/*.pt     │
                                              │  (baked in image)│
                                              └──────────────────┘

┌────────────┐   scrape /metrics 15s   ┌─────────────┐   query   ┌─────────┐
│ Prometheus │ ──────────────────────► │  FastAPI    │           │ Grafana │
│   :9090    │ ◄────────────────────── │  /metrics   │ ◄──────── │  :3001  │
└────────────┘                         └─────────────┘           └─────────┘
```

---

## 6. Deployment Architecture

### Inference-only deployment (minimal)
```bash
docker compose --profile inference up -d
```
Starts: API (:8001) + Frontend (:80) + Prometheus (:9090) + Grafana (:3001)
Requirements: Docker only. No Python, no data, no training infrastructure.

### Full stack
```bash
docker compose up -d
```
Adds: MLflow (:5000) + PostgreSQL (:5432) + Airflow (:8080)

---

## 7. Security Design

| Concern | Mitigation |
|---|---|
| Container privileges | Non-root `appuser` (uid 1000) in inference container |
| Input validation | FastAPI validates file upload content-type |
| Temp file cleanup | `finally` block deletes uploaded image after inference |
| Secrets | No hardcoded credentials; env vars via `docker-compose.yml` |
| CORS | Configured in FastAPI (permissive for dev, should be restricted in prod) |

---

## 8. Scalability Considerations

| Aspect | Current | Production Path |
|---|---|---|
| API workers | 1 (uvicorn single) | `--workers N` or Gunicorn |
| Model loading | Single process, shared memory | Ray Serve or TorchServe for multi-GPU |
| Horizontal scaling | Not configured | Add load balancer + multiple API containers |
| Data pipeline | Sequential DVC | Spark for large-scale data engineering |
