# Operations Runbook — Medical Image Classification MLOps

## 1. Service Overview

| Service | Port | Purpose |
|---|---|---|
| FastAPI (inference) | 8001 | REST API for predictions |
| MLflow | 5000 | Experiment tracking & model registry |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Metrics dashboards |
| Airflow | 8080 | Pipeline orchestration (optional) |
| Frontend | 80 / 3000 | Web UI |

---

## 2. Starting Services

### Local (development)

```bash
source venv/bin/activate

# MLflow
mlflow server --host 127.0.0.1 --port 5000 &

# API
MLFLOW_TRACKING_URI=http://localhost:5000 \
  uvicorn src.api.app:app --host 0.0.0.0 --port 8001 &

# Frontend
cd frontend && python -m http.server 8000
```

### Docker Compose (production)

```bash
docker compose up -d
docker compose logs -f api
```

---

## 3. Running the Pipeline

### Full pipeline (recommended)

```bash
PATH=venv/bin:$PATH MLFLOW_TRACKING_URI=http://localhost:5000 dvc repro
```

### Single stage

```bash
PATH=venv/bin:$PATH dvc repro --single-item <stage_name>
```

### Force re-run a stage

```bash
PATH=venv/bin:$PATH dvc repro --force <stage_name>
```

### Stage order

```
ingestion → validate → preprocess → feature_engineering
                                  → eda
                                  → split → train_pneumonia ─┐
                                          → train_brain      ├→ evaluate → model_optimization
                                                             │           → experiments_init
                                                             └→ monitor
```

---

## 4. Health Checks

```bash
# System + model health
python -m src.monitoring.health_check
cat reports/health_check_report.json

# API liveness
curl http://localhost:8001/healthz

# API readiness (models loaded)
curl http://localhost:8001/readyz

# Prometheus metrics
curl http://localhost:8001/metrics
```

---

## 5. Making Predictions

```bash
# Pneumonia
curl -X POST http://localhost:8001/predict \
  -F "file=@path/to/chest_xray.jpeg" \
  -F "model_type=pneumonia"

# Brain tumor
curl -X POST http://localhost:8001/predict \
  -F "file=@path/to/mri.jpg" \
  -F "model_type=brain"
```

Expected response:
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.998,
  "inference_time_seconds": 0.021,
  "model_type": "pneumonia"
}
```

---

## 6. Monitoring & Alerts

### Run monitoring manually

```bash
PATH=venv/bin:$PATH MLFLOW_TRACKING_URI=http://localhost:5000 \
  python -m src.monitoring.monitor
cat reports/monitoring_report.json
```

### Alert thresholds (alert_rules.yml)

| Alert | Condition |
|---|---|
| API down | No response for 1 min |
| High latency | p95 > 200ms |
| Accuracy drop | > 3% below baseline |
| Data drift | L1 distance > 0.15 |

### Check drift status

```bash
python -c "
import json
r = json.load(open('reports/monitoring_report.json'))
print('Overall:', r['comparison']['overall_status'])
for m in r['comparison']['checks']:
    print(m['dataset'], '-', m['status'])
"
```

---

## 7. Retraining

### Manual retrain

```bash
# Both models
PATH=venv/bin:$PATH MLFLOW_TRACKING_URI=http://localhost:5000 \
  dvc repro train_pneumonia train_brain evaluate monitor

# Single model
PATH=venv/bin:$PATH dvc repro --force train_pneumonia
```

### Automated retrain trigger

```bash
python -m src.retraining.auto_retrain
```

Triggers retrain when:
- Accuracy drops below 90%
- Drift score exceeds 0.35
- 7+ days since last retrain

---

## 8. MLflow Model Registry

```bash
# List registered models
mlflow models list

# Promote to production
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage('pneumonia-classifier', '1', 'Production')
"

# Load production model
python -c "
import mlflow.pytorch
model = mlflow.pytorch.load_model('models:/pneumonia-classifier/Production')
"
```

---

## 9. Rollback

### Model rollback via registry

```bash
python -c "
from src.versioning.model_registry import get_model_registry
registry = get_model_registry()
registry.rollback_model('pneumonia_resnet50', reason='accuracy_drop')
"
```

### DVC rollback to previous pipeline state

```bash
git log --oneline          # find target commit
git checkout <commit> -- dvc.lock
dvc checkout              # restore data/model artifacts to that state
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` in dvc repro | Run with `PATH=venv/bin:$PATH dvc repro` |
| MLflow artifact error | Set `MLFLOW_TRACKING_URI=http://localhost:5000` |
| MPS quantization error | Already fixed — quantization forced to CPU |
| API returns 422 | Ensure `model_type` form field is `pneumonia` or `brain` |
| Model not loaded on startup | Check model files exist in `models/` directory |
| DVC stage already tracked by git | Run `git rm -r --cached <path>` then retry |
