# Release Checklist

Use this checklist before promoting any model version to production.

---

## Pre-Release: Data & Training

- [ ] Dataset is validated — `reports/data_validation_report.json` shows `overall_valid: true`
- [ ] No corrupt images in any split
- [ ] Class distribution is as expected (no silent data corruption)
- [ ] DVC pipeline ran cleanly — `dvc repro` exits with code 0
- [ ] `dvc.lock` committed to git with updated checksums
- [ ] Training completed all epochs without NaN loss
- [ ] MLflow run is recorded — run_id is logged in `dvc.lock`

---

## Pre-Release: Model Quality

- [ ] Test accuracy meets threshold:
  - Pneumonia: ≥ 80% test accuracy, ≥ 0.75 F1
  - Brain Tumor: ≥ 90% test accuracy, ≥ 0.90 F1
- [ ] Validation loss is lower than training loss (no overfitting signal)
- [ ] Confusion matrix reviewed — no single class being ignored
- [ ] Model size ≤ 100 MB per model
- [ ] `reports/evaluation.txt` reviewed and values are reasonable

---

## Pre-Release: Infrastructure

- [ ] Health check passes — `python -m src.monitoring.health_check` returns `HEALTHY`
- [ ] `reports/health_check_report.json` shows:
  - `overall_status: healthy`
  - All model `inference_ok: true`
  - All model `latency_ms < 200`
- [ ] FastAPI starts without errors: `uvicorn src.api.app:app`
- [ ] `/healthz` returns `{"status": "alive"}`
- [ ] `/readyz` returns `{"ready": true, "models_loaded": true}`
- [ ] `/predict` tested with at least one real image per model type
- [ ] `/metrics` returns valid Prometheus output
- [ ] Docker image builds without errors: `docker build -f dockerfile .`
- [ ] Docker Compose stack starts cleanly: `docker compose up -d`

---

## Pre-Release: MLflow Registry

- [ ] Model is registered in MLflow Model Registry
- [ ] Model version is in `Staging` stage (transitioned from `None`)
- [ ] Model signature (input/output shape) is recorded
- [ ] Reproducibility artifacts logged (pip freeze, git commit, run context)
- [ ] Promote to Production stage:
  ```bash
  python -c "
  import mlflow
  client = mlflow.tracking.MlflowClient()
  client.transition_model_version_stage('pneumonia-classifier', 'VERSION', 'Production')
  client.transition_model_version_stage('brain-tumor-classifier', 'VERSION', 'Production')
  "
  ```

---

## Pre-Release: Monitoring Baseline

- [ ] Monitoring baseline established — `reports/monitoring_baseline.json` exists
- [ ] Drift thresholds confirmed in `monitoring_report.json`:
  - `drift_l1 < 0.15` for both datasets
  - `overall_status: healthy`
- [ ] Prometheus alert rules reviewed — `alert_rules.yml` thresholds appropriate

---

## Release

- [ ] Git tag created: `git tag -a v1.0.0 -m "Production release v1.0.0"`
- [ ] All source changes committed and pushed
- [ ] `dvc push` executed to sync artifacts to remote storage
- [ ] `CHANGELOG` or release notes written
- [ ] Team notified of deployment

---

## Post-Release

- [ ] Monitor `/metrics` for first 24 hours
- [ ] Check Prometheus for any fired alerts
- [ ] Run `python -m src.monitoring.monitor` after 24h and verify `overall_status: healthy`
- [ ] Confirm inference latency remains < 200ms under load
- [ ] Schedule next monitoring run (recommend: daily via cron)

---

## Rollback Procedure

If issues are found post-release:

```bash
# 1. Revert model in registry
python -c "
from src.versioning.model_registry import get_model_registry
get_model_registry().rollback_model('pneumonia_resnet50', reason='<reason>')
"

# 2. Revert DVC artifacts to previous state
git log --oneline                      # find previous good commit
git checkout <prev_commit> -- dvc.lock
PATH=venv/bin:$PATH dvc checkout       # restore model files

# 3. Restart API
pkill -f uvicorn
MLFLOW_TRACKING_URI=http://localhost:5000 \
  venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8001 &

# 4. Verify rollback
curl http://localhost:8001/healthz
python -m src.monitoring.health_check
```
