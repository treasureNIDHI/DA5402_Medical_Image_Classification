# Contributing Guide

## Getting Started

```bash
git clone <repo-url>
cd AI_Project_DA5402
/opt/homebrew/bin/python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Production-ready code |
| `develop` | Integration branch |
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `experiment/<name>` | ML experiments |

```bash
git checkout -b feature/your-feature-name
```

## Making Changes

### Code changes
- Follow existing patterns in the corresponding `src/` module
- No comments unless the WHY is non-obvious
- Run `black src/ --line-length=120` before committing

### Pipeline changes
- If you add/change a DVC stage, update `dvc.yaml`
- Run `dvc repro` end-to-end to verify the pipeline still completes
- Commit the updated `dvc.lock`

### Model changes
- Log all experiments to MLflow
- Record hyperparameter rationale in `docs/MODEL_SELECTION.md`
- Achieve at least baseline performance before opening a PR:
  - Pneumonia: ≥ 80% test accuracy
  - Brain tumor: ≥ 90% test accuracy

## Commit Messages

```
<type>: <short summary>

<optional body — the WHY, not the WHAT>

Co-Authored-By: ...
```

Types: `feat`, `fix`, `refactor`, `docs`, `ci`, `chore`

## Pull Requests

1. Ensure CI passes (all GitHub Actions jobs green)
2. Run the health check: `python -m src.monitoring.health_check`
3. Update `docs/` if you changed architecture or data handling
4. Fill in the PR template (summary + test plan)

## Running Tests

```bash
# Syntax check all modules
python -m py_compile src/**/*.py

# Full pipeline (requires data in data/raw/)
PATH=venv/bin:$PATH MLFLOW_TRACKING_URI=http://localhost:5000 dvc repro

# Health check
python -m src.monitoring.health_check

# API smoke test
uvicorn src.api.app:app --port 8001 &
curl http://localhost:8001/healthz
```

## Project Structure

```
src/
├── data/          # ingestion, validation, EDA, preprocessing, splitting
├── features/      # feature extraction and feature store
├── training/      # model training, optimization, experiments
├── evaluation/    # test set evaluation
├── monitoring/    # drift detection, health checks, latency
├── inference/     # prediction logic and model serving
├── api/           # FastAPI application
├── versioning/    # model registry
├── retraining/    # automated retraining pipeline
└── utils/         # reproducibility utilities
```

## Questions

Open a GitHub Issue with the `question` label.
