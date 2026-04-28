#!/bin/sh
# MLflow auto-init — creates experiments + registers models on every fresh start
# Uses Python urllib (no curl needed — works in python:3.11-slim)
set -e

echo "[mlflow-init] Waiting for MLflow..."
python3 - << 'WAIT'
import urllib.request, time, sys
for i in range(40):
    try:
        urllib.request.urlopen("http://mlflow:5000/health", timeout=3)
        print("[mlflow-init] MLflow ready")
        sys.exit(0)
    except Exception:
        time.sleep(3)
print("[mlflow-init] Timeout waiting for MLflow")
sys.exit(1)
WAIT

pip install mlflow==3.10.1 -q --no-cache-dir

python3 - << 'PYEOF'
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()

# Create experiments
experiments = [
    ("pneumonia-classification",   "ResNet-50 chest X-ray pneumonia detection"),
    ("brain-tumor-classification", "ResNet-50 brain MRI tumor classification"),
    ("model-monitoring",           "Drift detection and performance monitoring"),
]
exp_ids = {}
for name, desc in experiments:
    try:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            eid = client.create_experiment(name, tags={"description": desc})
            print(f"[mlflow-init] Created: {name}")
        else:
            eid = exp.experiment_id
            print(f"[mlflow-init] Exists: {name}")
        exp_ids[name] = eid
    except Exception as e:
        print(f"[mlflow-init] Warning: {e}")

# Log representative runs
configs = [
    ("pneumonia-classification", "pneumonia-resnet50",
     {"architecture":"resnet50","epochs":"5","lr":"0.0001","optimizer":"Adam","device":"mps"},
     {"train_loss":0.045,"val_loss":0.062,"accuracy":1.0,"f1_score":1.0,"test_accuracy":0.8141,"test_f1":0.7713}),
    ("brain-tumor-classification", "brain-resnet50",
     {"architecture":"resnet50","epochs":"5","lr":"0.0001","optimizer":"Adam","device":"mps"},
     {"train_loss":0.038,"val_loss":0.051,"accuracy":0.9688,"f1_score":0.9686,"test_accuracy":0.9269,"test_f1":0.9253}),
]
run_ids = {}
for exp_name, run_name, params, metrics in configs:
    try:
        existing = client.search_runs([exp_ids[exp_name]], f"tags.mlflow.runName = '{run_name}'")
        if existing:
            run_ids[exp_name] = existing[0].info.run_id
            print(f"[mlflow-init] Run exists: {run_name}")
        else:
            with mlflow.start_run(experiment_id=exp_ids[exp_name], run_name=run_name) as run:
                run_ids[exp_name] = run.info.run_id
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.set_tag("status", "completed")
                print(f"[mlflow-init] Created run: {run_name}")
    except Exception as e:
        print(f"[mlflow-init] Warning: {e}")

# Register models
model_map = {
    "pneumonia-classification":   "pneumonia-classifier",
    "brain-tumor-classification": "brain-tumor-classifier",
}
for exp_name, model_name in model_map.items():
    try:
        try:
            client.get_registered_model(model_name)
            print(f"[mlflow-init] Model registered: {model_name}")
        except Exception:
            client.create_registered_model(
                model_name,
                description=f"Production {model_name} — ResNet-50 (weights baked into Docker image)",
                tags={"framework":"pytorch","architecture":"resnet50","task":exp_name}
            )
            print(f"[mlflow-init] Registered new model: {model_name}")
        # Create a version pointing to the run
        if exp_name in run_ids:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                mv = client.create_model_version(
                    model_name,
                    source=f"runs:/{run_ids[exp_name]}/model",
                    run_id=run_ids[exp_name],
                    description="v1 — trained on Kaggle medical imaging datasets"
                )
                print(f"[mlflow-init] Created version {mv.version} for {model_name}")
    except Exception as e:
        print(f"[mlflow-init] Warning: {e}")

print("[mlflow-init] Done — experiments + model registry populated")
PYEOF
