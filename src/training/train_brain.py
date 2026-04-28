import torch
import torch.nn as nn
from torchvision import models
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
import os
import pickle

from src.training.dataset import get_dataloaders
from src.utils.reproducibility import log_reproducibility_context, set_global_seed

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

def train_model(data_dir):
    set_global_seed(SEED)

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("brain-tumor-classification")

    os.makedirs("mlflow_artifacts", exist_ok=True)

    with mlflow.start_run(run_name="brain-resnet50") as run:
        run_id = run.info.run_id
        log_reproducibility_context(SEED, {"model_type": "brain"})
        start_time = time.time()

        train_loader, val_loader, _ = get_dataloaders(data_dir, seed=SEED)

        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Log Parameters
        mlflow.log_params({
            "model_name": "brain_model",
            "model_architecture": "resnet50",
            "epochs": 5,
            "learning_rate": 0.0001,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "device": DEVICE,
            "batch_size": train_loader.batch_size
        })

        for epoch in range(5):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    preds = torch.argmax(out, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="weighted")
            print(f"Epoch {epoch} | Acc {acc:.4f} | F1 {f1:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }, step=epoch)

        # Artifacts
        report = classification_report(all_labels, all_preds)
        report_path = "mlflow_artifacts/brain_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")
        print("✓ Report logged")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure()
        plt.imshow(cm)
        plt.title("Brain Tumor Confusion Matrix")
        plt.colorbar()
        cm_path = "mlflow_artifacts/brain_cm.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")
        print("✓ Confusion matrix logged")

        # Save predictions
        predictions_data = {
            "predictions": all_preds,
            "labels": all_labels,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        pkl_path = "mlflow_artifacts/predictions.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(predictions_data, f)
        mlflow.log_artifact(pkl_path, artifact_path="predictions")
        print("✓ Predictions PKL logged")

        # Log training time
        mlflow.log_metric("training_time_sec", time.time() - start_time)
        print("✓ Training time logged")

        # Logic for Signature
        sample_input, _ = next(iter(val_loader))
        sample_input = sample_input[0:1].to(DEVICE)
        with torch.no_grad():
            sample_output = model(sample_input)
        signature = infer_signature(sample_input.cpu().numpy(), sample_output.cpu().numpy())

        # LOG MODEL (Using explicit method to avoid 404 error)
        artifact_path = "brain-model"
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature
        )

        # REGISTER MODEL & SAVE LOCALLY
        model_name = "brain-tumor-classifier"
        try:
            # Try to register with MLflow
            model_uri = f"runs:/{run_id}/{artifact_path}"
            mv = mlflow.register_model(model_uri, model_name)
            
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=True
            )
            print(f"✓ Model registered: {model_name} v{mv.version} → STAGING")
        except Exception as e:
            print(f"⚠️  MLflow registration skipped (Database may not be ready): {e}")
        
        # Always save model locally as backup
        try:
            os.makedirs("models/brain_tumor", exist_ok=True)
            local_model_path = "models/brain_tumor/brain_resnet50.pt"
            torch.save(model.state_dict(), local_model_path)
            print(f"✓ Model saved locally: {local_model_path}")
        except Exception as e:
            print(f"❌ Failed to save model locally: {e}")

if __name__ == "__main__":
    try:
        print("🚀 Starting brain tumor model training...")
        train_model("data/splits/brain_tumor")
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
