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

    # MLflow Setup - Works both locally and in Docker
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("pneumonia-classification")

    os.makedirs("mlflow_artifacts", exist_ok=True)

    with mlflow.start_run(run_name="pneumonia-resnet50") as run:
        run_id = run.info.run_id
        log_reproducibility_context(SEED, {"model_type": "pneumonia"})
        start_time = time.time()

        train_loader, val_loader, _ = get_dataloaders(data_dir, seed=SEED)

        # Model definition
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Log Parameters
        mlflow.log_params({
            "model_name": "pneumonia_model",
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
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)

            # Evaluation
            model.eval()
            val_loss = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            val_loss = val_loss / len(val_loader)

            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="weighted")
            print(f"Epoch {epoch} | Acc {accuracy:.4f} | F1 {f1:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }, step=epoch)

        # Save Artifacts
        report_path = "mlflow_artifacts/pneumonia_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(all_labels, all_preds))
        mlflow.log_artifact(report_path, artifact_path="reports")
        print("✓ Report logged")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure()
        plt.imshow(cm)
        plt.title("Pneumonia Confusion Matrix")
        plt.colorbar()
        cm_path = "mlflow_artifacts/pneumonia_cm.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")
        print("✓ Confusion matrix logged")

        # Save predictions
        predictions_data = {
            "predictions": all_preds,
            "labels": all_labels,
            "accuracy": accuracy,
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

        # --- SAFE REGISTRATION & STAGING ---
        # 1. Infer signature
        sample_input, _ = next(iter(val_loader))
        sample_input = sample_input[0:1].to(DEVICE)
        with torch.no_grad():
            sample_output = model(sample_input)
        signature = infer_signature(sample_input.cpu().numpy(), sample_output.cpu().numpy())

        # 2. Log Model (Explicitly without registered_model_name to avoid 404)
        artifact_path = "pneumonia-model"
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature
        )

        # 3. Save model to disk for deployment
        model_name = "pneumonia-classifier"
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
            os.makedirs("models/pneumonia", exist_ok=True)
            local_model_path = "models/pneumonia/pneumonia_resnet50.pt"
            torch.save(model.state_dict(), local_model_path)
            print(f"✓ Model saved locally: {local_model_path}")
        except Exception as e:
            print(f"❌ Failed to save model locally: {e}")

if __name__ == "__main__":
    try:
        print("🚀 Starting pneumonia model training...")
        train_model("data/processed/chest_xray")
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
