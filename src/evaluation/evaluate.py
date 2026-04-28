import torch
import torch.nn as nn
from torchvision import models

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.training.dataset import get_dataloaders
from src.utils.reproducibility import log_reproducibility_context
from sklearn.metrics import accuracy_score, f1_score
import mlflow


DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_path, data_dir, experiment):

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="evaluation"):
        log_reproducibility_context(42, {"experiment": experiment, "stage": "evaluation"})

        _, _, test_loader = get_dataloaders(data_dir)

        model = models.resnet50(weights=None)

        state_dict = torch.load(model_path, map_location=DEVICE)

        num_classes = state_dict["fc.weight"].shape[0]

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        preds_all = []
        labels_all = []

        with torch.no_grad():
            for x, y in test_loader:

                x = x.to(DEVICE)
                outputs = model(x)

                preds = torch.argmax(outputs, dim=1)

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(y.numpy())

        acc = accuracy_score(labels_all, preds_all)
        f1 = f1_score(labels_all, preds_all, average="macro")

        print("Test Accuracy:", acc)
        print("Test F1:", f1)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)

        mlflow.log_artifact("dvc.lock")
        mlflow.log_artifact("requirements.txt")
        mlflow.log_artifact("requirements-docker.txt")

        return acc, f1


if __name__ == "__main__":

    os.makedirs("reports", exist_ok=True)

    pneumonia_acc, pneumonia_f1 = evaluate(
        "models/pneumonia/pneumonia_resnet50.pt",
        "data/processed/chest_xray",
        "pneumonia-classification"
    )

    brain_acc, brain_f1 = evaluate(
        "models/brain_tumor/brain_resnet50.pt",
        "data/splits/brain_tumor",
        "brain-tumor-classification"
    )

    with open("reports/evaluation.txt", "w", encoding="utf-8") as report_file:
        report_file.write(f"pneumonia_test_accuracy: {pneumonia_acc}\n")
        report_file.write(f"pneumonia_test_f1: {pneumonia_f1}\n")
        report_file.write(f"brain_test_accuracy: {brain_acc}\n")
        report_file.write(f"brain_test_f1: {brain_f1}\n")