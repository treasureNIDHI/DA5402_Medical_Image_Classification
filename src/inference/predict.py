import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parents[2]

# image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# class labels
PNEUMONIA_CLASSES = ["NORMAL", "PNEUMONIA"]

BRAIN_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

CONFIDENCE_THRESHOLD = 0.7


# 🔥 MODEL LOADING (dynamic)
def load_model(model_path):

    model = models.resnet50(weights=None)

    state_dict = torch.load(model_path, map_location=DEVICE)

    num_classes = state_dict["fc.weight"].shape[0]

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, num_classes


try:
    pneumonia_model, _ = load_model(BASE_DIR / "models/pneumonia/pneumonia_resnet50.pt")
except Exception:
    print("Pneumonia model not available, skipping...")
    pneumonia_model = None

try:
    brain_model, brain_num_classes = load_model(BASE_DIR / "models/brain_tumor/brain_resnet50.pt")
    BRAIN_CLASSES = BRAIN_CLASSES[:brain_num_classes]
except Exception:
    print("Brain tumor model not available, skipping...")
    brain_model = None
    brain_num_classes = len(BRAIN_CLASSES)


# 🔥 SIMPLE MODALITY DETECTION
def detect_modality(image):
    img_gray = image.convert("L")
    img_array = np.array(img_gray)

    mean_intensity = img_array.mean()

    # heuristic:
    # X-ray images → brighter
    # MRI images → darker
    if mean_intensity > 100:
        return "pneumonia"
    else:
        return "brain"


def predict(image_path, model_type):

    image = Image.open(image_path).convert("RGB")

    # 🔥 detect modality
    detected_type = detect_modality(image)

    # 🚫 reject mismatch
    if detected_type != model_type:
        if model_type == "pneumonia":
            return {
                "prediction": "No pneumonia (input is not a chest X-ray)",
                "confidence": 0.0
            }
        elif model_type == "brain":
            return {
                "prediction": "No tumor (input is not a brain MRI)",
                "confidence": 0.0
            }

    image = transform(image).unsqueeze(0).to(DEVICE)

    if model_type == "pneumonia":
        model = pneumonia_model
        classes = PNEUMONIA_CLASSES
    elif model_type == "brain":
        model = brain_model
        classes = BRAIN_CLASSES
    else:
        raise ValueError("model_type must be pneumonia or brain")

    if model is None:
        raise RuntimeError(f"{model_type} model is not loaded")

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    pred_index = pred.item()
    confidence_val = confidence.item()

    # 🚫 confidence-based rejection (extra safety)
    if confidence_val < CONFIDENCE_THRESHOLD:
        if model_type == "pneumonia":
            return {
                "prediction": "No pneumonia detected (low confidence)",
                "confidence": confidence_val
            }
        else:
            return {
                "prediction": "No tumor detected (low confidence)",
                "confidence": confidence_val
            }

    # 🛡️ safety check
    if pred_index >= len(classes):
        return {
            "prediction": f"class_{pred_index}",
            "confidence": confidence_val
        }

    return {
        "prediction": classes[pred_index],
        "confidence": confidence_val
    }