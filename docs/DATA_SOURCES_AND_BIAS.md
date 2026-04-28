# Data Sources and Bias Analysis

## Datasets Used

### 1. Chest X-Ray Images (Pneumonia)

| Attribute | Details |
|---|---|
| **Source** | Kaggle: `paultimothymooney/chest-xray-pneumonia` |
| **Origin** | Guangzhou Women and Children's Medical Center, Guangzhou, China |
| **Publication** | Kermany et al., Cell 2018 |
| **License** | CC BY 4.0 |
| **Size** | 5,856 images (train + val + test) |
| **Classes** | NORMAL (1,583), PNEUMONIA (4,273) |
| **Image type** | Anterior-posterior chest X-rays, JPEG |
| **Patient age** | Pediatric patients (1–5 years) |

**Class distribution:**

| Split | NORMAL | PNEUMONIA | Total | Imbalance ratio |
|---|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 | 1:2.9 |
| Val | 8 | 8 | 16 | 1:1 |
| Test | 234 | 390 | 624 | 1:1.7 |

---

### 2. Brain Tumor MRI Dataset

| Attribute | Details |
|---|---|
| **Source** | Kaggle: `sartajbhuvaji/brain-tumor-classification-mri` |
| **License** | Open (Kaggle) |
| **Size** | 7,023 images (train + test) |
| **Classes** | glioma, meningioma, notumor, pituitary |
| **Image type** | MRI scans, JPG |

**Class distribution:**

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---|---|---|---|---|
| Training | 1,321 | 1,339 | 1,595 | 1,457 | 5,712 |
| Testing | 300 | 306 | 405 | 300 | 1,311 |

---

## Known Biases

### Chest X-Ray Dataset

| Bias Type | Description | Impact |
|---|---|---|
| **Population bias** | Pediatric patients only (1–5 years) | Model may not generalize to adults |
| **Geographic bias** | Single institution (Guangzhou) | Imaging equipment and protocols differ globally |
| **Class imbalance** | 3:1 PNEUMONIA:NORMAL in training | Model may over-predict PNEUMONIA |
| **Label quality** | Labels confirmed by physicians, but single-institution | Inter-rater variability not quantified |
| **Image quality** | Variable image quality removed by radiologists | Selection bias in what was excluded |

### Brain Tumor MRI Dataset

| Bias Type | Description | Impact |
|---|---|---|
| **Multi-source** | Images aggregated from multiple online sources | Inconsistent acquisition protocols |
| **No patient metadata** | Age, sex, scanner type unknown | Cannot assess demographic bias |
| **Label source** | Unclear ground truth methodology | Possible label noise |
| **Augmentation overlap** | Some images may be augmented versions of others | Risk of data leakage |

---

## Mitigation Strategies Applied

| Strategy | Applied To | Method |
|---|---|---|
| Class-weighted loss | Pneumonia imbalance | CrossEntropyLoss (equal weight — future work: use class weights) |
| Reproducible splits | Brain tumor | Fixed seed=42 for train/val split |
| Data validation | Both | `src/data/validation.py` checks integrity before training |
| Feature drift monitoring | Both | L1 distribution distance tracked in `monitoring_report.json` |

---

## Limitations and Disclaimers

1. **Not for clinical use**: Models are trained for research/demonstration purposes only. They have not been validated for clinical deployment.

2. **Out-of-distribution risk**: Performance on images from different scanners, patient populations, or imaging protocols is unknown and should be validated before deployment.

3. **Pediatric-only X-ray**: The pneumonia model is trained exclusively on pediatric X-rays. Adult chest X-rays may produce unreliable results.

4. **Confidence thresholding**: The API applies a 0.7 confidence threshold before returning a prediction. Below this threshold, "low confidence" is returned — this is a conservative safety measure, not a clinical negative result.

5. **Modality detection heuristic**: The API uses mean pixel intensity to distinguish X-ray vs MRI. This is a simple heuristic that may fail on edge cases.

---

## Data Lineage

```
Kaggle Download
    └── data/raw/chest_xray/      (original, DVC-tracked)
    └── data/raw/brain_tumor/     (original, DVC-tracked)
         ↓ src/data/preprocessing.py
    └── data/processed/           (resized 224×224, RGB, DVC-tracked)
         ↓ src/data/split.py
    └── data/splits/brain_tumor/  (train/val/test, DVC-tracked)
         ↓ src/training/train.py / train_brain.py
    └── models/                   (trained weights, DVC-tracked)
```

All intermediate artifacts are tracked by DVC. MD5 checksums in `dvc.lock` ensure full reproducibility.
