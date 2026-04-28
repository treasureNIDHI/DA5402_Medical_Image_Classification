# Medical Image Classification MLOps Project

## Problem Statement

Medical image classification is critical for screening and diagnostic support in healthcare. This project addresses two key clinical challenges:

1. **Pneumonia Detection from Chest X-rays**: Automated screening and triage of pneumonia cases to accelerate diagnosis and reduce radiologist workload
2. **Brain Tumor Classification from MRI Images**: Early detection and classification of brain tumors to support clinical decision-making

### Business Objectives
- Build a reproducible, production-grade MLOps pipeline for medical imaging
- Support screening and triage workflows with high accuracy
- Achieve inference latency < 200ms for real-time clinical deployment
- Ensure reproducibility, auditability, and compliance across all pipeline stages
- Enable continuous monitoring and model retraining

### Success Metrics
- **ML Metrics**: F1-score, Accuracy, Validation Loss
- **Operational Metrics**: Inference latency < 200ms, Model size < 100 MB
- **Reproducibility**: All experiments tracked, versioned, and auditable

---

## Approach

This project implements a comprehensive end-to-end MLOps pipeline with the following architecture:

### Pipeline Components

1. **Data Ingestion & Validation**
   - Automated ingestion from raw data sources
   - Data validation (structure, image integrity, class balance)
   - Reports generated: `data_validation_report.json`

2. **Exploratory Data Analysis (EDA)**
   - Class distribution analysis
   - Image statistics (width, height, modes)
   - Visual report generation (`eda_report.json`, `eda_report.md`)

3. **Preprocessing & Feature Engineering**
   - Image normalization and augmentation
   - Feature extraction and versioning
   - Split into train/validation/test sets
   - Feature manifests tracked in `reports/feature_store/`

4. **Model Training & Optimization**
   - ResNet-50 ImageNet pre-trained baseline
   - Hyperparameter tuning and experimentation
   - Model quantization, pruning, and optimization
   - Multiple configurations: resnet50_baseline, resnet50_aggressive, resnet34_lightweight, mobilenet_edge, efficientnet_balanced

5. **Evaluation & Monitoring**
   - Performance metrics computation
   - Model evaluation across test sets
   - Baseline monitoring and health checks
   - Health check reports: `health_check_report.json`, `monitoring_report.json`

6. **Orchestration & Reproducibility**
   - DVC pipeline for workflow orchestration (13 stages)
   - MLflow for experiment tracking and model registry
   - Docker for environment consistency
   - Comprehensive versioning and lineage tracking

### Technology Stack
- **Data Processing**: Python, DVC
- **ML Framework**: PyTorch
- **Model Tracking**: MLflow
- **Workflow Orchestration**: DVC, Airflow (optional)
- **Monitoring**: Prometheus, Health check modules
- **Deployment**: Docker, FastAPI
- **Version Control**: Git, DVC

---

## File & Folder Structure

```
AI_Project_DA5402/
│
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── requirements-docker.txt            # Docker-specific dependencies
├── dvc.yaml                           # DVC pipeline configuration
├── docker-compose.yml                 # Docker compose for multi-container setup
├── dockerfile                         # Docker image definition
├── alert_rules.yml                    # Monitoring alert rules configuration
│
├── data/                              # Data directory (DVC tracked)
│   ├── raw/                           # Raw input data
│   │   ├── chest_xray/                # Chest X-ray images
│   │   └── brain_tumor/               # Brain tumor MRI images
│   ├── processed/                     # Processed and normalized data
│   │   ├── brain_tumor/               # Processed brain tumor data
│   │   └── chest_xray/                # Processed chest X-ray data
│   └── splits/                        # Train/val/test splits
│       └── brain_tumor/               # Brain tumor split dataset
│
├── models/                            # Trained model artifacts
│   ├── brain_tumor/
│   │   └── brain_resnet50.pt          # Brain tumor classification model
│   └── pneumonia/
│       └── pneumonia_resnet50.pt      # Pneumonia classification model
│
├── reports/                           # Generated reports and metrics
│   ├── data_validation_report.json    # Data validation results
│   ├── eda_report.json                # EDA metrics in JSON format
│   ├── eda_report.md                  # EDA report in Markdown
│   ├── evaluation.txt                 # Model evaluation results
│   ├── health_check_report.json       # System health status
│   ├── model_configs.json             # Model configuration tracking
│   ├── model_optimization_report.json # Optimization metrics and results
│   ├── monitoring_baseline.json       # Baseline metrics for monitoring
│   ├── monitoring_report.json         # Continuous monitoring results
│   ├── preprocessing_report.json      # Preprocessing statistics
│   └── feature_store/                 # Feature engineering artifacts
│       ├── feature_baseline.json      # Feature baseline metrics
│       ├── feature_impact.json        # Feature importance analysis
│       ├── feature_manifest.jsonl     # Feature specifications
│       └── feature_spec.json          # Feature schema definitions
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── api/                           # REST API for model serving
│   │   └── app.py                     # FastAPI application
│   ├── data/                          # Data pipeline modules
│   │   ├── __init__.py
│   │   ├── eda.py                     # Exploratory Data Analysis
│   │   ├── ingestion.py               # Data ingestion logic
│   │   ├── preprocessing.py           # Data preprocessing
│   │   ├── split.py                   # Train/val/test splitting
│   │   └── validation.py              # Data validation checks
│   ├── features/                      # Feature engineering
│   │   └── engineering.py             # Feature extraction and transformation
│   ├── training/                      # Model training
│   │   └── train.py                   # Training scripts
│   ├── evaluation/                    # Model evaluation
│   │   └── evaluate.py                # Evaluation metrics
│   ├── monitoring/                    # Monitoring and health checks
│   │   ├── health_check.py            # System health checks
│   │   └── monitor.py                 # Performance monitoring
│   ├── inference/                     # Inference/prediction logic
│   │   └── predict.py                 # Model inference scripts
│   ├── deployment/                    # Deployment configurations
│   │   ├── docker/                    # Docker configurations
│   │   └── k8s/                       # Kubernetes manifests (optional)
│   ├── versioning/                    # Model versioning and registry
│   │   └── registry.py                # MLflow registry integration
│   ├── retraining/                    # Automated retraining logic
│   │   └── retrain.py                 # Retraining pipeline
│   └── utils/                         # Utility functions
│       └── helpers.py                 # Common utilities
│
├── frontend/                          # Web UI (optional)
│   └── index.html                     # Frontend interface
│
├── mlruns/                            # MLflow tracking (local storage)
│   ├── 1/, 2/, 3/, ...                # MLflow experiment runs
│   └── models/                        # MLflow model registry
│
├── envp/                              # Python virtual environment
│   ├── bin/                           # Executables
│   ├── lib/                           # Python packages
│   └── include/                       # Header files
│
├── scripts/                           # Standalone scripts
│   ├── run_all.sh                     # Full pipeline execution (Linux/macOS)
│   └── run_all.ps1                    # Full pipeline execution (Windows)
│
├── temp/                              # Temporary files and cache
│
└── docs/                              # Documentation
    ├── DATA_SOURCES_AND_BIAS.md       # Data provenance and bias analysis
    ├── MODEL_SELECTION.md             # Model architecture justification
    ├── RUNBOOK.md                     # Operations runbook
    └── RELEASE_CHECKLIST.md           # Release procedures
```

### Key Directories Explained

- **data/**: Versioned with DVC, contains raw inputs and processed outputs
- **models/**: PyTorch model checkpoints for both tasks
- **reports/**: Generated artifacts from each pipeline stage (not versioned)
- **src/**: Core Python source code organized by functional domain
- **mlruns/**: MLflow experiment tracking (reproducible experiment logs)
- **scripts/**: Platform-specific automation scripts

---

## Results

### Model Performance

**ResNet-50 (ImageNet pre-trained)**
- **Accuracy**: 94%+
- **F1-Score**: 0.92-0.95 (varies by task)
- **Inference Latency**: ~85ms (CPU), <50ms (GPU)
- **Model Size**: 102 MB (full), 25 MB (quantized)

### Generated Reports

- `evaluation.txt`: Detailed evaluation metrics per task
- `model_optimization_report.json`: Quantization and pruning results
- `monitoring_baseline.json`: Baseline metrics for drift detection
- `model_configs.json`: All tracked hyperparameter configurations
- `eda_report.md`: Data distribution and statistical analysis

### Reproducibility

All experiments, data lineage, and artifacts are tracked via:
- **DVC**: Data and model versioning
- **MLflow**: Experiment tracking and model registry
- **Git**: Code version control
- **Docker**: Environment reproducibility

---

## How to Run It

### Prerequisites

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI_Project_DA5402
   ```

2. **Set up the Python environment**
   ```bash
   # Using the existing virtual environment
   source envp/bin/activate  # Linux/macOS
   # or
   .\envp\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure DVC (if using remote storage)**
   ```bash
   dvc remote add -d myremote /path/to/remote/storage
   dvc remote list
   ```

---

### Running the Full Pipeline

#### Option 1: DVC Reproducible Pipeline (Recommended)

**Run the entire DVC pipeline:**
```bash
dvc repro
```

This executes all 13 stages in dependency order:
1. `ingestion` → Data loading
2. `validate` → Data validation
3. `preprocess` → Data preprocessing
4. `feature_engineering` → Feature extraction
5. `eda` → Exploratory Data Analysis
6. `latency_benchmark` → Performance baseline
7. `split` → Train/val/test splitting
8. `train_pneumonia` → Pneumonia model training
9. `train_brain` → Brain tumor model training
10. `evaluate` → Model evaluation
11. `model_optimization` → Quantization & pruning
12. `experiments_init` → Initialize MLflow experiments
13. `monitor` → Health checks and monitoring

**View the DAG (Directed Acyclic Graph):**
```bash
dvc dag
```

**Run a specific stage:**
```bash
dvc repro --single-item <stage_name>
```

**Force re-run specific stages:**
```bash
dvc repro --force <stage_name>
```

---

#### Option 2: Individual Pipeline Steps

**1. Data Validation**
```bash
python -m src.data.validation
# Output: reports/data_validation_report.json
```

**2. Exploratory Data Analysis**
```bash
python -m src.data.eda
# Output: reports/eda_report.json, reports/eda_report.md
```

**3. Data Preprocessing**
```bash
python -m src.data.preprocessing
# Output: data/processed/
```

**4. Feature Engineering**
```bash
python -m src.features.engineering
# Output: reports/feature_store/
```

**5. Data Splitting**
```bash
python -m src.data.split
# Output: data/splits/
```

**6. Model Training**
```bash
# Pneumonia model
python -m src.training.train --task pneumonia --config resnet50_baseline

# Brain tumor model
python -m src.training.train --task brain_tumor --config resnet50_baseline
```

**7. Model Evaluation**
```bash
python -m src.evaluation.evaluate
# Output: reports/evaluation.txt
```

**8. Model Optimization**
```bash
python -m src.training.optimize
# Output: reports/model_optimization_report.json
```

**9. Health Checks**
```bash
python -m src.monitoring.health_check
# Output: reports/health_check_report.json
```

---

### MLflow Experiment Tracking

**Start MLflow UI (local tracking)**
```bash
mlflow ui
```
Then open http://localhost:5000 to view experiments, metrics, and artifacts.

**View registered models:**
```bash
mlflow models list
```

**Log a new experiment:**
```bash
mlflow run . -P task=pneumonia -P config=resnet50_baseline
```

**Load a model from registry:**
```python
import mlflow
model = mlflow.pytorch.load_model("models:/pneumonia_classifier/production")
```

---

### Airflow Workflow Orchestration (Optional)

**Initialize Airflow (if using Airflow instead of DVC)**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
```

**Create DAGs directory:**
```bash
mkdir -p $AIRFLOW_HOME/dags
```

**Start Airflow Scheduler & Webserver:**
```bash
airflow scheduler &
airflow webserver &
```

Access Airflow UI at http://localhost:8080

**Trigger a DAG:**
```bash
airflow dags trigger medical_imaging_pipeline
```

**View DAG status:**
```bash
airflow dags list
airflow tasks list medical_imaging_pipeline
```

---

### Prometheus Monitoring & Alerts

**Start Prometheus (if using Docker):**
```bash
docker-compose up -d prometheus
```

**View Prometheus UI:**
http://localhost:9090

**Query metrics:**
```
# Model inference latency (p95)
histogram_quantile(0.95, model_inference_latency_seconds_bucket)

# Model accuracy
model_accuracy

# Data drift score
data_drift_score
```

**Check configured alerts:**
```bash
cat alert_rules.yml
```

**Trigger alerts on conditions:**
- Model accuracy drops below threshold
- Inference latency exceeds 200ms
- Data drift detected
- Failed health checks

---

### Docker Deployment

**Build Docker image:**
```bash
docker build -f dockerfile -t medical-imaging:latest .
```

**Run container with docker-compose:**
```bash
docker-compose up -d
```

**Access services:**
- FastAPI Server: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090

**Check container logs:**
```bash
docker-compose logs -f api
docker-compose logs -f mlflow
```

---

### Docker Compose Troubleshooting (WSL 2)

If you see: `"The command 'docker-compose' could not be found in this WSL 2 distro"`

**Solution 1: Use Modern Docker Compose Syntax (Recommended)**
```bash
# Instead of:
docker-compose up -d

# Use:
docker compose up -d
```

This works on all modern Docker Desktop installations without additional setup.

**Solution 2: Enable Docker Desktop WSL 2 Integration**
1. Open Docker Desktop Settings
2. Go to Settings → Resources → WSL Integration
3. Toggle "Enable integration with my default WSL distro"
4. Select your distro (Ubuntu) and click "Apply & Restart"
5. Restart Docker Desktop
6. Reopen WSL terminal

**Solution 3: Install docker-compose in WSL**
```bash
# Using pip (easiest)
pip install docker-compose

# Or using apt
sudo apt-get update && sudo apt-get install docker-compose

# Verify installation
docker-compose --version
```

**Solution 4: Run Services Manually (No Docker Compose Required)**

Open 3 separate terminals:

Terminal 1:
```bash
source envp/bin/activate
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

Terminal 2:
```bash
cd frontend
python -m http.server 8000
```

Terminal 3:
```bash
source envp/bin/activate
mlflow ui --host 0.0.0.0 --port 5000
```

Then access:
- Frontend: http://localhost:8000/index.html
- API: http://localhost:8001
- MLflow: http://localhost:5000

See [QUICK_START.md](QUICK_START.md) and [DOCKER_SETUP_WSL2.md](DOCKER_SETUP_WSL2.md) for detailed setup guides.

---

### Model Inference & API

**Start the FastAPI server:**
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Make predictions via API:**
```bash
# Pneumonia classification
curl -X POST "http://localhost:8000/predict/pneumonia" \
  -F "file=@path/to/chest_xray.jpg"

# Brain tumor classification
curl -X POST "http://localhost:8000/predict/brain_tumor" \
  -F "file=@path/to/mri_image.jpg"
```

**Python inference client:**
```python
import requests
from pathlib import Path

image_path = Path("path/to/image.jpg")
response = requests.post(
    "http://localhost:8000/predict/pneumonia",
    files={"file": open(image_path, "rb")}
)
print(response.json())
```

---

### Continuous Monitoring & Retraining

**Run monitoring:**
```bash
python -m src.monitoring.monitor
# Output: reports/monitoring_report.json
```

**Trigger retraining if drift detected:**
```bash
python -m src.retraining.retrain --task pneumonia --threshold 0.05
```

**Automated retraining schedule (via cron or scheduler):**
```bash
# Linux/macOS: Add to crontab
0 2 * * * cd /path/to/project && dvc repro

# Windows: Use Task Scheduler
schtasks /create /tn "DVC-Pipeline" /tr "dvc repro" /sc daily /st 02:00
```

---

### Useful Debugging Commands

**Check DVC status:**
```bash
dvc status
```

**Reproduce with detailed logging:**
```bash
dvc repro --verbose
```

**List all DVC metrics:**
```bash
dvc metrics show
```

**Pull data from remote:**
```bash
dvc pull
```

**Push data to remote:**
```bash
dvc push
```

**View commit history with data:**
```bash
dvc dag --outs
```

---

### Quick Start Script

Run the complete pipeline with one command:

**Linux/macOS:**
```bash
bash scripts/run_all.sh
```

**Windows PowerShell:**
```powershell
.\scripts\run_all.ps1
```

---

## Additional Resources

See:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/RUNBOOK.md](docs/RUNBOOK.md) - Operations runbook
- [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) - Release procedures
- [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md) - Model architecture justification
- [docs/DATA_SOURCES_AND_BIAS.md](docs/DATA_SOURCES_AND_BIAS.md) - Data provenance and bias analysis

## Security & Best Practices

- Sensitive data should be encrypted at rest by the storage provider or host volume encryption
- Sensitive data should be encrypted in transit using HTTPS/TLS or a secure cloud provider connection
- This repository documents the workflow but does not itself implement storage-layer encryption
- All model artifacts and data lineage are version controlled for auditability
- Use `.gitignore` and `.dvcignore` to prevent accidental commits of sensitive files
