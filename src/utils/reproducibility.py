from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import random
import subprocess
import sys
import tempfile

import mlflow
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_git_commit() -> str:
    try:
        commit = subprocess.getoutput("git rev-parse HEAD").strip()
        return commit if commit else "unknown"
    except:
        return "unknown"


def _write_temp_text(prefix: str, content: str, suffix: str = ".txt") -> str:
    with tempfile.NamedTemporaryFile("w", delete=False, prefix=prefix, suffix=suffix) as handle:
        handle.write(content)
        return handle.name


def log_reproducibility_context(seed: int, extra: dict[str, str] | None = None) -> None:
    run = mlflow.active_run()
    if run is None:
        return

    # SAFE DVC VERSION
    try:
        dvc_version = importlib_metadata.version("dvc")
    except:
        dvc_version = "not_installed"

    context = {
        "git_commit": get_git_commit(),
        "mlflow_run_id": run.info.run_id,
        "seed": str(seed),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": importlib_metadata.version("torchvision"),
        "mlflow_version": importlib_metadata.version("mlflow"),
        "dvc_version": dvc_version,
        "numpy_version": np.__version__,
    }

    if extra:
        context.update(extra)

    mlflow.set_tags(context)

    freeze = subprocess.getoutput(f'"{sys.executable}" -m pip freeze')
    run_context = json.dumps(context, indent=2, sort_keys=True)

    freeze_path = _write_temp_text("pip-freeze-", freeze)
    metadata_path = _write_temp_text("run-context-", run_context, suffix=".json")

    mlflow.log_artifact(freeze_path, artifact_path="reproducibility")
    mlflow.log_artifact(metadata_path, artifact_path="reproducibility")


# ✅ RESTORED FUNCTIONS (THIS FIXES YOUR ERROR)

def build_dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)