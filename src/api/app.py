import logging
import shutil
import uuid
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("medical_api")

from src.inference.predict import predict
from src.deployment.health import (
    get_liveness,
    get_readiness,
    get_startup,
    record_inference_time,
    save_health_check_report,
)

# ✅ IMPORT THIS (IMPORTANT)

from src.api.prometheus_middleware import add_prometheus_middleware 

app = FastAPI(
    title="Medical Classification API",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# ✅ ADD PROMETHEUS MIDDLEWARE (CRITICAL FIX)
add_prometheus_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# HEALTH ENDPOINTS
# =========================

@app.get("/healthz")
def liveness_probe():
    return get_liveness()


@app.get("/readyz")
def readiness_probe():
    return get_readiness()


@app.get("/startup")
def startup_probe():
    return get_startup()


@app.get("/health")
def health():
    liveness = get_liveness()
    readiness = get_readiness()
    return {
        "status": "healthy" if liveness["status"] == "alive" else "unhealthy",
        "liveness": liveness["status"],
        "readiness": readiness["status"],
        "details": readiness,
    }


@app.get("/ready")
def ready():
    readiness = get_readiness()
    return {
        "status": readiness["status"],
        "ready": readiness["ready"],
        "details": readiness,
    }


# =========================
# METRICS ENDPOINT (FIXED)
# =========================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =========================
# API ENDPOINTS
# =========================

@app.get("/")
def home():
    return {"message": "Medical Classification API"}


@app.post("/predict")
async def predict_api(
    file: UploadFile = File(...),
    model_type: str = Form(...),
):
    import os
    file_location = f"temp/{uuid.uuid4()}.jpg"
    os.makedirs("temp", exist_ok=True)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_time = time.time()

    try:
        logger.info("Prediction request: model_type=%s file=%s", model_type, file_location)
        result = predict(file_location, model_type)
        inference_time = time.time() - start_time
        record_inference_time(inference_time, success=True)
        logger.info("Prediction result: %s confidence=%.3f latency=%.3fs",
                    result.get("prediction"), result.get("confidence", 0), inference_time)
        return {
            **result,
            "inference_time_seconds": inference_time,
            "model_type": model_type,
        }

    except ValueError as e:
        inference_time = time.time() - start_time
        record_inference_time(inference_time, success=False)
        logger.warning("Invalid request: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    except RuntimeError as e:
        inference_time = time.time() - start_time
        record_inference_time(inference_time, success=False)
        logger.error("Inference error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        inference_time = time.time() - start_time
        record_inference_time(inference_time, success=False)
        logger.exception("Unexpected error during prediction")
        raise

    finally:
        try:
            os.remove(file_location)
        except OSError:
            pass


@app.on_event("shutdown")
async def shutdown_event():
    save_health_check_report()
    print("Health check report saved")