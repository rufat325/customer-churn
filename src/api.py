"""FastAPI service wrapping the churn model.

Run from the project root:

    python -m uvicorn src.api:app --reload
"""

import logging
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import (
    ModelNotAvailable,
    churn_threshold,
    load_model,
    load_model_card,
    score_batch,
)


MAX_BATCH_SIZE = 1000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("churn-api")


# ============================================================
# Model loading
# ============================================================
# The artifact is not tracked in git, so a fresh checkout may not have one.
# Rather than crashing the process at import time with a bare traceback, the
# failure is recorded and surfaced through /health, which lets the Docker
# health check mark the container unhealthy for the right reason.

try:
    model = load_model()
    model_error = None
except ModelNotAvailable as error:
    model = None
    model_error = str(error)
    logger.error("model_load_failed error=%r", str(error).splitlines()[0])

model_card = load_model_card()
threshold = churn_threshold(model_card)

if model is not None:
    logger.info(
        "model_loaded threshold=%.2f calibration=%s",
        threshold,
        (model_card or {}).get("calibration", "unknown"),
    )


app = FastAPI(
    title="Customer Churn Prediction API",
    version="2.0.0",
    description=(
        "Predicts whether a customer will place no order in the next 30 "
        "days. Probabilities are calibrated, and the decision threshold is "
        "chosen from an explicit cost model rather than left at 0.50. See "
        "the project README for feature definitions."
    ),
)


# ============================================================
# Request logging
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()

    response = await call_next(request)

    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )

    response.headers["X-Request-ID"] = request_id
    return response


# ============================================================
# Schemas
# ============================================================

class Customer(BaseModel):
    age: float = Field(..., description="Customer age in years.")
    country: str = Field(..., description="Two-letter country code, e.g. DE.")
    total_orders: float = Field(
        ..., ge=0, description="Orders placed on or before the prediction date."
    )
    total_spent: float = Field(
        ..., ge=0, description="Total order value over the same period."
    )
    days_since_last_order: float = Field(
        ...,
        ge=0,
        description=(
            "Days between the last order and the prediction date. For a "
            "customer who has never ordered, pass tenure_days: that is how "
            "the training data encodes it."
        ),
    )
    has_previous_order: int = Field(
        ..., ge=0, le=1, description="1 if the customer has ever ordered, else 0."
    )
    total_events: float = Field(..., ge=0)
    add_to_cart_count: float = Field(..., ge=0)
    checkout_count: float = Field(..., ge=0)
    login_count: float = Field(..., ge=0)
    product_view_count: float = Field(..., ge=0)
    tenure_days: float = Field(
        ..., ge=0, description="Days since signup, as of the prediction date."
    )
    events_last_30_days: float = Field(
        ...,
        ge=0,
        description="Website events in the 30 days ending on the prediction date.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "country": "DE",
                    "total_orders": 2,
                    "total_spent": 150.0,
                    "days_since_last_order": 75,
                    "has_previous_order": 1,
                    "total_events": 5,
                    "add_to_cart_count": 1,
                    "checkout_count": 0,
                    "login_count": 2,
                    "product_view_count": 2,
                    "tenure_days": 180,
                    "events_last_30_days": 0,
                }
            ]
        }
    }


class CustomerBatch(BaseModel):
    customers: list[Customer] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Between 1 and {MAX_BATCH_SIZE} customers.",
    )


class Prediction(BaseModel):
    churn_probability: float = Field(
        ..., description="Calibrated probability of no order in the next 30 days."
    )
    predicted_churn: int = Field(
        ..., description="1 if the probability is at or above the operating threshold."
    )
    risk_level: str = Field(..., description="LOW, MEDIUM or HIGH.")


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    count: int


def _unavailable() -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": model_error})


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "detail": model_error,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/model")
def model_info():
    """The model card: what was trained, how, and at which operating point."""

    if model is None:
        return _unavailable()

    if model_card is None:
        return {
            "detail": "No model card found. Retrain to generate one.",
            "operating_threshold": threshold,
        }

    return model_card


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    if model is None:
        return _unavailable()

    return score_batch(model, [customer.model_dump()], threshold)[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(batch: CustomerBatch):
    if model is None:
        return _unavailable()

    payload = [customer.model_dump() for customer in batch.customers]
    predictions = score_batch(model, payload, threshold)

    return {"predictions": predictions, "count": len(predictions)}
