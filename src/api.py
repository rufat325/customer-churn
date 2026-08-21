"""FastAPI service wrapping the churn model.

Run from the project root:

    python -m uvicorn src.api:app --reload

Then open http://127.0.0.1:8000 for the dashboard, or /docs for the API.

Two ways in, on purpose
-----------------------
``/customers/{id}`` is the one to use. It takes an identifier, looks the
features up in the feature store, and scores them with the same code that
built the training set. Nothing can drift.

``/predict`` takes raw features instead. It survives because what-if analysis
genuinely needs it -- "what would this customer's risk be if they had ordered
last week" -- and because other systems may hold their own features. But it
puts the burden of computing them correctly on the caller, which is how
training/serving skew starts, so it is not the path a UI should take.
"""

import logging
import sys
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, StoreNotAvailable
from src.security import (
    auth_enabled,
    check_api_key,
    check_password,
    check_rate_limit,
    client_ip,
)
from src.scoring import (
    ModelNotAvailable,
    churn_threshold,
    load_model,
    load_model_card,
    score_batch,
)


MAX_BATCH_SIZE = 1000

DASHBOARD_PATH = Path(__file__).resolve().parent / "static" / "index.html"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("churn-api")


# ============================================================
# Startup
# ============================================================
# Model and feature store fail independently. A missing model is fatal to
# every scoring path; a missing dataset only disables the customer lookups,
# and /predict still works. Reporting them separately means /health says
# which one is actually broken.

try:
    model = load_model()
    model_error = None
except ModelNotAvailable as error:
    model = None
    model_error = str(error)
    logger.error("model_load_failed error=%r", str(error).splitlines()[0])

model_card = load_model_card()
threshold = churn_threshold(model_card)

store = None
store_error = None

if model is not None:
    try:
        store = FeatureStore.load(model)
        logger.info(
            "feature_store_loaded customers=%d prediction_date=%s",
            len(store),
            store.prediction_date,
        )
    except StoreNotAvailable as error:
        store_error = str(error)
        logger.warning("feature_store_unavailable error=%r", str(error).splitlines()[0])
else:
    store_error = "Model unavailable, so no features were scored."


app = FastAPI(
    title="Customer Churn Prediction API",
    version="3.0.0",
    description=(
        "Churn risk for a customer base. Prefer /customers/{id}, which looks "
        "features up by identifier; /predict takes raw features and exists "
        "for what-if analysis. A dashboard is served at /."
    ),
)


@app.middleware("http")
async def enforce_limits(request: Request, call_next):
    """Rate limit every request, and apply auth when it is configured."""

    try:
        check_password(request)
        remaining, _ = check_rate_limit(request)
    except HTTPException as rejection:
        logger.warning(
            "rejected ip=%s path=%s status=%d",
            client_ip(request), request.url.path, rejection.status_code,
        )
        return JSONResponse(
            status_code=rejection.status_code,
            content={"detail": rejection.detail},
            headers=rejection.headers or {},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


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
    total_orders: float = Field(..., ge=0)
    total_spent: float = Field(..., ge=0)
    days_since_last_order: float = Field(
        ...,
        ge=0,
        description=(
            "Days between the last order and the prediction date. For a "
            "customer who has never ordered, pass tenure_days: that is how "
            "the training data encodes it."
        ),
    )
    has_previous_order: int = Field(..., ge=0, le=1)
    total_events: float = Field(..., ge=0)
    add_to_cart_count: float = Field(..., ge=0)
    checkout_count: float = Field(..., ge=0)
    login_count: float = Field(..., ge=0)
    product_view_count: float = Field(..., ge=0)
    tenure_days: float = Field(..., ge=0)
    events_last_30_days: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 35, "country": "DE", "total_orders": 2,
                "total_spent": 150.0, "days_since_last_order": 75,
                "has_previous_order": 1, "total_events": 5,
                "add_to_cart_count": 1, "checkout_count": 0, "login_count": 2,
                "product_view_count": 2, "tenure_days": 180,
                "events_last_30_days": 0,
            }]
        }
    }


class CustomerBatch(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    churn_probability: float
    predicted_churn: int
    risk_level: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    count: int


def _no_model() -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": model_error})


def _no_store() -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": store_error})


# ============================================================
# Dashboard
# ============================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    if not DASHBOARD_PATH.exists():
        return HTMLResponse(
            "<h1>Dashboard not found</h1>"
            "<p>Expected src/static/index.html. The JSON API is at /docs.</p>",
            status_code=404,
        )

    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


# ============================================================
# Service metadata
# ============================================================

@app.get("/health")
def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "feature_store_loaded": False,
                "detail": model_error,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_store_loaded": store is not None,
        "customers_scored": len(store) if store else 0,
        "protection": auth_enabled(),
    }


@app.get("/model")
def model_info():
    """The model card: what was trained, how, and at which operating point."""

    if model is None:
        return _no_model()

    if model_card is None:
        return {
            "detail": "No model card found. Retrain to generate one.",
            "operating_threshold": threshold,
        }

    return model_card


# ============================================================
# Customer lookups -- the path a UI should use
# ============================================================

@app.get("/summary")
def summary():
    """Portfolio-level view of the scored customer base."""

    if store is None:
        return _no_store()

    return store.summary()


@app.get("/customers")
def list_customers(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_level: str | None = Query(None, pattern="^(HIGH|MEDIUM|LOW)$"),
    country: str | None = Query(None, max_length=2),
    q: str | None = Query(None, max_length=32, description="Customer id fragment."),
    sort: str = Query("risk", pattern="^(risk|risk_asc|spend|orders|recency)$"),
):
    """Ranked worklist. Defaults to riskiest first."""

    if store is None:
        return _no_store()

    return store.list_customers(
        limit=limit, offset=offset, risk_level=risk_level,
        country=country, query=q, sort=sort,
    )


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    """One customer: profile, stored features, and current churn score."""

    if store is None:
        return _no_store()

    profile = store.profile(customer_id)

    if profile is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"No scored customer with id {customer_id}."},
        )

    return profile


@app.get("/customers/{customer_id}/drivers")
def get_drivers(customer_id: int, top: int = Query(5, ge=1, le=10)):
    """
    What is pushing this customer's score up or down.

    One-at-a-time ablation against the population median, not SHAP -- see
    the docstring in feature_store.drivers for what that does and does not
    support.
    """

    if store is None:
        return _no_store()

    if not store.exists(customer_id):
        return JSONResponse(
            status_code=404,
            content={"detail": f"No scored customer with id {customer_id}."},
        )

    return {
        "customer_id": customer_id,
        "method": "one-at-a-time ablation against the population median",
        "drivers": store.drivers(model, customer_id, top=top),
    }


# ============================================================
# Raw-feature scoring -- for what-if analysis
# ============================================================

@app.post("/predict", response_model=Prediction,
          dependencies=[Depends(check_api_key)])
def predict(customer: Customer):
    if model is None:
        return _no_model()

    return score_batch(model, [customer.model_dump()], threshold)[0]


@app.post("/predict/batch", response_model=BatchPrediction,
          dependencies=[Depends(check_api_key)])
def predict_batch(batch: CustomerBatch):
    if model is None:
        return _no_model()

    payload = [customer.model_dump() for customer in batch.customers]
    predictions = score_batch(model, payload, threshold)

    return {"predictions": predictions, "count": len(predictions)}
