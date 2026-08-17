from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)


# ============================================================
# Load model
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


# ============================================================
# Request schema
# ============================================================

class Customer(BaseModel):
    age: float
    country: str
    total_orders: float
    total_spent: float
    days_since_last_order: float
    has_previous_order: int
    total_events: float
    add_to_cart_count: float
    checkout_count: float
    login_count: float
    product_view_count: float
    tenure_days: float
    events_last_30_days: float


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(customer: Customer):

    customer_df = pd.DataFrame([customer.model_dump()])

    probability = model.predict_proba(
        customer_df
    )[0, 1]

    prediction = int(
        probability >= 0.5
    )

    if probability >= 0.75:
        risk_level = "HIGH"
    elif probability >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "churn_probability": round(
            float(probability),
            4,
        ),
        "predicted_churn": prediction,
        "risk_level": risk_level,
    }