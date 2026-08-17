from pathlib import Path

import joblib
import pandas as pd


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
# Load trained model
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# Example customer
# ============================================================

customer = pd.DataFrame([{
    "age": 35,
    "country": "DE",
    "total_orders": 2,
    "total_spent": 150.00,
    "days_since_last_order": 75,
    "has_previous_order": 1,
    "total_events": 5,
    "add_to_cart_count": 1,
    "checkout_count": 0,
    "login_count": 2,
    "product_view_count": 2,
    "tenure_days": 180,
    "events_last_30_days": 0,
}])


# ============================================================
# Prediction
# ============================================================

churn_probability = model.predict_proba(
    customer
)[0, 1]

prediction = int(
    churn_probability >= 0.5
)


# ============================================================
# Output
# ============================================================

print("CUSTOMER CHURN PREDICTION")
print("--------------------------")
print(
    f"Churn probability: {churn_probability:.2%}"
)

print(
    f"Predicted churn: {prediction}"
)

if prediction == 1:
    print("Risk level: HIGH")
else:
    print("Risk level: LOW")