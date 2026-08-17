from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def make_test_customer():
    return pd.DataFrame([{
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


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_can_load():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_produces_valid_probability():
    model = joblib.load(MODEL_PATH)
    customer = make_test_customer()

    probability = model.predict_proba(customer)[0, 1]

    assert 0.0 <= probability <= 1.0


def test_model_produces_binary_prediction():
    model = joblib.load(MODEL_PATH)
    customer = make_test_customer()

    prediction = model.predict(customer)[0]

    assert prediction in [0, 1]