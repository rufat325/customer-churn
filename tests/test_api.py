import pytest
from fastapi.testclient import TestClient

from src.api import MAX_BATCH_SIZE, app
from src.scoring import MODEL_PATH, churn_threshold, risk_level


client = TestClient(app)


requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"No model at {MODEL_PATH}. Run: python src/train.py",
)


def make_customer():
    return {
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


@requires_model
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


@requires_model
def test_predict_endpoint():
    response = client.post("/predict", json=make_customer())

    assert response.status_code == 200

    result = response.json()

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["predicted_churn"] in [0, 1]
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


@requires_model
def test_predict_response_is_internally_consistent():
    result = client.post("/predict", json=make_customer()).json()

    probability = result["churn_probability"]
    threshold = churn_threshold()

    assert result["predicted_churn"] == int(probability >= threshold)
    assert result["risk_level"] == risk_level(probability, threshold)


@requires_model
def test_batch_endpoint_matches_single_endpoint():
    customers = [make_customer() for _ in range(3)]
    customers[1]["total_orders"] = 40
    customers[2]["days_since_last_order"] = 400

    response = client.post("/predict/batch", json={"customers": customers})

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 3
    assert len(body["predictions"]) == 3

    for customer, batched in zip(customers, body["predictions"]):
        single = client.post("/predict", json=customer).json()
        assert batched == single


@requires_model
def test_batch_endpoint_rejects_empty_list():
    response = client.post("/predict/batch", json={"customers": []})

    assert response.status_code == 422


@requires_model
def test_batch_endpoint_rejects_oversized_batch():
    customers = [make_customer()] * (MAX_BATCH_SIZE + 1)

    response = client.post("/predict/batch", json={"customers": customers})

    assert response.status_code == 422


@requires_model
def test_model_endpoint_exposes_the_card():
    response = client.get("/model")

    assert response.status_code == 200

    card = response.json()

    assert "operating_threshold" in card
    assert 0.0 < card["operating_threshold"] < 1.0
    assert card["calibration"] in ("sigmoid", "isotonic")
    assert card["features"]["input"]


@requires_model
def test_response_carries_a_request_id():
    response = client.get("/health")

    assert response.headers.get("X-Request-ID")


def test_predict_rejects_invalid_customer():
    customer = make_customer()
    customer["age"] = "not-a-number"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_missing_field():
    customer = make_customer()
    del customer["tenure_days"]

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_negative_counts():
    customer = make_customer()
    customer["total_orders"] = -1

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


@requires_model
def test_predict_rejects_derived_feature_injection():
    # Unlike the other rejection tests, this one expects a scored 200 rather
    # than a 422 from schema validation, so it needs a loaded model.
    # Derived features are computed in the pipeline. A caller supplying one
    # should be ignored, not allowed to override the computation.
    customer = make_customer()
    customer["orders_per_day"] = 999.0

    response = client.post("/predict", json=customer)

    assert response.status_code == 200
    assert response.json() == client.post("/predict", json=make_customer()).json()
