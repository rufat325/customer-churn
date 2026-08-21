import pytest
from fastapi.testclient import TestClient

from src.api import MAX_BATCH_SIZE, app
from src.feature_store import DATA_DIR
from src.scoring import MODEL_PATH, churn_threshold, risk_level


client = TestClient(app)


requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"No model at {MODEL_PATH}. Run: python src/train.py",
)

DATA_PRESENT = all(
    (DATA_DIR / f"{name}.csv").exists()
    for name in ("customers", "orders", "website_events")
)

requires_store = pytest.mark.skipif(
    not (MODEL_PATH.exists() and DATA_PRESENT),
    reason="Needs data and a model for the feature store.",
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

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert "feature_store_loaded" in body


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


# ---------------------------------------------------------------------
# Customer lookup -- the path a UI should use
# ---------------------------------------------------------------------


@requires_store
def test_dashboard_is_served_at_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Churn Risk Console" in response.text


@requires_store
def test_summary_reports_the_whole_base():
    body = client.get("/summary").json()

    assert body["total_customers"] > 0
    assert sum(body["risk_counts"].values()) == body["total_customers"]
    assert 0 < body["operating_threshold"] < 1
    assert body["countries"]


@requires_store
def test_customer_listing_is_ranked_by_risk():
    body = client.get("/customers?limit=20").json()

    probabilities = [c["churn_probability"] for c in body["customers"]]

    assert len(body["customers"]) == 20
    assert probabilities == sorted(probabilities, reverse=True)
    assert body["total"] >= 20


@requires_store
def test_customer_listing_rejects_bad_parameters():
    assert client.get("/customers?limit=0").status_code == 422
    assert client.get("/customers?limit=99999").status_code == 422
    assert client.get("/customers?risk_level=EXTREME").status_code == 422
    assert client.get("/customers?sort=nonsense").status_code == 422


@requires_store
def test_customer_listing_filters_by_risk_band():
    body = client.get("/customers?risk_level=HIGH&limit=25").json()

    assert all(c["risk_level"] == "HIGH" for c in body["customers"])


@requires_store
def test_single_customer_lookup_needs_only_an_id():
    listed = client.get("/customers?limit=1").json()["customers"][0]

    body = client.get(f"/customers/{listed['customer_id']}").json()

    assert body["customer_id"] == listed["customer_id"]
    assert body["churn_probability"] == listed["churn_probability"]
    assert body["risk_level"] == listed["risk_level"]
    assert body["features"]


@requires_store
def test_lookup_agrees_with_raw_feature_scoring():
    # The two entry points must never disagree: /customers/{id} looks the
    # features up, /predict takes them from the caller. Given the same
    # features they have to produce the same score.
    listed = client.get("/customers?limit=1").json()["customers"][0]
    profile = client.get(f"/customers/{listed['customer_id']}").json()

    predicted = client.post("/predict", json=profile["features"]).json()

    assert predicted["churn_probability"] == pytest.approx(
        profile["churn_probability"]
    )
    assert predicted["risk_level"] == profile["risk_level"]


@requires_store
def test_unknown_customer_returns_404():
    response = client.get("/customers/999999999")

    assert response.status_code == 404
    assert "999999999" in response.json()["detail"]


@requires_store
def test_drivers_explain_the_score():
    listed = client.get("/customers?limit=1").json()["customers"][0]

    body = client.get(f"/customers/{listed['customer_id']}/drivers").json()

    assert body["customer_id"] == listed["customer_id"]
    assert "ablation" in body["method"]
    assert body["drivers"]

    for driver in body["drivers"]:
        assert {"feature", "value", "contribution", "percentile"} <= set(driver)


@requires_store
def test_drivers_for_unknown_customer_returns_404():
    assert client.get("/customers/999999999/drivers").status_code == 404
