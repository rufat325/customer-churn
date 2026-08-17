from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_predict_endpoint():
    customer = {
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

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "predicted_churn" in result
    assert "risk_level" in result

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["predicted_churn"] in [0, 1]
    assert result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]


def test_predict_rejects_invalid_customer():
    customer = {
        "age": "not-a-number",
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

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422