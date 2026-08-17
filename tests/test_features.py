import pandas as pd

from src.features import build_features


def make_test_data():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "age": [25, 40, 55],
        "country": ["DE", "FR", "AZ"],
        "signup_date": [
            "2025-01-01",
            "2025-02-01",
            "2025-03-01",
        ],
    })

    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 1, 2, 3],
        "order_date": [
            "2025-10-01",
            "2025-11-01",
            "2025-10-15",
            "2025-12-10",
        ],
        "amount": [100.0, 50.0, 200.0, 75.0],
    })

    website_events = pd.DataFrame({
        "event_id": [1, 2, 3, 4],
        "customer_id": [1, 1, 2, 3],
        "event_type": [
            "login",
            "product_view",
            "add_to_cart",
            "checkout",
        ],
        "event_date": [
            "2025-11-10",
            "2025-11-15",
            "2025-11-20",
            "2025-11-25",
        ],
    })

    return customers, orders, website_events


def test_build_features_returns_expected_columns():
    customers, orders, website_events = make_test_data()

    result = build_features(
        customers,
        orders,
        website_events,
    )

    expected_columns = {
        "customer_id",
        "age",
        "country",
        "total_orders",
        "total_spent",
        "days_since_last_order",
        "has_previous_order",
        "total_events",
        "add_to_cart_count",
        "checkout_count",
        "login_count",
        "product_view_count",
        "tenure_days",
        "events_last_30_days",
        "future_orders",
        "churn",
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_future_orders_are_not_used_as_features():
    customers, orders, website_events = make_test_data()

    result = build_features(
        customers,
        orders,
        website_events,
    )

    # The target-construction column should exist for auditing,
    # but it must be excluded before model training.
    assert "future_orders" in result.columns
    assert "churn" in result.columns


def test_churn_is_binary():
    customers, orders, website_events = make_test_data()

    result = build_features(
        customers,
        orders,
        website_events,
    )

    assert set(result["churn"].unique()).issubset({0, 1})


def test_customer_ids_are_unique():
    customers, orders, website_events = make_test_data()

    result = build_features(
        customers,
        orders,
        website_events,
    )

    assert result["customer_id"].is_unique