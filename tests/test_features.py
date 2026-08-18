import pandas as pd
import pytest

from src.features import (
    PREDICTION_DATE,
    TARGET_WINDOW_DAYS,
    build_features,
)


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


@pytest.fixture
def built():
    return build_features(*make_test_data())


def test_build_features_returns_expected_columns(built):
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

    assert expected_columns.issubset(set(built.columns))


def test_future_orders_are_not_used_as_features(built):
    # The target-construction column should exist for auditing,
    # but it must be excluded before model training.
    assert "future_orders" in built.columns
    assert "churn" in built.columns


def test_churn_is_binary(built):
    assert set(built["churn"].unique()).issubset({0, 1})


def test_customer_ids_are_unique(built):
    assert built["customer_id"].is_unique


# ---------------------------------------------------------------------
# Target window boundaries
# ---------------------------------------------------------------------
# Regression tests for the off-by-one that shifted the target window a day
# early: it counted orders placed ON the prediction date as future
# behaviour, and dropped the final day of the target window entirely.


def _single_order_on(day: pd.Timestamp) -> pd.DataFrame:
    customers = pd.DataFrame({
        "customer_id": [1],
        "age": [30],
        "country": ["DE"],
        "signup_date": ["2025-01-01"],
    })

    orders = pd.DataFrame({
        "order_id": [1],
        "customer_id": [1],
        "order_date": [day],
        "amount": [10.0],
    })

    events = pd.DataFrame({
        "event_id": pd.Series([], dtype="int64"),
        "customer_id": pd.Series([], dtype="int64"),
        "event_type": pd.Series([], dtype="object"),
        "event_date": pd.Series([], dtype="datetime64[ns]"),
    })

    return build_features(customers, orders, events)


def test_order_on_prediction_date_counts_as_history():
    result = _single_order_on(PREDICTION_DATE)

    assert result.loc[0, "total_orders"] == 1
    assert result.loc[0, "has_previous_order"] == 1
    assert result.loc[0, "days_since_last_order"] == 0
    assert result.loc[0, "future_orders"] == 0
    assert result.loc[0, "churn"] == 1


def test_order_on_first_day_of_target_window_counts_as_future():
    result = _single_order_on(PREDICTION_DATE + pd.Timedelta(days=1))

    assert result.loc[0, "total_orders"] == 0
    assert result.loc[0, "future_orders"] == 1
    assert result.loc[0, "churn"] == 0


def test_order_on_last_day_of_target_window_counts_as_future():
    last_day = PREDICTION_DATE + pd.Timedelta(days=TARGET_WINDOW_DAYS)

    result = _single_order_on(last_day)

    assert result.loc[0, "future_orders"] == 1
    assert result.loc[0, "churn"] == 0


def test_order_after_target_window_is_ignored():
    beyond = PREDICTION_DATE + pd.Timedelta(days=TARGET_WINDOW_DAYS + 1)

    result = _single_order_on(beyond)

    assert result.loc[0, "total_orders"] == 0
    assert result.loc[0, "future_orders"] == 0
    assert result.loc[0, "churn"] == 1


def test_history_and_target_windows_do_not_overlap():
    # Every order on or before the prediction date must be history, and
    # every order in the following 30 days must be target. Counting the
    # same order in both is what makes a leaky target.
    for offset in range(-2, TARGET_WINDOW_DAYS + 2):
        day = PREDICTION_DATE + pd.Timedelta(days=offset)
        result = _single_order_on(day)

        in_history = result.loc[0, "total_orders"] == 1
        in_target = result.loc[0, "future_orders"] == 1

        assert not (in_history and in_target), (
            f"order on {day.date()} counted in both windows"
        )

        if offset <= 0:
            assert in_history, f"order on {day.date()} should be history"
        elif offset <= TARGET_WINDOW_DAYS:
            assert in_target, f"order on {day.date()} should be target"
        else:
            assert not in_history and not in_target


# ---------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------


def test_never_ordered_customer_gets_tenure_as_recency():
    customers = pd.DataFrame({
        "customer_id": [1],
        "age": [30],
        "country": ["DE"],
        "signup_date": ["2025-01-01"],
    })

    orders = pd.DataFrame({
        "order_id": pd.Series([], dtype="int64"),
        "customer_id": pd.Series([], dtype="int64"),
        "order_date": pd.Series([], dtype="datetime64[ns]"),
        "amount": pd.Series([], dtype="float64"),
    })

    events = pd.DataFrame({
        "event_id": pd.Series([], dtype="int64"),
        "customer_id": pd.Series([], dtype="int64"),
        "event_type": pd.Series([], dtype="object"),
        "event_date": pd.Series([], dtype="datetime64[ns]"),
    })

    result = build_features(customers, orders, events)

    assert result.loc[0, "has_previous_order"] == 0
    assert (
        result.loc[0, "days_since_last_order"]
        == result.loc[0, "tenure_days"]
    )


def test_count_features_have_no_missing_values(built):
    count_columns = [
        "total_orders",
        "total_spent",
        "total_events",
        "events_last_30_days",
        "has_previous_order",
        "login_count",
        "product_view_count",
        "add_to_cart_count",
        "checkout_count",
        "days_since_last_order",
    ]

    assert built[count_columns].isna().sum().sum() == 0
