import pandas as pd


PREDICTION_DATE = pd.Timestamp("2025-11-30")
HISTORY_CUTOFF = PREDICTION_DATE - pd.Timedelta(days=30)


def build_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    website_events: pd.DataFrame,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
) -> pd.DataFrame:
    """
    Build customer-level churn features and target.

    Historical information is restricted to data available before
    the prediction date. The churn target is based on orders during
    the 30-day period following the prediction date.
    """

    customers = customers.copy()
    orders = orders.copy()
    website_events = website_events.copy()

    # ---------------------------------------------------------
    # 1. Convert dates
    # ---------------------------------------------------------

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"]
    )

    orders["order_date"] = pd.to_datetime(
        orders["order_date"]
    )

    website_events["event_date"] = pd.to_datetime(
        website_events["event_date"]
    )

    # ---------------------------------------------------------
    # 2. Remove duplicate orders
    # ---------------------------------------------------------

    orders = orders.drop_duplicates()

    # ---------------------------------------------------------
    # 3. Customer eligibility
    # ---------------------------------------------------------
    # Customers must have been signed up for at least 30 days
    # before the prediction date.

    eligibility_date = prediction_date - pd.Timedelta(days=30)

    eligible_customers = customers[
        customers["signup_date"] <= eligibility_date
    ].copy()

    # ---------------------------------------------------------
    # 4. Historical orders
    # ---------------------------------------------------------

    historical_orders = orders[
        orders["order_date"] < prediction_date
    ].copy()

    historical_orders = historical_orders[
        historical_orders["customer_id"].isin(
            eligible_customers["customer_id"]
        )
    ]

    order_features = (
        historical_orders
        .groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            total_spent=("amount", "sum"),
            last_order_date=("order_date", "max"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 5. Days since last order
    # ---------------------------------------------------------

    order_features["days_since_last_order"] = (
        prediction_date - order_features["last_order_date"]
    ).dt.days

    order_features["has_previous_order"] = 1

    # ---------------------------------------------------------
    # 6. Historical website events
    # ---------------------------------------------------------

    historical_events = website_events[
        website_events["event_date"] < prediction_date
    ].copy()

    historical_events = historical_events[
        historical_events["customer_id"].isin(
            eligible_customers["customer_id"]
        )
    ]

    # Total events
    total_events = (
        historical_events
        .groupby("customer_id")
        .size()
        .reset_index(name="total_events")
    )

    # Event-type counts
    event_counts = (
        historical_events
        .pivot_table(
            index="customer_id",
            columns="event_type",
            values="event_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    event_counts = event_counts.rename(
        columns={
            "login": "login_count",
            "product_view": "product_view_count",
            "add_to_cart": "add_to_cart_count",
            "checkout": "checkout_count",
        }
    )

    # Make sure all expected event columns exist
    expected_event_columns = [
        "login_count",
        "product_view_count",
        "add_to_cart_count",
        "checkout_count",
    ]

    for column in expected_event_columns:
        if column not in event_counts.columns:
            event_counts[column] = 0

    # ---------------------------------------------------------
    # 7. Recent website activity
    # ---------------------------------------------------------

    recent_events = historical_events[
        historical_events["event_date"]
        >= prediction_date - pd.Timedelta(days=30)
    ]

    recent_event_counts = (
        recent_events
        .groupby("customer_id")
        .size()
        .reset_index(name="events_last_30_days")
    )

    # ---------------------------------------------------------
    # 8. Combine customer features
    # ---------------------------------------------------------

    features = eligible_customers.copy()

    features = features.merge(
        order_features,
        on="customer_id",
        how="left",
    )

    features = features.merge(
        total_events,
        on="customer_id",
        how="left",
    )

    features = features.merge(
        event_counts,
        on="customer_id",
        how="left",
    )

    features = features.merge(
        recent_event_counts,
        on="customer_id",
        how="left",
    )

    # ---------------------------------------------------------
    # 9. Fill appropriate missing values
    # ---------------------------------------------------------

    features["total_orders"] = (
        features["total_orders"].fillna(0)
    )

    features["total_spent"] = (
        features["total_spent"].fillna(0)
    )

    features["total_events"] = (
        features["total_events"].fillna(0)
    )

    features["events_last_30_days"] = (
        features["events_last_30_days"].fillna(0)
    )

    features["has_previous_order"] = (
        features["has_previous_order"].fillna(0)
    )

    for column in expected_event_columns:
        features[column] = (
            features[column].fillna(0)
        )

    # ---------------------------------------------------------
    # 10. Customer tenure
    # ---------------------------------------------------------

    features["tenure_days"] = (
        prediction_date - features["signup_date"]
    ).dt.days

    # ---------------------------------------------------------
    # 11. Future orders
    # ---------------------------------------------------------
    # This is used ONLY to construct the target.
    # It must never be used as a model feature.

    future_end = prediction_date + pd.Timedelta(days=30)

    future_orders = orders[
        (orders["order_date"] >= prediction_date)
        & (orders["order_date"] < future_end)
    ]

    future_order_counts = (
        future_orders
        .groupby("customer_id")
        .size()
        .reset_index(name="future_orders")
    )

    features = features.merge(
        future_order_counts,
        on="customer_id",
        how="left",
    )

    features["future_orders"] = (
        features["future_orders"].fillna(0)
    )

    # ---------------------------------------------------------
    # 12. Create churn target
    # ---------------------------------------------------------

    features["churn"] = (
        features["future_orders"] == 0
    ).astype(int)

    return features