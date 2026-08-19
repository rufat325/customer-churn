"""Customer-level churn feature engineering.

The observation period and the target period are defined as two adjacent,
non-overlapping windows around ``PREDICTION_DATE``:

    history:  order_date <= PREDICTION_DATE
    target:   PREDICTION_DATE < order_date <= PREDICTION_DATE + TARGET_WINDOW_DAYS

Together they partition the timeline with no gap and no overlap, so no
transaction is either counted twice or silently dropped. Features are built
only from the history window; the target window is used only to build the
label and must never reach the model.
"""

import pandas as pd


# The last day of observed history, inclusive. Everything on or before this
# date is knowable at prediction time; everything after it is the target.
# Kept in step with PREDICTION_DAY in src/generate_data.py.
PREDICTION_DATE = pd.Timestamp("2025-12-31")

# Length of the target window, in days.
TARGET_WINDOW_DAYS = 30

# A customer must have signed up at least this long before the prediction
# date to have enough history to score.
MIN_TENURE_DAYS = 30

# Length of the "recent activity" lookback, in days.
RECENT_ACTIVITY_DAYS = 30

EVENT_COUNT_COLUMNS = [
    "login_count",
    "product_view_count",
    "add_to_cart_count",
    "checkout_count",
]


def build_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    website_events: pd.DataFrame,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
    target_window_days: int = TARGET_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build customer-level churn features and target.

    Returns one row per eligible customer, with the model features, the
    ``future_orders`` count used to derive the label, and the ``churn``
    label itself. Callers must drop ``future_orders`` (and the identifier
    and date columns) before training.
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
    # 3. Window boundaries
    # ---------------------------------------------------------
    # History is inclusive of the prediction date; the target window opens
    # the following day. Deriving both from the same anchor is what keeps
    # them adjacent. An earlier version of this module derived them
    # independently and shifted the target by one day (see README section
    # 10), which mislabelled 61 customers.

    target_start = prediction_date + pd.Timedelta(days=1)
    target_end = prediction_date + pd.Timedelta(days=target_window_days)

    recent_activity_start = prediction_date - pd.Timedelta(
        days=RECENT_ACTIVITY_DAYS - 1
    )

    # ---------------------------------------------------------
    # 4. Customer eligibility
    # ---------------------------------------------------------

    eligibility_date = prediction_date - pd.Timedelta(days=MIN_TENURE_DAYS)

    eligible_customers = customers[
        customers["signup_date"] <= eligibility_date
    ].copy()

    eligible_ids = eligible_customers["customer_id"]

    # ---------------------------------------------------------
    # 5. Historical orders
    # ---------------------------------------------------------

    historical_orders = orders[
        orders["order_date"] <= prediction_date
    ].copy()

    historical_orders = historical_orders[
        historical_orders["customer_id"].isin(eligible_ids)
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
    # 6. Days since last order
    # ---------------------------------------------------------

    order_features["days_since_last_order"] = (
        prediction_date - order_features["last_order_date"]
    ).dt.days

    order_features["has_previous_order"] = 1

    # ---------------------------------------------------------
    # 7. Historical website events
    # ---------------------------------------------------------

    historical_events = website_events[
        website_events["event_date"] <= prediction_date
    ].copy()

    historical_events = historical_events[
        historical_events["customer_id"].isin(eligible_ids)
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

    # Make sure all expected event columns exist, and drop any event type
    # the pivot produced that the model does not know about.
    for column in EVENT_COUNT_COLUMNS:
        if column not in event_counts.columns:
            event_counts[column] = 0

    event_counts = event_counts[["customer_id"] + EVENT_COUNT_COLUMNS]

    # ---------------------------------------------------------
    # 8. Recent website activity
    # ---------------------------------------------------------

    recent_events = historical_events[
        historical_events["event_date"] >= recent_activity_start
    ]

    recent_event_counts = (
        recent_events
        .groupby("customer_id")
        .size()
        .reset_index(name="events_last_30_days")
    )

    # ---------------------------------------------------------
    # 9. Combine customer features
    # ---------------------------------------------------------

    features = eligible_customers.copy()

    for frame in (
        order_features,
        total_events,
        event_counts,
        recent_event_counts,
    ):
        features = features.merge(
            frame,
            on="customer_id",
            how="left",
        )

    # ---------------------------------------------------------
    # 10. Customer tenure
    # ---------------------------------------------------------
    # Computed before the recency fill below, which depends on it.

    features["tenure_days"] = (
        prediction_date - features["signup_date"]
    ).dt.days

    # ---------------------------------------------------------
    # 11. Fill missing values
    # ---------------------------------------------------------
    # A missing count means the customer had no such activity, so zero is
    # the right fill. Recency is different: a customer who has never
    # ordered has no "days since last order" at all. Filling it with the
    # column median (the previous behaviour) made the highest-risk segment
    # look like an average recent buyer, so it is filled with tenure_days
    # instead: it has been their entire lifetime since they last ordered,
    # which is both true and monotonic with risk.

    zero_filled = [
        "total_orders",
        "total_spent",
        "total_events",
        "events_last_30_days",
        "has_previous_order",
        *EVENT_COUNT_COLUMNS,
    ]

    for column in zero_filled:
        features[column] = features[column].fillna(0)

    features["days_since_last_order"] = (
        features["days_since_last_order"].fillna(features["tenure_days"])
    )

    # ---------------------------------------------------------
    # 12. Target window
    # ---------------------------------------------------------
    # Used ONLY to construct the label. Never a model feature.

    future_orders = orders[
        (orders["order_date"] >= target_start)
        & (orders["order_date"] <= target_end)
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
    # 13. Create churn target
    # ---------------------------------------------------------

    features["churn"] = (
        features["future_orders"] == 0
    ).astype(int)

    return features
