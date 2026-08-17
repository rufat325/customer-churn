import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# 1. Customers
# ============================================================

n_customers = 5000

customers = pd.DataFrame({
    "customer_id": np.arange(10001, 10001 + n_customers),
    "age": np.random.randint(18, 70, n_customers),
    "country": np.random.choice(
        ["DE", "FR", "NL", "BE", "AT", "AZ"],
        n_customers
    ),
    "signup_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, n_customers),
        unit="D"
    )
})

# Hidden persistent customer activity level.
# This is deliberately NOT saved in the customer table.
customers["activity_level"] = np.random.beta(
    2.5,
    2.0,
    n_customers
)

# Missing ages
missing_age_indices = np.random.choice(
    customers.index,
    size=100,
    replace=False
)

customers.loc[missing_age_indices, "age"] = np.nan


# ============================================================
# 2. Historical orders
# ============================================================

n_historical_orders = 18000

historical_weights = (
    0.05 + customers["activity_level"] ** 2
)

historical_weights = (
    historical_weights / historical_weights.sum()
)

historical_customer_ids = np.random.choice(
    customers["customer_id"],
    size=n_historical_orders,
    replace=True,
    p=historical_weights
)

historical_orders = pd.DataFrame({
    "order_id": np.arange(
        50001,
        50001 + n_historical_orders
    ),
    "customer_id": historical_customer_ids,
    "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(
        np.random.randint(0, 334, n_historical_orders),
        unit="D"
    ),
    "amount": np.round(
        np.random.gamma(
            shape=2.0,
            scale=50.0,
            size=n_historical_orders
        ),
        2
    )
})


# ============================================================
# 3. Future orders
# ============================================================

n_future_orders = 2200

# Future behavior is driven by the SAME persistent
# customer activity level, but with randomness.
future_weights = (
    0.01 + customers["activity_level"] ** 3
)

future_weights = (
    future_weights / future_weights.sum()
)

future_customer_ids = np.random.choice(
    customers["customer_id"],
    size=n_future_orders,
    replace=True,
    p=future_weights
)

future_orders = pd.DataFrame({
    "order_id": np.arange(
        70001,
        70001 + n_future_orders
    ),
    "customer_id": future_customer_ids,
    "order_date": pd.to_datetime("2025-12-01") + pd.to_timedelta(
        np.random.randint(0, 30, n_future_orders),
        unit="D"
    ),
    "amount": np.round(
        np.random.gamma(
            shape=2.0,
            scale=50.0,
            size=n_future_orders
        ),
        2
    )
})


# ============================================================
# 4. Combine orders
# ============================================================

orders = pd.concat(
    [historical_orders, future_orders],
    ignore_index=True
)


# ============================================================
# 5. Introduce 50 duplicate historical orders
# ============================================================

duplicates = historical_orders.sample(
    50,
    random_state=42
)

orders = pd.concat(
    [orders, duplicates],
    ignore_index=True
)


# ============================================================
# 6. Website events
# ============================================================

n_events = 50000

event_weights = (
    0.05 + customers["activity_level"] ** 1.5
)

event_weights = (
    event_weights / event_weights.sum()
)

event_customer_ids = np.random.choice(
    customers["customer_id"],
    size=n_events,
    replace=True,
    p=event_weights
)

website_events = pd.DataFrame({
    "event_id": np.arange(
        90001,
        90001 + n_events
    ),
    "customer_id": event_customer_ids,
    "event_type": np.random.choice(
        [
            "login",
            "product_view",
            "add_to_cart",
            "checkout"
        ],
        size=n_events,
        p=[0.30, 0.45, 0.15, 0.10]
    ),
    "event_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(
        np.random.randint(0, 364, n_events),
        unit="D"
    )
})


# ============================================================
# 7. Save raw datasets
# ============================================================

customers.drop(
    columns=["activity_level"]
).to_csv(
    "data/customers.csv",
    index=False
)

orders.to_csv(
    "data/orders.csv",
    index=False
)

website_events.to_csv(
    "data/website_events.csv",
    index=False
)


# ============================================================
# 8. Summary
# ============================================================

print("Development data V3 generated successfully.")
print(f"Customers: {len(customers)}")
print(f"Orders: {len(orders)}")
print(f"Website events: {len(website_events)}")