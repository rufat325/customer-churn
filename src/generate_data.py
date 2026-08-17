import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# 1. Customers
# -----------------------------

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

# Introduce some missing ages
missing_age_indices = np.random.choice(
    customers.index,
    size=100,
    replace=False
)
customers.loc[missing_age_indices, "age"] = np.nan


# -----------------------------
# 2. Orders
# -----------------------------

n_orders = 20000

orders = pd.DataFrame({
    "order_id": np.arange(50001, 50001 + n_orders),
    "customer_id": np.random.choice(
        customers["customer_id"],
        n_orders
    ),
    "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, n_orders),
        unit="D"
    ),
    "amount": np.round(
        np.random.gamma(shape=2.0, scale=50.0, size=n_orders),
        2
    )
})

# Introduce duplicate orders
duplicates = orders.sample(50, random_state=42)
orders = pd.concat([orders, duplicates], ignore_index=True)


# -----------------------------
# 3. Website events
# -----------------------------

n_events = 50000

website_events = pd.DataFrame({
    "event_id": np.arange(90001, 90001 + n_events),
    "customer_id": np.random.choice(
        customers["customer_id"],
        n_events
    ),
    "event_type": np.random.choice(
        ["login", "product_view", "add_to_cart", "checkout"],
        n_events
    ),
    "event_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, n_events),
        unit="D"
    )
})


# -----------------------------
# Save files
# -----------------------------

customers.to_csv("data/customers.csv", index=False)
orders.to_csv("data/orders.csv", index=False)
website_events.to_csv(
    "data/website_events.csv",
    index=False
)

print("Development data generated successfully.")
print(f"Customers: {len(customers)}")
print(f"Orders: {len(orders)}")
print(f"Website events: {len(website_events)}")