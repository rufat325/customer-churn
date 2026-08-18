"""Generate the synthetic development dataset.

Run from anywhere:

    python src/generate_data.py

Design
------
Each customer is assigned a hidden ``activity_level`` drawn from a Beta
distribution. It drives their order rate and their website activity rate, and
is dropped before the CSVs are written, so a model has to infer it from
observed behaviour rather than reading it off a column. That is what makes the
data learnable without being leaky.

Behaviour is generated as a rate process, not a lottery. For each customer the
expected number of events in a window is

    rate(activity_level) * exposure_days

where exposure is the overlap between the window and the customer's lifetime,
and the realised count is Poisson. Dates are then drawn uniformly inside the
customer's own eligible range.

This matters. An earlier version drew a fixed global number of orders, picked
*who* placed them from the activity weights, and then assigned each one a date
drawn uniformly across the whole year -- independently of when the customer
signed up. The result was that 48% of orders and 50% of website events predated
their customer's signup date, and order count was uncorrelated with tenure
(Spearman 0.006) when it should be strongly positive. Under this version no
event can precede its customer's signup, and exposure drives volume.

Timeline
--------
The layout below is what src/features.py must agree with::

    day 0   = 2025-01-01   start of observation
    day 333 = 2025-11-30   prediction date, last day of history (inclusive)
    day 334 = 2025-12-01   first day of the target window
    day 363 = 2025-12-30   last day of the target window

Changing these offsets means changing PREDICTION_DATE in src/features.py.

Programmatic use
----------------
``generate()`` returns the three frames with ``activity_level`` still attached,
which is what notebooks/02_headroom_analysis.ipynb uses to measure how much of
the hidden signal the model actually recovers. ``main()`` drops it before
writing the CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RANDOM_SEED = 42

START_DATE = pd.Timestamp("2025-01-01")

# Day offsets from START_DATE.
PREDICTION_DAY = 333        # 2025-11-30
TARGET_START_DAY = 334      # 2025-12-01
TARGET_END_DAY = 363        # 2025-12-30
LAST_SIGNUP_DAY = 364       # 2025-12-31

N_CUSTOMERS = 5000

# Approximate totals. Exact counts vary slightly because volumes are Poisson.
TARGET_HISTORICAL_ORDERS = 18000
TARGET_FUTURE_ORDERS = 2200
TARGET_EVENTS = 50000

N_MISSING_AGES = 100
N_DUPLICATE_ORDERS = 50

COUNTRIES = ["DE", "FR", "NL", "BE", "AT", "AZ"]

EVENT_TYPES = ["login", "product_view", "add_to_cart", "checkout"]
EVENT_TYPE_PROBABILITIES = [0.30, 0.45, 0.15, 0.10]

HIDDEN_COLUMN = "activity_level"


def draw_counts_and_dates(
    rng: np.random.Generator,
    signup_day: np.ndarray,
    weights: np.ndarray,
    window_start: int,
    window_end: int,
    target_total: int,
):
    """
    Draw per-customer event counts and dates inside a window.

    Exposure is the overlap between ``[window_start, window_end]`` and the
    customer's lifetime, so a customer who signed up late simply has less
    time in which to do anything. The per-day rate constant is solved so the
    expected total across all customers matches ``target_total``.

    Returns ``(customer_index, day_offset)``, both flat arrays with one entry
    per generated event.
    """

    effective_start = np.maximum(signup_day, window_start)
    exposure = np.maximum(window_end - effective_start + 1, 0)

    expected = weights * exposure

    # Solve the rate constant so the expected grand total lands on target.
    rate = target_total / expected.sum()

    counts = rng.poisson(rate * expected)

    customer_index = np.repeat(np.arange(len(counts)), counts)

    # Draw each date uniformly within that customer's own eligible range.
    starts = np.repeat(effective_start, counts)
    spans = np.repeat(exposure, counts)

    day_offset = starts + np.floor(
        rng.random(len(customer_index)) * spans
    ).astype(int)

    return customer_index, day_offset


def check_integrity(customers, orders, website_events) -> None:
    """
    Assert the dataset is internally coherent.

    Assertions rather than tests, because a generator that silently emits
    incoherent data is worse than one that refuses to run.
    """

    signup_by_id = customers.set_index("customer_id")["signup_date"]

    assert (
        orders["order_date"].to_numpy()
        >= signup_by_id.loc[orders["customer_id"]].to_numpy()
    ).all(), "an order predates its customer's signup"

    assert (
        website_events["event_date"].to_numpy()
        >= signup_by_id.loc[website_events["customer_id"]].to_numpy()
    ).all(), "an event predates its customer's signup"

    prediction_date = START_DATE + pd.Timedelta(days=PREDICTION_DAY)
    target_end = START_DATE + pd.Timedelta(days=TARGET_END_DAY)

    history = orders[orders["order_id"] < 70000]
    future = orders[orders["order_id"] >= 70000]

    assert history["order_date"].max() <= prediction_date, (
        "a historical order falls inside the target window"
    )
    assert future["order_date"].min() > prediction_date, (
        "a future order falls inside the history window"
    )
    assert future["order_date"].max() <= target_end, (
        "a future order falls beyond the target window"
    )


def generate(seed: int = RANDOM_SEED):
    """
    Build the three raw frames.

    The returned ``customers`` frame still carries the hidden
    ``activity_level`` column. ``main()`` drops it before writing to disk;
    the headroom analysis keeps it to measure the ceiling.
    """

    rng = np.random.default_rng(seed)

    # ---- Customers ------------------------------------------------
    signup_day = rng.integers(0, LAST_SIGNUP_DAY + 1, N_CUSTOMERS)

    customers = pd.DataFrame({
        "customer_id": np.arange(10001, 10001 + N_CUSTOMERS),
        "age": rng.integers(18, 70, N_CUSTOMERS).astype(float),
        "country": rng.choice(COUNTRIES, N_CUSTOMERS),
        "signup_date": START_DATE + pd.to_timedelta(signup_day, unit="D"),
    })

    # Hidden persistent activity level. Deliberately not saved to CSV.
    customers[HIDDEN_COLUMN] = rng.beta(2.5, 2.0, N_CUSTOMERS)

    # Missing ages, so the pipeline has real imputation work to do.
    missing_age_indices = rng.choice(
        N_CUSTOMERS, size=N_MISSING_AGES, replace=False
    )
    customers.loc[missing_age_indices, "age"] = np.nan

    activity = customers[HIDDEN_COLUMN].to_numpy()
    customer_ids = customers["customer_id"].to_numpy()

    # Ordering is rarer and more sharply driven by activity than browsing.
    order_weights = 0.05 + activity ** 2
    future_order_weights = 0.01 + activity ** 3
    event_weights = 0.05 + activity ** 1.5

    # ---- Historical orders ----------------------------------------
    hist_index, hist_day = draw_counts_and_dates(
        rng, signup_day, order_weights,
        window_start=0,
        window_end=PREDICTION_DAY,
        target_total=TARGET_HISTORICAL_ORDERS,
    )

    historical_orders = pd.DataFrame({
        "order_id": np.arange(50001, 50001 + len(hist_index)),
        "customer_id": customer_ids[hist_index],
        "order_date": START_DATE + pd.to_timedelta(hist_day, unit="D"),
        "amount": np.round(rng.gamma(2.0, 50.0, len(hist_index)), 2),
    })

    # ---- Future orders --------------------------------------------
    # Driven by the same persistent activity level, so history genuinely
    # predicts the future -- but with independent randomness, so the ceiling
    # is well short of perfect.
    future_index, future_day = draw_counts_and_dates(
        rng, signup_day, future_order_weights,
        window_start=TARGET_START_DAY,
        window_end=TARGET_END_DAY,
        target_total=TARGET_FUTURE_ORDERS,
    )

    future_orders = pd.DataFrame({
        "order_id": np.arange(70001, 70001 + len(future_index)),
        "customer_id": customer_ids[future_index],
        "order_date": START_DATE + pd.to_timedelta(future_day, unit="D"),
        "amount": np.round(rng.gamma(2.0, 50.0, len(future_index)), 2),
    })

    orders = pd.concat([historical_orders, future_orders], ignore_index=True)

    # Inject duplicate rows, so de-duplication is a real step downstream.
    duplicates = historical_orders.sample(
        N_DUPLICATE_ORDERS, random_state=seed
    )
    orders = pd.concat([orders, duplicates], ignore_index=True)

    # ---- Website events -------------------------------------------
    event_index, event_day = draw_counts_and_dates(
        rng, signup_day, event_weights,
        window_start=0,
        window_end=TARGET_END_DAY,
        target_total=TARGET_EVENTS,
    )

    website_events = pd.DataFrame({
        "event_id": np.arange(90001, 90001 + len(event_index)),
        "customer_id": customer_ids[event_index],
        "event_type": rng.choice(
            EVENT_TYPES,
            size=len(event_index),
            p=EVENT_TYPE_PROBABILITIES,
        ),
        "event_date": START_DATE + pd.to_timedelta(event_day, unit="D"),
    })

    check_integrity(customers, orders, website_events)

    return customers, orders, website_events


def main() -> int:
    # git does not track empty directories, so a fresh clone has no data/.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    customers, orders, website_events = generate()

    customers.drop(columns=[HIDDEN_COLUMN]).to_csv(
        DATA_DIR / "customers.csv", index=False
    )
    orders.to_csv(DATA_DIR / "orders.csv", index=False)
    website_events.to_csv(DATA_DIR / "website_events.csv", index=False)

    history = int((orders["order_id"] < 70000).sum()) - N_DUPLICATE_ORDERS
    future = int((orders["order_id"] >= 70000).sum())

    print(f"Development data written to {DATA_DIR}")
    print(f"Customers:      {len(customers):,}")
    print(f"Orders:         {len(orders):,} "
          f"({history:,} history + {future:,} target "
          f"+ {N_DUPLICATE_ORDERS} duplicates)")
    print(f"Website events: {len(website_events):,}")
    print("Integrity:      no order or event precedes its customer's signup")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
