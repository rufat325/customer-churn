"""Generate the synthetic development dataset.

Run from anywhere:

    python src/generate_data.py

Design
------
Every customer carries hidden traits that are dropped before the CSVs are
written, so a model has to infer them from observed behaviour:

``activity_level``  how often they order
``drift``           how that propensity changes over time
``browse_bias``     how much they browse relative to how much they buy

Behaviour is an inhomogeneous Poisson process. For each customer and each day
of their lifetime the intensity is

    base_rate(activity_level) * exp(drift * years_since_signup)

and the realised count is Poisson. Dates therefore fall only inside a
customer's own lifetime, and exposure drives volume.

Why one process across the whole timeline
-----------------------------------------
An earlier version generated history with weights ``activity ** 2`` and the
future target window with ``activity ** 3``. That is a regime change at the
prediction date: the future obeyed a different law than the past, which is
unrealistic and makes any walk-forward backtest meaningless, since each
prediction date would be predicting a different process. One stationary
process per customer, modulated by a slow personal drift, replaces it.

The drift term is what makes temporal validation interesting. Without it a
model trained at any date generalises to any other date by construction, and a
walk-forward harness could never detect decay because there would be none.

Timeline
--------
::

    day   0  2024-01-01  observation starts
    day 730  2025-12-31  primary prediction date
    day 790  2026-02-29  last signup (the newest are too new to score)
    day 820  2026-03-31  data ends (90-day target window fits)
    day 850  2026-04-30  retention campaign window ends

The 90-day headroom means target windows of 30, 60 and 90 days can all be
evaluated from the same prediction date, which is what the target redefinition
study needs.

The retention campaign
----------------------
``campaign.csv`` is a simulated randomised experiment, run strictly after the
observational data ends so it cannot contradict it. Customers are randomised
to treatment or control; treated customers get their order intensity
multiplied by ``1 + uplift`` for 30 days.

Uplift is driven by ``browse_bias``, not by ``activity_level``. That is
deliberate: it makes the customers worth *treating* a different group from the
customers most likely to churn, which is the entire point of uplift modelling
and cannot be demonstrated if the two coincide. Some customers have negative
uplift -- contacting them makes things worse.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RANDOM_SEED = 42

START_DATE = pd.Timestamp("2024-01-01")

# Day offsets from START_DATE.
LAST_SIGNUP_DAY = 790       # 2026-02-29
PREDICTION_DAY = 730        # 2025-12-31
DATA_END_DAY = 820          # 2026-03-31
CAMPAIGN_START_DAY = 821    # 2026-04-01
CAMPAIGN_END_DAY = 850      # 2026-04-30

MAX_TARGET_WINDOW_DAYS = DATA_END_DAY - PREDICTION_DAY   # 90

N_CUSTOMERS = 5000

# Approximate totals; exact counts vary because volumes are Poisson.
TARGET_ORDERS = 35000
TARGET_EVENTS = 90000

N_MISSING_AGES = 100
N_DUPLICATE_ORDERS = 50

COUNTRIES = ["DE", "FR", "NL", "BE", "AT", "AZ"]

EVENT_TYPES = ["login", "product_view", "add_to_cart", "checkout"]
EVENT_TYPE_PROBABILITIES = [0.30, 0.45, 0.15, 0.10]

# Spread of the per-customer propensity drift, in log-rate per year.
DRIFT_SCALE = 0.35

# Spread of the browsing-versus-buying trait.
BROWSE_BIAS_SCALE = 0.60

# Treated customers get intensity multiplied by (1 + uplift) during the
# campaign window. The intercept is negative so a minority are harmed by
# contact.
UPLIFT_SLOPE = 0.90
UPLIFT_INTERCEPT = -0.20

HIDDEN_COLUMNS = ["activity_level", "drift", "browse_bias"]


def _daily_intensity(
    base_rate: np.ndarray,
    drift: np.ndarray,
    signup_day: np.ndarray,
    first_day: int,
    last_day: int,
) -> np.ndarray:
    """
    Per-customer, per-day Poisson intensity over ``[first_day, last_day]``.

    Zero before a customer's signup, so nothing can be dated before they
    existed.
    """

    days = np.arange(first_day, last_day + 1)

    years_since_signup = (days[None, :] - signup_day[:, None]) / 365.0

    intensity = (
        base_rate[:, None]
        * np.exp(drift[:, None] * years_since_signup)
    ).astype(np.float32)

    intensity[days[None, :] < signup_day[:, None]] = 0.0

    return intensity


def _draw_events(
    intensity: np.ndarray,
    first_day: int,
    rng: np.random.Generator,
):
    """Draw Poisson counts per cell and expand to (customer_index, day)."""

    counts = rng.poisson(intensity)

    customer_index, day_index = np.nonzero(counts)
    repeats = counts[customer_index, day_index]

    return (
        np.repeat(customer_index, repeats),
        np.repeat(day_index + first_day, repeats),
    )


def expected_orders(
    customers: pd.DataFrame,
    first_day: int,
    last_day: int,
) -> np.ndarray:
    """
    True expected order count per customer in ``[first_day, last_day]``.

    Computed from the hidden traits, so it is the quantity a perfect model
    would know. Since churn is "no order in the window", the true churn
    probability is ``exp(-expected_orders)`` -- a monotone transform, so
    ranking by this value is the Bayes-optimal ranking and its ROC-AUC is
    the ceiling for any model of this target.

    ``customers`` must still carry the hidden columns, i.e. come from
    ``generate()`` rather than from the CSV.
    """

    signup_day = (
        (customers["signup_date"] - START_DATE).dt.days.to_numpy()
    )

    base_rate = 0.05 + customers["activity_level"].to_numpy() ** 2
    drift = customers["drift"].to_numpy()

    # The generator normalises intensity across the full timeline, so the
    # same constant has to be recovered here for the window to be on the
    # right scale.
    full_timeline = _daily_intensity(
        base_rate, drift, signup_day, 0, DATA_END_DAY
    )
    scale = TARGET_ORDERS / full_timeline.sum()

    window = _daily_intensity(
        base_rate, drift, signup_day, first_day, last_day
    )

    return window.sum(axis=1) * scale


def check_integrity(customers, orders, website_events, campaign) -> None:
    """
    Assert the dataset is internally coherent.

    Assertions rather than tests, because a generator that silently emits
    incoherent data is worse than one that refuses to run.
    """

    signup_by_id = customers.set_index("customer_id")["signup_date"]

    for frame, column, label in (
        (orders, "order_date", "order"),
        (website_events, "event_date", "event"),
    ):
        assert (
            frame[column].to_numpy()
            >= signup_by_id.loc[frame["customer_id"]].to_numpy()
        ).all(), f"an {label} predates its customer's signup"

    data_end = START_DATE + pd.Timedelta(days=DATA_END_DAY)

    assert orders["order_date"].max() <= data_end, "order beyond the data window"
    assert website_events["event_date"].max() <= data_end, (
        "event beyond the data window"
    )

    assert set(campaign["treated"].unique()) <= {0, 1}
    assert set(campaign["ordered_in_campaign"].unique()) <= {0, 1}
    assert len(campaign) == len(customers)


def generate(
    seed: int = RANDOM_SEED,
    shock_day: int | None = None,
    shock_factor: float = 1.0,
):
    """
    Build the four raw frames.

    ``shock_day`` and ``shock_factor`` inject a population-wide change in
    order intensity from that day onward. The default dataset has no shock:
    per-customer drift is a fixed personal trait, so the mapping from
    observed history to future behaviour is stationary and a stale model
    never decays. That is a correct property of this process, but it leaves
    a monitoring harness untested, so the walk-forward notebook regenerates
    a shocked world to prove the harness can actually see a regime change.

    The returned ``customers`` frame still carries the hidden trait columns.
    ``main()`` drops them before writing to disk; the headroom, target and
    uplift analyses keep them to measure ceilings.
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

    activity = rng.beta(2.5, 2.0, N_CUSTOMERS)
    drift = rng.normal(0.0, DRIFT_SCALE, N_CUSTOMERS)
    browse_bias = rng.normal(0.0, BROWSE_BIAS_SCALE, N_CUSTOMERS)

    customers["activity_level"] = activity
    customers["drift"] = drift
    customers["browse_bias"] = browse_bias

    missing_age_indices = rng.choice(
        N_CUSTOMERS, size=N_MISSING_AGES, replace=False
    )
    customers.loc[missing_age_indices, "age"] = np.nan

    customer_ids = customers["customer_id"].to_numpy()

    # ---- Orders ---------------------------------------------------
    order_base = 0.05 + activity ** 2

    order_intensity = _daily_intensity(
        order_base, drift, signup_day, 0, DATA_END_DAY
    )
    # Captured before normalising in place: the campaign window has to be
    # scaled by the same constant, or its rates are on a different scale
    # entirely.
    order_scale = TARGET_ORDERS / order_intensity.sum()
    order_intensity *= order_scale

    if shock_day is not None:
        order_intensity[:, shock_day:] *= shock_factor

    order_index, order_day = _draw_events(order_intensity, 0, rng)

    orders = pd.DataFrame({
        "order_id": np.arange(50001, 50001 + len(order_index)),
        "customer_id": customer_ids[order_index],
        "order_date": START_DATE + pd.to_timedelta(order_day, unit="D"),
        "amount": np.round(rng.gamma(2.0, 50.0, len(order_index)), 2),
    }).sort_values("order_date", kind="stable", ignore_index=True)

    orders["order_id"] = np.arange(50001, 50001 + len(orders))

    # Inject duplicate rows, so de-duplication is a real step downstream.
    duplicates = orders.sample(N_DUPLICATE_ORDERS, random_state=seed)
    orders = pd.concat([orders, duplicates], ignore_index=True)

    # ---- Website events -------------------------------------------
    # Browsing is driven by the same activity level, modulated by the
    # customer's browse-versus-buy trait.
    event_base = (0.05 + activity ** 1.5) * np.exp(browse_bias)

    event_intensity = _daily_intensity(
        event_base, drift, signup_day, 0, DATA_END_DAY
    )
    event_intensity *= TARGET_EVENTS / event_intensity.sum()

    event_index, event_day = _draw_events(event_intensity, 0, rng)

    website_events = pd.DataFrame({
        "event_id": np.arange(90001, 90001 + len(event_index)),
        "customer_id": customer_ids[event_index],
        "event_type": rng.choice(
            EVENT_TYPES, size=len(event_index), p=EVENT_TYPE_PROBABILITIES
        ),
        "event_date": START_DATE + pd.to_timedelta(event_day, unit="D"),
    })

    # ---- Retention campaign (randomised) --------------------------
    # Persuadability rises with browse_bias: customers who browse a lot
    # relative to how much they buy are interested but hesitant, and a nudge
    # moves them. Risk of churning is driven by activity_level instead, so
    # the two populations differ.
    persuadability = 1.0 / (1.0 + np.exp(-2.0 * browse_bias))
    uplift = UPLIFT_SLOPE * persuadability + UPLIFT_INTERCEPT

    treated = rng.integers(0, 2, N_CUSTOMERS)

    campaign_intensity = _daily_intensity(
        order_base, drift, signup_day, CAMPAIGN_START_DAY, CAMPAIGN_END_DAY
    )
    campaign_intensity *= order_scale

    expected_orders = campaign_intensity.sum(axis=1)
    treated_expectation = expected_orders * (1.0 + uplift * treated)

    ordered_in_campaign = (
        rng.poisson(treated_expectation) > 0
    ).astype(int)

    campaign = pd.DataFrame({
        "customer_id": customer_ids,
        "treated": treated,
        "ordered_in_campaign": ordered_in_campaign,
        # Retained here only so the uplift notebook can measure its ceiling;
        # main() drops it, exactly like the other hidden traits.
        "true_uplift": uplift,
    })

    check_integrity(customers, orders, website_events, campaign)

    return customers, orders, website_events, campaign


def main() -> int:
    # git does not track empty directories, so a fresh clone has no data/.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    customers, orders, website_events, campaign = generate()

    customers.drop(columns=HIDDEN_COLUMNS).to_csv(
        DATA_DIR / "customers.csv", index=False
    )
    orders.to_csv(DATA_DIR / "orders.csv", index=False)
    website_events.to_csv(DATA_DIR / "website_events.csv", index=False)
    campaign.drop(columns=["true_uplift"]).to_csv(
        DATA_DIR / "campaign.csv", index=False
    )

    prediction_date = START_DATE + pd.Timedelta(days=PREDICTION_DAY)

    print(f"Development data written to {DATA_DIR}")
    print(f"Customers:      {len(customers):,}")
    print(f"Orders:         {len(orders):,} "
          f"(incl. {N_DUPLICATE_ORDERS} duplicates)")
    print(f"Website events: {len(website_events):,}")
    print(f"Campaign:       {len(campaign):,} customers, "
          f"{campaign['treated'].sum():,} treated")
    print(f"Span:           {START_DATE.date()} to "
          f"{(START_DATE + pd.Timedelta(days=DATA_END_DAY)).date()}")
    print(f"Prediction date:{prediction_date.date()} "
          f"(up to {MAX_TARGET_WINDOW_DAYS}-day target window)")
    print("Integrity:      no order or event precedes its customer's signup")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
