"""Tests for the synthetic data generator.

The generator asserts its own integrity at build time, but these pin the
guarantees downstream code relies on, and act as regression tests for two
earlier defects: dates drawn independently of signup, and a regime change in
the data-generating process at the prediction date.
"""

import numpy as np
import pandas as pd
import pytest

from src.generate_data import (
    CAMPAIGN_END_DAY,
    CAMPAIGN_START_DAY,
    DATA_END_DAY,
    HIDDEN_COLUMNS,
    MAX_TARGET_WINDOW_DAYS,
    PREDICTION_DAY,
    START_DATE,
    expected_orders,
    generate,
)


@pytest.fixture(scope="module")
def dataset():
    return generate()


def test_hidden_traits_are_present_in_memory(dataset):
    customers, _, _, _ = dataset

    for column in HIDDEN_COLUMNS:
        assert column in customers.columns

    assert customers["activity_level"].between(0, 1).all()


def test_no_order_precedes_signup(dataset):
    customers, orders, _, _ = dataset

    merged = orders.merge(
        customers[["customer_id", "signup_date"]], on="customer_id"
    )

    violations = (merged["order_date"] < merged["signup_date"]).sum()

    assert violations == 0, f"{violations} orders precede their customer's signup"


def test_no_event_precedes_signup(dataset):
    customers, _, events, _ = dataset

    merged = events.merge(
        customers[["customer_id", "signup_date"]], on="customer_id"
    )

    violations = (merged["event_date"] < merged["signup_date"]).sum()

    assert violations == 0, f"{violations} events precede their customer's signup"


def test_tenure_and_order_count_are_positively_related(dataset):
    # Regression test: dates drawn independently of signup made exposure and
    # volume unrelated (Spearman 0.006).
    customers, orders, _, _ = dataset

    counts = orders.groupby("customer_id").size().rename("n_orders")

    merged = customers.merge(counts, on="customer_id", how="left")
    merged["n_orders"] = merged["n_orders"].fillna(0)
    merged["tenure"] = (
        START_DATE + pd.Timedelta(days=PREDICTION_DAY) - merged["signup_date"]
    ).dt.days

    rho = merged["n_orders"].corr(merged["tenure"], method="spearman")

    assert rho > 0.3, f"tenure and order count barely related (rho={rho:.3f})"


def test_target_window_fits_inside_the_data(dataset):
    _, orders, _, _ = dataset

    assert MAX_TARGET_WINDOW_DAYS >= 90, (
        "not enough runway for the 90-day target study"
    )
    assert PREDICTION_DAY + MAX_TARGET_WINDOW_DAYS <= DATA_END_DAY

    assert orders["order_date"].max() <= START_DATE + pd.Timedelta(
        days=DATA_END_DAY
    )


def test_generation_is_deterministic():
    a_customers, a_orders, _, a_campaign = generate(seed=7)
    b_customers, b_orders, _, b_campaign = generate(seed=7)

    pd.testing.assert_frame_equal(a_customers, b_customers)
    pd.testing.assert_frame_equal(a_orders, b_orders)
    pd.testing.assert_frame_equal(a_campaign, b_campaign)


def test_different_seeds_give_different_data():
    _, orders_a, _, _ = generate(seed=1)
    _, orders_b, _, _ = generate(seed=2)

    assert len(orders_a) != len(orders_b) or not orders_a.equals(orders_b)


def test_dataset_retains_deliberate_defects(dataset):
    customers, orders, _, _ = dataset

    assert customers["age"].isna().sum() > 0
    assert orders.duplicated().sum() > 0


# ---------------------------------------------------------------------
# Expected orders (the oracle)
# ---------------------------------------------------------------------


def test_expected_orders_matches_realised_volume(dataset):
    # The oracle intensity should reproduce the realised order count in the
    # same window, up to Poisson noise.
    customers, orders, _, _ = dataset

    window_start, window_end = PREDICTION_DAY + 1, PREDICTION_DAY + 30

    predicted = expected_orders(customers, window_start, window_end).sum()

    lo = START_DATE + pd.Timedelta(days=window_start)
    hi = START_DATE + pd.Timedelta(days=window_end)
    actual = (
        (orders.drop_duplicates()["order_date"] >= lo)
        & (orders.drop_duplicates()["order_date"] <= hi)
    ).sum()

    assert predicted == pytest.approx(actual, rel=0.15), (
        f"expected {predicted:.0f} orders, observed {actual}"
    )


def test_expected_orders_grows_with_window_length(dataset):
    customers, _, _, _ = dataset

    short = expected_orders(customers, PREDICTION_DAY + 1, PREDICTION_DAY + 30)
    long = expected_orders(customers, PREDICTION_DAY + 1, PREDICTION_DAY + 90)

    assert (long >= short - 1e-9).all()
    assert long.sum() > short.sum() * 2


def test_expected_orders_is_zero_before_signup():
    customers, _, _, _ = generate(seed=3)

    newest = customers.nlargest(50, "signup_date")
    signup_day = (newest["signup_date"] - START_DATE).dt.days.min()

    # A window entirely before the newest customers existed.
    result = expected_orders(newest, 0, max(int(signup_day) - 1, 0))

    assert np.allclose(result, 0.0)


# ---------------------------------------------------------------------
# Randomised campaign
# ---------------------------------------------------------------------


def test_campaign_is_randomised_and_balanced(dataset):
    customers, _, _, campaign = dataset

    assert len(campaign) == len(customers)
    assert set(campaign["treated"].unique()) == {0, 1}

    share = campaign["treated"].mean()
    assert 0.45 < share < 0.55, f"treatment share {share:.3f} is not balanced"


def test_campaign_shows_a_positive_average_treatment_effect(dataset):
    _, _, _, campaign = dataset

    treated = campaign[campaign["treated"] == 1]["ordered_in_campaign"]
    control = campaign[campaign["treated"] == 0]["ordered_in_campaign"]

    assert treated.mean() - control.mean() > 0.01


def test_campaign_contains_sleeping_dogs(dataset):
    # Some customers must be harmed by contact, or uplift modelling has
    # nothing interesting to find.
    _, _, _, campaign = dataset

    assert (campaign["true_uplift"] < 0).mean() > 0.05


def test_uplift_is_uncorrelated_with_churn_risk(dataset):
    # The entire point of the uplift work: the customers worth treating are
    # not the customers most likely to churn.
    customers, _, _, campaign = dataset

    merged = campaign.merge(
        customers[["customer_id", "activity_level"]], on="customer_id"
    )

    rho = merged["true_uplift"].corr(
        merged["activity_level"], method="spearman"
    )

    assert abs(rho) < 0.15, (
        f"uplift and churn driver are correlated (rho={rho:.3f}); "
        "the uplift demonstration would be confounded"
    )


def test_campaign_window_follows_the_observational_data(dataset):
    _, orders, _, _ = dataset

    assert CAMPAIGN_START_DAY > DATA_END_DAY
    assert CAMPAIGN_END_DAY > CAMPAIGN_START_DAY

    assert orders["order_date"].max() < START_DATE + pd.Timedelta(
        days=CAMPAIGN_START_DAY
    )


# ---------------------------------------------------------------------
# Population shock
# ---------------------------------------------------------------------


def test_shock_reduces_order_volume():
    _, normal, _, _ = generate(shock_day=550, shock_factor=1.0)
    _, shocked, _, _ = generate(shock_day=550, shock_factor=0.5)

    assert len(shocked) < len(normal)


def test_shock_only_affects_days_after_it(dataset):
    _, baseline, _, _ = dataset
    _, shocked, _, _ = generate(shock_day=550, shock_factor=0.1)

    cutoff = START_DATE + pd.Timedelta(days=550)

    before_baseline = (baseline["order_date"] < cutoff).sum()
    before_shocked = (shocked["order_date"] < cutoff).sum()

    # Same seed and same intensities before the shock, so volumes should be
    # close; only the post-shock period collapses.
    assert before_shocked == pytest.approx(before_baseline, rel=0.05)

    after_baseline = (baseline["order_date"] >= cutoff).sum()
    after_shocked = (shocked["order_date"] >= cutoff).sum()

    assert after_shocked < after_baseline * 0.5
