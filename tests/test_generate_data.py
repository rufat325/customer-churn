"""Tests for the synthetic data generator.

The generator asserts its own integrity at build time, but these pin the
guarantees that downstream code relies on, and act as regression tests for
the temporally incoherent version that preceded it.
"""

import pandas as pd
import pytest

from src.generate_data import (
    HIDDEN_COLUMN,
    PREDICTION_DAY,
    START_DATE,
    TARGET_END_DAY,
    TARGET_START_DAY,
    generate,
)


@pytest.fixture(scope="module")
def dataset():
    return generate()


@pytest.fixture(scope="module")
def prediction_date():
    return START_DATE + pd.Timedelta(days=PREDICTION_DAY)


def test_hidden_driver_is_present_in_memory(dataset):
    customers, _, _ = dataset

    assert HIDDEN_COLUMN in customers.columns
    assert customers[HIDDEN_COLUMN].between(0, 1).all()


def test_no_order_precedes_signup(dataset):
    customers, orders, _ = dataset

    merged = orders.merge(
        customers[["customer_id", "signup_date"]], on="customer_id"
    )

    violations = (merged["order_date"] < merged["signup_date"]).sum()

    assert violations == 0, f"{violations} orders precede their customer's signup"


def test_no_event_precedes_signup(dataset):
    customers, _, events = dataset

    merged = events.merge(
        customers[["customer_id", "signup_date"]], on="customer_id"
    )

    violations = (merged["event_date"] < merged["signup_date"]).sum()

    assert violations == 0, f"{violations} events precede their customer's signup"


def test_tenure_and_order_count_are_positively_related(dataset):
    # The bug this guards against: dates drawn independently of signup made
    # exposure and volume unrelated (Spearman 0.006).
    customers, orders, _ = dataset

    counts = orders.groupby("customer_id").size().rename("n_orders")

    merged = customers.merge(counts, on="customer_id", how="left")
    merged["n_orders"] = merged["n_orders"].fillna(0)
    merged["tenure"] = (
        START_DATE + pd.Timedelta(days=PREDICTION_DAY) - merged["signup_date"]
    ).dt.days

    rho = merged["n_orders"].corr(merged["tenure"], method="spearman")

    assert rho > 0.3, f"tenure and order count are barely related (rho={rho:.3f})"


def test_history_and_target_blocks_are_separated(dataset, prediction_date):
    _, orders, _ = dataset

    history = orders[orders["order_id"] < 70000]
    future = orders[orders["order_id"] >= 70000]

    assert history["order_date"].max() <= prediction_date
    assert future["order_date"].min() > prediction_date
    assert future["order_date"].max() <= START_DATE + pd.Timedelta(
        days=TARGET_END_DAY
    )
    assert future["order_date"].min() >= START_DATE + pd.Timedelta(
        days=TARGET_START_DAY
    )


def test_generation_is_deterministic():
    first_customers, first_orders, _ = generate(seed=7)
    second_customers, second_orders, _ = generate(seed=7)

    pd.testing.assert_frame_equal(first_customers, second_customers)
    pd.testing.assert_frame_equal(first_orders, second_orders)


def test_different_seeds_give_different_data():
    _, orders_a, _ = generate(seed=1)
    _, orders_b, _ = generate(seed=2)

    assert len(orders_a) != len(orders_b) or not orders_a.equals(orders_b)


def test_dataset_retains_deliberate_defects(dataset):
    customers, orders, _ = dataset

    # Missing ages and duplicate orders exist on purpose, so the pipeline has
    # real cleaning work to do.
    assert customers["age"].isna().sum() > 0
    assert orders.duplicated().sum() > 0
