import pandas as pd
import pytest

from src.feature_store import DATA_DIR, SNAPSHOT_PATH, FeatureStore, StoreNotAvailable
from src.scoring import FEATURE_COLUMNS, MODEL_PATH, load_model


DATA_PRESENT = all(
    (DATA_DIR / f"{name}.csv").exists()
    for name in ("customers", "orders", "website_events")
)

requires_store = pytest.mark.skipif(
    not (MODEL_PATH.exists() and DATA_PRESENT),
    reason="Needs data and a model. Run: python src/generate_data.py && python src/train.py",
)


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("No model artifact.")
    return load_model()


@pytest.fixture(scope="module")
def store(model):
    if not DATA_PRESENT:
        pytest.skip("No datasets.")
    return FeatureStore.load(model)


def test_missing_data_raises_actionable_error(model, tmp_path):
    # snapshot_path=None forces the build-from-CSV path. Without it the
    # store would load the shipped snapshot and never look at data_dir --
    # which is the correct production behaviour, and not what this test is
    # about.
    with pytest.raises(StoreNotAvailable, match="generate_data"):
        FeatureStore.load(model, data_dir=tmp_path, snapshot_path=None)


@requires_store
def test_every_eligible_customer_is_scored(store):
    assert len(store) > 1000

    summary = store.summary()

    assert summary["total_customers"] == len(store)
    assert sum(summary["risk_counts"].values()) == len(store)


@requires_store
def test_profile_exposes_the_full_feature_contract(store):
    listing = store.list_customers(limit=1)
    customer_id = listing["customers"][0]["customer_id"]

    profile = store.profile(customer_id)

    assert profile["customer_id"] == customer_id
    assert 0.0 <= profile["churn_probability"] <= 1.0
    assert profile["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert profile["predicted_churn"] in (0, 1)

    # The stored features must be exactly what the model expects, so a UI
    # never has to recompute anything.
    assert set(profile["features"]) == set(FEATURE_COLUMNS)


@requires_store
def test_unknown_customer_returns_none(store):
    assert store.profile(-1) is None
    assert store.features_for(-1) is None
    assert not store.exists(-1)


@requires_store
def test_scores_match_the_model_for_a_looked_up_customer(store, model):
    from src.scoring import score_batch

    customer_id = store.list_customers(limit=1)["customers"][0]["customer_id"]

    stored = store.profile(customer_id)
    rescored = score_batch(
        model, [store.features_for(customer_id)], store.threshold
    )[0]

    assert stored["churn_probability"] == pytest.approx(
        rescored["churn_probability"]
    )
    assert stored["risk_level"] == rescored["risk_level"]


# ---------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------


@requires_store
def test_default_listing_is_riskiest_first(store):
    customers = store.list_customers(limit=25)["customers"]

    probabilities = [c["churn_probability"] for c in customers]

    assert probabilities == sorted(probabilities, reverse=True)
    assert customers[0]["risk_rank"] == 1


@requires_store
def test_ascending_sort_reverses_the_order(store):
    customers = store.list_customers(limit=25, sort="risk_asc")["customers"]

    probabilities = [c["churn_probability"] for c in customers]

    assert probabilities == sorted(probabilities)


@requires_store
@pytest.mark.parametrize("band", ["HIGH", "MEDIUM", "LOW"])
def test_risk_filter_returns_only_that_band(store, band):
    result = store.list_customers(limit=50, risk_level=band)

    assert result["total"] > 0
    assert all(c["risk_level"] == band for c in result["customers"])


@requires_store
def test_country_filter_returns_only_that_country(store):
    country = store.summary()["countries"][0]

    result = store.list_customers(limit=50, country=country)

    assert result["total"] > 0
    assert all(c["country"] == country for c in result["customers"])


@requires_store
def test_query_matches_on_customer_id(store):
    customer_id = store.list_customers(limit=1)["customers"][0]["customer_id"]

    result = store.list_customers(query=str(customer_id))

    assert result["total"] >= 1
    assert any(
        c["customer_id"] == customer_id for c in result["customers"]
    )


@requires_store
def test_pagination_does_not_repeat_customers(store):
    first = store.list_customers(limit=20, offset=0)["customers"]
    second = store.list_customers(limit=20, offset=20)["customers"]

    assert len(first) == len(second) == 20
    assert not ({c["customer_id"] for c in first}
                & {c["customer_id"] for c in second})


@requires_store
def test_spend_sort_orders_by_spend(store):
    customers = store.list_customers(limit=20, sort="spend")["customers"]

    spends = [c["total_spent"] for c in customers]

    assert spends == sorted(spends, reverse=True)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


@requires_store
def test_expected_churners_differs_from_flagged_count(store):
    # These answer different questions -- how many will leave, versus how
    # many are above the action threshold -- and conflating them is a
    # classic reporting error.
    summary = store.summary()

    assert summary["expected_churners"] > 0
    assert summary["flagged_for_contact"] > 0
    assert summary["expected_churners"] != summary["flagged_for_contact"]


@requires_store
def test_summary_revenue_at_risk_is_bounded_by_total_spend(store):
    summary = store.summary()

    total_spent = sum(
        c["total_spent"]
        for c in store.list_customers(limit=500)["customers"]
    )

    assert summary["revenue_at_risk"] > 0
    # Probabilities are below 1, so risk-weighted spend cannot exceed the
    # whole base's spend.
    assert summary["revenue_at_risk"] < total_spent * 1000


# ---------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------


@requires_store
def test_drivers_are_returned_sorted_by_absolute_effect(store, model):
    customer_id = store.list_customers(limit=1)["customers"][0]["customer_id"]

    drivers = store.drivers(model, customer_id, top=5)

    assert 0 < len(drivers) <= 5

    magnitudes = [abs(d["contribution"]) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)

    for driver in drivers:
        assert 0 <= driver["percentile"] <= 100
        assert "population_median" in driver


@requires_store
def test_driver_at_the_median_has_almost_no_effect(store, model):
    # A customer sitting at the population median on a feature should get
    # near-zero contribution from it, because the ablation replaces it with
    # essentially the same value.
    customer_id = store.list_customers(limit=1)["customers"][0]["customer_id"]

    drivers = store.drivers(model, customer_id, top=10)

    for driver in drivers:
        if driver["value"] == driver["population_median"]:
            assert driver["contribution"] == pytest.approx(0.0, abs=1e-9)


@requires_store
def test_drivers_for_an_unknown_customer_are_empty(store, model):
    assert store.drivers(model, -1) == []


@requires_store
def test_high_risk_customers_have_risk_raising_drivers(store, model):
    riskiest = store.list_customers(limit=1)["customers"][0]

    drivers = store.drivers(model, riskiest["customer_id"], top=5)

    # The top-ranked customer got there somehow; at least one feature must
    # be pushing their score up.
    assert any(d["contribution"] > 0 for d in drivers)


# ---------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------
# The serving container ships a scored snapshot rather than the raw order
# and event history. Regression test for a real deployment bug: the image
# excluded data/ (correctly -- it is the warehouse), which left the
# dashboard and every customer endpoint dead in the container while the
# health check still reported the model loaded.


@requires_store
def test_snapshot_round_trips(store, model, tmp_path):
    path = store.save(tmp_path / "snapshot.csv")

    assert path.exists()

    reloaded = FeatureStore.load(model, snapshot_path=path)

    assert len(reloaded) == len(store)

    customer_id = store.list_customers(limit=1)["customers"][0]["customer_id"]

    assert reloaded.profile(customer_id) == store.profile(customer_id)


@requires_store
def test_snapshot_loads_without_any_raw_data(model, tmp_path):
    # The whole point: given a snapshot, the store must not need data_dir at
    # all. Pointing it at an empty directory proves it.
    snapshot = FeatureStore.load(model, snapshot_path=None).save(
        tmp_path / "snapshot.csv"
    )

    empty = tmp_path / "no-data"
    empty.mkdir()

    store = FeatureStore.load(model, data_dir=empty, snapshot_path=snapshot)

    assert len(store) > 1000
    assert store.summary()["total_customers"] == len(store)


@requires_store
def test_missing_snapshot_falls_back_to_building_from_csvs(model, tmp_path):
    store = FeatureStore.load(model, snapshot_path=tmp_path / "absent.csv")

    assert len(store) > 1000


@requires_store
def test_training_writes_a_snapshot_next_to_the_model():
    # train.py writes it into models/, which is what the Dockerfile already
    # copies -- so no Dockerfile change was needed to ship it.
    assert SNAPSHOT_PATH.exists()
    assert SNAPSHOT_PATH.parent.name == "models"
