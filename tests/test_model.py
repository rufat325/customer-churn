import pandas as pd
import pytest

from src.scoring import (
    DEFAULT_CHURN_THRESHOLD,
    DERIVED_FEATURES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    ModelNotAvailable,
    add_rate_features,
    churn_threshold,
    load_model,
    risk_level,
    score,
    score_batch,
)


# The artifact is not tracked in git, so on a fresh clone these tests would
# fail for a reason that has nothing to do with the code. Skip with an
# actionable message instead.
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"No model at {MODEL_PATH}. Run: python src/train.py",
)


def make_test_customer():
    return {
        "age": 35,
        "country": "DE",
        "total_orders": 2,
        "total_spent": 150.00,
        "days_since_last_order": 75,
        "has_previous_order": 1,
        "total_events": 5,
        "add_to_cart_count": 1,
        "checkout_count": 0,
        "login_count": 2,
        "product_view_count": 2,
        "tenure_days": 180,
        "events_last_30_days": 0,
    }


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"No model at {MODEL_PATH}. Run: python src/train.py")

    return load_model()


@requires_model
def test_model_artifact_exists():
    assert MODEL_PATH.exists()


@requires_model
def test_model_can_load(model):
    assert model is not None


@requires_model
def test_model_produces_valid_probability(model):
    probability = model.predict_proba(
        pd.DataFrame([make_test_customer()])
    )[0, 1]

    assert 0.0 <= probability <= 1.0


@requires_model
def test_model_accepts_only_the_documented_input_columns(model):
    # The pipeline derives its rate features internally, so a caller must
    # never have to supply them.
    frame = pd.DataFrame([make_test_customer()])[FEATURE_COLUMNS]

    assert model.predict_proba(frame).shape == (1, 2)


@requires_model
def test_score_returns_consistent_result(model):
    result = score(model, make_test_customer())

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["predicted_churn"] in (0, 1)
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    threshold = churn_threshold()

    assert result["predicted_churn"] == int(
        result["churn_probability"] >= threshold
    )
    assert result["risk_level"] == risk_level(
        result["churn_probability"], threshold
    )


@requires_model
def test_score_batch_matches_single_scoring(model):
    customers = [make_test_customer() for _ in range(3)]
    customers[1]["total_orders"] = 40
    customers[2]["days_since_last_order"] = 400

    batch = score_batch(model, customers)

    assert len(batch) == 3

    for customer, batched in zip(customers, batch):
        assert batched == score(model, customer)


@requires_model
def test_score_batch_handles_empty_input(model):
    assert score_batch(model, []) == []


@requires_model
def test_score_rejects_missing_features(model):
    incomplete = make_test_customer()
    del incomplete["tenure_days"]

    with pytest.raises(ValueError, match="tenure_days"):
        score(model, incomplete)


def test_load_model_raises_actionable_error(tmp_path):
    with pytest.raises(ModelNotAvailable, match="train.py"):
        load_model(tmp_path / "does_not_exist.joblib")


# ---------------------------------------------------------------------
# Risk banding
# ---------------------------------------------------------------------


def test_risk_bands_at_default_threshold():
    t = DEFAULT_CHURN_THRESHOLD

    assert risk_level(0.90, t) == "HIGH"
    assert risk_level(0.75, t) == "HIGH"
    assert risk_level(0.60, t) == "MEDIUM"
    assert risk_level(0.50, t) == "MEDIUM"
    assert risk_level(0.10, t) == "LOW"
    assert risk_level(0.00, t) == "LOW"


def test_all_risk_bands_reachable_at_any_threshold():
    # Regression test: with fixed cut points at 0.50/0.75, a tuned threshold
    # above 0.75 made MEDIUM unreachable.
    for threshold in (0.10, 0.50, 0.83, 0.95):
        bands = {
            risk_level(p / 100, threshold) for p in range(0, 101)
        }

        assert bands == {"LOW", "MEDIUM", "HIGH"}, (
            f"threshold {threshold} produced only {bands}"
        )


def test_risk_band_never_contradicts_prediction():
    for threshold in (0.10, 0.50, 0.83, 0.95):
        for p in [i / 100 for i in range(101)]:
            predicted = int(p >= threshold)
            band = risk_level(p, threshold)

            if predicted:
                assert band in ("MEDIUM", "HIGH")
            else:
                assert band == "LOW"


def test_churn_threshold_falls_back_without_card():
    assert churn_threshold({}) == DEFAULT_CHURN_THRESHOLD
    assert churn_threshold({"operating_threshold": "not-a-number"}) == (
        DEFAULT_CHURN_THRESHOLD
    )
    assert churn_threshold({"operating_threshold": 0.83}) == 0.83


# ---------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------


def test_feature_columns_are_unique():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert not set(FEATURE_COLUMNS) & set(DERIVED_FEATURES)


def test_add_rate_features_adds_every_derived_column():
    result = add_rate_features(pd.DataFrame([make_test_customer()]))

    for column in DERIVED_FEATURES:
        assert column in result.columns

    assert result.loc[0, "orders_per_day"] == 2 / 180
    assert result.loc[0, "spend_per_order"] == 75.0


def test_add_rate_features_survives_zero_denominators():
    empty_customer = {**make_test_customer()}
    empty_customer.update(
        total_orders=0, total_spent=0, total_events=0,
        checkout_count=0, add_to_cart_count=0, tenure_days=0,
    )

    result = add_rate_features(pd.DataFrame([empty_customer]))

    assert result[DERIVED_FEATURES].notna().all().all()
    assert result.loc[0, "spend_per_order"] == 0.0


def test_add_rate_features_does_not_mutate_input():
    original = pd.DataFrame([make_test_customer()])
    before = list(original.columns)

    add_rate_features(original)

    assert list(original.columns) == before
