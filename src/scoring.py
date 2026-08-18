"""Shared scoring contract.

Single source of truth for where the model lives, which columns it expects,
how raw inputs are enriched, and how a probability becomes a decision. The
training script and the HTTP service both go through this module so the two
can never drift apart.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"


# ------------------------------------------------------------------
# Feature contract
# ------------------------------------------------------------------
# These 13 columns are what a caller supplies. They are the API request
# schema and the raw output of build_features().

NUMERIC_FEATURES = [
    "age",
    "total_orders",
    "total_spent",
    "days_since_last_order",
    "has_previous_order",
    "total_events",
    "add_to_cart_count",
    "checkout_count",
    "login_count",
    "product_view_count",
    "tenure_days",
    "events_last_30_days",
]

CATEGORICAL_FEATURES = [
    "country",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Derived features are computed inside the pipeline, not requested from the
# caller: they are deterministic functions of the columns above, so asking a
# client to supply them would only create an opportunity to disagree with
# training. Adding them here automatically adds them at inference time.
DERIVED_FEATURES = [
    "orders_per_day",
    "events_per_day",
    "spend_per_order",
    "checkout_rate",
    "cart_rate",
    "recency_ratio",
]

MODEL_NUMERIC_FEATURES = NUMERIC_FEATURES + DERIVED_FEATURES


def add_rate_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add exposure-normalised rate features.

    Raw counts conflate how active a customer is with how long they have
    been around. A customer with 200 days of history and 10 orders is not
    the same as one with 40 days of history and 10 orders, but
    ``total_orders`` cannot tell them apart. Dividing by tenure separates
    rate from exposure.

    This is a module-level function on purpose: it is referenced by the
    pickled pipeline, so it has to be importable when the model is loaded.
    """

    X = X.copy()

    tenure = X["tenure_days"].clip(lower=1)
    events = X["total_events"].clip(lower=1)
    orders = X["total_orders"].clip(lower=1)

    X["orders_per_day"] = X["total_orders"] / tenure
    X["events_per_day"] = X["total_events"] / tenure
    X["spend_per_order"] = np.where(
        X["total_orders"] > 0, X["total_spent"] / orders, 0.0
    )
    X["checkout_rate"] = X["checkout_count"] / events
    X["cart_rate"] = X["add_to_cart_count"] / events
    X["recency_ratio"] = X["days_since_last_order"] / tenure

    return X


# ------------------------------------------------------------------
# Decision thresholds
# ------------------------------------------------------------------

# Used when no tuned threshold has been persisted. train.py selects an
# operating threshold from the cost model and writes it to the model card;
# this is only the fallback.
DEFAULT_CHURN_THRESHOLD = 0.50


class ModelNotAvailable(RuntimeError):
    """Raised when the serialised model cannot be found or loaded."""


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the trained pipeline.

    Raises ``ModelNotAvailable`` with an actionable message rather than a
    bare traceback, because the artifact is not tracked in git and a fresh
    clone will not have one until ``src/train.py`` has been run.
    """

    if not model_path.exists():
        raise ModelNotAvailable(
            f"No model artifact at {model_path}. "
            "Generate data and train first:\n"
            "    python src/generate_data.py\n"
            "    python src/train.py"
        )

    try:
        return joblib.load(model_path)
    except Exception as error:
        raise ModelNotAvailable(
            f"Could not load the model at {model_path}: {error}"
        ) from error


def load_model_card(card_path: Path = MODEL_CARD_PATH) -> dict | None:
    """Load the model card written by train.py, or None if absent."""

    if not card_path.exists():
        return None

    try:
        return json.loads(card_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def churn_threshold(card: dict | None = None) -> float:
    """
    The operating threshold: the cost-optimal cutoff chosen at training
    time, falling back to 0.50 when no model card is available.
    """

    if card is None:
        card = load_model_card()

    if not card:
        return DEFAULT_CHURN_THRESHOLD

    value = card.get("operating_threshold")

    if not isinstance(value, (int, float)):
        return DEFAULT_CHURN_THRESHOLD

    return float(value)


def risk_level(probability: float, threshold: float | None = None) -> str:
    """
    Map a churn probability onto a risk band.

    The bands are defined relative to the operating threshold, not at fixed
    cut points: LOW is everything below it, and the actionable region above
    it is split in half into MEDIUM and HIGH. Anchoring to the threshold
    means the band can never contradict ``predicted_churn``, and all three
    bands stay reachable whatever the cost model chooses. (Fixed cut points
    at 0.50/0.75 left MEDIUM empty once the tuned threshold rose above
    0.75.)

    At the default threshold of 0.50 this reduces to the familiar
    LOW < 0.50 <= MEDIUM < 0.75 <= HIGH.
    """

    if threshold is None:
        threshold = churn_threshold()

    if probability < threshold:
        return "LOW"

    high_cutoff = threshold + (1.0 - threshold) / 2.0

    return "HIGH" if probability >= high_cutoff else "MEDIUM"


def _frame(customers: list[dict]) -> pd.DataFrame:
    for customer in customers:
        missing = [c for c in FEATURE_COLUMNS if c not in customer]

        if missing:
            raise ValueError(
                f"Missing required feature(s): {', '.join(missing)}"
            )

    return pd.DataFrame(
        [{c: customer[c] for c in FEATURE_COLUMNS} for customer in customers]
    )


def score_batch(
    model,
    customers: list[dict],
    threshold: float | None = None,
) -> list[dict]:
    """Score many customers in one pass."""

    if not customers:
        return []

    if threshold is None:
        threshold = churn_threshold()

    probabilities = model.predict_proba(_frame(customers))[:, 1]

    return [
        {
            "churn_probability": round(float(p), 4),
            "predicted_churn": int(p >= threshold),
            "risk_level": risk_level(float(p), threshold),
        }
        for p in probabilities
    ]


def score(model, customer: dict, threshold: float | None = None) -> dict:
    """Score a single customer."""

    return score_batch(model, [customer], threshold)[0]
