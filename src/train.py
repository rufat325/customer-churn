"""Reproducible training pipeline.

Run from the project root:

    python src/train.py

Steps: load data, build features, split, tune a gradient boosting classifier
by cross-validated grid search, calibrate its probabilities, choose an
operating threshold from an explicit cost model, evaluate against a
majority-class baseline, and save the fitted pipeline plus a model card.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import bootstrap_metric, format_ci, paired_bootstrap
from src.features import PREDICTION_DATE, TARGET_WINDOW_DAYS, build_features
from src.scoring import (
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    FEATURE_COLUMNS,
    MODEL_CARD_PATH,
    MODEL_NUMERIC_FEATURES,
    MODEL_PATH,
    add_rate_features,
    risk_level,
)


DATA_DIR = PROJECT_ROOT / "data"

CUSTOMERS_PATH = DATA_DIR / "customers.csv"
ORDERS_PATH = DATA_DIR / "orders.csv"
EVENTS_PATH = DATA_DIR / "website_events.csv"

RANDOM_STATE = 42

NON_FEATURE_COLUMNS = [
    "customer_id",
    "signup_date",
    "last_order_date",
    "future_orders",
    "churn",
]

PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3],
    "model__min_samples_leaf": [5, 10],
}


# ------------------------------------------------------------------
# Cost model
# ------------------------------------------------------------------
# An operating threshold should come from what a decision is worth, not
# from the 0.50 default.
#
#   contact a customer  -> costs offer_cost
#   they were churning
#   and accept          -> earns margin (with probability accept_rate)
#
# So a true positive is worth margin * accept_rate - offer_cost, a false
# positive costs offer_cost, and doing nothing is worth zero either way.
#
# The optimal threshold depends entirely on these numbers, and the honest
# way to show that is to sweep them rather than to quote one figure. With a
# cheap offer, contacting the entire base is already profitable at a 68%
# churn rate and a model barely helps. As the offer gets expensive, blanket
# contact turns loss-making and targeting is the only thing that works.

class Scenario(NamedTuple):
    name: str
    margin: float
    accept_rate: float
    offer_cost: float

    @property
    def value_true_positive(self) -> float:
        return self.margin * self.accept_rate - self.offer_cost

    @property
    def value_false_positive(self) -> float:
        return -self.offer_cost


SCENARIOS = [
    Scenario("cheap offer", margin=50.0, accept_rate=0.30, offer_cost=5.0),
    Scenario("standard offer", margin=50.0, accept_rate=0.30, offer_cost=12.0),
    Scenario("premium offer", margin=50.0, accept_rate=0.30, offer_cost=20.0),
]

# The scenario whose threshold is shipped in the model card.
DEFAULT_SCENARIO = "standard offer"


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, MODEL_NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        # Derived features are computed here rather than demanded from the
        # caller, so the API contract stays at the raw columns and train and
        # serve cannot disagree.
        ("rates", FunctionTransformer(add_rate_features)),
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def evaluate(y_true, y_pred, y_prob=None) -> dict:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # None rather than NaN: these land in the JSON model card, and NaN is not
    # valid JSON (Python will happily emit it, but strict parsers and the API
    # serialiser both reject it).
    result["roc_auc"] = (
        roc_auc_score(y_true, y_prob) if y_prob is not None else None
    )
    result["brier"] = (
        brier_score_loss(y_true, y_prob) if y_prob is not None else None
    )

    return result


METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier"]


def print_metrics_table(rows: dict) -> None:
    header = f"{'':<26}" + "".join(f"{m:>11}" for m in METRIC_ORDER)
    print(header)
    print("-" * len(header))

    for name, values in rows.items():
        line = f"{name:<26}"
        for metric in METRIC_ORDER:
            value = values.get(metric)
            line += "        n/a" if value is None else f"{value:>11.4f}"
        print(line)


def expected_value(y_true, y_prob, threshold: float, scenario: Scenario) -> float:
    """Total expected value of acting on every customer above `threshold`."""

    contacted = y_prob >= threshold

    true_positives = int(np.sum(contacted & (y_true == 1)))
    false_positives = int(np.sum(contacted & (y_true == 0)))

    return (
        true_positives * scenario.value_true_positive
        + false_positives * scenario.value_false_positive
    )


def choose_threshold(y_true, y_prob, scenario: Scenario) -> tuple[float, float]:
    """Pick the threshold maximising expected value under `scenario`."""

    candidates = np.round(np.arange(0.01, 1.00, 0.01), 2)

    values = [expected_value(y_true, y_prob, t, scenario) for t in candidates]
    best = int(np.argmax(values))

    return float(candidates[best]), float(values[best])


def reliability_table(y_true, y_prob, bins: int = 5) -> list[dict]:
    """Predicted vs observed churn rate, by probability bin."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)

    rows = []

    for b in range(bins):
        mask = index == b

        if not mask.any():
            continue

        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(mask.sum()),
            "mean_predicted": float(np.mean(y_prob[mask])),
            "observed": float(np.mean(y_true[mask])),
        })

    return rows


def main() -> int:
    missing = [
        path
        for path in (CUSTOMERS_PATH, ORDERS_PATH, EVENTS_PATH)
        if not path.exists()
    ]

    if missing:
        print(
            "ERROR: missing input data: "
            + ", ".join(str(p) for p in missing)
            + "\nGenerate it first:\n    python src/generate_data.py",
            file=sys.stderr,
        )
        return 1

    print("Loading data...")

    customers = pd.read_csv(CUSTOMERS_PATH)
    orders = pd.read_csv(ORDERS_PATH)
    website_events = pd.read_csv(EVENTS_PATH)

    print("Building features...")

    features = build_features(
        customers=customers,
        orders=orders,
        website_events=website_events,
    )

    print(f"Prediction date:    {PREDICTION_DATE.date()}")
    print(f"Target window:      {TARGET_WINDOW_DAYS} days after prediction date")
    print(f"Eligible customers: {len(features)}")
    print(f"Churn rate:         {features['churn'].mean():.4f}")

    X = features.drop(columns=NON_FEATURE_COLUMNS)
    y = features["churn"]

    leaked = set(X.columns) & set(NON_FEATURE_COLUMNS)
    assert not leaked, f"target-derived columns leaked into X: {leaked}"

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE,
    )

    print()
    print("Dataset split:")
    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    # --------------------------------------------------------------
    # Hyperparameter search
    # --------------------------------------------------------------

    print()
    print("Running hyperparameter search...")

    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print()
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------
    # Gradient boosting optimises a ranking-friendly loss, so its raw scores
    # are not probabilities. Calibration is fitted with internal CV on the
    # training split only, leaving validation clean for threshold choice.

    print()
    print("Calibrating probabilities...")

    uncalibrated = grid_search.best_estimator_
    uncalibrated_val_prob = uncalibrated.predict_proba(X_val)[:, 1]

    calibrators = {}

    for method in ("sigmoid", "isotonic"):
        candidate = CalibratedClassifierCV(
            estimator=clone(uncalibrated),
            method=method,
            cv=5,
        )
        candidate.fit(X_train, y_train)

        val_prob = candidate.predict_proba(X_val)[:, 1]
        calibrators[method] = (
            candidate,
            brier_score_loss(y_val, val_prob),
        )

    print(f"  uncalibrated Brier (validation): "
          f"{brier_score_loss(y_val, uncalibrated_val_prob):.4f}")

    for method, (_, brier) in calibrators.items():
        print(f"  {method:<12} Brier (validation): {brier:.4f}")

    calibration_method = min(calibrators, key=lambda m: calibrators[m][1])
    model = calibrators[calibration_method][0]

    print(f"  selected: {calibration_method}")

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    print()
    print("Reliability on validation (calibrated):")
    print(f"  {'bin':<12}{'n':>6}{'predicted':>12}{'observed':>11}")
    for row in reliability_table(y_val.to_numpy(), val_prob):
        print(f"  {row['bin']:<12}{row['n']:>6}"
              f"{row['mean_predicted']:>12.3f}{row['observed']:>11.3f}")

    # --------------------------------------------------------------
    # Operating threshold
    # --------------------------------------------------------------

    print()
    print("Threshold sensitivity: chosen on validation, valued on test")
    print(f"  {'scenario':<16}{'TP':>7}{'FP':>7}{'thresh':>8}"
          f"{'contact all':>13}{'targeted':>10}{'uplift':>9}")
    print("  " + "-" * 70)

    scenario_rows = []

    for scenario in SCENARIOS:
        chosen, _ = choose_threshold(y_val.to_numpy(), val_prob, scenario)

        all_value = expected_value(y_test.to_numpy(), test_prob, 0.0, scenario)
        targeted = expected_value(y_test.to_numpy(), test_prob, chosen, scenario)

        scenario_rows.append({
            "scenario": scenario.name,
            "value_true_positive": scenario.value_true_positive,
            "value_false_positive": scenario.value_false_positive,
            "threshold": chosen,
            "test_value_contact_all": all_value,
            "test_value_targeted": targeted,
            "test_uplift": targeted - all_value,
        })

        print(f"  {scenario.name:<16}{scenario.value_true_positive:>+7.0f}"
              f"{scenario.value_false_positive:>+7.0f}{chosen:>8.2f}"
              f"{all_value:>13,.0f}{targeted:>10,.0f}"
              f"{targeted - all_value:>+9,.0f}")

    default_scenario = next(s for s in SCENARIOS if s.name == DEFAULT_SCENARIO)
    threshold, _ = choose_threshold(y_val.to_numpy(), val_prob, default_scenario)

    print()
    print(f"  shipped threshold: {threshold:.2f} (from '{DEFAULT_SCENARIO}')")

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    baseline_val = evaluate(y_val, np.ones(len(y_val), dtype=int))
    baseline_test = evaluate(y_test, np.ones(len(y_test), dtype=int))

    model_val = evaluate(y_val, (val_prob >= threshold).astype(int), val_prob)
    model_test = evaluate(y_test, (test_prob >= threshold).astype(int), test_prob)

    print()
    print(f"RESULTS (baseline = always predict churn; "
          f"model thresholded at {threshold:.2f})")
    print()
    print_metrics_table({
        "baseline / validation": baseline_val,
        "model / validation": model_val,
        "baseline / test": baseline_test,
        "model / test": model_test,
    })

    contact_all_test = expected_value(
        y_test.to_numpy(), test_prob, 0.0, default_scenario
    )
    default_test = expected_value(
        y_test.to_numpy(), test_prob, 0.50, default_scenario
    )
    tuned_test = expected_value(
        y_test.to_numpy(), test_prob, threshold, default_scenario
    )

    print()
    print(f"Campaign value on the {len(y_test)}-customer test split "
          f"('{DEFAULT_SCENARIO}'):")
    print(f"  contact everyone           {contact_all_test:>10,.0f}")
    print(f"  threshold 0.50             {default_test:>10,.0f}")
    print(f"  threshold {threshold:.2f} (tuned)     {tuned_test:>10,.0f}")

    # --------------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------------
    # A point estimate off 600-odd rows is not a four-significant-figure
    # number. Report the interval, and compare models with a paired test on
    # identical rows rather than by eyeballing whether two intervals overlap.

    auc_ci = bootstrap_metric(y_test.to_numpy(), test_prob)

    # The obvious business rule: rank by how long since they last ordered.
    # If the model cannot beat this, it is not earning its complexity.
    recency_score = X_test["days_since_last_order"].to_numpy()
    recency_ci = bootstrap_metric(y_test.to_numpy(), recency_score)

    versus_recency = paired_bootstrap(
        y_test.to_numpy(), recency_score, test_prob
    )

    print()
    print("Test ROC-AUC with 95% bootstrap confidence intervals:")
    print(f"  model              {format_ci(auc_ci, 4)}")
    print(f"  recency heuristic  {format_ci(recency_ci, 4)}")
    print(f"  difference         {format_ci(versus_recency, 4)}"
          f"   model wins {versus_recency['win_rate']:.1%} of resamples")
    print("  -> difference is "
          + ("significant" if versus_recency["significant"] else "NOT significant")
          + " at the 5% level")

    print()
    print("Permutation importance (validation, drop in ROC-AUC):")

    importance = permutation_importance(
        model, X_val, y_val,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    ranked = sorted(
        zip(X_val.columns, importance.importances_mean, importance.importances_std),
        key=lambda row: row[1],
        reverse=True,
    )

    importances = []
    for name, mean, std in ranked:
        importances.append({"feature": name, "mean": float(mean), "std": float(std)})
        if mean > 0.0005:
            print(f"  {name:<24}{mean:>8.4f}  +/- {std:.4f}")

    # --------------------------------------------------------------
    # Persist
    # --------------------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    card = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "GradientBoostingClassifier",
        "calibration": calibration_method,
        "best_params": {
            key.replace("model__", ""): value
            for key, value in grid_search.best_params_.items()
        },
        "cv_roc_auc": float(grid_search.best_score_),
        "operating_threshold": threshold,
        "cost_model": {
            "default_scenario": DEFAULT_SCENARIO,
            "margin_per_retained": default_scenario.margin,
            "accept_rate": default_scenario.accept_rate,
            "offer_cost": default_scenario.offer_cost,
            "value_true_positive": default_scenario.value_true_positive,
            "value_false_positive": default_scenario.value_false_positive,
        },
        "threshold_sensitivity": scenario_rows,
        "campaign_value_test": {
            "contact_everyone": contact_all_test,
            "threshold_0.50": default_test,
            "threshold_tuned": tuned_test,
        },
        "uncertainty_test": {
            "roc_auc": auc_ci,
            "recency_heuristic_roc_auc": recency_ci,
            "model_minus_recency": versus_recency,
        },
        "metrics": {
            "validation": model_val,
            "test": model_test,
            "baseline_validation": baseline_val,
            "baseline_test": baseline_test,
        },
        "reliability_validation": reliability_table(y_val.to_numpy(), val_prob),
        "permutation_importance_validation": importances,
        "features": {
            "input": FEATURE_COLUMNS,
            "derived_in_pipeline": DERIVED_FEATURES,
        },
        "data": {
            "prediction_date": str(PREDICTION_DATE.date()),
            "target_window_days": TARGET_WINDOW_DAYS,
            "eligible_customers": int(len(features)),
            "churn_rate": float(features["churn"].mean()),
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_val)),
            "n_test": int(len(X_test)),
        },
    }

    MODEL_CARD_PATH.write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )

    # Score the whole base and ship the result with the model, so the
    # serving container never needs the raw order and event history.
    from src.feature_store import FeatureStore

    snapshot_path = FeatureStore.load(model, snapshot_path=None).save()

    print()
    print(f"Model saved to:      {MODEL_PATH}")
    print(f"Model card saved to: {MODEL_CARD_PATH}")
    print(f"Feature snapshot:    {snapshot_path}")

    example = X_test.iloc[0].to_dict()
    print()
    print(f"Sanity check on one test row -> "
          f"p={test_prob[0]:.4f}, risk={risk_level(float(test_prob[0]), threshold)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
