"""A minimal feature store.

Why this exists
---------------
``POST /predict`` asks the caller for thirteen pre-computed features. That is
the wrong shape for a serving API and it is how training/serving skew happens:
the caller has to reimplement ``build_features`` exactly -- the same windows,
the same imputation, the same definitions -- and any drift between their
version and ours silently corrupts every prediction. Nothing errors. The
numbers are just quietly wrong.

Production systems solve this by keying features on an entity id. Features are
computed once, by the same code that produced the training set, and the serving
call passes ``customer_id`` rather than thirteen numbers it might compute
differently. That is what this module provides.

It is deliberately simple: a pandas frame held in memory, built at startup from
the same ``build_features`` used for training. A real deployment would put
Feast, Tecton, or a warehouse table here. The interface is what matters -- look
up by id, never recompute at the edge.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import PREDICTION_DATE, build_features
from src.scoring import FEATURE_COLUMNS, churn_threshold, risk_level, score_batch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# A scored snapshot, written by train.py and shipped in the image alongside
# the model. The serving container should not carry the raw order and event
# history: it is the warehouse, it is far larger than the features derived
# from it, and nothing at serve time needs it. Shipping the snapshot instead
# also means startup is a file read rather than a full feature build.
SNAPSHOT_PATH = PROJECT_ROOT / "models" / "feature_snapshot.csv"

SNAPSHOT_DATE_COLUMNS = ["signup_date", "last_order_date"]

# Columns describing who the customer is, as opposed to model inputs.
PROFILE_COLUMNS = ["customer_id", "age", "country", "signup_date"]

# Features shown as risk drivers, most globally important first. Taken from
# the permutation importances in the model card.
DRIVER_FEATURES = [
    "total_orders",
    "tenure_days",
    "total_events",
    "days_since_last_order",
    "events_last_30_days",
    "total_spent",
]


class StoreNotAvailable(RuntimeError):
    """Raised when the underlying datasets are missing."""


@dataclass
class Driver:
    feature: str
    value: float
    percentile: float
    contribution: float


class FeatureStore:
    """Customer features and scores, keyed by customer_id."""

    def __init__(self, frame: pd.DataFrame, threshold: float):
        self._frame = frame
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model,
        data_dir: Path = DATA_DIR,
        snapshot_path: Path | None = SNAPSHOT_PATH,
    ) -> "FeatureStore":
        """
        Load the scored customer base.

        Prefers a snapshot written by train.py; falls back to building from
        the raw CSVs, which is what happens in local development and what
        produces the snapshot in the first place.

        Either way the whole base is scored up front, because the useful
        question is "who are my 50 riskiest customers" and that cannot be
        answered one lookup at a time.
        """

        if snapshot_path is not None and snapshot_path.exists():
            frame = pd.read_csv(snapshot_path)

            for column in SNAPSHOT_DATE_COLUMNS:
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column])

            return cls(frame, churn_threshold())

        paths = {
            name: data_dir / f"{name}.csv"
            for name in ("customers", "orders", "website_events")
        }

        missing = [str(p) for p in paths.values() if not p.exists()]

        if missing:
            raise StoreNotAvailable(
                "Missing data files: "
                + ", ".join(missing)
                + "\nGenerate them first:\n    python src/generate_data.py"
            )

        frame = build_features(
            customers=pd.read_csv(paths["customers"]),
            orders=pd.read_csv(paths["orders"]),
            website_events=pd.read_csv(paths["website_events"]),
        )

        threshold = churn_threshold()

        scored = score_batch(
            model,
            frame[FEATURE_COLUMNS].to_dict("records"),
            threshold,
        )

        frame = frame.reset_index(drop=True)
        frame["churn_probability"] = [s["churn_probability"] for s in scored]
        frame["predicted_churn"] = [s["predicted_churn"] for s in scored]
        frame["risk_level"] = [s["risk_level"] for s in scored]

        frame["risk_rank"] = (
            frame["churn_probability"].rank(ascending=False, method="first")
            .astype(int)
        )

        return cls(frame, threshold)

    def save(self, snapshot_path: Path = SNAPSHOT_PATH) -> Path:
        """Write the scored frame so a serving container can skip the build."""

        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self._frame.to_csv(snapshot_path, index=False)

        return snapshot_path

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def prediction_date(self) -> str:
        return str(PREDICTION_DATE.date())

    def exists(self, customer_id: int) -> bool:
        return bool((self._frame["customer_id"] == customer_id).any())

    def _row(self, customer_id: int) -> pd.Series | None:
        match = self._frame[self._frame["customer_id"] == customer_id]

        return None if match.empty else match.iloc[0]

    def profile(self, customer_id: int) -> dict | None:
        """Everything the UI needs about one customer."""

        row = self._row(customer_id)

        if row is None:
            return None

        last_order = row.get("last_order_date")

        return {
            "customer_id": int(row["customer_id"]),
            "age": None if pd.isna(row["age"]) else float(row["age"]),
            "country": row["country"],
            "signup_date": str(pd.Timestamp(row["signup_date"]).date()),
            "last_order_date": (
                None if pd.isna(last_order)
                else str(pd.Timestamp(last_order).date())
            ),
            "churn_probability": float(row["churn_probability"]),
            "predicted_churn": int(row["predicted_churn"]),
            "risk_level": row["risk_level"],
            "risk_rank": int(row["risk_rank"]),
            "features": {
                column: (
                    None if pd.isna(row[column]) else float(row[column])
                ) if column != "country" else row[column]
                for column in FEATURE_COLUMNS
            },
        }

    def features_for(self, customer_id: int) -> dict | None:
        row = self._row(customer_id)

        if row is None:
            return None

        return {column: row[column] for column in FEATURE_COLUMNS}

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_customers(
        self,
        limit: int = 50,
        offset: int = 0,
        risk_level: str | None = None,
        country: str | None = None,
        query: str | None = None,
        sort: str = "risk",
    ) -> dict:
        """Filtered, sorted page of customers with their scores."""

        frame = self._frame

        if risk_level:
            frame = frame[frame["risk_level"] == risk_level.upper()]

        if country:
            frame = frame[frame["country"] == country.upper()]

        if query:
            text = str(query).strip()
            frame = frame[
                frame["customer_id"].astype(str).str.contains(text, na=False)
            ]

        ascending = sort == "risk_asc"
        column = {
            "risk": "churn_probability",
            "risk_asc": "churn_probability",
            "spend": "total_spent",
            "orders": "total_orders",
            "recency": "days_since_last_order",
        }.get(sort, "churn_probability")

        frame = frame.sort_values(column, ascending=ascending, kind="stable")

        total = len(frame)
        page = frame.iloc[offset:offset + limit]

        return {
            "total": int(total),
            "limit": int(limit),
            "offset": int(offset),
            "customers": [
                {
                    "customer_id": int(r["customer_id"]),
                    "country": r["country"],
                    "churn_probability": float(r["churn_probability"]),
                    "risk_level": r["risk_level"],
                    "risk_rank": int(r["risk_rank"]),
                    "total_orders": float(r["total_orders"]),
                    "total_spent": float(r["total_spent"]),
                    "days_since_last_order": float(r["days_since_last_order"]),
                }
                for _, r in page.iterrows()
            ],
        }

    # ------------------------------------------------------------------
    # Portfolio view
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        frame = self._frame

        counts = frame["risk_level"].value_counts().to_dict()

        return {
            "prediction_date": self.prediction_date,
            "operating_threshold": self.threshold,
            "total_customers": int(len(frame)),
            "risk_counts": {
                band: int(counts.get(band, 0))
                for band in ("HIGH", "MEDIUM", "LOW")
            },
            "flagged_for_contact": int(frame["predicted_churn"].sum()),
            # Expected churners is the sum of calibrated probabilities, not
            # the count above the threshold. Those answer different
            # questions: how many will leave, versus how many to act on.
            "expected_churners": round(
                float(frame["churn_probability"].sum()), 1
            ),
            "mean_churn_probability": round(
                float(frame["churn_probability"].mean()), 4
            ),
            "revenue_at_risk": round(
                float(
                    (frame["churn_probability"] * frame["total_spent"]).sum()
                ),
                2,
            ),
            "countries": sorted(frame["country"].dropna().unique().tolist()),
        }

    # ------------------------------------------------------------------
    # Local explanation
    # ------------------------------------------------------------------

    def drivers(self, model, customer_id: int, top: int = 5) -> list[dict]:
        """
        Why this customer scores as they do.

        One-at-a-time ablation: re-score the customer with a single feature
        replaced by the population median, and report how much the
        probability moves. Positive means the real value pushes risk up.

        This is not SHAP. It ignores interactions and does not sum to the
        prediction, and with correlated features it will over-credit each of
        a correlated pair. It is cheap, it is easy to explain to someone who
        will act on it, and it is directionally honest -- which is the right
        trade for a worklist. Anything used to justify a decision to a
        customer would need the real thing.
        """

        base_features = self.features_for(customer_id)

        if base_features is None:
            return []

        baseline = score_batch(
            model, [base_features], self.threshold
        )[0]["churn_probability"]

        variants, names = [], []

        for feature in DRIVER_FEATURES:
            if feature not in base_features:
                continue

            counterfactual = dict(base_features)
            counterfactual[feature] = float(self._frame[feature].median())

            variants.append(counterfactual)
            names.append(feature)

        if not variants:
            return []

        scored = score_batch(model, variants, self.threshold)

        results = []

        for feature, result in zip(names, scored):
            value = base_features[feature]

            results.append({
                "feature": feature,
                "value": float(value),
                "population_median": float(self._frame[feature].median()),
                "percentile": round(
                    float(
                        (self._frame[feature] < value).mean() * 100
                    ),
                    1,
                ),
                "contribution": round(
                    baseline - result["churn_probability"], 4
                ),
            })

        results.sort(key=lambda r: abs(r["contribution"]), reverse=True)

        return results[:top]
