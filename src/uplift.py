"""Uplift modelling: who should we actually contact?

A churn model answers "who is likely to leave". That is not the question a
retention campaign asks. The campaign should go to customers whose behaviour
the contact *changes*, and those are not the same people:

- **Sure things** will stay whether contacted or not. Contacting them costs
  money and buys nothing.
- **Lost causes** will leave whichever happens. Same problem.
- **Persuadables** stay only if contacted. These are the entire return.
- **Sleeping dogs** are annoyed by contact and leave *because* of it. Every
  one of these targeted is worse than doing nothing.

Ranking by churn probability finds the customers at risk. Ranking by uplift
finds the persuadables. When risk and persuadability are uncorrelated -- as
they are in this dataset by construction -- the churn model is close to
useless as a targeting rule, and this module shows that with numbers.

Estimating uplift needs randomised data, because it is a causal quantity: we
have to observe comparable customers under both arms. ``data/campaign.csv``
supplies that.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone


def fit_t_learner(estimator, X, treated, y):
    """
    Fit one outcome model per arm (the "T-learner").

    Predicted uplift is the difference of their predictions. Simple, and it
    makes the causal assumption explicit: each model sees only its own arm,
    so neither can confuse "was treated" with "was going to respond".
    """

    treated = np.asarray(treated).astype(bool)

    control_model = clone(estimator).fit(X[~treated], np.asarray(y)[~treated])
    treated_model = clone(estimator).fit(X[treated], np.asarray(y)[treated])

    return control_model, treated_model


def predict_uplift(models, X) -> np.ndarray:
    """Predicted change in response probability from treating."""

    control_model, treated_model = models

    return (
        treated_model.predict_proba(X)[:, 1]
        - control_model.predict_proba(X)[:, 1]
    )


def uplift_by_decile(y, treated, scores, bins: int = 10) -> pd.DataFrame:
    """
    Observed uplift within each decile of a targeting score.

    A useful sanity check: for a score that genuinely ranks uplift, the
    observed treated-minus-control difference should fall monotonically from
    the top decile down, and may go negative at the bottom.
    """

    frame = pd.DataFrame({
        "y": np.asarray(y),
        "treated": np.asarray(treated).astype(int),
        "score": np.asarray(scores),
    })

    frame["decile"] = pd.qcut(
        frame["score"].rank(method="first", ascending=False),
        bins,
        labels=range(1, bins + 1),
    )

    rows = []

    for decile, group in frame.groupby("decile", observed=True):
        t = group[group["treated"] == 1]["y"]
        c = group[group["treated"] == 0]["y"]

        rows.append({
            "decile": int(decile),
            "n": len(group),
            "treated_rate": t.mean() if len(t) else np.nan,
            "control_rate": c.mean() if len(c) else np.nan,
            "observed_uplift": (
                t.mean() - c.mean() if len(t) and len(c) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def qini_curve(y, treated, scores, points: int = 100):
    """
    Incremental responders gained by targeting the top fraction, by score.

    At each depth the treated and control responder counts are compared,
    rescaling control to the treated group size so the arms are comparable:

        incremental = responders_t - responders_c * (n_t / n_c)

    Returns ``(fractions, incremental)``.
    """

    y = np.asarray(y)
    treated = np.asarray(treated).astype(int)
    scores = np.asarray(scores)

    order = np.argsort(-scores)
    y, treated = y[order], treated[order]

    cum_treated = np.cumsum(treated)
    cum_control = np.cumsum(1 - treated)
    cum_y_treated = np.cumsum(y * treated)
    cum_y_control = np.cumsum(y * (1 - treated))

    n = len(y)
    cuts = np.unique(
        np.linspace(1, n, points).astype(int)
    ) - 1

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_control = np.where(
            cum_control[cuts] > 0,
            cum_y_control[cuts] * cum_treated[cuts] / cum_control[cuts],
            0.0,
        )

    incremental = cum_y_treated[cuts] - scaled_control

    return (cuts + 1) / n, incremental


def qini_score(y, treated, scores, points: int = 100) -> float:
    """
    Area between a strategy's Qini curve and random targeting.

    Random targeting is the straight line from the origin to the overall
    incremental gain, so this measures how much better than "contact people
    in arbitrary order" a ranking is. Higher is better; zero means no better
    than random.
    """

    fractions, incremental = qini_curve(y, treated, scores, points)

    random_line = fractions * incremental[-1]

    return float(np.trapezoid(incremental - random_line, fractions))
