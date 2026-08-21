"""Uncertainty quantification for model metrics.

A point estimate from a 624-row test split is not a precise number. Reporting
"ROC-AUC 0.7231" implies four significant figures of confidence that a sample
that size cannot support; the honest version is 0.72 with an interval attached.

Two functions here, and the distinction between them matters:

``bootstrap_metric`` gives a confidence interval for one model's score.

``paired_bootstrap`` compares two models. It resamples rows once and scores
both models on the same resample, which is the right test when both were
evaluated on identical data. Comparing two independent intervals and checking
whether they overlap is a different and weaker question, and it routinely
misses real differences: two models can have heavily overlapping intervals
while one beats the other on essentially every resample, because their errors
are correlated. Overlap is not a significance test.
"""

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_RESAMPLES = 5000
DEFAULT_ALPHA = 0.05


def _resample_indices(
    n: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.integers(0, n, size=(resamples, n))


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> dict:
    """
    Percentile bootstrap confidence interval for a single metric.

    Resamples that end up with only one class present are skipped, because
    ranking metrics are undefined there.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    rng = np.random.default_rng(seed)
    draws = _resample_indices(len(y_true), resamples, rng)

    values = []

    for index in draws:
        sample = y_true[index]

        if sample.min() == sample.max():
            continue

        values.append(metric(sample, y_score[index]))

    values = np.asarray(values)

    lower, upper = np.percentile(
        values, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    return {
        "estimate": float(metric(y_true, y_score)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "std_error": float(values.std()),
        "resamples": int(len(values)),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> dict:
    """
    Confidence interval for ``metric(b) - metric(a)`` on identical rows.

    ``win_rate`` is the fraction of resamples in which b beats a, which is
    often the more intuitive summary of the two.
    """

    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)

    rng = np.random.default_rng(seed)
    draws = _resample_indices(len(y_true), resamples, rng)

    differences = []

    for index in draws:
        sample = y_true[index]

        if sample.min() == sample.max():
            continue

        differences.append(
            metric(sample, y_score_b[index]) - metric(sample, y_score_a[index])
        )

    differences = np.asarray(differences)

    lower, upper = np.percentile(
        differences, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    return {
        "difference": float(
            metric(y_true, y_score_b) - metric(y_true, y_score_a)
        ),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "win_rate": float((differences > 0).mean()),
        "significant": bool(lower > 0 or upper < 0),
        "resamples": int(len(differences)),
    }


def format_ci(result: dict, digits: int = 3) -> str:
    """Render a bootstrap result as `estimate [lower, upper]`."""

    key = "estimate" if "estimate" in result else "difference"

    return (
        f"{result[key]:.{digits}f} "
        f"[{result['ci_lower']:.{digits}f}, {result['ci_upper']:.{digits}f}]"
    )
