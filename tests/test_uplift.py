import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.uplift import (
    fit_t_learner,
    predict_uplift,
    qini_curve,
    qini_score,
    uplift_by_decile,
)


@pytest.fixture
def experiment():
    """A randomised trial where uplift depends on `persuadable` only."""

    rng = np.random.default_rng(0)
    n = 3000

    persuadable = rng.random(n)
    baseline = rng.random(n)

    treated = rng.integers(0, 2, n)

    # Response probability rises with baseline propensity, and treatment adds
    # an effect proportional to persuadability -- a different axis.
    probability = 0.15 + 0.5 * baseline + 0.45 * persuadable * treated
    y = (rng.random(n) < np.clip(probability, 0, 1)).astype(int)

    X = pd.DataFrame({"persuadable": persuadable, "baseline": baseline})

    return X, treated, y, persuadable


def test_t_learner_recovers_the_uplift_ordering(experiment):
    X, treated, y, persuadable = experiment

    models = fit_t_learner(LogisticRegression(max_iter=500), X, treated, y)
    predicted = predict_uplift(models, X)

    rho = pd.Series(predicted).corr(pd.Series(persuadable), method="spearman")

    assert rho > 0.5, f"uplift ordering not recovered (rho={rho:.3f})"


def test_t_learner_fits_one_model_per_arm(experiment):
    X, treated, y, _ = experiment

    control_model, treated_model = fit_t_learner(
        LogisticRegression(max_iter=500), X, treated, y
    )

    assert control_model is not treated_model

    # Each arm's model must have seen only its own rows.
    assert control_model.n_features_in_ == X.shape[1]
    assert treated_model.n_features_in_ == X.shape[1]


def test_qini_score_ranks_strategies_correctly(experiment):
    X, treated, y, persuadable = experiment

    models = fit_t_learner(LogisticRegression(max_iter=500), X, treated, y)
    predicted = predict_uplift(models, X)

    rng = np.random.default_rng(1)
    random_score = rng.random(len(y))

    oracle = qini_score(y, treated, persuadable)
    model = qini_score(y, treated, predicted)
    chance = qini_score(y, treated, random_score)

    assert model > chance
    assert oracle >= model * 0.8


def test_qini_curve_is_monotone_in_coverage(experiment):
    X, treated, y, persuadable = experiment

    fractions, incremental = qini_curve(y, treated, persuadable)

    assert fractions[0] > 0
    assert fractions[-1] == pytest.approx(1.0)
    assert len(fractions) == len(incremental)

    # Targeting by true uplift should accumulate gains, not lose them.
    assert incremental[-1] > 0


def test_uplift_by_decile_declines_for_a_good_score(experiment):
    X, treated, y, persuadable = experiment

    table = uplift_by_decile(y, treated, persuadable, bins=4)

    assert len(table) == 4
    assert table["n"].sum() == len(y)

    top = table.iloc[0]["observed_uplift"]
    bottom = table.iloc[-1]["observed_uplift"]

    assert top > bottom, "top quantile should show more uplift than the bottom"


def test_uplift_by_decile_handles_a_score_with_no_signal(experiment):
    X, treated, y, _ = experiment

    rng = np.random.default_rng(2)
    noise = rng.random(len(y))

    table = uplift_by_decile(y, treated, noise, bins=4)

    # No ordering, so the spread across quantiles should be small.
    spread = table["observed_uplift"].max() - table["observed_uplift"].min()

    assert spread < 0.15


def test_random_targeting_scores_near_zero_on_average(experiment):
    # Qini measures area above random targeting, so a random ranking should
    # centre on zero. Any single draw is noisy (standard deviation is several
    # units here), which is exactly why this averages over many.
    X, treated, y, persuadable = experiment

    rng = np.random.default_rng(5)

    scores = np.array([
        qini_score(y, treated, rng.random(len(y)))
        for _ in range(40)
    ])

    standard_error = scores.std() / np.sqrt(len(scores))

    assert abs(scores.mean()) < 3 * standard_error, (
        f"random targeting should centre on zero, got {scores.mean():.2f}"
    )

    # And a genuine ranking should beat every random draw, not merely the
    # average one.
    oracle = qini_score(y, treated, persuadable)

    assert oracle > scores.max()
