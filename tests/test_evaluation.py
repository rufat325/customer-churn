import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import bootstrap_metric, format_ci, paired_bootstrap


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)

    y = rng.integers(0, 2, 500)
    # A score that carries real but imperfect signal.
    strong = y + rng.normal(0, 1.0, 500)
    weak = y + rng.normal(0, 3.0, 500)

    return y, weak, strong


def test_bootstrap_interval_brackets_the_point_estimate(scored):
    y, _, strong = scored

    result = bootstrap_metric(y, strong, resamples=500)

    assert result["ci_lower"] < result["estimate"] < result["ci_upper"]
    assert result["estimate"] == pytest.approx(roc_auc_score(y, strong))


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(1)

    def width(n):
        y = rng.integers(0, 2, n)
        score = y + rng.normal(0, 1.0, n)
        r = bootstrap_metric(y, score, resamples=400)
        return r["ci_upper"] - r["ci_lower"]

    assert width(2000) < width(200)


def test_bootstrap_is_reproducible(scored):
    y, _, strong = scored

    a = bootstrap_metric(y, strong, resamples=300, seed=7)
    b = bootstrap_metric(y, strong, resamples=300, seed=7)

    assert a == b


def test_paired_bootstrap_detects_a_real_difference(scored):
    y, weak, strong = scored

    result = paired_bootstrap(y, weak, strong, resamples=800)

    assert result["difference"] > 0
    assert result["significant"]
    assert result["win_rate"] > 0.9


def test_paired_bootstrap_finds_no_difference_between_identical_scores(scored):
    y, _, strong = scored

    result = paired_bootstrap(y, strong, strong, resamples=400)

    assert result["difference"] == pytest.approx(0.0)
    assert not result["significant"]


def test_paired_bootstrap_is_directional(scored):
    y, weak, strong = scored

    forward = paired_bootstrap(y, weak, strong, resamples=400)
    reverse = paired_bootstrap(y, strong, weak, resamples=400)

    assert forward["difference"] == pytest.approx(-reverse["difference"])


def test_paired_test_beats_comparing_overlapping_intervals():
    # The motivating case for the module: two models whose individual
    # confidence intervals overlap heavily, but where one reliably beats the
    # other on identical rows. Judging by overlap would miss it.
    rng = np.random.default_rng(3)

    y = rng.integers(0, 2, 600)
    shared_noise = rng.normal(0, 1.2, 600)

    worse = y + shared_noise
    better = y + shared_noise - 0.35 * rng.normal(0, 1, 600) + 0.35 * y

    a = bootstrap_metric(y, worse, resamples=600)
    b = bootstrap_metric(y, better, resamples=600)

    intervals_overlap = a["ci_upper"] > b["ci_lower"]

    paired = paired_bootstrap(y, worse, better, resamples=600)

    assert intervals_overlap, "test setup no longer produces overlap"
    assert paired["significant"], "paired test should still resolve it"


def test_format_ci_renders_both_result_shapes(scored):
    y, weak, strong = scored

    single = format_ci(bootstrap_metric(y, strong, resamples=200))
    difference = format_ci(paired_bootstrap(y, weak, strong, resamples=200))

    for text in (single, difference):
        assert "[" in text and "," in text and "]" in text
