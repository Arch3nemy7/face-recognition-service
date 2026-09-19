"""Metrics are pure functions of score arrays; accept iff score > threshold."""

import numpy as np
import pytest

from evaluation.metrics import (
    equal_error_rate,
    far_frr,
    min_impostors_for,
    score_summary,
    threshold_at_far,
)


def test_far_frr_counts_ties_as_rejections() -> None:
    genuine = [0.2, 0.5, 0.9]
    impostor = [0.1, 0.5, 0.6]
    far, frr = far_frr(genuine, impostor, threshold=0.5)
    assert far == pytest.approx(1 / 3)  # only 0.6 is accepted
    assert frr == pytest.approx(2 / 3)  # 0.2 and 0.5 are rejected


def test_eer_is_zero_when_classes_separate() -> None:
    rate, threshold = equal_error_rate([0.7, 0.8, 0.9], [0.1, 0.2, 0.3])
    assert rate == 0.0
    assert 0.3 <= threshold < 0.7


def test_eer_on_overlapping_scores() -> None:
    rate, threshold = equal_error_rate([0.2, 0.6, 0.8, 0.9], [0.1, 0.3, 0.5, 0.7])
    assert rate == pytest.approx(0.25)
    assert threshold == pytest.approx(0.5)


def test_threshold_at_far_admits_exactly_the_allowed_impostors() -> None:
    impostor = np.arange(1000) / 1000.0  # 0.000 .. 0.999
    threshold = threshold_at_far(impostor, 0.01)
    assert threshold == pytest.approx(0.989)
    assert float(np.mean(impostor > threshold)) == pytest.approx(0.01)


def test_threshold_at_far_refuses_without_enough_impostors() -> None:
    assert min_impostors_for(0.01) == 300
    assert threshold_at_far(np.linspace(0, 1, 299), 0.01) is None
    assert threshold_at_far(np.linspace(0, 1, 300), 0.01) is not None


@pytest.mark.parametrize("bad", [[], [0.1, float("nan")], [float("inf")]])
def test_invalid_scores_raise(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        far_frr(bad or [0.5], bad, 0.5)


def test_target_far_must_be_a_probability() -> None:
    with pytest.raises(ValueError):
        threshold_at_far([0.1] * 10, 0.0)


def test_score_summary_keys_and_values() -> None:
    summary = score_summary(np.linspace(0.0, 1.0, 101))
    assert set(summary) == {"n", "mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max"}
    assert summary["n"] == 101
    assert summary["p50"] == pytest.approx(0.5)
    assert summary["min"] == 0.0 and summary["max"] == 1.0
