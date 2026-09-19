"""Verification metrics over similarity scores (higher = more alike).

A pair is accepted when its score is strictly greater than the threshold,
mirroring the service's `distance < threshold` rule on cosine distance
(distance = 1 - similarity).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _as_array(scores: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("scores must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite")
    return arr


def far_frr(genuine, impostor, threshold: float) -> tuple[float, float]:
    """False accept rate and false reject rate at one threshold."""
    g = _as_array(genuine)
    i = _as_array(impostor)
    return float(np.mean(i > threshold)), float(np.mean(g <= threshold))


def equal_error_rate(genuine, impostor) -> tuple[float, float]:
    """(EER, threshold) at the observed score where FAR and FRR are closest."""
    g = np.sort(_as_array(genuine))
    i = np.sort(_as_array(impostor))
    thresholds = np.unique(np.concatenate([g, i]))
    frr = np.searchsorted(g, thresholds, side="right") / g.size
    far = (i.size - np.searchsorted(i, thresholds, side="right")) / i.size
    best = int(np.argmin(np.abs(far - frr)))
    return float((far[best] + frr[best]) / 2), float(thresholds[best])


def min_impostors_for(target_far: float) -> int:
    """Impostor pairs needed before a FAR claim means anything (rule of three, ~95%)."""
    return math.ceil(3 / target_far)


def threshold_at_far(impostor, target_far: float) -> float | None:
    """Lowest observed threshold whose FAR is <= target_far, or None if too few impostors."""
    if not 0 < target_far < 1:
        raise ValueError("target_far must be in (0, 1)")
    ranked = np.sort(_as_array(impostor))[::-1]
    if ranked.size < min_impostors_for(target_far):
        return None
    allowed = math.floor(target_far * ranked.size)  # impostors that may score above t
    return float(ranked[allowed])


def percentile_summary(values: Iterable[float] | np.ndarray) -> dict[str, float]:
    """`{n, p01, p05, p50, p95, p99}` over a non-empty 1-D sample (e.g. a quality metric)."""
    arr = _as_array(values)
    p01, p05, p50, p95, p99 = np.quantile(arr, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


def score_summary(scores) -> dict[str, float]:
    s = _as_array(scores)
    p01, p05, p50, p95, p99 = np.quantile(s, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(s.max()),
    }
