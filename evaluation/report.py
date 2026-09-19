"""Assemble evaluation numbers into a JSON report and a markdown comparison table."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict

import numpy as np

from .datasets import Pair
from .embedder import QUALITY_FIELDS, EmbeddingResult, EvalConfig, Role
from .metrics import (
    equal_error_rate,
    far_frr,
    min_impostors_for,
    percentile_summary,
    score_summary,
    threshold_at_far,
)
from .scoring import PairSet, score_pairs, score_pairs_full

FAR_TARGETS = (1e-2, 1e-3, 1e-4)

# A failed-to-acquire (FTA) pair -- an image that never embedded at all --
# is a rejection at every threshold: a genuine FTA is a missed check-in, an
# impostor FTA is a (correct) refusal. metrics.py rejects non-finite scores,
# so FTA pairs are represented with a finite sentinel score guaranteed to be
# below every possible cosine similarity (which lies in [-1, 1]) rather than
# with -inf/NaN, so they count as a reject at any threshold without special
# casing the metrics functions themselves. This gives the ISO/IEC 19795-1
# "generalised" FAR/FRR/EER, which are comparable across presets and models
# that fail to acquire different numbers of images -- unlike the scored-only
# numbers, which silently narrow to whatever each preset managed to embed.
FTA_SENTINEL = -2.0


def _with_fta(scores, fta_count: int) -> np.ndarray:
    return np.concatenate([scores, np.full(fta_count, FTA_SENTINEL, dtype=np.float64)])


def _quality_report(embeddings: Mapping[str, EmbeddingResult], roles: Mapping[str, Role] | None) -> dict | None:
    """`{role: {field: {n, p01, p05, p50, p95, p99}}}` over embedded images, ignoring None values.

    A role/field combination with no non-None values (e.g. `interocular_px` when no image
    had landmarks) is left out entirely rather than reported as an empty summary.
    """
    by_role: dict[str, dict[str, list[float]]] = {}
    for key, result in embeddings.items():
        if result.embedding is None or result.quality is None:
            continue
        role = (roles or {}).get(key, "selfie")
        bucket = by_role.setdefault(role, {field: [] for field in QUALITY_FIELDS})
        for field in QUALITY_FIELDS:
            value = result.quality.get(field)
            if value is not None:
                bucket[field].append(value)
    if not by_role:
        return None
    return {
        role: {field: percentile_summary(values) for field, values in fields.items() if values}
        for role, fields in by_role.items()
    }


def _group_by_left_identity(pairs: Sequence[Pair], identities: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Positions in `pairs` grouped by the identity of each pair's left (first) key."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (left, _right) in enumerate(pairs):
        identity = identities.get(left)
        if identity is not None:
            groups[identity].append(idx)
    return {identity: np.array(idxs, dtype=np.int64) for identity, idxs in groups.items()}


# The FAR targets that get a resampled threshold_at_far_including_fta CI: the two lowest of
# FAR_TARGETS (the harness's noisiest, per docs/evaluation.md §2's rule-of-three note). Derived
# from FAR_TARGETS rather than restated so the two can't silently drift apart.
_BOOTSTRAP_FAR_TARGETS = tuple(sorted(FAR_TARGETS)[:2])
assert set(_BOOTSTRAP_FAR_TARGETS) <= set(FAR_TARGETS)


def _searchsorted_self_right(sorted_arr: np.ndarray) -> np.ndarray:
    """`np.searchsorted(sorted_arr, sorted_arr, side="right")`, in O(n) instead of O(n log n).

    Exploits that `sorted_arr` is already sorted: every element's side="right" rank equals the
    1-based index of the *last* element in its run of equal values (duplicate values -- common
    here, since an identity drawn more than once in a resample contributes the same underlying
    scores more than once -- all share that run's end rank, not their own individual index).
    Computed by marking each run's end with its rank, then propagating that rank *backwards*
    over the rest of the run: reverse the array so "backwards" becomes "forwards", forward-fill
    each zero with the most recent nonzero via a cumulative-max-of-index trick, and reverse back.
    Both passes are O(n) with no comparisons against other elements needed.
    """
    n = sorted_arr.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    is_run_end = np.empty(n, dtype=bool)
    is_run_end[:-1] = sorted_arr[1:] != sorted_arr[:-1]
    is_run_end[-1] = True  # the last element always ends its run
    run_end_rank = np.where(is_run_end, np.arange(1, n + 1), 0)
    reversed_ranks = run_end_rank[::-1]
    last_nonzero_index = np.maximum.accumulate(np.where(reversed_ranks != 0, np.arange(n), 0))
    return reversed_ranks[last_nonzero_index][::-1]


def _resample_metrics(g: np.ndarray, imp: np.ndarray, service_threshold: float) -> tuple:
    """(eer_including_fta.rate, far, frr_including_fta, {target: threshold_at_far}) for one resample.

    Every value returned here is exact -- for any g/imp, `eer_including_fta.rate` equals
    `metrics.equal_error_rate(g, imp)[0]`, `far`/`frr_including_fta` equal the plain-mean
    definitions in `metrics.far_frr`, and each `threshold_at_far` value equals
    `metrics.threshold_at_far(imp, target)` -- see tests/test_eval_report_cli.py's
    `test_resample_metrics_matches_metrics_module_exactly` for a randomised property check.

    `far`/`frr_including_fta` need no sort at all (plain means at a fixed threshold).
    `threshold_at_far` for both FAR targets shares one sort of `imp`. The EER search needs
    FAR and FRR at every candidate threshold in `g ∪ imp` (the only points where either curve
    can change, so the minimum |FAR-FRR| over the continuum is always attained at one of them --
    exactly `metrics.equal_error_rate`'s `thresholds = np.unique(np.concatenate([g, i]))`, just
    without the dedup, which doesn't change the best achievable rate since duplicate thresholds
    repeat the same FAR/FRR pair). Naively locating every impostor score's own rank *within the
    impostor array* is an O(n log n) search of ~1M elements against itself -- the single most
    expensive step in a naive implementation at this dataset size. Since the impostor array is
    already sorted (needed for `threshold_at_far` anyway), that self-rank is instead computed in
    O(n) by `_searchsorted_self_right`. The other three lookups needed (impostor's rank against
    the small genuine array, and genuine's rank against both arrays) all have the *small* array
    as the search haystack or needle, so they're cheap regardless.
    """
    g_sorted = np.sort(g)
    i_sorted = np.sort(imp)
    m, n = g_sorted.size, i_sorted.size

    frr_including_fta = float(np.mean(g <= service_threshold))
    scored_impostor = imp[imp != FTA_SENTINEL]
    far = float(np.mean(scored_impostor > service_threshold)) if scored_impostor.size else None

    ranked = i_sorted[::-1]
    far_thresholds: dict[float, float | None] = {}
    for target in _BOOTSTRAP_FAR_TARGETS:
        if ranked.size >= min_impostors_for(target):
            allowed = math.floor(target * ranked.size)
            far_thresholds[target] = float(ranked[allowed])
        else:
            far_thresholds[target] = None

    # Candidates from g: haystack is the small array on both sides -- cheap either way.
    rank_g_in_g = np.searchsorted(g_sorted, g_sorted, side="right")
    rank_g_in_i = np.searchsorted(i_sorted, g_sorted, side="right")
    frr_at_g = rank_g_in_g / m
    far_at_g = (n - rank_g_in_i) / n

    # Candidates from imp: rank-in-imp is the O(n) self-rank; rank-in-g has the small array as
    # the haystack (needle is large, but binary search against ~300 elements is still cheap).
    rank_i_in_g = np.searchsorted(g_sorted, i_sorted, side="right")
    rank_i_in_i = _searchsorted_self_right(i_sorted)
    frr_at_i = rank_i_in_g / m
    far_at_i = (n - rank_i_in_i) / n

    # metrics.equal_error_rate breaks ties in |FAR-FRR| by taking the first minimum in *ascending
    # threshold order* (np.argmin over np.unique's sorted output) -- so the g- and imp-origin
    # candidates above must be interleaved in that same ascending value order before the argmin,
    # not just concatenated, or a tie could resolve to a different (and wrong) FAR/FRR pair even
    # though the |FAR-FRR| minimum itself matches. `positions` (already computed for the merge
    # above) gives exactly that interleaving via np.insert, applied in parallel to far/frr.
    all_far = np.insert(far_at_i, rank_g_in_i, far_at_g)
    all_frr = np.insert(frr_at_i, rank_g_in_i, frr_at_g)
    best = int(np.argmin(np.abs(all_far - all_frr)))
    eer_rate = float((all_far[best] + all_frr[best]) / 2)

    return eer_rate, far, frr_including_fta, far_thresholds


def _bootstrap_report(
    *,
    pairs: PairSet,
    vectors: Mapping[str, np.ndarray | None],
    identities: Mapping[str, str],
    service_threshold: float,
    bootstrap: int,
    seed: int,
) -> dict | None:
    """Identity-level bootstrap CIs for a handful of FTA-inclusive point estimates.

    Resamples identities with replacement (`bootstrap` draws of the same identity
    count each), rebuilds the genuine/impostor score sets from pairs whose *left*
    key's identity was drawn (with multiplicity, including FTA sentinels), and
    reports the 2.5th/97.5th percentile of each metric across resamples where it
    could be computed at all.
    """
    if bootstrap <= 0 or not identities:
        return None
    identity_list = np.array(sorted(set(identities.values())))
    n = int(identity_list.size)
    metric_keys = ["eer_including_fta.rate", "service_threshold.far", "service_threshold.frr_including_fta"] + [
        f"threshold_at_far_including_fta.{target:g}.similarity_threshold" for target in _BOOTSTRAP_FAR_TARGETS
    ]
    samples: dict[str, list[float]] = {key: [] for key in metric_keys}
    if n:
        genuine_full = score_pairs_full(pairs.genuine, vectors, FTA_SENTINEL)
        impostor_full = score_pairs_full(pairs.impostor, vectors, FTA_SENTINEL)
        genuine_groups = _group_by_left_identity(pairs.genuine, identities)
        impostor_groups = _group_by_left_identity(pairs.impostor, identities)
        empty = np.empty(0, dtype=np.int64)
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            drawn = rng.choice(identity_list, size=n, replace=True)
            genuine_idx = np.concatenate([genuine_groups.get(identity, empty) for identity in drawn])
            impostor_idx = np.concatenate([impostor_groups.get(identity, empty) for identity in drawn])
            g = genuine_full[genuine_idx]
            imp = impostor_full[impostor_idx]
            if g.size and imp.size:
                rate, far, frr_including_fta, far_thresholds = _resample_metrics(g, imp, service_threshold)
                samples["eer_including_fta.rate"].append(rate)
                samples["service_threshold.frr_including_fta"].append(frr_including_fta)
                if far is not None:
                    samples["service_threshold.far"].append(far)
                for target in _BOOTSTRAP_FAR_TARGETS:
                    threshold = far_thresholds[target]
                    if threshold is not None:
                        samples[f"threshold_at_far_including_fta.{target:g}.similarity_threshold"].append(threshold)
    ci: dict[str, dict] = {}
    for key in metric_keys:
        values = samples[key]
        if values:
            low, high = np.percentile(np.array(values, dtype=np.float64), [2.5, 97.5])
            ci[key] = {"low": float(low), "high": float(high), "resamples_used": len(values)}
        else:
            ci[key] = {"low": None, "high": None, "resamples_used": 0}
    return {"resamples_requested": bootstrap, "identities": n, "seed": seed, "ci": ci}


def build_report(
    *,
    dataset_name: str,
    config: EvalConfig,
    embeddings: Mapping[str, EmbeddingResult],
    pairs: PairSet,
    service_threshold: float,
    run: dict | None = None,
    roles: Mapping[str, Role] | None = None,
    identities: Mapping[str, str] | None = None,
    bootstrap: int = 0,
    seed: int = 0,
) -> dict:
    vectors = {key: result.embedding for key, result in embeddings.items()}
    genuine, genuine_fta = score_pairs(pairs.genuine, vectors)
    impostor, impostor_fta = score_pairs(pairs.impostor, vectors)
    failures = Counter(r.error_code for r in embeddings.values() if r.error_code)
    report: dict = {
        "dataset": dataset_name,
        "config": asdict(config),
        "run": run,
        "images": {
            "total": len(embeddings),
            "embedded": sum(1 for r in embeddings.values() if r.embedding is not None),
            "failed_by_code": dict(sorted(failures.items())),
        },
        "pairs": {
            "genuine_total": len(pairs.genuine),
            "genuine_scored": int(genuine.size),
            "genuine_fta": genuine_fta,
            "impostor_total": len(pairs.impostor),
            "impostor_scored": int(impostor.size),
            "impostor_fta": impostor_fta,
        },
        "genuine_scores": score_summary(genuine) if genuine.size else None,
        "impostor_scores": score_summary(impostor) if impostor.size else None,
        "quality": _quality_report(embeddings, roles),
        "eer": None,
        "eer_including_fta": None,
        "service_threshold": None,
        "threshold_at_far": {f"{target:g}": None for target in FAR_TARGETS},
        "threshold_at_far_including_fta": {f"{target:g}": None for target in FAR_TARGETS},
        "bootstrap": _bootstrap_report(
            pairs=pairs,
            vectors=vectors,
            identities=identities or {},
            service_threshold=service_threshold,
            bootstrap=bootstrap,
            seed=seed,
        ),
    }
    if not (genuine.size and impostor.size):
        return report
    rate, eer_threshold = equal_error_rate(genuine, impostor)
    report["eer"] = {"rate": rate, "similarity_threshold": eer_threshold}

    genuine_all = _with_fta(genuine, genuine_fta)
    impostor_all = _with_fta(impostor, impostor_fta)
    rate_fta, eer_threshold_fta = equal_error_rate(genuine_all, impostor_all)
    report["eer_including_fta"] = {"rate": rate_fta, "similarity_threshold": eer_threshold_fta}

    far, frr = far_frr(genuine, impostor, service_threshold)
    report["service_threshold"] = {
        "similarity": service_threshold,
        "far": far,
        "frr": frr,
        "frr_including_fta": (frr * genuine.size + genuine_fta) / len(pairs.genuine),
    }
    for target in FAR_TARGETS:
        threshold = threshold_at_far(impostor, target)
        if threshold is not None:
            report["threshold_at_far"][f"{target:g}"] = {
                "similarity_threshold": threshold,
                "cosine_distance_threshold": 1.0 - threshold,
                "frr": far_frr(genuine, impostor, threshold)[1],
            }
        threshold_fta = threshold_at_far(impostor_all, target)
        if threshold_fta is not None:
            report["threshold_at_far_including_fta"][f"{target:g}"] = {
                "similarity_threshold": threshold_fta,
                "cosine_distance_threshold": 1.0 - threshold_fta,
                "frr": far_frr(genuine_all, impostor_all, threshold_fta)[1],
            }
    return report


def _pct(value: float | None) -> str:
    """Percentage formatting that keeps tiny non-zero rates visible.

    `f"{100*value:.2f}%"` alone rounds any rate below 0.005% (5e-5) to "0.00%",
    which is indistinguishable from an exactly-zero rate -- a real problem for FAR
    at the service threshold, FAR in bootstrap CI rows, and EER, which can
    legitimately be a few parts per million on large real datasets. Below that
    cutoff this instead prints in scientific notation with 3 significant figures
    (e.g. "2.00e-04%" for 2 false accepts out of 1,000,000 impostor pairs), so a
    non-zero rate never prints identically to zero.
    """
    if value is None:
        return "n/a"
    if value == 0:
        return "0.00%"
    pct = 100 * value
    if abs(pct) < 0.01:
        return f"{pct:.2e}%"
    return f"{pct:.2f}%"


def _num(value: float | None) -> str:
    # A similarity threshold at or below -1 can only be the FTA sentinel (-2.0) surfacing --
    # every real cosine similarity lies in [-1, 1] -- meaning FTA pairs alone decided this
    # threshold, not a genuine separation between genuine and impostor scores.
    if value is not None and value <= -1:
        return "n/a (FTA-dominated)"
    return "n/a" if value is None else f"{value:.3f}"


def _at_far(report: dict, key: str) -> str:
    point = report["threshold_at_far"].get(key)
    if point is None:
        return "n/a"
    threshold_text = _num(point["similarity_threshold"])
    if threshold_text.startswith("n/a"):
        return threshold_text
    return f"{threshold_text} ({_pct(point['frr'])})"


def _format_bootstrap_value(key: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    return _num(value) if key.endswith(".similarity_threshold") else _pct(value)


def _render_bootstrap_table(report: dict) -> str:
    bootstrap = report.get("bootstrap")
    if not bootstrap:
        return ""
    lines = [
        f"\nBootstrap CIs ({report['dataset']} / {report['config']['name']} / {report['config']['model_name']}, "
        f"{bootstrap['identities']} identities, {bootstrap['resamples_requested']} resamples requested, "
        f"seed {bootstrap['seed']}):\n\n"
        "| metric | 95% CI low | 95% CI high | resamples used |\n|---|---|---|---|\n"
    ]
    for key, values in bootstrap["ci"].items():
        low = _format_bootstrap_value(key, values["low"])
        high = _format_bootstrap_value(key, values["high"])
        lines.append(f"| {key} | {low} | {high} | {values['resamples_used']}/{bootstrap['resamples_requested']} |\n")
    return "".join(lines)


def render_markdown(reports: Sequence[dict]) -> str:
    header = (
        "| dataset | config | model | images failed | genuine scored | EER (sim thr) | EER incl. FTA "
        "| FAR @ service | FRR @ service | FRR incl. FTA | sim thr @ FAR 1e-3 (FRR) | sim thr @ FAR 1e-4 (FRR) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in reports:
        eer = r["eer"]
        eer_text = "n/a" if eer is None else f"{_pct(eer['rate'])} ({_num(eer['similarity_threshold'])})"
        eer_fta = r["eer_including_fta"]
        eer_fta_text = "n/a" if eer_fta is None else f"{_pct(eer_fta['rate'])} ({_num(eer_fta['similarity_threshold'])})"
        service = r["service_threshold"] or {}
        failed = r["images"]["total"] - r["images"]["embedded"]
        rows.append(
            f"| {r['dataset']} | {r['config']['name']} | {r['config']['model_name']} "
            f"| {failed}/{r['images']['total']} "
            f"| {r['pairs']['genuine_scored']}/{r['pairs']['genuine_total']} "
            f"| {eer_text} | {eer_fta_text} "
            f"| {_pct(service.get('far'))} | {_pct(service.get('frr'))} | {_pct(service.get('frr_including_fta'))} "
            f"| {_at_far(r, '0.001')} | {_at_far(r, '0.0001')} |"
        )
    body = header + "\n".join(rows) + "\n"
    return body + "".join(_render_bootstrap_table(r) for r in reports)
