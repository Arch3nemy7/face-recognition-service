"""Turn a dataset plus embeddings into genuine and impostor similarity scores."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .datasets import Dataset, Pair


@dataclass(frozen=True)
class PairSet:
    genuine: tuple[Pair, ...]
    impostor: tuple[Pair, ...]


def _by_identity(dataset: Dataset) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in dataset.images:
        groups[dataset.identity[key]].append(key)
    return groups


def _derived_genuine(dataset: Dataset) -> list[Pair]:
    pairs: list[Pair] = []
    for keys in _by_identity(dataset).values():
        refs = [k for k in keys if k in dataset.references]
        probes = [k for k in keys if k not in dataset.references]
        if refs:
            pairs.extend((ref, probe) for ref in refs for probe in probes)
        else:
            pairs.extend(itertools.combinations(keys, 2))
    return pairs


def _cross_identity(dataset: Dataset, max_pairs: int, seed: int) -> list[Pair]:
    identity = dataset.identity
    if dataset.references:
        left = sorted(dataset.references)
        right = sorted(k for k in dataset.images if k not in dataset.references)
        ordered = True
    else:
        left = right = sorted(dataset.images)
        ordered = False

    def admissible(a: str, b: str) -> bool:
        return identity[a] != identity[b] and (ordered or a < b)

    if ordered:
        total = sum(1 for a in left for b in right if identity[a] != identity[b])
    else:
        counts = defaultdict(int)
        for key in left:
            counts[identity[key]] += 1
        total = (len(left) ** 2 - sum(n * n for n in counts.values())) // 2
    if total <= max_pairs:
        return [(a, b) for a in left for b in right if admissible(a, b)]

    rng = np.random.default_rng(seed)
    chosen: dict[Pair, None] = {}  # insertion-ordered, so the result depends only on the seed
    while len(chosen) < max_pairs:
        for i, j in zip(rng.integers(len(left), size=4096), rng.integers(len(right), size=4096), strict=True):
            a, b = left[i], right[j]
            if not ordered and b < a:
                a, b = b, a
            if admissible(a, b):
                chosen.setdefault((a, b), None)
                if len(chosen) == max_pairs:
                    break
    return list(chosen)


def build_pairs(dataset: Dataset, *, max_impostors: int, seed: int = 0, augment_impostors: bool = False) -> PairSet:
    """Build the genuine/impostor pairs to score.

    Explicit genuine and impostor pairs on the dataset (e.g. LFW's protocol
    pairs) are always kept in full; `max_impostors` never trims them. Only
    *derived* cross-identity impostor pairs -- produced when the dataset has
    no explicit impostor pairs, or when `augment_impostors=True` -- are capped
    at `max_impostors`; if more candidates exist than the cap, they are
    sampled uniformly without replacement using `numpy.random.default_rng(seed)`,
    so the result is deterministic for a given seed. A derived pair that
    duplicates an explicit impostor pair (in either order) is dropped rather
    than double-counted.
    """
    if dataset.genuine_pairs is None and dataset.references:
        identities_with_reference = {dataset.identity[key] for key in dataset.references}
        identities_without_reference = set(dataset.identity.values()) - identities_with_reference
        if identities_without_reference:
            raise ValueError(
                f"{dataset.name}: {len(identities_without_reference)} identities have no reference photo "
                f"(e.g. {sorted(identities_without_reference)[0]!r}) while others do; this would mix "
                "probe-vs-probe genuine pairs for the reference-less identities with reference-vs-probe "
                "impostor pairs for the rest. Give every identity a reference photo, or supply explicit "
                "genuine_pairs so pairing is unambiguous."
            )

    genuine = list(dataset.genuine_pairs) if dataset.genuine_pairs is not None else _derived_genuine(dataset)
    impostor = list(dataset.impostor_pairs or ())
    if dataset.impostor_pairs is None or augment_impostors:
        if not dataset.identity:
            raise ValueError(f"{dataset.name}: identities are unknown, so impostor pairs cannot be derived")
        seen = {frozenset(p) for p in impostor}
        for pair in _cross_identity(dataset, max_impostors, seed):
            if frozenset(pair) not in seen:
                impostor.append(pair)
    return PairSet(genuine=tuple(genuine), impostor=tuple(impostor))


def _is_usable(vector: np.ndarray | None) -> bool:
    """A vector is usable only if present, finite, and non-zero-length."""
    if vector is None:
        return False
    norm = np.linalg.norm(vector)
    return bool(np.isfinite(norm) and norm != 0)


def score_pairs(
    pairs: Sequence[Pair],
    vectors: Mapping[str, np.ndarray | None],
    chunk: int = 50_000,
) -> tuple[np.ndarray, int]:
    """Cosine similarity per scorable pair, and how many pairs had a missing/degenerate embedding.

    A pair is unscorable -- and counted in `failed` -- if either embedding is
    missing, non-finite (NaN/inf), or has zero L2 norm; such vectors would
    otherwise produce NaN similarities (and a RuntimeWarning) that silently
    behave like a reject.
    """
    scorable = [(a, b) for a, b in pairs if _is_usable(vectors.get(a)) and _is_usable(vectors.get(b))]
    failed = len(pairs) - len(scorable)
    parts: list[np.ndarray] = []
    for start in range(0, len(scorable), chunk):
        block = scorable[start : start + chunk]
        left = np.stack([vectors[a] for a, _ in block]).astype(np.float64)
        right = np.stack([vectors[b] for _, b in block]).astype(np.float64)
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        parts.append(np.einsum("ij,ij->i", left, right))
    return (np.concatenate(parts) if parts else np.empty(0)), failed


def score_pairs_full(
    pairs: Sequence[Pair],
    vectors: Mapping[str, np.ndarray | None],
    fta_value: float,
    chunk: int = 50_000,
) -> np.ndarray:
    """Cosine similarity per pair, kept in the pairs' original order and length.

    Unlike `score_pairs`, no pair is dropped: a pair with a missing/degenerate
    embedding gets `fta_value` instead. This lets a caller index the result by
    the same positions as `pairs` -- e.g. to resample pairs by identity for a
    bootstrap confidence interval, where the position of each pair must line up
    with which identity produced it.
    """
    result = np.full(len(pairs), fta_value, dtype=np.float64)
    positions = [idx for idx, (a, b) in enumerate(pairs) if _is_usable(vectors.get(a)) and _is_usable(vectors.get(b))]
    for start in range(0, len(positions), chunk):
        block = positions[start : start + chunk]
        left = np.stack([vectors[pairs[i][0]] for i in block]).astype(np.float64)
        right = np.stack([vectors[pairs[i][1]] for i in block]).astype(np.float64)
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        result[block] = np.einsum("ij,ij->i", left, right)
    return result
