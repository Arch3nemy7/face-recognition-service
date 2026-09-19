"""Embedding distance calculation and comparison utilities."""

import math
from typing import List, Tuple

import numpy as np

from ..schemas.api_schemas import MatchResult, ReferenceEmbedding


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize an embedding vector to unit length.

    Args:
        embedding: Embedding vector as numpy array

    Returns:
        Normalized embedding vector
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine distance between two embeddings.

    Cosine distance = 1 - cosine similarity
    Range: [0, 2], where 0 means identical and 2 means opposite

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine distance as float
    """
    # Normalize embeddings
    emb1_norm = normalize_embedding(embedding1)
    emb2_norm = normalize_embedding(embedding2)

    # Calculate cosine similarity
    cosine_sim = np.dot(emb1_norm, emb2_norm)

    # Clip to handle numerical errors
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    # Convert to distance (0 = identical, 2 = opposite)
    distance = 1.0 - cosine_sim

    return float(distance)


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Euclidean (L2) distance between the unit-normalised embeddings.

    Normalising first makes the distance depend only on the angle between the
    embeddings (raw InsightFace vectors have norms of ~9-30), so on unit
    vectors d_euclidean = sqrt(2 * d_cosine) and both metrics can share one
    threshold -- see euclidean_threshold_for.
    """
    diff = normalize_embedding(np.asarray(embedding1, dtype=np.float64)) - normalize_embedding(
        np.asarray(embedding2, dtype=np.float64)
    )
    return float(np.linalg.norm(diff))


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Cosine similarity in [-1, 1] (1 = same direction)."""
    return 1.0 - cosine_distance(embedding1, embedding2)


def euclidean_threshold_for(cosine_threshold: float) -> float:
    """The euclidean threshold that makes the same decision as a cosine one on unit vectors."""
    return math.sqrt(2.0 * cosine_threshold)


def match_threshold(metric: str, cosine_threshold: float) -> float:
    """Distance threshold for `metric`, derived from the single configured cosine threshold."""
    if metric == "cosine":
        return cosine_threshold
    if metric == "euclidean":
        return euclidean_threshold_for(cosine_threshold)
    raise ValueError(f"Unsupported distance metric: {metric}")


def calculate_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Calculate distance between two embeddings using specified metric.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        Distance as float

    Raises:
        ValueError: If metric is not supported
    """
    if metric == "cosine":
        return cosine_distance(embedding1, embedding2)
    elif metric == "euclidean":
        return euclidean_distance(embedding1, embedding2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """
    Convert distance to similarity score in range [0, 1].

    For cosine distance: similarity = 1 - (distance / 2)
    For euclidean distance: similarity = 1 / (1 + distance)

    Args:
        distance: Distance value
        metric: Distance metric used ('cosine' or 'euclidean')

    Returns:
        Similarity score in range [0, 1], where 1 is most similar
    """
    if metric == "cosine":
        # Cosine distance range is [0, 2]
        # Convert to similarity [0, 1]
        similarity = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, similarity))
    elif metric == "euclidean":
        # Euclidean distance range is [0, infinity)
        # Convert to similarity using inverse relationship
        similarity = 1.0 / (1.0 + distance)
        return max(0.0, min(1.0, similarity))
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def find_best_match(
    query_embedding: List[float],
    reference_embeddings: List[ReferenceEmbedding],
    metric: str = "cosine"
) -> Tuple[List[MatchResult], MatchResult]:
    """
    Rank reference embeddings by distance to the query (closest first).

    Returns (all matches sorted ascending by distance, the closest match).
    No threshold is applied here; callers decide what counts as a match.
    """
    query = np.asarray(query_embedding, dtype=np.float64)
    refs = np.asarray([ref.embedding for ref in reference_embeddings], dtype=np.float64)
    distances = batch_calculate_distances(query, refs, metric=metric)
    matches = [
        MatchResult(id=ref.id, distance=float(d), similarity=distance_to_similarity(float(d), metric=metric))
        for ref, d in zip(reference_embeddings, distances, strict=True)
    ]
    matches.sort(key=lambda m: m.distance)
    return matches, matches[0]


def batch_calculate_distances(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Distances from one query to many references, each row normalised on its own.

    Args:
        query_embedding: 1-D array [embedding_dim]
        reference_embeddings: 2-D array [num_refs, embedding_dim]
        metric: 'cosine' or 'euclidean'

    Returns:
        1-D array [num_refs]
    """
    query = normalize_embedding(np.asarray(query_embedding, dtype=np.float64))
    refs = np.asarray(reference_embeddings, dtype=np.float64)
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    refs = refs / norms
    if metric == "cosine":
        return 1.0 - np.clip(refs @ query, -1.0, 1.0)
    if metric == "euclidean":
        return np.linalg.norm(refs - query, axis=1)
    raise ValueError(f"Unsupported distance metric: {metric}")


def is_valid_embedding(embedding: List[float], expected_size: int = 512) -> bool:
    """
    Validate that an embedding has the correct size and valid values.

    Args:
        embedding: Embedding vector as list of floats
        expected_size: Expected embedding dimension

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(embedding, (list, np.ndarray)):
        return False

    if len(embedding) != expected_size:
        return False

    # Check for NaN or Inf values
    try:
        arr = np.array(embedding, dtype=np.float32)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return False
    except (ValueError, TypeError):
        return False

    return True
