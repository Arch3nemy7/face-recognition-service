"""Embedding distance calculation and comparison utilities."""

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
    Calculate Euclidean (L2) distance between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Euclidean distance as float
    """
    diff = embedding1 - embedding2
    distance = np.linalg.norm(diff)
    return float(distance)


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
    Find the best matching reference embedding for a query embedding.

    Args:
        query_embedding: Query embedding vector (512-dimensional list)
        reference_embeddings: List of reference embeddings with IDs
        metric: Distance metric to use ('cosine' or 'euclidean')

    Returns:
        Tuple of (all_matches, best_match) where:
        - all_matches: List of all MatchResult objects sorted by distance (ascending)
        - best_match: The best matching MatchResult (lowest distance)
    """
    # Convert query embedding to numpy array
    query_array = np.array(query_embedding, dtype=np.float32)

    # Calculate distances to all references
    matches: List[MatchResult] = []

    for ref in reference_embeddings:
        # Convert reference embedding to numpy array
        ref_array = np.array(ref.embedding, dtype=np.float32)

        # Calculate distance
        distance = calculate_distance(query_array, ref_array, metric=metric)

        # Convert to similarity
        similarity = distance_to_similarity(distance, metric=metric)

        # Create match result
        match = MatchResult(
            id=ref.id,
            distance=distance,
            similarity=similarity
        )
        matches.append(match)

    # Sort by distance (ascending - lower is better)
    matches.sort(key=lambda x: x.distance)

    # Best match is the first one (lowest distance)
    best_match = matches[0]

    return matches, best_match


def batch_calculate_distances(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Calculate distances between one query and multiple reference embeddings efficiently.

    Args:
        query_embedding: Query embedding (1D array of shape [embedding_dim])
        reference_embeddings: Reference embeddings (2D array of shape [num_refs, embedding_dim])
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        Array of distances (1D array of shape [num_refs])
    """
    if metric == "cosine":
        # Normalize embeddings
        query_norm = normalize_embedding(query_embedding)
        refs_norm = normalize_embedding(reference_embeddings)

        # Calculate cosine similarities (vectorized)
        cosine_sims = np.dot(refs_norm, query_norm)

        # Convert to distances
        distances = 1.0 - cosine_sims

        return distances

    elif metric == "euclidean":
        # Calculate Euclidean distances (vectorized)
        diffs = reference_embeddings - query_embedding
        distances = np.linalg.norm(diffs, axis=1)

        return distances

    else:
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
