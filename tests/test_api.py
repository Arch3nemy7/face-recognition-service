"""Tests for the face recognition API endpoints."""

import base64
import io
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from shelia_face_recognition_service.main import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_face_image_base64() -> str:
    """
    Create a sample face image for testing.

    Note: This is a synthetic image, not a real face.
    For real testing, use actual face images.
    """
    # Create a simple test image (RGB)
    img = Image.new('RGB', (200, 200), color='white')

    # Add a simple face-like pattern (circle for head, dots for eyes)
    pixels = img.load()
    center_x, center_y = 100, 100

    # Draw a circle (head)
    for x in range(200):
        for y in range(200):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if 40 < dist < 60:
                pixels[x, y] = (0, 0, 0)

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/jpeg;base64,{base64_str}"


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample 512-dimensional embedding."""
    # Create a normalized random embedding
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


class TestHealthEndpoints:
    """Tests for health and info endpoints."""

    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "service" in data
        assert "version" in data

    def test_health_check(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_model_info(self, client: TestClient):
        """Test the model info endpoint."""
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["embedding_size"] == 512
        assert data["backend"] == "insightface"
        assert "name" in data
        assert "device" in data


class TestEmbedEndpoint:
    """Tests for the embedding extraction endpoint."""

    def test_embed_endpoint_schema(self, client: TestClient, sample_face_image_base64: str):
        """Test that the embed endpoint accepts the correct schema."""
        response = client.post(
            "/api/v1/embed",
            json={"image": sample_face_image_base64}
        )
        # Note: This test may fail if no face is detected in the synthetic image
        # In real testing, use actual face images
        assert response.status_code in [200, 400]  # 400 if no face detected

    def test_embed_endpoint_invalid_base64(self, client: TestClient):
        """Test embed endpoint with invalid base64."""
        response = client.post(
            "/api/v1/embed",
            json={"image": "invalid_base64!!!"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_embed_endpoint_empty_image(self, client: TestClient):
        """Test embed endpoint with empty image."""
        response = client.post(
            "/api/v1/embed",
            json={"image": ""}
        )
        assert response.status_code == 422  # Validation error

    def test_embed_endpoint_missing_image(self, client: TestClient):
        """Test embed endpoint without image field."""
        response = client.post("/api/v1/embed", json={})
        assert response.status_code == 422  # Validation error


class TestCompareEndpoint:
    """Tests for the embedding comparison endpoint."""

    def test_compare_endpoint_cosine(self, client: TestClient, sample_embedding: list[float]):
        """Test compare endpoint with cosine distance."""
        query = sample_embedding
        ref1 = sample_embedding.copy()
        ref2 = np.random.randn(512).tolist()

        response = client.post(
            "/api/v1/compare",
            json={
                "query_embedding": query,
                "reference_embeddings": [
                    {"id": "user_001", "embedding": ref1},
                    {"id": "user_002", "embedding": ref2}
                ],
                "distance_metric": "cosine"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert "best_match" in data
        assert "distance_metric" in data
        assert len(data["matches"]) == 2
        assert data["distance_metric"] == "cosine"

        # Best match should be ref1 (same as query)
        assert data["best_match"]["id"] == "user_001"
        assert data["best_match"]["distance"] < 0.1  # Should be very close to 0

    def test_compare_endpoint_euclidean(self, client: TestClient, sample_embedding: list[float]):
        """Test compare endpoint with euclidean distance."""
        query = sample_embedding
        ref1 = sample_embedding.copy()

        response = client.post(
            "/api/v1/compare",
            json={
                "query_embedding": query,
                "reference_embeddings": [
                    {"id": "user_001", "embedding": ref1}
                ],
                "distance_metric": "euclidean"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["distance_metric"] == "euclidean"

    def test_compare_endpoint_invalid_metric(self, client: TestClient, sample_embedding: list[float]):
        """Test compare endpoint with invalid distance metric."""
        response = client.post(
            "/api/v1/compare",
            json={
                "query_embedding": sample_embedding,
                "reference_embeddings": [
                    {"id": "user_001", "embedding": sample_embedding}
                ],
                "distance_metric": "invalid"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_compare_endpoint_invalid_embedding_size(self, client: TestClient):
        """Test compare endpoint with wrong embedding size."""
        wrong_size_embedding = [0.1] * 256  # Wrong size

        response = client.post(
            "/api/v1/compare",
            json={
                "query_embedding": wrong_size_embedding,
                "reference_embeddings": [
                    {"id": "user_001", "embedding": [0.1] * 512}
                ],
                "distance_metric": "cosine"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_compare_endpoint_empty_references(self, client: TestClient, sample_embedding: list[float]):
        """Test compare endpoint with no reference embeddings."""
        response = client.post(
            "/api/v1/compare",
            json={
                "query_embedding": sample_embedding,
                "reference_embeddings": [],
                "distance_metric": "cosine"
            }
        )

        assert response.status_code == 422  # Validation error


class TestEmbeddingUtils:
    """Tests for embedding utility functions."""

    def test_cosine_distance_identical(self, sample_embedding: list[float]):
        """Test cosine distance between identical embeddings."""
        from shelia_face_recognition_service.utils.embedding_utils import cosine_distance

        emb = np.array(sample_embedding)
        distance = cosine_distance(emb, emb)

        # Distance between identical embeddings should be very close to 0
        assert distance < 0.001

    def test_euclidean_distance_identical(self, sample_embedding: list[float]):
        """Test euclidean distance between identical embeddings."""
        from shelia_face_recognition_service.utils.embedding_utils import euclidean_distance

        emb = np.array(sample_embedding)
        distance = euclidean_distance(emb, emb)

        # Distance between identical embeddings should be very close to 0
        assert distance < 0.001

    def test_distance_to_similarity(self):
        """Test distance to similarity conversion."""
        from shelia_face_recognition_service.utils.embedding_utils import distance_to_similarity

        # Cosine distance of 0 should give similarity of 1
        assert distance_to_similarity(0.0, "cosine") == 1.0

        # Euclidean distance of 0 should give similarity of 1
        assert distance_to_similarity(0.0, "euclidean") == 1.0

        # High distance should give low similarity
        assert distance_to_similarity(2.0, "cosine") == 0.0
        assert distance_to_similarity(10.0, "euclidean") < 0.1

    def test_is_valid_embedding(self):
        """Test embedding validation."""
        from shelia_face_recognition_service.utils.embedding_utils import is_valid_embedding

        # Valid embedding
        valid = [0.1] * 512
        assert is_valid_embedding(valid, 512) is True

        # Invalid size
        invalid_size = [0.1] * 256
        assert is_valid_embedding(invalid_size, 512) is False

        # With NaN
        with_nan = [0.1] * 511 + [float('nan')]
        assert is_valid_embedding(with_nan, 512) is False

        # With Inf
        with_inf = [0.1] * 511 + [float('inf')]
        assert is_valid_embedding(with_inf, 512) is False


class TestImageUtils:
    """Tests for image utility functions."""

    def test_validate_image_valid(self):
        """Test image validation with valid image."""
        from shelia_face_recognition_service.utils.image_utils import validate_image

        valid_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        is_valid, error = validate_image(valid_image)

        assert is_valid is True
        assert error is None

    def test_validate_image_too_small(self):
        """Test image validation with too small image."""
        from shelia_face_recognition_service.utils.image_utils import validate_image

        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        is_valid, error = validate_image(small_image)

        assert is_valid is False
        assert "too small" in error.lower()

    def test_validate_image_none(self):
        """Test image validation with None."""
        from shelia_face_recognition_service.utils.image_utils import validate_image

        is_valid, error = validate_image(None)

        assert is_valid is False
        assert error is not None

    def test_encode_decode_image(self):
        """Test encoding and decoding images."""
        from shelia_face_recognition_service.utils.image_utils import (
            decode_base64_image,
            encode_image_to_base64
        )

        # Create a test image
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Encode
        base64_str = encode_image_to_base64(original, format="JPEG")

        # Decode
        decoded = decode_base64_image(base64_str)

        # Check shape matches (JPEG is lossy, so exact match not expected)
        assert decoded.shape == original.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
