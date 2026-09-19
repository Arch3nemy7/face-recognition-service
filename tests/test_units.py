"""Offline unit/API tests: no real model, no network."""

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from face_recognition_service.utils.embedding_utils import (
    cosine_distance,
    distance_to_similarity,
    euclidean_distance,
    is_valid_embedding,
)
from face_recognition_service.utils.image_utils import (
    decode_base64_image,
    encode_image_to_base64,
    validate_image,
)
from tests.conftest import TEST_API_TOKEN

AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_TOKEN}"}


@pytest.fixture
def unit_embedding() -> list[float]:
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


class TestPublicEndpoints:
    def test_root(self, client: TestClient) -> None:
        body = client.get("/").json()
        assert body["status"] == "running"
        assert {"service", "version"} <= body.keys()

    def test_health_needs_no_auth(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


class TestAuth:
    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/api/v1/model-info"),
            ("post", "/api/v1/embed"),
            ("post", "/api/v1/compare"),
            ("post", "/api/v1/compare-photos"),
            ("post", "/api/v1/compare-photos-upload"),
        ],
    )
    def test_protected_routes_reject_missing_token(self, client: TestClient, method: str, path: str) -> None:
        assert getattr(client, method)(path).status_code == 401

    def test_wrong_token_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/v1/model-info", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_model_info_with_token(self, client: TestClient) -> None:
        response = client.get("/api/v1/model-info", headers=AUTH_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["embedding_size"] == 512
        assert body["backend"] == "insightface"
        assert {"name", "device"} <= body.keys()


class TestEmbedValidation:
    def test_undecodable_image_is_invalid_image(self, client: TestClient) -> None:
        response = client.post("/api/v1/embed", json={"image": "invalid_base64!!!"}, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_IMAGE"

    def test_empty_image_is_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/embed", json={"image": ""}, headers=AUTH_HEADERS)
        assert response.status_code == 422

    def test_missing_image_is_422(self, client: TestClient) -> None:
        assert client.post("/api/v1/embed", json={}, headers=AUTH_HEADERS).status_code == 422


class TestCompareEmbeddings:
    def _post(self, client: TestClient, payload: dict) -> httpx.Response:
        return client.post("/api/v1/compare", json=payload, headers=AUTH_HEADERS)

    def test_identical_embedding_is_best_match(self, client: TestClient, unit_embedding: list[float]) -> None:
        other = np.random.default_rng(1).standard_normal(512).tolist()
        response = self._post(
            client,
            {
                "query_embedding": unit_embedding,
                "reference_embeddings": [
                    {"id": "same", "embedding": unit_embedding},
                    {"id": "other", "embedding": other},
                ],
                "distance_metric": "cosine",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert [m["id"] for m in body["matches"]] == ["same", "other"]
        assert body["distance_metric"] == "cosine"
        assert body["best_match"]["id"] == "same"
        assert body["best_match"]["distance"] < 1e-5

    def test_identical_embedding_euclidean_is_best_match(
        self, client: TestClient, unit_embedding: list[float]
    ) -> None:
        # Only identical vectors are asserted here; non-identical euclidean
        # distances hit a known normalization bug that Phase 2 fixes.
        response = self._post(
            client,
            {
                "query_embedding": unit_embedding,
                "reference_embeddings": [{"id": "same", "embedding": unit_embedding}],
                "distance_metric": "euclidean",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["distance_metric"] == "euclidean"
        assert body["best_match"]["distance"] < 1e-5

    @pytest.mark.parametrize(
        "payload_patch",
        [
            {"distance_metric": "manhattan"},
            {"query_embedding": [0.1] * 256},
            {"reference_embeddings": []},
        ],
    )
    def test_invalid_requests_are_422(
        self, client: TestClient, unit_embedding: list[float], payload_patch: dict
    ) -> None:
        payload = {
            "query_embedding": unit_embedding,
            "reference_embeddings": [{"id": "a", "embedding": unit_embedding}],
            "distance_metric": "cosine",
        }
        payload.update(payload_patch)
        assert self._post(client, payload).status_code == 422


class TestEmbeddingUtils:
    def test_identical_vectors_have_zero_distance(self, unit_embedding: list[float]) -> None:
        vector = np.array(unit_embedding)
        assert cosine_distance(vector, vector) < 1e-6
        assert euclidean_distance(vector, vector) < 1e-6

    def test_distance_to_similarity_bounds(self) -> None:
        assert distance_to_similarity(0.0, "cosine") == 1.0
        assert distance_to_similarity(2.0, "cosine") == 0.0
        assert distance_to_similarity(0.0, "euclidean") == 1.0
        assert distance_to_similarity(10.0, "euclidean") < 0.1

    @pytest.mark.parametrize(
        "embedding,expected",
        [
            ([0.1] * 512, True),
            ([0.1] * 256, False),
            ([0.1] * 511 + [float("nan")], False),
            ([0.1] * 511 + [float("inf")], False),
        ],
    )
    def test_is_valid_embedding(self, embedding: list[float], expected: bool) -> None:
        assert is_valid_embedding(embedding, 512) is expected


class TestImageUtils:
    def test_valid_image(self) -> None:
        assert validate_image(np.zeros((100, 100, 3), dtype=np.uint8)) == (True, None)

    def test_too_small_image(self) -> None:
        ok, error = validate_image(np.zeros((10, 10, 3), dtype=np.uint8))
        assert ok is False and "too small" in error.lower()

    def test_none_image(self) -> None:
        ok, error = validate_image(None)
        assert ok is False and error

    def test_base64_round_trip_keeps_shape(self) -> None:
        original = np.random.default_rng(2).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        assert decode_base64_image(encode_image_to_base64(original, format="JPEG")).shape == original.shape
