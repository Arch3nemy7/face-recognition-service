"""Unified error envelope (Task 1, §3.1).

Every failure path -- FaceServiceError subclasses, plain starlette
HTTPException, FastAPI RequestValidationError, and any unhandled Exception --
must produce a body with exactly the keys {"error", "detail", "error_code",
"image"}, matching `ErrorResponse`. This suite is fully offline (the model
singleton, `fetch_image_from_url` and `decode_base64_image` are stubbed via
the `client`/`fake_model` fixtures in tests/conftest.py).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import face_recognition_service.main as main_module
from face_recognition_service.config import settings
from face_recognition_service.schemas.api_schemas import ErrorCode
from tests.conftest import _face_result

AUTH_HEADERS = {"Authorization": f"Bearer {settings.api_token}"}
ENVELOPE_KEYS = {"error", "detail", "error_code", "image"}


def _assert_envelope(body: dict) -> None:
    assert set(body.keys()) == ENVELOPE_KEYS


class TestComparePhotosRequestErrors:
    """400s raised directly in the compare-photos route."""

    def test_blank_image1_is_invalid_request(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "   ", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 400
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.INVALID_REQUEST
        assert "cannot be empty" in body["detail"]
        assert body["image"] is None

    def test_bad_url_scheme_is_invalid_request(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "ftp://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 400
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.INVALID_REQUEST

    def test_bad_distance_metric_is_invalid_request(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "manhattan"},
            files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 400
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.INVALID_REQUEST


class TestAuthErrors:
    def test_missing_authorization_header(self, client: TestClient) -> None:
        response = client.get("/api/v1/model-info")
        # Whatever status the service returns today for a missing
        # Authorization header (HTTPBearer with auto_error=False routes
        # through verify_token, which raises 401 for both missing and wrong
        # tokens) -- asserted unchanged.
        assert response.status_code == 401
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.UNAUTHORIZED
        assert response.headers.get("www-authenticate") == "Bearer"

    def test_wrong_token(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/model-info",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.UNAUTHORIZED
        assert response.headers.get("www-authenticate") == "Bearer"


class TestValidationError:
    def test_missing_image2_field_is_422_validation_error(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 422
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.VALIDATION_ERROR
        assert isinstance(body["detail"], list)


class TestRoutingErrors:
    def test_unknown_route_is_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/does-not-exist", headers=AUTH_HEADERS)
        assert response.status_code == 404
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.NOT_FOUND

    def test_get_on_post_only_route_is_405(self, client: TestClient) -> None:
        response = client.get("/api/v1/embed", headers=AUTH_HEADERS)
        assert response.status_code == 405
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.METHOD_NOT_ALLOWED


class TestUnexpectedErrorsAreGeneric:
    def test_unexpected_runtime_error_in_route_is_generic_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _boom(url: str) -> np.ndarray:
            raise RuntimeError("secret-internal-text")

        monkeypatch.setattr(main_module, "fetch_image_from_url", _boom)

        with caplog.at_level(logging.ERROR):
            response = client.post(
                "/api/v1/compare-photos",
                data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
                files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 500
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.PROCESSING_ERROR
        raw_text = response.text
        assert "secret-internal-text" not in raw_text
        assert any(
            record.exc_info is not None for record in caplog.records
        ), "expected a log record emitted with exc_info"

    def test_catch_all_handler_on_route_with_no_try_except(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom():
            raise RuntimeError("secret-internal-text")

        monkeypatch.setattr(main_module, "get_model", _boom)

        with TestClient(main_module.app, raise_server_exceptions=False) as raw_client:
            response = raw_client.get("/api/v1/model-info", headers=AUTH_HEADERS)

        assert response.status_code == 500
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.PROCESSING_ERROR
        assert "secret-internal-text" not in response.text


class TestCompareInvalidEmbedding:
    def test_value_error_from_find_best_match_is_invalid_embedding(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ReferenceEmbedding/CompareRequest pin every embedding to exactly
        # 512 floats at the schema layer, so a real dimension mismatch can't
        # reach the route -- find_best_match is monkeypatched to raise the
        # ValueError it would raise for one directly.
        def _raise(**kwargs: object) -> None:
            raise ValueError("embedding dimensions do not match")

        monkeypatch.setattr(main_module, "find_best_match", _raise)

        response = client.post(
            "/api/v1/compare",
            json={
                "query_embedding": [0.1] * 512,
                "reference_embeddings": [{"id": "ref1", "embedding": [0.2] * 512}],
                "distance_metric": "cosine",
            },
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 400
        body = response.json()
        _assert_envelope(body)
        assert body["error_code"] == ErrorCode.INVALID_EMBEDDING


class TestHealthUnhealthy503:
    def test_health_returns_503_when_model_not_loaded(
        self, client: TestClient, fake_model: MagicMock
    ) -> None:
        fake_model.is_loaded.return_value = False

        response = client.get("/api/v1/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"
        assert body["model_loaded"] is False


class TestNoneSafeScoreLogging:
    def test_analyze_result_with_none_det_score_does_not_500(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_model: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(main_module, "decode_base64_image", lambda b64: np.zeros((64, 64, 3)))
        fake_model.analyze.return_value = _face_result(np.ones(512, dtype=np.float32), None)

        with caplog.at_level(logging.DEBUG, logger="face_recognition_service.main"):
            response = client.post(
                "/api/v1/embed",
                json={"image": "data:image/jpeg;base64,AAAA"},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 200
