"""Mocked contract tests for face-service error responses.

Unlike tests/test_api.py, this suite never loads InsightFace: the model
singleton, `fetch_image_from_url` and `decode_base64_image` are stubbed so it
runs fully offline. It exercises every row of the error contract -- the split
between NO_FACE_DETECTED and FACE_LOW_QUALITY, the reference/selfie `image`
tag on compare-photos, REFERENCE_UNAVAILABLE vs SERVICE_UNAVAILABLE for a bad
vs unreachable reference URL, the MODEL_NOT_LOADED/PROCESSING_ERROR status
codes, and the HTTPException status bug (a 400 raised inside `try` must not
become a 500).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from httpx import Response

import face_recognition_service.main as main_module
from face_recognition_service.config import settings
from face_recognition_service.models import face_model as face_model_module
from face_recognition_service.models.face_model import FaceModelError, FaceRecognitionModel
from face_recognition_service.schemas.api_schemas import ErrorCode
from face_recognition_service.utils import image_utils as image_utils_module
from face_recognition_service.utils.image_utils import ImageProcessingError, fetch_image_from_url

AUTH_HEADERS = {"Authorization": f"Bearer {settings.api_token}"}
DUMMY_IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# FaceRecognitionModel.get_embedding: NO_FACE_DETECTED vs FACE_LOW_QUALITY
#
# Exercised directly against the model class (no FastAPI involved) since
# this is where the quality split actually happens; the endpoint tests below
# mock get_embedding itself and so can't observe this logic.
# ---------------------------------------------------------------------------


class _FakeFace:
    """Stand-in for an insightface Face: only the attributes get_embedding reads."""

    def __init__(self, det_score: float, embedding: np.ndarray | None = None) -> None:
        self.det_score = det_score
        self.bbox = (0.0, 0.0, 10.0, 10.0)
        self.embedding = embedding if embedding is not None else np.ones(512, dtype=np.float32)


def _model_with_faces(faces: list) -> FaceRecognitionModel:
    """A loaded FaceRecognitionModel whose InsightFace `.get()` call is stubbed."""
    model = FaceRecognitionModel()
    model.model = SimpleNamespace(get=lambda image: faces)
    return model


class TestGetEmbeddingQualitySplit:
    """face_model.get_embedding must tell 'no face' apart from 'low quality'."""

    def test_no_faces_at_all_raises_no_face_detected(self) -> None:
        model = _model_with_faces([])
        with pytest.raises(FaceModelError) as exc_info:
            model.get_embedding(DUMMY_IMAGE)
        assert exc_info.value.error_code == ErrorCode.NO_FACE_DETECTED

    def test_faces_below_quality_raise_face_low_quality_with_best_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pin the threshold rather than relying on the .env value (this repo's
        # dev .env sets min_face_quality permissively low).
        monkeypatch.setattr(face_model_module.settings, "min_face_quality", 0.7)
        model = _model_with_faces([_FakeFace(det_score=0.3), _FakeFace(det_score=0.55)])
        with pytest.raises(FaceModelError) as exc_info:
            model.get_embedding(DUMMY_IMAGE)
        assert exc_info.value.error_code == ErrorCode.FACE_LOW_QUALITY
        # Dev message carries the best of the rejected scores (0.55), not 0.3.
        assert "0.55" in exc_info.value.message

    def test_face_meeting_quality_succeeds(self) -> None:
        model = _model_with_faces([_FakeFace(det_score=0.95)])
        embedding, score = model.get_embedding(DUMMY_IMAGE)
        assert score == pytest.approx(0.95)
        assert len(embedding) == 512

    def test_model_not_loaded_raises_model_not_loaded(self) -> None:
        model = FaceRecognitionModel()  # .model stays None until load()
        with pytest.raises(FaceModelError) as exc_info:
            model.get_embedding(DUMMY_IMAGE)
        assert exc_info.value.error_code == ErrorCode.MODEL_NOT_LOADED


# ---------------------------------------------------------------------------
# fetch_image_from_url: a signed reference URL's query string (auth) must
# never come back in the client-facing error, since the reference photo URL
# is frequently a presigned link carrying its own credentials.
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal requests.Response stand-in: only what raise_for_status() needs."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        error = requests.exceptions.HTTPError(f"{self.status_code} error for url: ...")
        error.response = self
        raise error


class TestFetchImageFromUrlRedactsSignedQueryString:
    """The dev-facing `error` message must drop the URL's query string."""

    SIGNED_URL = "https://photos.example.com/uploads/photo123.jpg?sig=super-secret-token&exp=999"
    SECRET = "super-secret-token"
    REDACTED_HOST_AND_PATH = "photos.example.com/uploads/photo123.jpg"

    def test_timeout_omits_query_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_timeout(*args: object, **kwargs: object) -> None:
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(image_utils_module.requests, "get", _raise_timeout)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.SIGNED_URL)

        # A timeout is an unreachable-upstream fault (SERVICE_UNAVAILABLE),
        # not a REFERENCE_UNAVAILABLE on this specific link -- see
        # TestFetchImageFromUrlReferenceVsServiceFault below for the full split.
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        assert self.SECRET not in exc_info.value.message
        assert "?" not in exc_info.value.message
        assert self.REDACTED_HOST_AND_PATH in exc_info.value.message

    def test_404_response_omits_query_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A real 4xx, via raise_for_status() as requests actually raises it
        # (HTTPError.response set to the Response) -- str(e) on that
        # exception embeds the full requested URL, query string and all,
        # which is exactly what must not reach the client.
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda *a, **k: _FakeResponse(404)
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.SIGNED_URL)

        assert exc_info.value.error_code == ErrorCode.REFERENCE_UNAVAILABLE
        assert self.SECRET not in exc_info.value.message
        assert "?" not in exc_info.value.message
        assert self.REDACTED_HOST_AND_PATH in exc_info.value.message


class TestFetchImageFromUrlReferenceVsServiceFault:
    """Only a definite 4xx on the reference link is REFERENCE_UNAVAILABLE.

    A timeout, connection/DNS/TLS failure, or the far end's own 5xx is an
    unreachable-upstream fault (SERVICE_UNAVAILABLE), since it's not evidence
    that specific photo/link is bad and must not read as "the reference
    photo is broken".
    """

    URL = "https://photos.example.com/uploads/photo123.jpg"

    def test_404_is_reference_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda *a, **k: _FakeResponse(404)
        )
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_UNAVAILABLE

    def test_403_is_reference_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda *a, **k: _FakeResponse(403)
        )
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_UNAVAILABLE

    def test_timeout_is_service_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*args: object, **kwargs: object) -> None:
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(image_utils_module.requests, "get", _raise)
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_connection_error_is_service_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*args: object, **kwargs: object) -> None:
            # DNS failures and TLS/SSLError surface as (subclasses of)
            # ConnectionError from requests.
            raise requests.exceptions.ConnectionError("connection refused")

        monkeypatch.setattr(image_utils_module.requests, "get", _raise)
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_ssl_error_is_service_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*args: object, **kwargs: object) -> None:
            raise requests.exceptions.SSLError("certificate verify failed")

        monkeypatch.setattr(image_utils_module.requests, "get", _raise)
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_500_is_service_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda *a, **k: _FakeResponse(500)
        )
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE


# ---------------------------------------------------------------------------
# API-level contract: main.py's routing of error_code -> status/image
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_model() -> MagicMock:
    """Stand-in for the FaceRecognitionModel singleton every endpoint resolves."""
    model = MagicMock()
    model.is_loaded.return_value = True
    model.model_name = "fake"
    model.get_model_info.return_value = {
        "name": "fake",
        "embedding_size": 512,
        "backend": "insightface",
        "device": "cpu",
    }
    return model


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock) -> Generator[TestClient, None, None]:
    """TestClient whose lifespan installs `fake_model` instead of loading InsightFace."""

    def _fake_initialize_model() -> None:
        face_model_module._model_instance = fake_model

    monkeypatch.setattr(main_module, "initialize_model", _fake_initialize_model)
    monkeypatch.setattr(main_module, "cleanup_model", lambda: None)
    # Every scenario below controls fetch/decode directly; preprocessing is
    # not what this suite is testing, so make it a no-op by default.
    monkeypatch.setattr(main_module, "preprocess_image", lambda image: image)

    with TestClient(main_module.app) as test_client:
        yield test_client

    face_model_module._model_instance = None


def _compare_photos(client: TestClient, *, image1: str = "https://example.com/ref.jpg") -> Response:
    return client.post(
        "/api/v1/compare-photos",
        data={"image1": image1, "distance_metric": "cosine"},
        files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
        headers=AUTH_HEADERS,
    )


class TestComparePhotosReferenceErrors:
    """Reference-image (image1) failures: tagged image="reference"."""

    def test_reference_fetch_404_maps_to_reference_unavailable(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A definite 4xx on the reference link itself (404, 403, 410, ...)
        # is REFERENCE_UNAVAILABLE -- see fetch_image_from_url's HTTPError
        # handler and TestFetchImageFromUrlReferenceVsServiceFault for the
        # full split against SERVICE_UNAVAILABLE (timeout/DNS/TLS/5xx).
        def _raise_not_found(url: str) -> np.ndarray:
            raise ImageProcessingError("404 Not Found", ErrorCode.REFERENCE_UNAVAILABLE)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_not_found)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.REFERENCE_UNAVAILABLE
        assert response.json()["image"] == "reference"

    def test_reference_unreadable_image_maps_to_invalid_image(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # URL was reachable but the bytes don't decode -- unreadable input,
        # not "unreachable", so it stays INVALID_IMAGE (not REFERENCE_UNAVAILABLE).
        def _raise_invalid(url: str) -> np.ndarray:
            raise ImageProcessingError("bad bytes", ErrorCode.INVALID_IMAGE)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_invalid)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.INVALID_IMAGE
        assert response.json()["image"] == "reference"

    def test_reference_unsupported_format_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_unsupported(url: str) -> np.ndarray:
            raise ImageProcessingError("bad format", ErrorCode.UNSUPPORTED_FORMAT)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_unsupported)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.UNSUPPORTED_FORMAT
        assert response.json()["image"] == "reference"

    def test_reference_too_large_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_too_large(url: str) -> np.ndarray:
            raise ImageProcessingError("too big", ErrorCode.IMAGE_TOO_LARGE)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_too_large)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        assert response.json()["image"] == "reference"

    def test_reference_no_face_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        fake_model.get_embedding.side_effect = FaceModelError("no face", ErrorCode.NO_FACE_DETECTED)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.NO_FACE_DETECTED
        assert response.json()["image"] == "reference"

    def test_reference_low_quality_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        fake_model.get_embedding.side_effect = FaceModelError("blurry", ErrorCode.FACE_LOW_QUALITY)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.FACE_LOW_QUALITY
        assert response.json()["image"] == "reference"


class TestComparePhotosSelfieErrors:
    """Selfie-image (image2) failures: tagged image="selfie", reference already ok."""

    def _ok_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)

    def test_selfie_invalid_image_tagged_selfie(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        self._ok_reference(monkeypatch)
        fake_model.get_embedding.return_value = (np.ones(512, dtype=np.float32), 0.9)

        def _raise_invalid(b64: str) -> np.ndarray:
            raise ImageProcessingError("bad selfie bytes", ErrorCode.INVALID_IMAGE)

        monkeypatch.setattr(main_module, "decode_base64_image", _raise_invalid)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.INVALID_IMAGE
        assert response.json()["image"] == "selfie"

    def test_selfie_no_face_tagged_selfie(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        self._ok_reference(monkeypatch)
        monkeypatch.setattr(main_module, "decode_base64_image", lambda b64: DUMMY_IMAGE)
        # First call (reference) succeeds, second call (selfie) fails.
        fake_model.get_embedding.side_effect = [
            (np.ones(512, dtype=np.float32), 0.9),
            FaceModelError("no face", ErrorCode.NO_FACE_DETECTED),
        ]

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.NO_FACE_DETECTED
        assert response.json()["image"] == "selfie"

    def test_selfie_low_quality_tagged_selfie(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        self._ok_reference(monkeypatch)
        monkeypatch.setattr(main_module, "decode_base64_image", lambda b64: DUMMY_IMAGE)
        fake_model.get_embedding.side_effect = [
            (np.ones(512, dtype=np.float32), 0.9),
            FaceModelError("blurry", ErrorCode.FACE_LOW_QUALITY),
        ]

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.FACE_LOW_QUALITY
        assert response.json()["image"] == "selfie"


class TestComparePhotosServerProblems:
    """Server-side faults: no `image` tag, and their own HTTP status."""

    def test_model_not_loaded_returns_503_untagged(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate the singleton never having been initialised.
        face_model_module._model_instance = None

        response = _compare_photos(client)

        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == ErrorCode.MODEL_NOT_LOADED
        assert body["image"] is None

    def test_unexpected_error_returns_500_processing_error_untagged(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        # A raw (non-FaceModelError/ImageProcessingError) exception surfacing
        # mid-reference-stage must still land as untagged 500 PROCESSING_ERROR,
        # not get swallowed into the reference tag.
        def _boom(url: str) -> np.ndarray:
            raise RuntimeError("network stack exploded")

        monkeypatch.setattr(main_module, "fetch_image_from_url", _boom)

        response = _compare_photos(client)

        assert response.status_code == 500
        body = response.json()
        assert body["error_code"] == ErrorCode.PROCESSING_ERROR
        assert body["image"] is None
        # No raw exception text/stack leaks unexpectedly-shaped; envelope
        # still matches ErrorResponse (error/detail/error_code/image), not
        # FastAPI's bare {"detail": ...}.
        assert set(body.keys()) == {"error", "detail", "error_code", "image"}

    def test_reference_timeout_returns_503_untagged(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A timeout on the reference fetch is an unreachable-upstream fault,
        # not evidence that specific photo/link is bad -- it must route like
        # a real outage (untagged, 503), so a caller can treat it as a
        # transient failure instead of asking the photo's owner to reupload.
        def _raise_timeout(url: str) -> np.ndarray:
            raise ImageProcessingError("timed out", ErrorCode.SERVICE_UNAVAILABLE)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_timeout)

        response = _compare_photos(client)

        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == ErrorCode.SERVICE_UNAVAILABLE
        assert body["image"] is None

    def test_reference_connection_error_returns_503_untagged(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_connection_error(url: str) -> np.ndarray:
            raise ImageProcessingError("unreachable", ErrorCode.SERVICE_UNAVAILABLE)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_connection_error)

        response = _compare_photos(client)

        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == ErrorCode.SERVICE_UNAVAILABLE
        assert body["image"] is None

    def test_reference_upstream_500_returns_503_untagged(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 5xx from the far end (e.g. a CDN) is their outage, not a bad link/photo.
        def _raise_upstream_500(url: str) -> np.ndarray:
            raise ImageProcessingError("HTTP 500", ErrorCode.SERVICE_UNAVAILABLE)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_upstream_500)

        response = _compare_photos(client)

        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == ErrorCode.SERVICE_UNAVAILABLE
        assert body["image"] is None


class TestComparePhotosValidationKeepsOwnStatus:
    """HTTPException raised inside `try` must not be swallowed into a 500."""

    def test_blank_image1_url_stays_400(self, client: TestClient) -> None:
        # A literal empty string is dropped by the multipart encoder before
        # it reaches FastAPI (which then 422s on the missing field, a
        # separate concern); whitespace-only exercises the same `.strip()`
        # emptiness check in compare_photos without that encoder quirk.
        response = _compare_photos(client, image1="   ")
        assert response.status_code == 400

    def test_bad_url_scheme_stays_400(self, client: TestClient) -> None:
        response = _compare_photos(client, image1="ftp://example.com/ref.jpg")
        assert response.status_code == 400

    def test_bad_distance_metric_stays_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "manhattan"},
            files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 400

    def test_successful_match_is_unaffected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        # Sanity check that the happy path still works after wrapping each
        # image stage in its own try/except.
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        monkeypatch.setattr(main_module, "decode_base64_image", lambda b64: DUMMY_IMAGE)
        embedding = np.ones(512, dtype=np.float32)
        fake_model.get_embedding.side_effect = [(embedding, 0.95), (embedding, 0.93)]

        response = _compare_photos(client)

        assert response.status_code == 200
        body = response.json()
        assert body["match"] is True
        assert body["image1_detection_score"] == pytest.approx(0.95)
        assert body["image2_detection_score"] == pytest.approx(0.93)


class TestComparePhotosUploadSameContract:
    """/compare-photos-upload uses the same codes; image1~reference, image2~selfie."""

    def test_first_image_error_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        def _decode(b64: str) -> np.ndarray:
            raise ImageProcessingError("bad image1", ErrorCode.INVALID_IMAGE)

        monkeypatch.setattr(main_module, "decode_base64_image", _decode)

        response = client.post(
            "/api/v1/compare-photos-upload",
            files={
                "image1": ("first.jpg", b"first-bytes", "image/jpeg"),
                "image2": ("second.jpg", b"second-bytes", "image/jpeg"),
            },
            data={"distance_metric": "cosine"},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.INVALID_IMAGE
        assert response.json()["image"] == "reference"

    def test_second_image_error_tagged_selfie(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "decode_base64_image", lambda b64: DUMMY_IMAGE)
        fake_model.get_embedding.side_effect = [
            (np.ones(512, dtype=np.float32), 0.9),
            FaceModelError("no face", ErrorCode.NO_FACE_DETECTED),
        ]

        response = client.post(
            "/api/v1/compare-photos-upload",
            files={
                "image1": ("first.jpg", b"first-bytes", "image/jpeg"),
                "image2": ("second.jpg", b"second-bytes", "image/jpeg"),
            },
            data={"distance_metric": "cosine"},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.NO_FACE_DETECTED
        assert response.json()["image"] == "selfie"

    def test_bad_distance_metric_stays_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/compare-photos-upload",
            files={
                "image1": ("first.jpg", b"first-bytes", "image/jpeg"),
                "image2": ("second.jpg", b"second-bytes", "image/jpeg"),
            },
            data={"distance_metric": "manhattan"},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 400


class TestEmbedSameContract:
    """/embed uses the same codes; `image` is left null (single-image endpoint)."""

    def test_no_face_detected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "decode_base64_image", lambda b64: DUMMY_IMAGE)
        fake_model.get_embedding.side_effect = FaceModelError("no face", ErrorCode.NO_FACE_DETECTED)

        response = client.post(
            "/api/v1/embed",
            json={"image": "data:image/jpeg;base64,AAAA"},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == ErrorCode.NO_FACE_DETECTED
        assert body["image"] is None

    def test_unexpected_error_returns_500_via_error_response_envelope(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(b64: str) -> np.ndarray:
            raise RuntimeError("decoder crashed")

        monkeypatch.setattr(main_module, "decode_base64_image", _boom)

        response = client.post(
            "/api/v1/embed",
            json={"image": "data:image/jpeg;base64,AAAA"},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 500
        body = response.json()
        assert body["error_code"] == ErrorCode.PROCESSING_ERROR
        assert body["image"] is None
        assert set(body.keys()) == {"error", "detail", "error_code", "image"}
