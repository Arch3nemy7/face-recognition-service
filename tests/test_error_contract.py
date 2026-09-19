"""Mocked contract tests for face-service error responses (Task 1).

Unlike tests/test_api.py, this suite never loads InsightFace: the model
singleton, `fetch_image_from_url`, `decode_base64_image` (/embed) and
`decode_image_bytes` (upload paths) are stubbed so it runs fully offline. It exercises every row of the error contract in
.superpowers/sdd/face-errors-tasks/global-constraints.md -- the split between
NO_FACE_DETECTED and FACE_LOW_QUALITY, the reference/selfie `image` tag on
compare-photos, REFERENCE_UNAVAILABLE for an unreachable reference URL, the
MODEL_NOT_LOADED/PROCESSING_ERROR status codes, and the HTTPException status
bug (a 400 raised inside `try` must not become a 500).
"""

from __future__ import annotations

import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from httpx import Response

import face_recognition_service.main as main_module
from face_recognition_service.config import settings
from face_recognition_service.models import face_model as face_model_module
from face_recognition_service.models.face_model import (
    FaceModelError,
    FaceRecognitionModel,
)
from face_recognition_service.schemas.api_schemas import ErrorCode
from face_recognition_service.utils import image_utils as image_utils_module
from face_recognition_service.utils.image_utils import (
    ImageProcessingError,
    fetch_image_from_url,
)
from tests.conftest import _face_result

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
        self.kps = None
        self.embedding = embedding if embedding is not None else np.ones(512, dtype=np.float32)


def _model_with_faces(faces: list) -> FaceRecognitionModel:
    """A loaded FaceRecognitionModel whose detector and recognizer are stubbed."""
    model = FaceRecognitionModel()
    model.model = SimpleNamespace()  # anything non-None counts as loaded
    model._detect = lambda image: faces
    model._embed = lambda image, face: face.embedding
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
# never come back in the client-facing error, since the reference photo URL is
# frequently a presigned link carrying its own credentials.
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

    SIGNED_URL = "https://reference.example.com/photos/photo123.jpg?sig=super-secret-token&exp=999"
    SECRET = "super-secret-token"
    REDACTED_HOST_AND_PATH = "reference.example.com/photos/photo123.jpg"

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
    unreachable-upstream fault (SERVICE_UNAVAILABLE) -- controller ruling
    correcting the earlier contract, since it's not evidence that specific
    photo/link is bad and must not read as "your reference photo is broken".
    """

    URL = "https://reference.example.com/photos/photo123.jpg"

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


class TestFetchErrorsDoNotChainTheSignedUrl:
    """A logged traceback must not resurrect the query string `error` redacts.

    requests puts the full URL (query string included) in its exception
    messages. The redacted ImageProcessingError is raised inside the
    `except`, so without `from None` Python chains the original exception
    and any `logger.exception(...)` / traceback print shows it again.
    """

    SIGNED_URL = "https://reference.example.com/photos/photo123.jpg?sig=super-secret-token&exp=999"
    SECRET = "super-secret-token"

    @staticmethod
    def _formatted(exc: BaseException) -> str:
        return "".join(traceback.format_exception(exc))

    @pytest.mark.parametrize(
        "exc_type",
        [
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.TooManyRedirects,
        ],
    )
    def test_transport_errors_are_not_chained(
        self, monkeypatch: pytest.MonkeyPatch, exc_type: type[Exception]
    ) -> None:
        def _raise(*args: object, **kwargs: object) -> None:
            raise exc_type(f"Max retries exceeded with url: {self.SIGNED_URL}")

        monkeypatch.setattr(image_utils_module.requests, "get", _raise)
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.SIGNED_URL)
        assert self.SECRET not in self._formatted(exc_info.value)

    @pytest.mark.parametrize("status_code", [404, 503])
    def test_http_errors_are_not_chained(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        signed_url = self.SIGNED_URL

        class _Response:
            def __init__(self) -> None:
                self.status_code = status_code

            def raise_for_status(self) -> None:
                error = requests.exceptions.HTTPError(f"{status_code} Error for url: {signed_url}")
                error.response = self
                raise error

        monkeypatch.setattr(image_utils_module.requests, "get", lambda *a, **k: _Response())
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(self.SIGNED_URL)
        assert self.SECRET not in self._formatted(exc_info.value)


# ---------------------------------------------------------------------------
# API-level contract: main.py's routing of error_code -> status/image
# ---------------------------------------------------------------------------


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
        fake_model.analyze.side_effect = FaceModelError("no face", ErrorCode.NO_FACE_DETECTED)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.NO_FACE_DETECTED
        assert response.json()["image"] == "reference"

    def test_reference_low_quality_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        fake_model.analyze.side_effect = FaceModelError("blurry", ErrorCode.FACE_LOW_QUALITY)

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
        fake_model.analyze.return_value = _face_result(np.ones(512, dtype=np.float32), 0.9)

        def _raise_invalid(data: bytes) -> np.ndarray:
            raise ImageProcessingError("bad selfie bytes", ErrorCode.INVALID_IMAGE)

        monkeypatch.setattr(main_module, "decode_image_bytes", _raise_invalid)

        response = _compare_photos(client)

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.INVALID_IMAGE
        assert response.json()["image"] == "selfie"

    def test_selfie_no_face_tagged_selfie(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        self._ok_reference(monkeypatch)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        # First call (reference) succeeds, second call (selfie) fails.
        fake_model.analyze.side_effect = [
            _face_result(np.ones(512, dtype=np.float32), 0.9),
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
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        fake_model.analyze.side_effect = [
            _face_result(np.ones(512, dtype=np.float32), 0.9),
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
        # Controller ruling: a timeout on the reference fetch is an
        # unreachable-upstream fault, not evidence that specific photo/link
        # is bad -- it must route like a real outage (untagged, 503), which
        # the backend then maps to ERR-INT-008, not the caller-owned
        # HR-037 "reupload your reference photo" path.
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
        # A 5xx from the reference host/CDN side is their outage, not a bad link/photo.
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
        body = response.json()
        assert body["error_code"] == ErrorCode.INVALID_REQUEST
        assert "manhattan" in body["detail"]

    def test_successful_match_is_unaffected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        # Sanity check that the happy path still works after wrapping each
        # image stage in its own try/except.
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        embedding = np.ones(512, dtype=np.float32)
        fake_model.analyze.side_effect = [
            _face_result(embedding, 0.95),
            _face_result(embedding, 0.93),
        ]

        response = _compare_photos(client)

        assert response.status_code == 200
        body = response.json()
        assert body["match"] is True
        assert body["image1_detection_score"] == pytest.approx(0.95)
        assert body["image2_detection_score"] == pytest.approx(0.93)

    def test_compare_photos_returns_quality_and_analyses_with_roles(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        """/compare-photos must analyse image1 as the reference and image2 as
        the selfie, and surface each side's (optional) quality metrics."""
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        embedding = np.ones(512, dtype=np.float32)
        quality1 = face_model_module.FaceQuality(
            face_size_px=120.0,
            interocular_px=40.0,
            roll_deg=1.0,
            yaw_proxy=0.05,
            blur_variance=500.0,
            embedding_norm=20.0,
            faces_considered=1,
            second_face_ratio=0.0,
        )
        quality2 = face_model_module.FaceQuality(
            face_size_px=100.0,
            interocular_px=35.0,
            roll_deg=-2.0,
            yaw_proxy=-0.1,
            blur_variance=300.0,
            embedding_norm=18.0,
            faces_considered=1,
            second_face_ratio=0.0,
        )
        fake_model.analyze.side_effect = [
            _face_result(embedding, 0.95, quality=quality1),
            _face_result(embedding, 0.93, quality=quality2),
        ]

        response = _compare_photos(client)

        assert response.status_code == 200
        body = response.json()
        assert body["image1_quality"]["face_size_px"] == pytest.approx(120.0)
        assert body["image1_quality"]["interocular_px"] == pytest.approx(40.0)
        assert body["image2_quality"]["face_size_px"] == pytest.approx(100.0)
        assert body["image2_quality"]["interocular_px"] == pytest.approx(35.0)
        roles = [call.kwargs.get("role") for call in fake_model.analyze.call_args_list]
        assert roles == ["reference", "selfie"]


class TestComparePhotosUploadSameContract:
    """/compare-photos-upload uses the same codes; image1~reference, image2~selfie."""

    def test_first_image_error_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        def _decode(data: bytes) -> np.ndarray:
            raise ImageProcessingError("bad image1", ErrorCode.INVALID_IMAGE)

        monkeypatch.setattr(main_module, "decode_image_bytes", _decode)

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
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        fake_model.analyze.side_effect = [
            _face_result(np.ones(512, dtype=np.float32), 0.9),
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
        fake_model.analyze.side_effect = FaceModelError("no face", ErrorCode.NO_FACE_DETECTED)

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
