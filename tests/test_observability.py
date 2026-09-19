"""Tests for Phase 3 robustness, Task 5 (S3.5): request IDs, the one
PII-free summary log line per comparison, and the Docker CMD that lets
LOG_LEVEL reach uvicorn.

Runs fully offline, the same way tests/test_error_contract.py does: the
model singleton, `fetch_image_from_url` and `decode_image_bytes` are
stubbed via `main_module` monkeypatches, never touching the network or
InsightFace.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import face_recognition_service.main as main_module
from face_recognition_service.config import settings
from face_recognition_service.schemas.api_schemas import ErrorCode
from face_recognition_service.utils.image_utils import ImageProcessingError
from tests.conftest import _face_result

AUTH_HEADERS = {"Authorization": f"Bearer {settings.api_token}"}
DUMMY_IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)
_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _compare_photos(client: TestClient, *, image1: str = "https://example.com/ref.jpg") -> "object":
    return client.post(
        "/api/v1/compare-photos",
        data={"image1": image1, "distance_metric": "cosine"},
        files={"image2": ("selfie-secret-name.jpg", b"selfie-bytes", "image/jpeg")},
        headers=AUTH_HEADERS,
    )


def _successful_compare_photos(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock):
    monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
    monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
    embedding = np.ones(512, dtype=np.float32)
    fake_model.analyze.side_effect = [
        _face_result(embedding, 0.95),
        _face_result(embedding, 0.93),
    ]
    return _compare_photos(client)


class TestRequestIdHeader:
    def test_generated_when_absent(self, client: TestClient) -> None:
        response = client.get("/")
        request_id = response.headers.get("x-request-id")
        assert request_id is not None
        assert _HEX32.match(request_id)

    def test_valid_inbound_id_is_echoed(self, client: TestClient) -> None:
        response = client.get("/", headers={"X-Request-ID": "my-valid-id.123"})
        assert response.headers.get("x-request-id") == "my-valid-id.123"

    def test_invalid_inbound_id_is_replaced(self, client: TestClient) -> None:
        response = client.get("/", headers={"X-Request-ID": "bad id!"})
        request_id = response.headers.get("x-request-id")
        assert request_id != "bad id!"
        assert _HEX32.match(request_id)

        too_long = "a" * 65
        response = client.get("/", headers={"X-Request-ID": too_long})
        request_id = response.headers.get("x-request-id")
        assert request_id != too_long
        assert _HEX32.match(request_id)

    def test_body_too_large_still_carries_request_id(self, client: TestClient) -> None:
        oversize = b"x" * (settings.max_request_body_bytes + 1)
        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", oversize, "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        request_id = response.headers.get("x-request-id")
        assert request_id is not None
        assert _HEX32.match(request_id)


class TestRequestIdPropagatesIntoThreadpool:
    def test_id_appears_on_log_records_emitted_in_threadpool_helper(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        helper_logger = logging.getLogger("face_recognition_service.main")

        def _load_reference_and_log(url: str):
            helper_logger.info("threadpool helper ran")
            return DUMMY_IMAGE

        monkeypatch.setattr(main_module, "fetch_image_from_url", _load_reference_and_log)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        embedding = np.ones(512, dtype=np.float32)
        fake_model.analyze.side_effect = [
            _face_result(embedding, 0.95),
            _face_result(embedding, 0.93),
        ]

        with caplog.at_level(logging.INFO):
            response = _compare_photos(client, image1="https://example.com/ref.jpg")

        assert response.status_code == 200
        request_id = response.headers["x-request-id"]

        helper_records = [r for r in caplog.records if r.message == "threadpool helper ran"]
        assert len(helper_records) == 1
        assert helper_records[0].request_id == request_id


class TestSummaryLogLine:
    def _summary_records(self, caplog: pytest.LogCaptureFixture) -> list:
        return [r for r in caplog.records if r.name == "face_recognition_service.summary"]

    def test_successful_compare_photos_emits_one_match_summary(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            response = _successful_compare_photos(client, monkeypatch, fake_model)

        assert response.status_code == 200
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert "endpoint=compare-photos" in message
        assert "outcome=match" in message
        for field in (
            "endpoint",
            "outcome",
            "error_code",
            "image",
            "match",
            "cosine_similarity",
            "threshold",
            "det1",
            "det2",
            "fetch_ms",
            "decode_ms",
            "queue_ms",
            "infer_ms",
            "total_ms",
            "q1_face_px",
            "q2_face_px",
            "q2_blur",
        ):
            assert f"{field}=" in message, f"missing field {field!r} in summary: {message}"

    def test_reference_404_emits_error_summary(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _raise_not_found(url: str) -> np.ndarray:
            raise ImageProcessingError("404 Not Found", ErrorCode.REFERENCE_UNAVAILABLE)

        monkeypatch.setattr(main_module, "fetch_image_from_url", _raise_not_found)

        with caplog.at_level(logging.INFO):
            response = _compare_photos(client)

        assert response.status_code == 400
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert "outcome=error" in message
        assert f"error_code={ErrorCode.REFERENCE_UNAVAILABLE}" in message
        assert "image=reference" in message

    def test_no_summary_record_contains_url_host_or_filename(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            response = _successful_compare_photos(client, monkeypatch, fake_model)
        assert response.status_code == 200

        forbidden = ("example.com", "selfie-secret-name.jpg")
        for record in caplog.records:
            text = record.getMessage()
            for needle in forbidden:
                assert needle not in text, f"{needle!r} leaked into log record: {text!r}"

    def test_service_busy_summary_includes_queue_ms(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from face_recognition_service.concurrency import ServiceBusyError

        async def _raise_busy(*args, **kwargs):
            raise ServiceBusyError()

        # Simulate the gate timing out on its very first acquire.
        class _AlwaysBusyGate:
            def slot(self):
                from contextlib import asynccontextmanager

                @asynccontextmanager
                async def _cm():
                    raise ServiceBusyError()
                    yield  # pragma: no cover

                return _cm()

        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        monkeypatch.setattr(main_module, "_inference_gate", lambda: _AlwaysBusyGate())

        with caplog.at_level(logging.INFO):
            response = _compare_photos(client)

        assert response.status_code == 503
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert f"error_code={ErrorCode.SERVICE_BUSY}" in message
        assert "queue_ms=" in message
        assert "queue_ms=-" not in message

    def test_embed_endpoint_emits_ok_summary(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(main_module, "decode_base64_image", lambda b64: DUMMY_IMAGE)
        embedding = np.ones(512, dtype=np.float32)
        fake_model.analyze.return_value = _face_result(embedding, 0.9)

        with caplog.at_level(logging.INFO):
            response = client.post(
                "/api/v1/embed",
                json={"image": "ZmFrZQ=="},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 200
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert "endpoint=embed" in message
        assert "outcome=ok" in message

    def test_compare_photos_upload_emits_summary(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        embedding = np.ones(512, dtype=np.float32)
        fake_model.analyze.side_effect = [
            _face_result(embedding, 0.95),
            _face_result(embedding, 0.93),
        ]

        with caplog.at_level(logging.INFO):
            response = client.post(
                "/api/v1/compare-photos-upload",
                files={
                    "image1": ("first-secret.jpg", b"first-bytes", "image/jpeg"),
                    "image2": ("second-secret.jpg", b"second-bytes", "image/jpeg"),
                },
                data={"distance_metric": "cosine"},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 200
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert "endpoint=compare-photos-upload" in message
        assert "outcome=match" in message
        for needle in ("first-secret.jpg", "second-secret.jpg"):
            assert needle not in message


class TestBodyTooLargeSummary:
    """S3.6 fix 3: MaxBodySizeMiddleware's 400 IMAGE_TOO_LARGE burns a
    selfie try the same as any other 400 image error, but -- since it's a
    pure-ASGI middleware short-circuit that runs before any route code --
    it never went through a route's `finally: log_summary(...)`, so it was
    invisible to the one-summary-line-per-request observability guarantee.
    """

    def _summary_records(self, caplog: pytest.LogCaptureFixture) -> list:
        return [r for r in caplog.records if r.name == "face_recognition_service.summary"]

    def test_compare_photos_emits_summary(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        oversize = b"x" * (settings.max_request_body_bytes + 1)
        with caplog.at_level(logging.INFO):
            response = client.post(
                "/api/v1/compare-photos",
                data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
                files={"image2": ("selfie.jpg", oversize, "image/jpeg")},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert "endpoint=compare-photos-upload" not in message
        assert "endpoint=compare-photos" in message
        assert "outcome=error" in message
        assert f"error_code={ErrorCode.IMAGE_TOO_LARGE}" in message
        assert "image=-" in message

    def test_compare_photos_upload_emits_summary(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        oversize = b"x" * (settings.max_request_body_bytes + 1)
        with caplog.at_level(logging.INFO):
            response = client.post(
                "/api/v1/compare-photos-upload",
                files={
                    "image1": ("first.jpg", b"first-bytes", "image/jpeg"),
                    "image2": ("second.jpg", oversize, "image/jpeg"),
                },
                data={"distance_metric": "cosine"},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert "endpoint=compare-photos-upload" in message
        assert "outcome=error" in message
        assert f"error_code={ErrorCode.IMAGE_TOO_LARGE}" in message
        assert "image=-" in message

    def test_embed_emits_summary(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        oversize = "x" * (settings.max_request_body_bytes + 1)
        with caplog.at_level(logging.INFO):
            response = client.post(
                "/api/v1/embed",
                json={"image": oversize},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        records = self._summary_records(caplog)
        assert len(records) == 1
        message = records[0].message
        assert "endpoint=embed" in message
        assert "outcome=error" in message
        assert f"error_code={ErrorCode.IMAGE_TOO_LARGE}" in message
        assert "image=-" in message


class TestCatchAllOutsideExceptionMiddleware:
    """S3.6 fix 4: an exception FastAPI's own `@app.exception_handler(Exception)`
    never sees.

    Starlette pulls a handler registered for the bare `Exception` class out
    of `ExceptionMiddleware` and hands it to `ServerErrorMiddleware` instead
    (`key in (500, Exception)` in `Starlette.build_middleware_stack`) --
    which sits *outside* every `add_middleware` call, including
    `RequestIdMiddleware`. So when a route raises before
    `ExceptionMiddleware` (inside `RequestIdMiddleware`) ever gets a chance
    to run its own handler -- e.g. a dependency-free route like
    `model_info()` calling `get_model()` directly, with no route-level
    try/except of its own -- the exception unwinds straight past
    `RequestIdMiddleware`'s `try/finally` (which has already reset
    `request_id_var` by the time `ServerErrorMiddleware` catches it) and
    `ServerErrorMiddleware` responds using the *original* `send` it was
    given, not `RequestIdMiddleware`'s `send_with_request_id` wrapper -- so
    the response carries no `X-Request-ID`, and even the "Unhandled
    exception" log line for the SAME id is impossible.
    """

    def test_500_from_route_with_no_route_level_handler_carries_request_id(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _boom():
            raise RuntimeError("secret-internal-text")

        monkeypatch.setattr(main_module, "get_model", _boom)

        with caplog.at_level(logging.ERROR):
            with TestClient(main_module.app, raise_server_exceptions=False) as raw_client:
                response = raw_client.get(
                    "/api/v1/model-info",
                    headers={**AUTH_HEADERS, "X-Request-ID": "catch-all-test-id"},
                )

        assert response.status_code == 500
        body = response.json()
        assert body["error_code"] == ErrorCode.PROCESSING_ERROR
        assert "secret-internal-text" not in response.text

        request_id = response.headers.get("x-request-id")
        assert request_id == "catch-all-test-id"

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(r.exc_info is not None for r in error_records), (
            "expected the exception to be logged with exc_info"
        )
        assert any(getattr(r, "request_id", None) == request_id for r in error_records), (
            "expected the logged exception to carry the same request_id as the response"
        )

    def test_existing_response_in_flight_is_not_swallowed(
        self, client: TestClient
    ) -> None:
        # A normal request must be completely unaffected by wrapping the
        # inner app call in try/except -- no double-send, no swallowed
        # response.
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers.get("x-request-id") is not None


class TestDockerfileUsesModuleInvocation:
    def test_cmd_uses_python_module_invocation(self) -> None:
        dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
        text = dockerfile.read_text()
        assert 'CMD ["python", "-m", "face_recognition_service.main"]' in text
