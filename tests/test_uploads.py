"""Bounded uploads, no base64 round trip (Phase 3 robustness, Task 2, S3.2).

Covers `image_utils.decode_image_bytes` directly, `main._read_upload`'s size
check on the upload-path routes (/compare-photos' image2, both images on
/compare-photos-upload), and `MaxBodySizeMiddleware`'s whole-request-body
cap -- both the declared-Content-Length short-circuit and the streamed/
chunked fallback. Offline only: the model singleton, `fetch_image_from_url`
and `decode_image_bytes` are stubbed via the `client`/`fake_model` fixtures
in tests/conftest.py, same as tests/test_error_contract.py.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import face_recognition_service.main as main_module
from face_recognition_service.config import settings
from face_recognition_service.schemas.api_schemas import ErrorCode
from face_recognition_service.utils.image_utils import (
    ImageProcessingError,
    decode_image_bytes,
)
from tests.conftest import _face_result

AUTH_HEADERS = {"Authorization": f"Bearer {settings.api_token}"}
DUMMY_IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)


def _ok_reference(monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock) -> None:
    """Make the reference (image1) side of /compare-photos succeed, so a
    selfie-side assertion isn't muddied by a reference failure."""
    monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
    fake_model.analyze.return_value = _face_result(np.ones(512, dtype=np.float32), 0.9)


class TestDecodeImageBytesUnit:
    """decode_image_bytes: the size/empty checks that used to live only in
    decode_base64_image, now reachable without a base64 round trip."""

    def test_empty_bytes_is_invalid_image(self) -> None:
        with pytest.raises(ImageProcessingError) as exc_info:
            decode_image_bytes(b"")
        assert exc_info.value.error_code == ErrorCode.INVALID_IMAGE

    def test_too_many_bytes_is_image_too_large(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "max_image_size", 10)
        with pytest.raises(ImageProcessingError) as exc_info:
            decode_image_bytes(b"x" * 11)
        assert exc_info.value.error_code == ErrorCode.IMAGE_TOO_LARGE


class TestComparePhotosSelfieUploadSize:
    """/compare-photos: image2 (selfie) is read straight off the multipart
    part -- _read_upload -- with no base64 round trip."""

    def test_oversize_selfie_is_400_tagged_selfie_reference_processed_first(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(settings, "max_image_size", 1000)
        _ok_reference(monkeypatch, fake_model)

        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"s" * 1001, "image/jpeg")},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        assert body["image"] == "selfie"
        # Reference is processed (and embedded) before the selfie is even
        # read -- reference-first order is preserved -- so analyze() was
        # called exactly once, for the reference.
        assert fake_model.analyze.call_count == 1
        assert fake_model.analyze.call_args.kwargs.get("role") == "reference"

    def test_selfie_at_exactly_the_limit_reaches_decode_image_bytes(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(settings, "max_image_size", 1000)
        _ok_reference(monkeypatch, fake_model)

        selfie_bytes = b"s" * 1000
        seen: list[bytes] = []

        def _decode(data: bytes) -> np.ndarray:
            seen.append(data)
            return DUMMY_IMAGE

        monkeypatch.setattr(main_module, "decode_image_bytes", _decode)

        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", selfie_bytes, "image/jpeg")},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 200
        assert seen == [selfie_bytes]

    def test_decode_base64_image_is_not_used_on_compare_photos(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        _ok_reference(monkeypatch, fake_model)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)

        def _boom(b64: str) -> np.ndarray:
            raise AssertionError("decode_base64_image must not be called by /compare-photos")

        monkeypatch.setattr(main_module, "decode_base64_image", _boom)

        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 200


class TestComparePhotosUploadSize:
    """/compare-photos-upload: both images are read via _read_upload."""

    def test_oversize_image1_is_400_tagged_reference(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(settings, "max_image_size", 1000)

        response = client.post(
            "/api/v1/compare-photos-upload",
            files={
                "image1": ("first.jpg", b"r" * 1001, "image/jpeg"),
                "image2": ("second.jpg", b"s" * 10, "image/jpeg"),
            },
            data={"distance_metric": "cosine"},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        assert body["image"] == "reference"
        fake_model.analyze.assert_not_called()

    def test_decode_base64_image_is_not_used_on_compare_photos_upload(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        fake_model.analyze.return_value = _face_result(np.ones(512, dtype=np.float32), 0.9)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)

        def _boom(b64: str) -> np.ndarray:
            raise AssertionError("decode_base64_image must not be called by /compare-photos-upload")

        monkeypatch.setattr(main_module, "decode_base64_image", _boom)

        response = client.post(
            "/api/v1/compare-photos-upload",
            files={
                "image1": ("first.jpg", b"first-bytes", "image/jpeg"),
                "image2": ("second.jpg", b"second-bytes", "image/jpeg"),
            },
            data={"distance_metric": "cosine"},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 200


class TestMaxRequestBodyMiddleware:
    """MaxBodySizeMiddleware: the whole-request-body cap, independent of any
    single image's own limit."""

    def test_declared_content_length_over_limit_is_400_before_the_route_runs(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "max_request_body_bytes", 100)

        def _unreachable(url: str) -> np.ndarray:
            raise AssertionError("the route must never run for an oversize declared body")

        monkeypatch.setattr(main_module, "fetch_image_from_url", _unreachable)

        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"s" * 500, "image/jpeg")},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        body = response.json()
        assert set(body.keys()) == {"error", "detail", "error_code", "image"}
        assert body["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        assert body["image"] is None

    def test_chunked_body_without_content_length_over_limit_is_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end via TestClient/httpx: a `content=` generator produces a
        request with no Content-Length header, exercising the declared-length
        short-circuit's absence (it's skipped) and falling through to the
        streaming counter. Note httpx's ASGI transport drains the generator
        and hands it to the app as a single `http.request` message, so this
        does not by itself prove the counter accumulates *across* multiple
        receive() calls -- see the raw-ASGI test below for that."""
        monkeypatch.setattr(settings, "max_request_body_bytes", 500)

        def _generate_body():
            chunk = b"a" * 200
            for _ in range(10):
                yield chunk

        response = client.post(
            "/api/v1/compare-photos-upload",
            content=_generate_body(),
            headers={
                **AUTH_HEADERS,
                "Content-Type": "multipart/form-data; boundary=doesnotmatter",
            },
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        assert body["image"] is None

    def test_streamed_body_crosses_limit_across_many_asgi_messages(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        """Drive `main.app` directly at the raw ASGI level with a *valid*
        multipart body split into many small `http.request` messages
        (more_body=True until the last), and no Content-Length header at
        all -- the case TestClient's transport can't produce (see the test
        above). Proves `limited_receive`'s byte counter actually accumulates
        across messages rather than only ever seeing one big chunk: the
        limit is set well below the full body, so the overflow can only be
        hit after several messages have already been counted.

        Reuses the `client` fixture (entered, not called) purely to run its
        lifespan -- installing `fake_model` as the model singleton via
        `initialize_model`/`face_model_module._model_instance` -- since
        `main.app` is invoked directly here, bypassing TestClient/httpx
        entirely.
        """
        _ok_reference(monkeypatch, fake_model)

        # A real multipart body, built the same way TestClient would, so the
        # only thing under test is the chunking/streaming behaviour -- not
        # whether malformed input trips something else first.
        built = client.build_request(
            "POST",
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"s" * 2000, "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        body = built.read()
        content_type = built.headers["content-type"]
        assert len(body) > 1000  # sanity: big enough to cross a 500-byte limit over several chunks

        monkeypatch.setattr(settings, "max_request_body_bytes", 500)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/compare-photos",
            "raw_path": b"/api/v1/compare-photos",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", content_type.encode("latin-1")),
                (b"authorization", AUTH_HEADERS["Authorization"].encode("latin-1")),
            ],
            "client": ("testclient", 12345),
            "server": ("testserver", 80),
        }

        chunk_size = 64
        remaining = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
        assert len(remaining) > 500 // chunk_size + 1  # crosses the limit only after several messages

        async def receive():
            if remaining:
                chunk = remaining.pop(0)
                return {"type": "http.request", "body": chunk, "more_body": bool(remaining)}
            return {"type": "http.disconnect"}

        sent: list[dict] = []

        async def send(message: dict) -> None:
            sent.append(message)

        asyncio.run(main_module.app(scope, receive, send))

        assert len(sent) == 2
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 400
        assert sent[1]["type"] == "http.response.body"
        payload = json.loads(sent[1]["body"])
        assert payload["error_code"] == ErrorCode.IMAGE_TOO_LARGE
        assert payload["image"] is None

    def test_body_under_the_limit_is_untouched(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        _ok_reference(monkeypatch, fake_model)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)

        # Build the request first so the limit can be pinned right at this
        # request's actual (declared) Content-Length -- a limit generously
        # above the request would pass trivially and not exercise the
        # `declared_length > limit` boundary at all.
        request = client.build_request(
            "POST",
            "/api/v1/compare-photos",
            data={"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        content_length = int(request.headers["content-length"])
        monkeypatch.setattr(settings, "max_request_body_bytes", content_length)

        response = client.send(request)

        assert response.status_code == 200
