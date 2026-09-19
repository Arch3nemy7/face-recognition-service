"""Inference off the event loop behind a bounded queue (Phase 3 robustness,
Task 3, S3.3).

Fetch/decode/inference run in the threadpool (`run_in_threadpool`), and only
a limited number of `model.analyze` calls run at once
(`concurrency.InferenceGate`). This suite proves:

  - the event loop stays responsive while a request is blocked in a fetch or
    an analyze call (health check keeps answering);
  - the gate is never held during the fetch, only around `analyze`;
  - a saturated gate times out into 503 SERVICE_BUSY with a Retry-After
    header, not a hang;
  - the gate is released after an error, so the next request isn't starved;
  - /embed is gated too;
  - reference errors still get reported (and tagged) before selfie errors;
  - `InferenceGate` itself admits up to its limit and times out beyond it;
  - a cancelled request's permit is not released until its threadpool work
    actually finishes, even under a native `asyncio.Task.cancel()` (Fix
    round 1 -- see task-3-report.md).

Offline only, same style as tests/test_error_contract.py and
tests/test_uploads.py: the model singleton and `fetch_image_from_url` are
stubbed so no real InsightFace or network call ever happens. Every blocking
wait uses `threading.Event` with a <=5s safety timeout so a bug fails fast
instead of hanging the suite; blocking side_effects are released in
`finally`.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import face_recognition_service.main as main_module
from face_recognition_service.concurrency import (
    InferenceGate,
    ServiceBusyError,
    run_in_threadpool_shielded,
)
from face_recognition_service.config import settings
from face_recognition_service.models import face_model as face_model_module
from face_recognition_service.models.face_model import FaceModelError
from face_recognition_service.schemas.api_schemas import ErrorCode
from tests.conftest import _face_result

AUTH_HEADERS = {"Authorization": f"Bearer {settings.api_token}"}
DUMMY_IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)
WAIT_TIMEOUT = 5.0

# A 1x1 PNG, base64-encoded, for /embed.
TINY_PNG_B64 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
        "53de0000000c4944415478da6360000002000155ec52720000000049454e44"
        "ae426082"
    )
).decode("ascii")


def _compare_photos_files() -> dict:
    return {"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")}


def _compare_photos_data() -> dict:
    return {"image1": "https://example.com/ref.jpg", "distance_metric": "cosine"}


@pytest.fixture
def busy_client(monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock):
    """Like conftest's `client`, but with a tiny inference queue timeout so
    saturation tests don't need to wait 20s for the default to expire."""
    monkeypatch.setattr(settings, "max_concurrent_inference", 1)
    monkeypatch.setattr(settings, "inference_queue_timeout", 0.2)
    monkeypatch.setattr(settings, "busy_retry_after_seconds", 5)

    def _fake_initialize_model() -> None:
        face_model_module._model_instance = fake_model

    monkeypatch.setattr(main_module, "initialize_model", _fake_initialize_model)
    monkeypatch.setattr(main_module, "cleanup_model", lambda: None)
    monkeypatch.setattr(main_module, "preprocess_image", lambda image: image)

    with TestClient(main_module.app) as test_client:
        yield test_client

    face_model_module._model_instance = None


class TestEventLoopStaysResponsive:
    """1. Health check keeps answering while /compare-photos is blocked in a
    patched fetch (proves the fetch runs off the event loop)."""

    def test_health_check_responsive_during_blocked_fetch(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        def _blocking_fetch(url: str):
            started.set()
            if not release.wait(timeout=WAIT_TIMEOUT):
                raise TimeoutError("test never released the blocked fetch")
            return DUMMY_IMAGE

        monkeypatch.setattr(main_module, "fetch_image_from_url", _blocking_fetch)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        fake_model.analyze.return_value = _face_result(np.ones(512, dtype=np.float32), 0.9)

        result: dict = {}

        def _do_request():
            result["response"] = client.post(
                "/api/v1/compare-photos",
                data=_compare_photos_data(),
                files=_compare_photos_files(),
                headers=AUTH_HEADERS,
            )

        thread = threading.Thread(target=_do_request)
        try:
            thread.start()
            assert started.wait(timeout=WAIT_TIMEOUT), "fetch never started"

            start = time.monotonic()
            health_response = client.get("/api/v1/health")
            elapsed = time.monotonic() - start

            assert health_response.status_code == 200
            assert elapsed < 1.0
        finally:
            release.set()
            thread.join(timeout=WAIT_TIMEOUT)
            assert not thread.is_alive(), "background request thread never finished"

        assert result["response"].status_code == 200


class TestGateSaturation:
    """2. A saturated gate times out into 503 SERVICE_BUSY with Retry-After,
    not a hang."""

    def test_saturated_gate_returns_503_service_busy(
        self, busy_client: TestClient, fake_model: MagicMock
    ) -> None:
        entered = threading.Event()
        release = threading.Event()

        def _blocking_analyze(image, role):
            entered.set()
            if not release.wait(timeout=WAIT_TIMEOUT):
                raise TimeoutError("test never released the blocked analyze")
            return _face_result(np.ones(512, dtype=np.float32), 0.9)

        fake_model.analyze.side_effect = _blocking_analyze

        result: dict = {}

        def _do_request_a():
            result["a"] = busy_client.post(
                "/api/v1/embed",
                json={"image": TINY_PNG_B64},
                headers=AUTH_HEADERS,
            )

        thread = threading.Thread(target=_do_request_a)
        try:
            thread.start()
            assert entered.wait(timeout=WAIT_TIMEOUT), "request A never entered analyze"

            # B can't get a slot within the (monkeypatched, 0.2s) timeout.
            response_b = busy_client.post(
                "/api/v1/embed",
                json={"image": TINY_PNG_B64},
                headers=AUTH_HEADERS,
            )
        finally:
            release.set()
            thread.join(timeout=WAIT_TIMEOUT)
            assert not thread.is_alive(), "background request thread never finished"

        assert response_b.status_code == 503
        body = response_b.json()
        assert body["error_code"] == "SERVICE_BUSY"
        assert body["image"] is None
        assert response_b.headers.get("Retry-After") == "5"
        assert result["a"].status_code == 200


class TestComparePhotosSaturation:
    """3. A saturated gate returns 503 SERVICE_BUSY on /compare-photos too
    (not just /embed): A blocks inside analyze via /compare-photos, B's own
    /compare-photos call can't get a slot within the timeout."""

    def test_compare_photos_saturated_returns_503_service_busy(
        self, busy_client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)

        entered = threading.Event()
        release = threading.Event()

        def _blocking_analyze(image, role):
            entered.set()
            if not release.wait(timeout=WAIT_TIMEOUT):
                raise TimeoutError("test never released the blocked analyze")
            return _face_result(np.ones(512, dtype=np.float32), 0.9)

        fake_model.analyze.side_effect = _blocking_analyze

        result: dict = {}

        def _do_request_a():
            result["a"] = busy_client.post(
                "/api/v1/compare-photos",
                data=_compare_photos_data(),
                files=_compare_photos_files(),
                headers=AUTH_HEADERS,
            )

        thread = threading.Thread(target=_do_request_a)
        try:
            thread.start()
            assert entered.wait(timeout=WAIT_TIMEOUT), "request A never entered analyze"

            # B can't get a slot within the (monkeypatched, 0.2s) timeout.
            response_b = busy_client.post(
                "/api/v1/compare-photos",
                data=_compare_photos_data(),
                files=_compare_photos_files(),
                headers=AUTH_HEADERS,
            )
        finally:
            release.set()
            thread.join(timeout=WAIT_TIMEOUT)
            assert not thread.is_alive(), "background request thread never finished"

        assert response_b.status_code == 503
        body = response_b.json()
        assert body["error_code"] == "SERVICE_BUSY"
        assert body["image"] is None
        assert response_b.headers.get("Retry-After") == "5"
        assert result["a"].status_code == 200


class TestGateNotHeldDuringFetch:
    """4. The gate is never held during the fetch: while A is blocked in
    fetch_image_from_url (before it ever reaches the gate), B's analysis
    still completes."""

    def test_fetch_does_not_hold_the_gate(
        self, busy_client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        def _blocking_fetch(url: str):
            started.set()
            if not release.wait(timeout=WAIT_TIMEOUT):
                raise TimeoutError("test never released the blocked fetch")
            return DUMMY_IMAGE

        monkeypatch.setattr(main_module, "fetch_image_from_url", _blocking_fetch)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)
        fake_model.analyze.return_value = _face_result(np.ones(512, dtype=np.float32), 0.9)

        result: dict = {}

        def _do_request_a():
            result["a"] = busy_client.post(
                "/api/v1/compare-photos",
                data=_compare_photos_data(),
                files=_compare_photos_files(),
                headers=AUTH_HEADERS,
            )

        thread = threading.Thread(target=_do_request_a)
        try:
            thread.start()
            assert started.wait(timeout=WAIT_TIMEOUT), "fetch never started"

            # B (a plain /embed call) should complete fine -- the gate is
            # free because A hasn't reached it yet.
            response_b = busy_client.post(
                "/api/v1/embed",
                json={"image": TINY_PNG_B64},
                headers=AUTH_HEADERS,
            )
            assert response_b.status_code == 200
        finally:
            release.set()
            thread.join(timeout=WAIT_TIMEOUT)
            assert not thread.is_alive(), "background request thread never finished"

        assert result["a"].status_code == 200


class TestGateReleasedAfterError:
    """5. The gate is released after an error: A's analyze raises
    NO_FACE_DETECTED, then B still succeeds (sequential is enough to prove
    release-on-exception; no threading needed). Uses `busy_client` (0.2s
    queue timeout) rather than the default 20s, so a leaked permit fails
    this test in ~0.2s instead of 20s."""

    def test_gate_released_after_analyze_error(
        self, busy_client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: DUMMY_IMAGE)

        fake_model.analyze.side_effect = FaceModelError("no face", ErrorCode.NO_FACE_DETECTED)
        response_a = busy_client.post(
            "/api/v1/compare-photos",
            data=_compare_photos_data(),
            files=_compare_photos_files(),
            headers=AUTH_HEADERS,
        )
        assert response_a.status_code == 400
        assert response_a.json()["error_code"] == "NO_FACE_DETECTED"

        fake_model.analyze.side_effect = None
        fake_model.analyze.return_value = _face_result(np.ones(512, dtype=np.float32), 0.9)
        response_b = busy_client.post(
            "/api/v1/compare-photos",
            data=_compare_photos_data(),
            files=_compare_photos_files(),
            headers=AUTH_HEADERS,
        )
        assert response_b.status_code == 200, response_b.json()


class TestEmbedIsGated:
    """6. /embed is gated too -- the saturated case returns 503 SERVICE_BUSY."""

    def test_embed_saturated_returns_503(self, busy_client: TestClient, fake_model: MagicMock) -> None:
        entered = threading.Event()
        release = threading.Event()

        def _blocking_analyze(image, role):
            entered.set()
            if not release.wait(timeout=WAIT_TIMEOUT):
                raise TimeoutError("test never released the blocked analyze")
            return _face_result(np.ones(512, dtype=np.float32), 0.9)

        fake_model.analyze.side_effect = _blocking_analyze

        def _do_request_a():
            busy_client.post("/api/v1/embed", json={"image": TINY_PNG_B64}, headers=AUTH_HEADERS)

        thread = threading.Thread(target=_do_request_a)
        try:
            thread.start()
            assert entered.wait(timeout=WAIT_TIMEOUT), "request A never entered analyze"

            response_b = busy_client.post(
                "/api/v1/embed", json={"image": TINY_PNG_B64}, headers=AUTH_HEADERS
            )
        finally:
            release.set()
            thread.join(timeout=WAIT_TIMEOUT)
            assert not thread.is_alive(), "background request thread never finished"

        assert response_b.status_code == 503
        assert response_b.json()["error_code"] == "SERVICE_BUSY"


class TestReferencePrecedesSelfie:
    """7. Reference errors still get reported (and tagged) before selfie
    errors: reference no-face plus an oversize selfie -> tagged reference,
    because the selfie is never even read once the reference's analyze has
    already failed."""

    def test_reference_error_wins_over_oversize_selfie(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock
    ) -> None:
        monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: DUMMY_IMAGE)
        monkeypatch.setattr(settings, "max_image_size", 1)
        fake_model.analyze.side_effect = FaceModelError("no face", ErrorCode.NO_FACE_DETECTED)

        response = client.post(
            "/api/v1/compare-photos",
            data=_compare_photos_data(),
            files={"image2": ("selfie.jpg", b"way-too-many-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == "NO_FACE_DETECTED"
        assert body["image"] == "reference"


class TestInferenceGateUnit:
    """8. InferenceGate on its own: limit 2 admits two holders; the third
    times out into ServiceBusyError."""

    @pytest.mark.asyncio
    async def test_limit_two_admits_two_then_third_times_out(self) -> None:
        gate = InferenceGate(limit=2, timeout=0.2)

        async with gate.slot():
            async with gate.slot():
                with pytest.raises(ServiceBusyError):
                    async with gate.slot():
                        pass  # pragma: no cover -- should never be reached

    @pytest.mark.asyncio
    async def test_slot_released_after_use(self) -> None:
        gate = InferenceGate(limit=1, timeout=0.2)

        async with gate.slot():
            pass

        # The slot from above was released, so this should not time out.
        async with gate.slot():
            pass

    @pytest.mark.asyncio
    async def test_slot_released_after_exception(self) -> None:
        gate = InferenceGate(limit=1, timeout=0.2)

        with pytest.raises(ValueError):
            async with gate.slot():
                raise ValueError("boom")

        async with gate.slot():
            pass


class TestGateSurvivesNativeTaskCancellation:
    """9. Fix round 1 regression test: a cancelled request's permit must not
    be released until its threadpool work actually finishes -- otherwise a
    second request can start inference concurrently past the gate's limit.

    Reviewer's `gate_race.py` stress test found 3 concurrent inference
    threads with `limit=1` after cancelling request tasks. The cause: a
    plain `await run_in_threadpool(...)` made while holding a gate slot, even
    wrapped directly in `with anyio.CancelScope(shield=True):`, does NOT
    delay a native `asyncio.Task.cancel()`'s `CancelledError` past the
    background thread's completion -- verified with a standalone repro
    (`anyio.CancelScope(shield=True)` around a bare `run_in_threadpool`
    still let `CancelledError` surface in ~0ms while the thread kept
    running). `asyncio.shield` around a real `asyncio.Task`, re-awaited in a
    `finally` (see `concurrency.run_in_threadpool_shielded`), does protect
    it: cancelling the *caller* only cancels its own await of the shield,
    never the wrapped task, so the permit-holding code can block on that
    same task until the thread genuinely finishes before letting
    `CancelledError` propagate into `InferenceGate.slot()`'s cleanup.

    This test exercises `InferenceGate` + `run_in_threadpool_shielded`
    directly (no TestClient/HTTP layer -- cancelling a request task cleanly
    through TestClient's portal isn't straightforward, and isn't needed to
    prove the gate's own contract)."""

    @pytest.mark.asyncio
    async def test_cancelled_request_blocks_next_until_thread_finishes(self) -> None:
        gate = InferenceGate(limit=1, timeout=WAIT_TIMEOUT)

        entered_a = threading.Event()
        release_a = threading.Event()
        finished_a = threading.Event()
        entered_b = threading.Event()

        def _blocking_a() -> str:
            entered_a.set()
            if not release_a.wait(timeout=WAIT_TIMEOUT):
                raise TimeoutError("test never released request A's thread")
            finished_a.set()
            return "a-done"

        def _blocking_b() -> str:
            entered_b.set()
            return "b-done"

        async def _request_a() -> None:
            async with gate.slot():
                await run_in_threadpool_shielded(_blocking_a)

        async def _request_b() -> str:
            async with gate.slot():
                return await run_in_threadpool_shielded(_blocking_b)

        task_a = asyncio.ensure_future(_request_a())
        try:
            for _ in range(500):
                if entered_a.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered_a.is_set(), "request A's thread never started"

            # Cancel A's task mid-analyze -- its thread is still running.
            task_a.cancel()

            task_b = asyncio.ensure_future(_request_b())
            try:
                # B must NOT be able to start its thread yet: A's slot is
                # still held because A's (cancelled) thread hasn't finished.
                await asyncio.sleep(0.1)
                assert not entered_b.is_set(), (
                    "request B started its thread before request A's "
                    "(cancelled) thread finished -- the gate leaked a permit"
                )

                release_a.set()

                with pytest.raises(asyncio.CancelledError):
                    await task_a
                assert finished_a.is_set(), "A's thread should have finished before A's task resolved"

                result_b = await asyncio.wait_for(task_b, timeout=WAIT_TIMEOUT)
                assert result_b == "b-done"
                assert entered_b.is_set()
            finally:
                release_a.set()
                if not task_b.done():
                    task_b.cancel()
        finally:
            release_a.set()
            if not task_a.done():
                task_a.cancel()
