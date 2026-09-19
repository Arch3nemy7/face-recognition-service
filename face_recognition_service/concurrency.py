"""Bounded concurrency for inference (Phase 3 robustness, Task 3, S3.3).

`model.analyze` is CPU-bound and runs in a threadpool (main.py's
`_analyze`, via `starlette.concurrency.run_in_threadpool`) so it never blocks
the event loop. That alone isn't enough on a single-worker deployment: an
unbounded number of concurrent requests would still pile up threads and
starve the process. `InferenceGate` caps how many `analyze` calls run at
once and fails fast -- 503 SERVICE_BUSY -- instead of queuing forever when
saturated.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, TypeVar

from starlette.concurrency import run_in_threadpool

from .config import settings
from .errors import FaceServiceError
from .schemas.api_schemas import ErrorCode

T = TypeVar("T")


class ServiceBusyError(FaceServiceError):
    """Raised by `InferenceGate.slot()` when no slot frees up within its
    timeout. The service is up but saturated -- distinct from
    SERVICE_UNAVAILABLE (a downstream fetch failure)."""

    def __init__(self) -> None:
        super().__init__(
            "Service is busy processing other requests; please retry shortly",
            ErrorCode.SERVICE_BUSY,
            image=None,
            headers={"Retry-After": str(settings.busy_retry_after_seconds)},
        )


async def run_in_threadpool_shielded(func: Callable[..., T], *args: object) -> T:
    """Run `func(*args)` in the threadpool, guaranteeing the worker thread
    finishes before this call can be reported as cancelled to its caller.

    Use this (never a bare `run_in_threadpool`) for every threadpool call
    made while holding an `InferenceGate` slot -- otherwise a cancelled
    request can free the slot while its thread is still running, letting a
    second request start inference concurrently past the configured limit.

    Why plain `run_in_threadpool` isn't enough: a native
    `asyncio.Task.cancel()` (as opposed to an anyio-native cancel-scope
    cancellation) delivers `CancelledError` into the awaiting coroutine
    immediately, regardless of anyio's own `CancelScope(shield=True)` --
    verified empirically (see task-3-report.md, "Fix round 1"): wrapping a
    bare `await run_in_threadpool(...)` in `with anyio.CancelScope(shield=
    True):` did *not* delay the CancelledError past the background thread's
    completion. `asyncio.shield`, wrapping a real `asyncio.Task`, does: a
    cancelled *caller* only cancels its own await of the shield, never the
    wrapped task, so the `finally` below can re-await that same task and
    block until the thread genuinely finishes -- before letting
    `CancelledError` propagate any further, e.g. into
    `InferenceGate.slot()`'s own cleanup, so the gate is never released
    while a thread it's gating is still running.
    """
    task = asyncio.ensure_future(run_in_threadpool(func, *args))
    try:
        return await asyncio.shield(task)
    finally:
        if not task.done():
            await asyncio.shield(task)


class InferenceGate:
    """Bounds how many inference calls run concurrently.

    Wraps an `asyncio.Semaphore` behind a timed acquire, so a saturated
    service returns 503 SERVICE_BUSY instead of queuing a request
    indefinitely. Must be constructed inside a running event loop (e.g. in
    `lifespan`, or lazily on first use) -- never at import time, since
    `asyncio.Semaphore` binds to the loop that creates it.
    """

    def __init__(self, limit: int, timeout: float) -> None:
        self._semaphore = asyncio.Semaphore(limit)
        self._timeout = timeout

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Async context manager: holds one of the gate's slots for its
        duration.

        Raises:
            ServiceBusyError: no slot became free within `timeout` seconds.
        """
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise ServiceBusyError() from None
        try:
            yield
        finally:
            self._semaphore.release()
