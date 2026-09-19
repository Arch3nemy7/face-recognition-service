"""Observability primitives for Phase 3 robustness, Task 5 (S3.5).

Three pieces:

- A per-request ID (`request_id_var`, `RequestIdMiddleware`): every request
  gets a `X-Request-ID` -- either an inbound one that already matches the
  expected shape, or a freshly generated one -- carried through the request
  via a `ContextVar` so every log line emitted while handling it (including
  from a `run_in_threadpool`/`run_in_threadpool_shielded` worker, since both
  copy the current context when they start) can be tied back to it.
- `StageTimer`, a tiny millisecond stopwatch for the fetch/decode/queue/infer
  stages of a comparison.
- `log_summary`, which emits exactly one PII-free `key=value` log record per
  comparison request, success or failure.

Nothing here ever logs a URL, host, path, filename, embedding or image
bytes -- only IDs, codes, scores and durations.
"""

import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .schemas.api_schemas import ErrorCode, ErrorResponse

# Default "-" (never None) so the shared log format's [%(request_id)s]
# always has something to print, even for a log line emitted outside any
# request (e.g. during lifespan startup/shutdown).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Conservative charset for a header value: alphanumerics plus the three
# separators commonly used in request-id schemes (dot, underscore, hyphen).
# Anything else (or an empty/over-long value) is treated as absent and
# replaced with a freshly generated ID, rather than echoed back verbatim
# into a log line and a response header.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

summary_logger = logging.getLogger("face_recognition_service.summary")
_logger = logging.getLogger(__name__)

# Pre-serialised once, at import time: the generic 500 envelope
# RequestIdMiddleware sends for an exception that never reached FastAPI's own
# `@app.exception_handler(Exception)` (see RequestIdMiddleware's docstring for
# why that handler can't be relied on here). Same shape and wording as
# main.py's own catch-all handler -- never the real exception text, which may
# contain internals not meant for a client.
_INTERNAL_ERROR_BODY = json.dumps(
    ErrorResponse(
        error="Internal server error",
        error_code=ErrorCode.PROCESSING_ERROR,
    ).model_dump()
).encode("utf-8")


class RequestIdFilter(logging.Filter):
    """Attaches the current request's ID to every log record it handles, as
    `record.request_id`. Installed on the root logger's own handlers by
    `install_request_id_filter` -- see that function's docstring for why a
    `logging.setLogRecordFactory` hook is also installed alongside it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def install_request_id_filter() -> None:
    """Make `record.request_id` available on every log record, however it's
    captured.

    Call once, after `logging.basicConfig(...)` has created the root
    logger's handlers.

    A `logging.Filter` only runs for records that reach the handler it's
    attached to, so attaching `RequestIdFilter` to the root logger's own
    handlers (the `%(request_id)s` in the shared format string) covers
    normal application logging. It does *not* cover a handler attached
    later and independently of the root logger's handler list -- notably
    pytest's `caplog`, which installs its own handler. Rather than special-
    case every such handler, `logging.setLogRecordFactory` is used as the
    robust, handler-independent option: it wraps record *creation* itself,
    so `request_id` is guaranteed to exist on every `LogRecord` (app logger,
    uvicorn's own loggers, caplog's handler, ...) the moment it's created --
    never only after it happens to reach one particular handler. The two are
    kept together (belt and suspenders): the filter is cheap, explicit about
    intent on the handlers it's attached to, and harmless to keep even
    though the record factory alone already satisfies the requirement.
    """
    factory = logging.getLogRecordFactory()

    def _request_id_record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = factory(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(_request_id_record_factory)

    request_id_filter = RequestIdFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(request_id_filter)


class RequestIdMiddleware:
    """Pure-ASGI middleware assigning every HTTP request a request ID.

    Must be added last (`app.add_middleware(RequestIdMiddleware)` after
    every other `add_middleware` call) so Starlette's insert-at-front
    semantics make it the *outermost* middleware -- otherwise a response
    substituted by an inner middleware (e.g. `MaxBodySizeMiddleware`'s 400
    IMAGE_TOO_LARGE for an oversize body) would never get the header added
    by an outer one.

    An inbound `X-Request-ID` matching `_VALID_REQUEST_ID` is echoed back
    unchanged; anything else (absent, malformed, oversized) gets a fresh
    `uuid4().hex`. Either way the ID is: (1) stored in `request_id_var` for
    the lifetime of the request, so it propagates into log records emitted
    anywhere while handling it, and (2) added as the `X-Request-ID` response
    header by wrapping `send` -- not by mutating a `Response` object, since
    this must also cover the raw `http.response.start` messages
    `MaxBodySizeMiddleware` sends directly.

    Also the last line of defence for an exception that never reaches
    FastAPI's own `@app.exception_handler(Exception)`. Starlette pulls a
    handler registered for the bare `Exception` class out of
    `ExceptionMiddleware` and hands it to `ServerErrorMiddleware` instead
    (`Starlette.build_middleware_stack`: `if key in (500, Exception):
    error_handler = value`) -- and `ServerErrorMiddleware` sits *outside*
    every `add_middleware` call, this one included. A route with no
    dependency/middleware of its own to catch a bug (e.g. `model_info()`
    calling `get_model()` directly) can therefore raise all the way past
    this middleware's `try/finally` before `ServerErrorMiddleware` ever
    catches it -- by which point `request_id_var` has already been reset and
    `ServerErrorMiddleware` responds using the plain `send` it was given,
    not `send_with_request_id`, so the response would carry no
    `X-Request-ID` and the exception would log under `[-]`. Catching it here
    instead (while the ID is still live) and answering with the same
    PROCESSING_ERROR envelope `unhandled_exception_handler` in main.py uses
    keeps both consistent regardless of which layer actually catches a given
    bug. If a response has already started going out, it's too late to
    substitute a different one (a second `http.response.start` would violate
    the ASGI protocol) -- re-raise so `ServerErrorMiddleware` can still log
    it, same as before this fix.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = None
        for name, value in scope.get("headers") or []:
            if name == b"x-request-id":
                inbound = value.decode("latin-1")
                break

        if inbound and _VALID_REQUEST_ID.match(inbound):
            request_id = inbound
        else:
            request_id = uuid.uuid4().hex

        token = request_id_var.set(request_id)
        encoded_request_id = request_id.encode("latin-1")
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", encoded_request_id))
            await send(message)

        try:
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception:
                if response_started:
                    raise
                # No path/method here (see this module's own "never log a
                # URL, host, path, filename" rule) -- exc_info carries the
                # real traceback for debugging, and request_id (attached to
                # every record via install_request_id_filter) ties it back
                # to this response.
                _logger.exception("Unhandled exception outside route-level exception handling")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"x-request-id", encoded_request_id),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": _INTERNAL_ERROR_BODY})
        finally:
            request_id_var.reset(token)


class StageTimer:
    """Millisecond stopwatch for the named stages of one request.

    `with timer.stage("fetch"): ...` times a block and *adds* its duration
    to that stage's running total (rather than overwriting it), so a stage
    entered more than once per request -- e.g. `infer` covering two
    `_analyze` calls on /compare-photos -- accumulates correctly.
    `timer.record(name, ms)` sets/accumulates a duration measured elsewhere
    (queue_ms is measured around `gate.slot()`, not inside a `with` block,
    since it must still be captured when acquiring the slot itself raises
    `ServiceBusyError`).
    """

    def __init__(self) -> None:
        self._ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.record(name, (time.monotonic() - start) * 1000)

    def record(self, name: str, ms: float) -> None:
        self._ms[name] = self._ms.get(name, 0.0) + ms

    def get(self, name: str) -> Optional[float]:
        return self._ms.get(name)


def _fmt_ms(value: Optional[float]) -> str:
    return "-" if value is None else str(int(round(value)))


def _fmt_score(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def log_summary(
    *,
    endpoint: str,
    outcome: str,
    error_code: Optional[str] = None,
    image: Optional[str] = None,
    match: Optional[bool] = None,
    cosine_similarity: Optional[float] = None,
    threshold: Optional[float] = None,
    det1: Optional[float] = None,
    det2: Optional[float] = None,
    timer: Optional[StageTimer] = None,
    q1_face_px: Optional[float] = None,
    q2_face_px: Optional[float] = None,
    q2_blur: Optional[float] = None,
) -> None:
    """Emit the one PII-free summary log record for a comparison request.

    Never pass (or add) a URL, host, path, filename, embedding or image
    bytes here -- only IDs, codes, scores and durations. `request_id` itself
    isn't a field below because it's already attached to every record (this
    one included) via `RequestIdFilter`/the record factory and printed by
    the shared log format.
    """
    fetch_ms = timer.get("fetch") if timer else None
    decode_ms = timer.get("decode") if timer else None
    queue_ms = timer.get("queue") if timer else None
    infer_ms = timer.get("infer") if timer else None
    total_ms = timer.get("total") if timer else None

    fields = {
        "endpoint": endpoint,
        "outcome": outcome,
        "error_code": error_code or "-",
        "image": image or "-",
        "match": "-" if match is None else str(match).lower(),
        "cosine_similarity": _fmt_score(cosine_similarity),
        "threshold": _fmt_score(threshold),
        "det1": _fmt_score(det1),
        "det2": _fmt_score(det2),
        "fetch_ms": _fmt_ms(fetch_ms),
        "decode_ms": _fmt_ms(decode_ms),
        "queue_ms": _fmt_ms(queue_ms),
        "infer_ms": _fmt_ms(infer_ms),
        "total_ms": _fmt_ms(total_ms),
        "q1_face_px": _fmt_score(q1_face_px),
        "q2_face_px": _fmt_score(q2_face_px),
        "q2_blur": _fmt_score(q2_blur),
    }
    summary_logger.info(" ".join(f"{key}={value}" for key, value in fields.items()))
