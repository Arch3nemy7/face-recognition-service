"""FastAPI application for face recognition microservice."""

import json
import logging
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import verify_token
from .concurrency import InferenceGate, ServiceBusyError, run_in_threadpool_shielded
from .config import settings
from .errors import FaceServiceError
from .models.face_model import (
    FaceModelError,
    FaceQuality,
    cleanup_model,
    get_model,
    initialize_model,
)
from .observability import (
    RequestIdMiddleware,
    StageTimer,
    install_request_id_filter,
    log_summary,
)
from .schemas.api_schemas import (
    ComparePhotosResponse,
    CompareRequest,
    CompareResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorCode,
    ErrorResponse,
    FaceQualityResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .utils.embedding_utils import (
    calculate_distance,
    cosine_similarity,
    distance_to_similarity,
    find_best_match,
    match_threshold,
)
from .utils.image_utils import (
    ImageProcessingError,
    decode_base64_image,
    decode_image_bytes,
    fetch_image_from_url,
    preprocess_image,
)

# Configure logging. request_id is always present on every record (see
# observability.install_request_id_filter), so it can sit directly in the
# shared format string without risking a KeyError for a record logged
# outside any request.
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s",
)
install_request_id_filter()
logger = logging.getLogger(__name__)


def _too_large_envelope_body() -> bytes:
    """The same {error, detail, error_code, image} envelope the FaceServiceError
    handler produces for IMAGE_TOO_LARGE, pre-serialised for use before the
    app (and its handlers) ever runs."""
    return json.dumps(
        ErrorResponse(
            error="Request body exceeds the maximum allowed size",
            error_code=ErrorCode.IMAGE_TOO_LARGE,
        ).model_dump()
    ).encode("utf-8")


# Only routes that actually accept image bytes burn a selfie/reference try
# on an oversize body; MaxBodySizeMiddleware runs ahead of routing, so the
# endpoint name has to be derived from the raw ASGI path rather than read
# off a matched route.
_BODY_SIZE_LOGGED_ENDPOINTS = {
    f"{settings.api_v1_prefix}/compare-photos": "compare-photos",
    f"{settings.api_v1_prefix}/compare-photos-upload": "compare-photos-upload",
    f"{settings.api_v1_prefix}/embed": "embed",
}


async def _send_body_too_large(send: Send, path: str) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status.HTTP_400_BAD_REQUEST,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": _too_large_envelope_body()})
    # A 400 IMAGE_TOO_LARGE here burns a try the exact same way one raised
    # from inside a route does, but this middleware runs before any route's
    # own `finally: log_summary(...)` -- without this, that summary line
    # (the "exactly one PII-free record per comparison request" guarantee)
    # would simply never be emitted for this outcome.
    endpoint = _BODY_SIZE_LOGGED_ENDPOINTS.get(path)
    if endpoint is not None:
        log_summary(
            endpoint=endpoint,
            outcome="error",
            error_code=ErrorCode.IMAGE_TOO_LARGE,
        )


class _BodySizeLimitExceeded(Exception):
    """Internal signal only -- never leaves MaxBodySizeMiddleware. Raised by
    the wrapped `receive` once the streamed body crosses
    settings.max_request_body_bytes, so it unwinds whatever is reading the
    body (Starlette's form/multipart parser, `request.body()`, ...)."""


class MaxBodySizeMiddleware:
    """Pure ASGI middleware enforcing settings.max_request_body_bytes on every
    HTTP request, so an oversize body is rejected the same way an oversize
    image is: 400 IMAGE_TOO_LARGE, `image: null`, never a 413 (a 413 would
    read as an outage to the calling application -- see the global contract).

    Two cases:
      - Content-Length is declared and already over the limit: answered
        before `self.app` is ever called, so no route/auth/dependency code
        runs and the body is never read.
      - No usable Content-Length (chunked transfer, or a client that lies):
        the wrapped `receive` counts bytes as the body streams in and raises
        `_BodySizeLimitExceeded` once the running total crosses the limit.

    That exception does NOT reach this middleware's own `except` as a 500 (or
    at all, most of the time): FastAPI 0.141's routing.py catches *any*
    exception raised while it reads/parses the body, inside
    `get_request_handler`'s body-reading block (`except Exception` around
    line 469 in routing.py) and re-raises it as
    `HTTPException(400, "There was an error parsing the body")`. That
    HTTPException is then handled by our own `http_exception_handler`
    (registered on `StarletteHTTPException`), which sends a real response --
    status 400, but `error_code=INVALID_REQUEST`, not `IMAGE_TOO_LARGE`, and
    with none of the "never leaks internals" care IMAGE_TOO_LARGE gets
    elsewhere. So it does escape this middleware's `try/except
    _BodySizeLimitExceeded` (that `except` only catches the rarer case where
    nothing downstream ever touches the body, e.g. no Content-Type/route
    matched it at all -- it exists as a fallback, not the primary path).

    To still answer with our own IMAGE_TOO_LARGE envelope instead of FastAPI's
    generic INVALID_REQUEST one, `send` is wrapped too (`guarded_send`): once
    the overflow flag is set, every message the inner app (its own
    HTTPException handler included) tries to send is replaced with the one
    IMAGE_TOO_LARGE response, sent exactly once -- unless a real response has
    already started going out to the client (`response_started`), in which
    case it is too late to swap the status; see M1 below. Removing this send
    wrap (e.g. "just catch `_BodySizeLimitExceeded` after `self.app(...)`")
    would silently regress every overflow to a 400 INVALID_REQUEST response
    instead of 400 IMAGE_TOO_LARGE -- still a 400 (so the outage
    contract for the calling application holds), but the wrong error_code for a client trying to
    distinguish "your photo was too big" from "your request was malformed".

    M1: if the response has already started by the time the overflow is
    detected, this middleware cannot send a different status without
    violating the ASGI protocol (a second `http.response.start` is invalid).
    Rather than silently truncating the in-flight response by dropping its
    remaining messages, the overflow is re-raised so the connection fails
    loudly (visible in logs/as a client-side error) instead of looking like
    a clean but wrong response. This path is not reachable by the current
    routes -- both upload routes read the whole body before any response is
    produced -- but is handled defensively for future streaming routes.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = settings.max_request_body_bytes
        declared_length = None
        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                declared_length = value
                break
        if declared_length is not None:
            try:
                declared = int(declared_length)
            except ValueError:
                declared = None
            if declared is not None and declared > limit:
                await _send_body_too_large(send, scope["path"])
                return

        total = 0
        overflowed = False
        response_started = False
        overflow_sent = False

        async def limited_receive():
            nonlocal total, overflowed
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body") or b"")
                if total > limit:
                    overflowed = True
                    raise _BodySizeLimitExceeded()
            return message

        async def guarded_send(message):
            nonlocal response_started, overflow_sent
            if overflowed:
                if overflow_sent:
                    # Already sent our one substitute response -- drop
                    # anything further the app (or its own exception
                    # handlers) tries to send.
                    return
                if response_started:
                    # M1: a real response is already in flight -- too late
                    # to swap its status without sending a second
                    # http.response.start, which the ASGI protocol forbids.
                    # Fail loudly instead of silently dropping the rest of
                    # that response (see the class docstring).
                    raise _BodySizeLimitExceeded()
                overflow_sent = True
                await _send_body_too_large(send, scope["path"])
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodySizeLimitExceeded:
            if response_started:
                raise
            if not overflow_sent:
                await _send_body_too_large(send, scope["path"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.

    Loads the face recognition model at startup and cleans up at shutdown.
    """
    # Startup
    logger.info("Starting up face recognition service...")
    if settings.enhance_image is not None and settings.enhance_mode is None:
        logger.warning("ENHANCE_IMAGE is deprecated; set ENHANCE_MODE=always|detect_fallback|off")
    try:
        initialize_model()
        # asyncio.Semaphore binds to the running loop, so the gate is built
        # here (inside lifespan, once the loop is up) rather than at import
        # time.
        app.state.inference_gate = InferenceGate(
            settings.max_concurrent_inference, settings.inference_queue_timeout
        )
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down face recognition service...")
    cleanup_model()
    # The semaphore inside this gate is bound to the loop that's about to
    # stop; clear it so a later lifecycle (e.g. TestClient reused across a
    # loop restart) can never reuse a semaphore bound to a dead loop --
    # `_inference_gate()` will build a fresh one lazily if needed.
    app.state.inference_gate = None
    logger.info("Service shut down successfully")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Stateless face recognition microservice for embedding extraction and comparison",
    lifespan=lifespan,
)


def _cors_kwargs() -> dict:
    """CORSMiddleware kwargs, factored out so both the real app and tests
    share one definition. Always `allow_credentials=False`: the consumer
    authenticates with a bearer token, not cookies, so credentialed CORS
    buys nothing, and pairing it with the default wildcard
    `allow_origins=["*"]` would be unsafe (browsers reject that combination
    anyway, but this keeps the config honest for any narrower origin list
    too).
    """
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": False,
        "allow_methods": settings.cors_methods,
        "allow_headers": settings.cors_headers,
    }


# Add CORS middleware
if settings.cors_enabled:
    app.add_middleware(CORSMiddleware, **_cors_kwargs())

# Added last (see add_middleware's insert-at-front semantics) so it ends up
# outermost relative to CORS: body-size enforcement runs before CORS and
# before any route/auth/dependency code.
app.add_middleware(MaxBodySizeMiddleware)

# Added last of all -- and so, by the same insert-at-front semantics,
# outermost of everything, including MaxBodySizeMiddleware. That's required
# for the X-Request-ID response header to land on every response, including
# the 400 IMAGE_TOO_LARGE that MaxBodySizeMiddleware sends by substituting
# `send` directly rather than returning a normal Response.
app.add_middleware(RequestIdMiddleware)


# Error codes that describe a server-side fault rather than a problem with a
# specific photo. compare-photos never tags these with `image` (see
# _tag_image_error below), and they get their own HTTP status. Renamed from
# _SERVER_FAULT_CODES: it only drives tagging now, not status.
_UNTAGGED_CODES = {
    ErrorCode.MODEL_NOT_LOADED,
    ErrorCode.PROCESSING_ERROR,
    ErrorCode.SERVICE_UNAVAILABLE,
    ErrorCode.SERVICE_BUSY,
    ErrorCode.REFERENCE_URL_NOT_ALLOWED,
}

# starlette/FastAPI HTTPException.status_code -> our ErrorCode, for the
# generic HTTPException handler below (auth failures, 404s, 405s, and any
# plain `raise HTTPException(...)` still used directly in a route).
_STATUS_TO_ERROR_CODE = {
    status.HTTP_400_BAD_REQUEST: ErrorCode.INVALID_REQUEST,
    status.HTTP_401_UNAUTHORIZED: ErrorCode.UNAUTHORIZED,
    status.HTTP_403_FORBIDDEN: ErrorCode.FORBIDDEN,
    status.HTTP_404_NOT_FOUND: ErrorCode.NOT_FOUND,
    status.HTTP_405_METHOD_NOT_ALLOWED: ErrorCode.METHOD_NOT_ALLOWED,
    status.HTTP_422_UNPROCESSABLE_ENTITY: ErrorCode.VALIDATION_ERROR,
}


def _status_for_error_code(error_code: Optional[str]) -> int:
    """Map a face-service error code to its HTTP status.

    MODEL_NOT_LOADED and SERVICE_UNAVAILABLE are down dependencies (503) --
    the latter is a reference fetch that couldn't be reached at all
    (timeout/DNS/TLS/connection/5xx), as opposed to a definite 4xx on that
    link (REFERENCE_UNAVAILABLE, still a 400 below). SERVICE_BUSY (also 503)
    is our own capacity, not a dependency -- the bounded inference queue
    (concurrency.InferenceGate) timed out waiting for a free slot.
    PROCESSING_ERROR and any other unmapped server fault stay 500; every
    image-specific code is a 400 the caller can fix by retaking or replacing
    the photo.
    """
    if error_code in (ErrorCode.MODEL_NOT_LOADED, ErrorCode.SERVICE_UNAVAILABLE, ErrorCode.SERVICE_BUSY):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if error_code == ErrorCode.PROCESSING_ERROR:
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    return status.HTTP_400_BAD_REQUEST


def _tag_image_error(exc: "FaceModelError | ImageProcessingError", image: str) -> None:
    """Mark which photo (reference/selfie) an image-specific error came from.

    Server faults are left untagged so a 503/500 is never misread as "your
    photo is bad".
    """
    if exc.error_code not in _UNTAGGED_CODES:
        exc.image = image


async def _read_upload(upload: UploadFile) -> bytes:
    """Read an uploaded file's bytes, capped at settings.max_image_size, with
    no base64 round trip.

    Checks `upload.size` first (Starlette's multipart parser fills it in as
    the part is written) so an oversize file is usually rejected without
    reading a single byte here; `read(max_image_size + 1)` is the fallback
    for when `.size` is unknown, and also catches a `.size` that undercounts.

    Raises:
        ImageProcessingError: IMAGE_TOO_LARGE if the file exceeds
            settings.max_image_size.
    """
    if upload.size is not None and upload.size > settings.max_image_size:
        raise ImageProcessingError(
            f"Image size ({upload.size} bytes) exceeds maximum allowed "
            f"({settings.max_image_size} bytes)",
            ErrorCode.IMAGE_TOO_LARGE,
        )

    data = await upload.read(settings.max_image_size + 1)
    if len(data) > settings.max_image_size:
        raise ImageProcessingError(
            f"Image size exceeds maximum allowed ({settings.max_image_size} bytes)",
            ErrorCode.IMAGE_TOO_LARGE,
        )
    return data


def _inference_gate() -> InferenceGate:
    """`app.state.inference_gate`, created lazily if lifespan hasn't run
    (e.g. a route called directly in a test without the app's lifespan)."""
    gate = getattr(app.state, "inference_gate", None)
    if gate is None:
        gate = InferenceGate(settings.max_concurrent_inference, settings.inference_queue_timeout)
        app.state.inference_gate = gate
    return gate


# Sync helpers run off the event loop via `run_in_threadpool`. Each calls
# the corresponding main-module global by its bare name (not a local alias),
# so a test's `monkeypatch.setattr(main_module, "fetch_image_from_url", ...)`
# etc. still takes effect -- the name is looked up fresh on every call.
def _load_reference(url: str):
    """Fetch + preprocess the reference photo (no gate -- see the ordering
    note on each route: the gate is never held during the fetch)."""
    return preprocess_image(fetch_image_from_url(url))


def _load_upload(data: bytes):
    """Decode + preprocess an already-read upload's bytes."""
    return preprocess_image(decode_image_bytes(data))


def _analyze(model, image, role: str):
    """Run the model's (CPU-bound) analysis for one image/role."""
    return model.analyze(image, role=role)


def _quality_response(quality: Optional[FaceQuality]) -> Optional[FaceQualityResponse]:
    """Map the model's internal FaceQuality onto the API's response schema.

    Purely informational -- unaffected by whether any quality gate is
    enabled -- so a client can start observing real distributions before any
    gate is turned on. None when quality wasn't computed (or None was set
    directly by a test double).
    """
    if quality is None:
        return None
    return FaceQualityResponse(
        face_size_px=quality.face_size_px,
        interocular_px=quality.interocular_px,
        roll_deg=quality.roll_deg,
        yaw_proxy=quality.yaw_proxy,
        blur_variance=quality.blur_variance,
        embedding_norm=quality.embedding_norm,
        faces_considered=quality.faces_considered,
        second_face_ratio=quality.second_face_ratio,
    )


# Single handler for every face-service error (FaceModelError and
# ImageProcessingError both subclass FaceServiceError -- see errors.py).
@app.exception_handler(FaceServiceError)
async def face_service_error_handler(request, exc: FaceServiceError):
    """Handle FaceServiceError (and its FaceModelError/ImageProcessingError subclasses)."""
    return JSONResponse(
        status_code=_status_for_error_code(exc.error_code),
        content=ErrorResponse(
            error=exc.message,
            error_code=exc.error_code,
            image=exc.image,
        ).model_dump(),
        headers=exc.headers,
    )


# Every plain HTTPException still raised directly (auth failures in
# auth.py, FastAPI's own 404/405 for unmatched routes/methods, and any
# `raise HTTPException(...)` left in a route) goes through the same
# envelope, with the status/headers it was already given.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    """Handle starlette/FastAPI HTTPException with the unified envelope."""
    error_code = _STATUS_TO_ERROR_CODE.get(exc.status_code, ErrorCode.HTTP_ERROR)
    error_message = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=error_message,
            detail=exc.detail,
            error_code=error_code,
        ).model_dump(),
        headers=exc.headers,
    )


# FastAPI's request-validation failures (missing/invalid fields) keep their
# 422 status but carry the same envelope, with the full pydantic error list
# preserved in `detail` (JSON-encodable via jsonable_encoder).
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    """Handle FastAPI RequestValidationError with the unified envelope."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Request validation failed",
            detail=jsonable_encoder(exc.errors()),
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump(),
    )


# Catch-all for any exception not handled above -- an unexpected bug
# anywhere (including a route with no try/except of its own). Never leaks
# the exception's own text to the client; the traceback is only logged.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Handle any unhandled exception as a generic 500, logging the real cause."""
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_code=ErrorCode.PROCESSING_ERROR,
        ).model_dump(),
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get(
    f"{settings.api_v1_prefix}/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status and model information

    Note:
        This endpoint does not require authentication for monitoring purposes
    """
    model = get_model()
    is_loaded = model.is_loaded()

    body = HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        model_name=model.model_name if is_loaded else None,
    )
    if not is_loaded:
        # A monitoring probe should see this as a failed dependency, not a
        # 200 that happens to say "unhealthy" in the body.
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())
    return body


@app.get(
    f"{settings.api_v1_prefix}/model-info",
    response_model=ModelInfoResponse,
    tags=["Info"],
    dependencies=[Depends(verify_token)],
)
async def model_info():
    """
    Get information about the loaded face recognition model.

    Returns:
        ModelInfoResponse with model metadata

    Security:
        Requires valid Bearer token in Authorization header
    """
    model = get_model()
    info = model.get_model_info()

    return ModelInfoResponse(
        name=info["name"],
        embedding_size=info["embedding_size"],
        backend=info["backend"],
        device=info["device"],
    )


@app.post(
    f"{settings.api_v1_prefix}/embed",
    response_model=EmbedResponse,
    tags=["Face Recognition"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_token)],
)
async def extract_embedding(request: EmbedRequest):
    """
    Extract face embedding from an image.

    This endpoint accepts a base64-encoded image, detects a face,
    and returns the face embedding vector.

    Args:
        request: EmbedRequest with base64-encoded image

    Returns:
        EmbedResponse with 512-dimensional embedding vector

    Raises:
        HTTPException: If image processing or face detection fails

    Security:
        Requires valid Bearer token in Authorization header
    """
    request_start = time.monotonic()
    timer = StageTimer()
    outcome = "error"
    error_code = None
    image_tag = None
    det1 = None
    q2_face_px = None
    q2_blur = None
    try:
        # Decode + preprocess image -- off the event loop, no gate needed
        # for either.
        with timer.stage("decode"):
            image = await run_in_threadpool(decode_base64_image, request.image)
            image = await run_in_threadpool(preprocess_image, image)

        # Get model and extract embedding. /embed is only ever called with a
        # selfie-style photo (a live capture), never the reference photo.
        # Analysis runs behind the bounded inference gate, off the event
        # loop.
        model = get_model()
        queue_start = time.monotonic()
        try:
            async with _inference_gate().slot():
                timer.record("queue", (time.monotonic() - queue_start) * 1000)
                with timer.stage("infer"):
                    result = await run_in_threadpool_shielded(_analyze, model, image, "selfie")
        except ServiceBusyError:
            timer.record("queue", (time.monotonic() - queue_start) * 1000)
            raise

        # Convert embedding to list
        embedding_list = result.embedding.tolist()

        det1 = result.det_score
        if result.quality is not None:
            q2_face_px = result.quality.face_size_px
            q2_blur = result.quality.blur_variance
        outcome = "ok"

        return EmbedResponse(
            embedding=embedding_list,
            face_detected=True,
            detection_score=result.det_score,
            quality=_quality_response(result.quality),
        )

    except FaceServiceError as exc:
        # These are handled by custom exception handlers (FaceModelError,
        # ImageProcessingError, and ServiceBusyError all subclass
        # FaceServiceError -- catching the base lets a ServiceBusyError from
        # the inference gate pass through untouched too).
        error_code = exc.error_code
        image_tag = exc.image
        raise
    except Exception as e:
        # Never echo str(e) to the client -- only the log gets the real
        # cause (with traceback, via exc_info=True below).
        logger.exception("Unexpected error in extract_embedding")
        error_code = ErrorCode.PROCESSING_ERROR
        # Route through the same ErrorResponse envelope (and 500 status) as
        # every other unexpected failure, instead of FastAPI's bare {"detail": ...}.
        raise FaceModelError(
            "Internal server error",
            ErrorCode.PROCESSING_ERROR,
        ) from e
    finally:
        timer.record("total", (time.monotonic() - request_start) * 1000)
        log_summary(
            endpoint="embed",
            outcome=outcome,
            error_code=error_code,
            image=image_tag,
            det1=det1,
            timer=timer,
            q2_face_px=q2_face_px,
            q2_blur=q2_blur,
        )


@app.post(
    f"{settings.api_v1_prefix}/compare",
    response_model=CompareResponse,
    tags=["Face Recognition"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_token)],
)
async def compare_embeddings(request: CompareRequest):
    """
    Compare a query embedding against multiple reference embeddings.

    This endpoint calculates distances between the query embedding
    and all reference embeddings, returning sorted results with the best match.

    Args:
        request: CompareRequest with query and reference embeddings

    Returns:
        CompareResponse with sorted matches and best match

    Raises:
        HTTPException: If comparison fails

    Security:
        Requires valid Bearer token in Authorization header
    """
    try:
        logger.debug(
            f"Comparing query embedding with {len(request.reference_embeddings)} references"
        )

        # Find best match
        all_matches, best_match = find_best_match(
            query_embedding=request.query_embedding,
            reference_embeddings=request.reference_embeddings,
            metric=request.distance_metric,
        )

        logger.info(
            f"Best match: {best_match.id} "
            f"(distance: {best_match.distance:.4f}, similarity: {best_match.similarity:.4f})"
        )

        return CompareResponse(
            matches=all_matches,
            best_match=best_match,
            distance_metric=request.distance_metric,
        )

    except ValueError as e:
        logger.error(f"Validation error during comparison: {str(e)}")
        raise FaceModelError(str(e), ErrorCode.INVALID_EMBEDDING) from e
    except Exception as e:
        logger.exception("Unexpected error in compare_embeddings")
        raise FaceModelError(
            "Internal server error",
            ErrorCode.PROCESSING_ERROR,
        ) from e


@app.post(
    f"{settings.api_v1_prefix}/compare-photos",
    response_model=ComparePhotosResponse,
    tags=["Face Recognition"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_token)],
)
async def compare_photos(
    image1: str = Form(..., description="First image URL (http:// or https://)"),
    image2: UploadFile = File(..., description="Second image file"),
    distance_metric: str = Form("cosine", description="Distance metric: 'cosine' or 'euclidean'"),
):
    """
    Compare two photos directly and return similarity score.

    This endpoint accepts one image URL for the first image and a file upload for the second image.
    It fetches/processes both images, extracts face embeddings, and compares them to determine if they match.

    Args:
        image1: First image URL (http:// or https://)
        image2: Second image file to upload
        distance_metric: Distance metric to use ('cosine' or 'euclidean')

    Returns:
        ComparePhotosResponse with match result, similarity score, and distance

    Raises:
        HTTPException: If image processing or face detection fails

    Security:
        Requires valid Bearer token in Authorization header
    """
    request_start = time.monotonic()
    timer = StageTimer()
    outcome = "error"
    error_code = None
    image_tag = None
    detection_score1 = None
    detection_score2 = None
    match_similarity = None
    match_threshold_value = None
    is_match = None
    q1_face_px = None
    q2_face_px = None
    q2_blur = None
    try:
        # Validate image1 URL
        if not image1 or not image1.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image URL cannot be empty",
            )

        image1 = image1.strip()
        if not image1.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image URL must start with http:// or https://",
            )

        # Validate distance metric
        if distance_metric.lower() not in ["cosine", "euclidean"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Distance metric must be 'cosine' or 'euclidean', got '{distance_metric}'",
            )
        metric = distance_metric.lower()

        # Get the model once, before touching either photo, so a down model
        # is reported as MODEL_NOT_LOADED (image: null) rather than getting
        # mistaken for a reference-photo problem.
        model = get_model()

        # Fetch + preprocess the reference photo off the event loop, before
        # ever touching the gate -- a slow/hanging fetch must not hold up
        # every other request's inference slot.
        try:
            with timer.stage("fetch"):
                img1 = await run_in_threadpool(_load_reference, image1)
        except (FaceModelError, ImageProcessingError) as exc:
            _tag_image_error(exc, "reference")
            raise

        queue_start = time.monotonic()
        try:
            async with _inference_gate().slot():
                timer.record("queue", (time.monotonic() - queue_start) * 1000)
                try:
                    with timer.stage("infer"):
                        result1 = await run_in_threadpool_shielded(_analyze, model, img1, "reference")
                except (FaceModelError, ImageProcessingError) as exc:
                    _tag_image_error(exc, "reference")
                    raise
                detection_score1 = result1.det_score

                try:
                    # Read and process second (selfie) image from upload,
                    # then embed -- straight off the multipart part, no
                    # base64 round trip.
                    image2_bytes = await _read_upload(image2)
                    with timer.stage("decode"):
                        img2 = await run_in_threadpool_shielded(_load_upload, image2_bytes)
                    with timer.stage("infer"):
                        result2 = await run_in_threadpool_shielded(_analyze, model, img2, "selfie")
                except (FaceModelError, ImageProcessingError) as exc:
                    _tag_image_error(exc, "selfie")
                    raise
                detection_score2 = result2.det_score
        except ServiceBusyError:
            timer.record("queue", (time.monotonic() - queue_start) * 1000)
            raise

        if result1.quality is not None:
            q1_face_px = result1.quality.face_size_px
        if result2.quality is not None:
            q2_face_px = result2.quality.face_size_px
            q2_blur = result2.quality.blur_variance

        # Calculate distance between embeddings
        distance = calculate_distance(result1.embedding, result2.embedding, metric=metric)

        # Convert distance to similarity
        similarity = distance_to_similarity(distance, metric=metric)

        # Determine if it's a match against the single configured cosine
        # threshold, translated into this metric's units.
        threshold = match_threshold(metric, settings.cosine_match_threshold)
        is_match = distance < threshold
        match_similarity = cosine_similarity(result1.embedding, result2.embedding)
        match_threshold_value = threshold
        outcome = "match" if is_match else "no_match"

        return ComparePhotosResponse(
            match=is_match,
            similarity=similarity,
            distance=distance,
            distance_metric=metric,
            threshold=threshold,
            cosine_similarity=match_similarity,
            image1_detection_score=detection_score1,
            image2_detection_score=detection_score2,
            image1_quality=_quality_response(result1.quality),
            image2_quality=_quality_response(result2.quality),
        )

    except HTTPException as exc:
        # Request/validation errors raised above (empty URL, bad scheme, bad
        # distance_metric) keep their own status -- without this, the
        # `except Exception` below would swallow a 400 into a 500.
        error_code = _STATUS_TO_ERROR_CODE.get(exc.status_code, ErrorCode.HTTP_ERROR)
        raise
    except FaceServiceError as exc:
        # These are handled by custom exception handlers (FaceModelError,
        # ImageProcessingError, and ServiceBusyError all subclass
        # FaceServiceError -- catching the base lets a ServiceBusyError from
        # the inference gate pass through untouched too).
        error_code = exc.error_code
        image_tag = exc.image
        raise
    except Exception as e:
        logger.exception("Unexpected error in compare_photos")
        error_code = ErrorCode.PROCESSING_ERROR
        raise FaceModelError(
            "Internal server error",
            ErrorCode.PROCESSING_ERROR,
        ) from e
    finally:
        timer.record("total", (time.monotonic() - request_start) * 1000)
        log_summary(
            endpoint="compare-photos",
            outcome=outcome,
            error_code=error_code,
            image=image_tag,
            match=is_match,
            cosine_similarity=match_similarity,
            threshold=match_threshold_value,
            det1=detection_score1,
            det2=detection_score2,
            timer=timer,
            q1_face_px=q1_face_px,
            q2_face_px=q2_face_px,
            q2_blur=q2_blur,
        )


@app.post(
    f"{settings.api_v1_prefix}/compare-photos-upload",
    response_model=ComparePhotosResponse,
    tags=["Face Recognition"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_token)],
)
async def compare_photos_upload(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file"),
    distance_metric: str = Form("cosine", description="Distance metric: 'cosine' or 'euclidean'"),
):
    """
    Compare two photos using file upload (convenient for testing in Swagger UI).

    This endpoint is designed for easy testing via the interactive API documentation.
    It accepts file uploads directly and converts them to base64 internally.

    For programmatic API usage, prefer the /compare-photos endpoint which accepts
    base64-encoded images in JSON format.

    Args:
        image1: First image file to upload
        image2: Second image file to upload
        distance_metric: Distance metric to use ('cosine' or 'euclidean')

    Returns:
        ComparePhotosResponse with match result, similarity score, and distance

    Raises:
        HTTPException: If image processing or face detection fails

    Security:
        Requires valid Bearer token in Authorization header
    """
    request_start = time.monotonic()
    timer = StageTimer()
    outcome = "error"
    error_code = None
    image_tag = None
    detection_score1 = None
    detection_score2 = None
    match_similarity = None
    match_threshold_value = None
    is_match = None
    q1_face_px = None
    q2_face_px = None
    q2_blur = None
    try:
        # Validate distance metric
        if distance_metric.lower() not in ["cosine", "euclidean"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Distance metric must be 'cosine' or 'euclidean', got '{distance_metric}'",
            )
        metric = distance_metric.lower()

        # Process using the same logic as compare_photos, including the
        # same reference/selfie image tagging on errors (image1 mirrors the
        # reference role, image2 the selfie, even though both are uploads
        # here rather than a URL fetch) and the same no-base64-round-trip
        # reading of the multipart part.
        model = get_model()

        # Read both uploads off the event loop first (reading is cheap and
        # async-native; no gate needed for it), then decode + analyze each
        # inside one gate slot, reference before selfie -- same tag order as
        # before.
        try:
            image1_bytes = await _read_upload(image1)
        except (FaceModelError, ImageProcessingError) as exc:
            _tag_image_error(exc, "reference")
            raise
        try:
            image2_bytes = await _read_upload(image2)
        except (FaceModelError, ImageProcessingError) as exc:
            _tag_image_error(exc, "selfie")
            raise

        queue_start = time.monotonic()
        try:
            async with _inference_gate().slot():
                timer.record("queue", (time.monotonic() - queue_start) * 1000)
                try:
                    with timer.stage("decode"):
                        img1 = await run_in_threadpool_shielded(_load_upload, image1_bytes)
                    with timer.stage("infer"):
                        result1 = await run_in_threadpool_shielded(_analyze, model, img1, "reference")
                except (FaceModelError, ImageProcessingError) as exc:
                    _tag_image_error(exc, "reference")
                    raise
                detection_score1 = result1.det_score

                try:
                    with timer.stage("decode"):
                        img2 = await run_in_threadpool_shielded(_load_upload, image2_bytes)
                    with timer.stage("infer"):
                        result2 = await run_in_threadpool_shielded(_analyze, model, img2, "selfie")
                except (FaceModelError, ImageProcessingError) as exc:
                    _tag_image_error(exc, "selfie")
                    raise
                detection_score2 = result2.det_score
        except ServiceBusyError:
            timer.record("queue", (time.monotonic() - queue_start) * 1000)
            raise

        if result1.quality is not None:
            q1_face_px = result1.quality.face_size_px
        if result2.quality is not None:
            q2_face_px = result2.quality.face_size_px
            q2_blur = result2.quality.blur_variance

        # Calculate distance
        distance = calculate_distance(result1.embedding, result2.embedding, metric=metric)

        # Convert to similarity
        similarity = distance_to_similarity(distance, metric=metric)

        # Determine if it's a match against the single configured cosine
        # threshold, translated into this metric's units.
        threshold = match_threshold(metric, settings.cosine_match_threshold)
        is_match = distance < threshold
        match_similarity = cosine_similarity(result1.embedding, result2.embedding)
        match_threshold_value = threshold
        outcome = "match" if is_match else "no_match"

        return ComparePhotosResponse(
            match=is_match,
            similarity=similarity,
            distance=distance,
            distance_metric=metric,
            threshold=threshold,
            cosine_similarity=match_similarity,
            image1_detection_score=detection_score1,
            image2_detection_score=detection_score2,
            image1_quality=_quality_response(result1.quality),
            image2_quality=_quality_response(result2.quality),
        )

    except HTTPException as exc:
        # Request/validation errors raised above (bad distance_metric) keep
        # their own status -- without this, `except Exception` below would
        # swallow a 400 into a 500.
        error_code = _STATUS_TO_ERROR_CODE.get(exc.status_code, ErrorCode.HTTP_ERROR)
        raise
    except FaceServiceError as exc:
        # These are handled by custom exception handlers (FaceModelError,
        # ImageProcessingError, and ServiceBusyError all subclass
        # FaceServiceError -- catching the base lets a ServiceBusyError from
        # the inference gate pass through untouched too).
        error_code = exc.error_code
        image_tag = exc.image
        raise
    except Exception as e:
        logger.exception("Unexpected error in compare_photos_upload")
        error_code = ErrorCode.PROCESSING_ERROR
        raise FaceModelError(
            "Internal server error",
            ErrorCode.PROCESSING_ERROR,
        ) from e
    finally:
        timer.record("total", (time.monotonic() - request_start) * 1000)
        log_summary(
            endpoint="compare-photos-upload",
            outcome=outcome,
            error_code=error_code,
            image=image_tag,
            match=is_match,
            cosine_similarity=match_similarity,
            threshold=match_threshold_value,
            det1=detection_score1,
            det2=detection_score2,
            timer=timer,
            q1_face_px=q1_face_px,
            q2_face_px=q2_face_px,
            q2_blur=q2_blur,
        )


if __name__ == "__main__":
    import uvicorn

    # Always reload=False: this runs as the container's PID 1
    # (`CMD ["python", "-m", "face_recognition_service.main"]`), and
    # uvicorn's reloader spawns a supervisor subprocess that doesn't forward
    # signals the same way a single process does, which breaks graceful
    # shutdown under `docker stop`. DEBUG=true must not change that. For
    # local dev auto-reload, run uvicorn directly instead:
    #   uvicorn face_recognition_service.main:app --reload
    uvicorn.run(
        "face_recognition_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level,
    )
