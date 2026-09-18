"""FastAPI application for face recognition microservice."""

import base64
import logging
from contextlib import asynccontextmanager
from typing import Annotated, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import verify_token
from .config import settings
from .models.face_model import (
    FaceModelError,
    cleanup_model,
    get_model,
    initialize_model,
)
from .schemas.api_schemas import (
    ComparePhotosRequest,
    ComparePhotosResponse,
    ComparePhotosUploadRequest,
    CompareRequest,
    CompareResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .utils.embedding_utils import calculate_distance, distance_to_similarity, find_best_match
from .utils.image_utils import (
    ImageProcessingError,
    _redact_url,
    decode_base64_image,
    fetch_image_from_url,
    preprocess_image,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.

    Loads the face recognition model at startup and cleans up at shutdown.
    """
    # Startup
    logger.info("Starting up face recognition service...")
    try:
        initialize_model()
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down face recognition service...")
    cleanup_model()
    logger.info("Service shut down successfully")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Stateless face recognition microservice for embedding extraction and comparison",
    lifespan=lifespan,
)

# Add CORS middleware
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )


# Error codes that describe a server-side fault rather than a problem with a
# specific photo. compare-photos never tags these with `image` (see
# _tag_image_error below), and they get their own HTTP status.
_SERVER_FAULT_CODES = {
    ErrorCode.MODEL_NOT_LOADED,
    ErrorCode.PROCESSING_ERROR,
    ErrorCode.SERVICE_UNAVAILABLE,
}


def _status_for_error_code(error_code: Optional[str]) -> int:
    """Map a face-service error code to its HTTP status.

    MODEL_NOT_LOADED and SERVICE_UNAVAILABLE are down dependencies (503) --
    the latter is a reference fetch that couldn't be reached at all
    (timeout/DNS/TLS/connection/5xx), as opposed to a definite 4xx on that
    link (REFERENCE_UNAVAILABLE, still a 400 below). PROCESSING_ERROR and
    any other unmapped server fault stay 500; every image-specific code is a
    400 the caller can fix by retaking or replacing the photo.
    """
    down_dependency_codes = (
        ErrorCode.MODEL_NOT_LOADED,
        ErrorCode.SERVICE_UNAVAILABLE,
    )
    if error_code in down_dependency_codes:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if error_code == ErrorCode.PROCESSING_ERROR:
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    return status.HTTP_400_BAD_REQUEST


def _tag_image_error(
    exc: "FaceModelError | ImageProcessingError", image: str
) -> None:
    """Mark which photo (reference/selfie) an image-specific error came from.

    Server faults are left untagged so a 503/500 is never misread as "your
    photo is bad".
    """
    if exc.error_code not in _SERVER_FAULT_CODES:
        exc.image = image


# Custom exception handler for FaceModelError
@app.exception_handler(FaceModelError)
async def face_model_error_handler(request, exc: FaceModelError):
    """Handle FaceModelError exceptions."""
    return JSONResponse(
        status_code=_status_for_error_code(exc.error_code),
        content=ErrorResponse(
            error=exc.message,
            error_code=exc.error_code,
            image=exc.image,
        ).model_dump(),
    )


# Custom exception handler for ImageProcessingError
@app.exception_handler(ImageProcessingError)
async def image_processing_error_handler(request, exc: ImageProcessingError):
    """Handle ImageProcessingError exceptions."""
    return JSONResponse(
        status_code=_status_for_error_code(exc.error_code),
        content=ErrorResponse(
            error=exc.message,
            error_code=exc.error_code,
            image=exc.image,
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

    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        model_name=model.model_name if is_loaded else None,
    )


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
    try:
        # Decode image from base64
        logger.debug("Decoding base64 image...")
        image = decode_base64_image(request.image)

        # Preprocess image
        logger.debug("Preprocessing image...")
        image = preprocess_image(image)

        # Get model and extract embedding
        logger.debug("Extracting face embedding...")
        model = get_model()
        embedding, detection_score = model.get_embedding(image, return_detection_info=True)

        # Convert embedding to list
        embedding_list = embedding.tolist()

        logger.info(f"Successfully extracted embedding (size: {len(embedding_list)})")

        return EmbedResponse(
            embedding=embedding_list,
            face_detected=True,
            detection_score=detection_score,
        )

    except (FaceModelError, ImageProcessingError):
        # These are handled by custom exception handlers
        raise
    except Exception as e:
        logger.error(f"Unexpected error during embedding extraction: {str(e)}")
        # Route through the same ErrorResponse envelope (and 500 status) as
        # every other unexpected failure, instead of FastAPI's bare
        # {"detail": ...}.
        raise FaceModelError(
            f"Internal server error: {str(e)}",
            ErrorCode.PROCESSING_ERROR,
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error during comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


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

        # Get the model once, before touching either photo, so a down model
        # is reported as MODEL_NOT_LOADED (image: null) rather than getting
        # mistaken for a reference-photo problem.
        model = get_model()

        logger.debug(f"Fetching first image from URL: {_redact_url(image1)}")
        try:
            # Fetch and process first (reference) image from URL, then embed
            img1 = fetch_image_from_url(image1)
            img1 = preprocess_image(img1)
            embedding1, detection_score1 = model.get_embedding(
                img1, return_detection_info=True
            )
        except (FaceModelError, ImageProcessingError) as exc:
            _tag_image_error(exc, "reference")
            raise
        logger.debug(f"First image processed (detection score: {detection_score1:.4f})")

        logger.debug(f"Reading second image file: {image2.filename}")
        try:
            # Read and process second (selfie) image from upload, then embed
            image2_bytes = await image2.read()
            image2_b64 = base64.b64encode(image2_bytes).decode("utf-8")
            img2 = decode_base64_image(image2_b64)
            img2 = preprocess_image(img2)
            embedding2, detection_score2 = model.get_embedding(
                img2, return_detection_info=True
            )
        except (FaceModelError, ImageProcessingError) as exc:
            _tag_image_error(exc, "selfie")
            raise
        logger.debug(f"Second image processed (detection score: {detection_score2:.4f})")

        # Calculate distance between embeddings
        logger.debug(f"Calculating {distance_metric} distance...")
        distance = calculate_distance(embedding1, embedding2, metric=distance_metric.lower())

        # Convert distance to similarity
        similarity = distance_to_similarity(distance, metric=distance_metric.lower())

        # Determine if it's a match based on configurable thresholds
        if distance_metric.lower() == "cosine":
            is_match = distance < settings.cosine_match_threshold
        else:  # euclidean
            is_match = distance < settings.euclidean_match_threshold

        logger.info(
            f"Comparison complete: match={is_match}, "
            f"similarity={similarity:.4f}, distance={distance:.4f}"
        )

        return ComparePhotosResponse(
            match=is_match,
            similarity=similarity,
            distance=distance,
            distance_metric=distance_metric.lower(),
            image1_detection_score=detection_score1,
            image2_detection_score=detection_score2,
        )

    except HTTPException:
        # Request/validation errors raised above (empty URL, bad scheme, bad
        # distance_metric) keep their own status -- without this, the
        # `except Exception` below would swallow a 400 into a 500.
        raise
    except (FaceModelError, ImageProcessingError):
        # These are handled by custom exception handlers
        raise
    except Exception as e:
        logger.error(f"Unexpected error during photo comparison: {str(e)}")
        raise FaceModelError(
            f"Internal server error: {str(e)}",
            ErrorCode.PROCESSING_ERROR,
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
    distance_metric: str = "cosine",
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
    try:
        # Validate distance metric
        if distance_metric.lower() not in ["cosine", "euclidean"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Distance metric must be 'cosine' or 'euclidean', got '{distance_metric}'",
            )

        # Read and encode first image
        logger.debug(f"Reading first image: {image1.filename}")
        image1_bytes = await image1.read()
        image1_b64 = base64.b64encode(image1_bytes).decode("utf-8")

        # Read and encode second image
        logger.debug(f"Reading second image: {image2.filename}")
        image2_bytes = await image2.read()
        image2_b64 = base64.b64encode(image2_bytes).decode("utf-8")

        # Create request object
        request = ComparePhotosUploadRequest(
            image1=image1_b64,
            image2=image2_b64,
            distance_metric=distance_metric.lower(),
        )

        # Process using the same logic as compare_photos, including the
        # same reference/selfie image tagging on errors (image1 mirrors the
        # reference role, image2 the selfie, even though both are uploads
        # here rather than a URL fetch).
        model = get_model()

        logger.debug("Processing first image...")
        try:
            img1 = decode_base64_image(request.image1)
            img1 = preprocess_image(img1)
            embedding1, detection_score1 = model.get_embedding(
                img1, return_detection_info=True
            )
        except (FaceModelError, ImageProcessingError) as exc:
            _tag_image_error(exc, "reference")
            raise
        logger.debug(f"First image processed (detection score: {detection_score1:.4f})")

        logger.debug("Processing second image...")
        try:
            img2 = decode_base64_image(request.image2)
            img2 = preprocess_image(img2)
            embedding2, detection_score2 = model.get_embedding(
                img2, return_detection_info=True
            )
        except (FaceModelError, ImageProcessingError) as exc:
            _tag_image_error(exc, "selfie")
            raise
        logger.debug(f"Second image processed (detection score: {detection_score2:.4f})")

        # Calculate distance
        logger.debug(f"Calculating {request.distance_metric} distance...")
        distance = calculate_distance(embedding1, embedding2, metric=request.distance_metric)

        # Convert to similarity
        similarity = distance_to_similarity(distance, metric=request.distance_metric)

        # Determine match
        if request.distance_metric == "cosine":
            is_match = distance < settings.cosine_match_threshold
        else:  # euclidean
            is_match = distance < settings.euclidean_match_threshold

        logger.info(
            f"Upload comparison complete: match={is_match}, "
            f"similarity={similarity:.4f}, distance={distance:.4f}"
        )

        return ComparePhotosResponse(
            match=is_match,
            similarity=similarity,
            distance=distance,
            distance_metric=request.distance_metric,
            image1_detection_score=detection_score1,
            image2_detection_score=detection_score2,
        )

    except HTTPException:
        # Request/validation errors raised above (bad distance_metric) keep
        # their own status -- without this, `except Exception` below would
        # swallow a 400 into a 500.
        raise
    except (FaceModelError, ImageProcessingError):
        # These are handled by custom exception handlers
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload comparison: {str(e)}")
        raise FaceModelError(
            f"Internal server error: {str(e)}",
            ErrorCode.PROCESSING_ERROR,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "face_recognition_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level,
    )
