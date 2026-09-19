"""Pydantic schemas for API requests and responses."""

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbedRequest(BaseModel):
    """Request schema for face embedding extraction."""

    image: str = Field(
        ...,
        description="Base64-encoded image string",
        min_length=1
    )

    @field_validator("image")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the image string is not empty."""
        if not v or not v.strip():
            raise ValueError("Image data cannot be empty")
        return v.strip()


class FaceQualityResponse(BaseModel):
    """Quality metrics for the chosen face. Always informational: reported
    regardless of whether any quality gate is enabled, and gates default off
    (see README). Landmark-based fields are null when the face has no
    detected landmarks."""

    face_size_px: Optional[float] = Field(
        None, description="Shorter side of the face bounding box, in pixels"
    )
    interocular_px: Optional[float] = Field(
        None, description="Distance between the two eye landmarks, in pixels"
    )
    roll_deg: Optional[float] = Field(
        None, description="In-plane head tilt, in degrees (0 = level)"
    )
    yaw_proxy: Optional[float] = Field(
        None, description="Left/right head-turn proxy; ~0 is frontal, sign gives direction"
    )
    blur_variance: Optional[float] = Field(
        None, description="Laplacian variance of the aligned face crop; lower means blurrier"
    )
    embedding_norm: Optional[float] = Field(
        None, description="L2 norm of the raw embedding, before unit normalisation"
    )
    faces_considered: Optional[int] = Field(
        None, description="Number of detected faces meeting the detector's quality threshold"
    )
    second_face_ratio: Optional[float] = Field(
        None,
        description="Area of the next-largest qualifying face relative to the chosen face's area; 0 if none",
    )


class EmbedResponse(BaseModel):
    """Response schema for face embedding extraction."""

    embedding: list[float] = Field(
        ...,
        description="Face embedding vector (512-dimensional)",
        min_length=512,
        max_length=512
    )
    face_detected: bool = Field(
        ...,
        description="Whether a face was successfully detected"
    )
    detection_score: Optional[float] = Field(
        None,
        description="Confidence score of face detection (0-1)",
        ge=0.0,
        le=1.0
    )
    quality: Optional[FaceQualityResponse] = Field(
        None,
        description="Informational face quality metrics for the detected face",
    )


class ReferenceEmbedding(BaseModel):
    """A reference embedding with an identifier."""

    id: str = Field(
        ...,
        description="Unique identifier for this embedding (e.g., user ID)",
        min_length=1
    )
    embedding: list[float] = Field(
        ...,
        description="Face embedding vector (512-dimensional)",
        min_length=512,
        max_length=512
    )


class CompareRequest(BaseModel):
    """Request schema for comparing embeddings."""

    query_embedding: list[float] = Field(
        ...,
        description="Query face embedding to compare (512-dimensional)",
        min_length=512,
        max_length=512
    )
    reference_embeddings: list[ReferenceEmbedding] = Field(
        ...,
        description="List of reference embeddings to compare against",
        min_length=1
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric to use: 'cosine' or 'euclidean'"
    )

    @field_validator("distance_metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate distance metric."""
        allowed = {"cosine", "euclidean"}
        if v.lower() not in allowed:
            raise ValueError(f"Distance metric must be one of {allowed}")
        return v.lower()


class MatchResult(BaseModel):
    """A single match result."""

    id: str = Field(..., description="Identifier of the matched reference")
    distance: float = Field(..., description="Distance value (lower is more similar)")
    similarity: float = Field(
        ...,
        description="Similarity score (0-1, higher is more similar)",
        ge=0.0,
        le=1.0
    )


class CompareResponse(BaseModel):
    """Response schema for embedding comparison."""

    matches: list[MatchResult] = Field(
        ...,
        description="List of all matches sorted by distance (best first)"
    )
    best_match: MatchResult = Field(
        ...,
        description="The best matching reference (lowest distance)"
    )
    distance_metric: str = Field(
        ...,
        description="The distance metric used for comparison"
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    name: str = Field(..., description="Model name")
    embedding_size: int = Field(..., description="Embedding vector dimension")
    backend: str = Field(..., description="Backend framework (insightface)")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")


class ComparePhotosRequest(BaseModel):
    """Request schema for comparing two photos directly."""

    image1: str = Field(
        ...,
        description="First image URL (http:// or https://)",
        min_length=1
    )
    image2: str = Field(
        ...,
        description="Second image URL (http:// or https://)",
        min_length=1
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric to use: 'cosine' or 'euclidean'"
    )

    @field_validator("image1", "image2")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that the image URL is not empty and has valid format."""
        if not v or not v.strip():
            raise ValueError("Image URL cannot be empty")
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Image URL must start with http:// or https://")
        return v

    @field_validator("distance_metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate distance metric."""
        allowed = {"cosine", "euclidean"}
        if v.lower() not in allowed:
            raise ValueError(f"Distance metric must be one of {allowed}")
        return v.lower()


class ComparePhotosUploadRequest(BaseModel):
    """Request schema for comparing two photos via file upload (base64-encoded images)."""

    image1: str = Field(
        ...,
        description="First image as base64-encoded string",
        min_length=1
    )
    image2: str = Field(
        ...,
        description="Second image as base64-encoded string",
        min_length=1
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric to use: 'cosine' or 'euclidean'"
    )

    @field_validator("image1", "image2")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the image data is not empty."""
        if not v or not v.strip():
            raise ValueError("Image data cannot be empty")
        return v.strip()

    @field_validator("distance_metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate distance metric."""
        allowed = {"cosine", "euclidean"}
        if v.lower() not in allowed:
            raise ValueError(f"Distance metric must be one of {allowed}")
        return v.lower()


class ComparePhotosResponse(BaseModel):
    """Response schema for comparing two photos."""

    match: bool = Field(
        ...,
        description="Whether the faces match (based on typical threshold)"
    )
    similarity: float = Field(
        ...,
        description="Similarity score (0-1, higher is more similar)",
        ge=0.0,
        le=1.0
    )
    distance: float = Field(
        ...,
        description="Distance value between embeddings (lower is more similar)"
    )
    distance_metric: str = Field(
        ...,
        description="The distance metric used for comparison"
    )
    threshold: Optional[float] = Field(
        None,
        description="Distance threshold used for `match`, in the units of `distance_metric`; "
        "the photos match when distance < threshold",
    )
    cosine_similarity: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Raw cosine similarity of the two face embeddings (-1..1), independent of distance_metric",
    )
    image1_detection_score: Optional[float] = Field(
        None,
        description="Face detection confidence for first image (0-1)",
        ge=0.0,
        le=1.0
    )
    image2_detection_score: Optional[float] = Field(
        None,
        description="Face detection confidence for second image (0-1)",
        ge=0.0,
        le=1.0
    )
    image1_quality: Optional[FaceQualityResponse] = Field(
        None,
        description="Informational face quality metrics for the first (reference) image",
    )
    image2_quality: Optional[FaceQualityResponse] = Field(
        None,
        description="Informational face quality metrics for the second (selfie) image",
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[Union[str, list]] = Field(None, description="Detailed error information")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    # Which photo an image-specific error is about. Server-side faults
    # (model not loaded, unexpected processing error) leave this null so a
    # caller never mistakes "we broke" for "your photo is bad".
    image: Optional[Literal["reference", "selfie"]] = Field(
        None,
        description="Which photo the error is about ('reference' or 'selfie'), when applicable",
    )


# Error codes for standardized error handling
class ErrorCode:
    """Standard error codes."""

    INVALID_IMAGE = "INVALID_IMAGE"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    FACE_LOW_QUALITY = "FACE_LOW_QUALITY"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INVALID_EMBEDDING = "INVALID_EMBEDDING"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    REFERENCE_UNAVAILABLE = "REFERENCE_UNAVAILABLE"
    # The reference URL couldn't even be reached (timeout, DNS, TLS,
    # connection refused, or the far end's own 5xx) -- a server-side/
    # upstream fault, not evidence the specific photo/link is bad. Distinct
    # from REFERENCE_UNAVAILABLE (a definite 4xx on that link), which *is*
    # the enrolled person's problem to fix via a new reference photo.
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    # The bounded inference queue (concurrency.InferenceGate) timed out
    # waiting for a free slot -- the service is up but saturated. Distinct
    # from SERVICE_UNAVAILABLE (a downstream fetch failure): this is our own
    # capacity, not a dependency.
    SERVICE_BUSY = "SERVICE_BUSY"
    # The reference URL's host (or a redirect target's host) failed the
    # host/IP policy: not on a non-empty allowlist, or resolved to an
    # always-blocked address (link-local/multicast/unspecified/reserved) or,
    # with REFERENCE_URL_ALLOW_PRIVATE=false, a private/loopback one. Left
    # untagged like SERVICE_UNAVAILABLE (see main._UNTAGGED_CODES) -- the
    # backend shows both as a plain outage, which is correct whether this is
    # a config mismatch or someone probing internal addresses.
    REFERENCE_URL_NOT_ALLOWED = "REFERENCE_URL_NOT_ALLOWED"

    # Generic HTTP-layer codes, produced by the catch-all exception handlers
    # in main.py rather than raised directly by model/image-processing code.
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    HTTP_ERROR = "HTTP_ERROR"  # any other HTTPException status
