"""Pydantic schemas for API requests and responses."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


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


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")


# Error codes for standardized error handling
class ErrorCode:
    """Standard error codes."""

    INVALID_IMAGE = "INVALID_IMAGE"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INVALID_EMBEDDING = "INVALID_EMBEDDING"
    PROCESSING_ERROR = "PROCESSING_ERROR"
