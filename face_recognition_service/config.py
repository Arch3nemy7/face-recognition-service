"""Configuration settings for the face recognition service."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Settings
    app_name: str = "Face Recognition Service"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # Security Settings
    api_token: str

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    # CORS Settings
    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]

    # Model Settings
    model_name: str = "buffalo_l"  # Options: antelopev2 (best), buffalo_l, buffalo_sc
    detection_threshold: float = 0.5       # InsightFace det_thresh: minimum score to consider a face detected
    min_face_quality: float = 0.7          # Minimum det_score to accept embedding (0.5–1.0, higher = stricter quality gate)
    embedding_size: int = 512
    cosine_match_threshold: float = 0.5   # Lower = stricter. 0.4 conservative, 0.5 recommended, 0.6 permissive
    euclidean_match_threshold: float = 1.1  # Adjusted proportionally from 1.0

    # Device Settings
    device: Literal["cpu", "cuda"] = "cpu"
    providers: list[str] | None = None  # ONNX Runtime providers

    # Image Processing Settings
    max_image_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_image_formats: set[str] = {"jpg", "jpeg", "png", "bmp", "webp"}
    enhance_image: bool = True  # Apply CLAHE + auto-gamma correction before face detection

    # Performance Settings
    request_timeout: int = 30  # seconds

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set ONNX Runtime providers based on device
        if self.providers is None:
            if self.device == "cuda":
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.providers = ["CPUExecutionProvider"]


# Global settings instance
settings = Settings()
