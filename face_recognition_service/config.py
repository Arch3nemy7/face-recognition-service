"""Configuration settings for the face recognition service."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # `model_name` is a real setting, not pydantic internals.
        protected_namespaces=("settings_",),
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
    # Off by default: the calling application talks to this service
    # server-to-server with a bearer token, not from a browser, so CORS
    # has nothing to enable for the normal deployment.
    cors_enabled: bool = False
    cors_origins: list[str] = ["*"]
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]

    # Model Settings
    model_name: str = "antelopev2"  # Options: antelopev2 (best, production default), buffalo_l, buffalo_sc
    detection_threshold: float = 0.5       # InsightFace det_thresh: minimum score to consider a face detected
    min_face_quality: float = 0.7          # Minimum det_score to accept embedding (0.5–1.0, higher = stricter quality gate)
    embedding_size: int = 512
    # Match when cosine distance < this. Euclidean uses sqrt(2 * this) on unit vectors, so both metrics agree.
    cosine_match_threshold: float = 0.5   # Lower = stricter. 0.4 conservative, 0.5 recommended, 0.6 permissive
    det_size: int = 640  # detector input size (square)
    pad_retry_ratio: float = 0.5  # when nothing is detected, retry on a copy padded by this fraction of the longer side (0 = off)
    ort_intra_op_threads: int = 0  # ONNX Runtime threads per inference (0 = runtime default; set to the CPU quota)
    ort_inter_op_threads: int = 0  # ONNX Runtime parallel-operator threads (0 = runtime default)

    # Device Settings
    # `providers` derives from this unless PROVIDERS is set explicitly; an explicit PROVIDERS always wins.
    device: Literal["cpu", "cuda"] = "cpu"
    providers: list[str] | None = None  # ONNX Runtime providers

    # Image Processing Settings
    max_image_size: int = 10 * 1024 * 1024  # 10 MB of encoded bytes
    # Hard cap on the whole request body (both images plus multipart/base64
    # overhead), enforced by a raw ASGI middleware before any route runs.
    # Must exceed two images plus base64/multipart overhead.
    max_request_body_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 50_000_000  # decoded width*height; checked from the header before decoding
    max_image_side: int = 2048  # downscale so the longer side is at most this (0 = keep original size)
    allowed_image_formats: set[str] = {"jpg", "jpeg", "png", "bmp", "webp", "mpo"}  # MPO = many phone JPEGs
    # Apply CLAHE + auto-gamma correction before face detection. Deprecated in
    # favour of `enhance_mode`; kept only so ENHANCE_IMAGE=true/false set as
    # an actual process/container environment variable keeps meaning what it
    # always has (see `effective_enhance_mode`). This only takes effect if
    # the variable actually reaches this process: under the shipped
    # docker-compose.yml, ENHANCE_IMAGE is not in the container's
    # `environment:` list and there's no `env_file:`, so setting it in the
    # host .env is silently ignored there -- use ENHANCE_MODE instead.
    enhance_image: bool | None = None
    # "always": detect and embed on the enhanced copy (today's ENHANCE_IMAGE=true).
    # "off": never enhance.
    # "detect_fallback": detect on the original first; only fall back to the
    #   enhanced copy when nothing is found there; always embed from the
    #   original pixels.
    enhance_mode: Literal["off", "detect_fallback", "always"] | None = None

    # Face Quality Gates -- all default to 0/off. A limit of 0 disables that
    # gate. Gates never reject a reference photo (a rejection there is a
    # serious problem for the enrolled person, since it blocks them until
    # someone intervenes) unless `quality_gates_apply_to_reference` is set.
    # Values are reported on every result regardless, so real distributions
    # can calibrate these later.
    min_face_size_px: float = 0  # reject if the bbox's shorter side is below this
    min_interocular_px: float = 0  # reject if eye-to-eye distance is below this
    min_blur_variance: float = 0  # reject if the aligned crop's Laplacian variance is below this
    max_abs_yaw_proxy: float = 0  # reject if |yaw_proxy| exceeds this
    max_abs_roll_deg: float = 0  # reject if |roll_deg| exceeds this
    min_embedding_norm: float = 0  # reject if the raw embedding norm is below this
    quality_gates_apply_to_reference: bool = False  # opt-in: also gate the reference photo, not just the selfie
    # "largest": always accept the dominant face (today's behaviour).
    # "reject_ambiguous": raise MULTIPLE_FACES_DETECTED when another
    #   qualifying face is close in score and size to the chosen one.
    multi_face_policy: Literal["largest", "reject_ambiguous"] = "largest"
    multi_face_area_ratio: float = 0.5  # second face must be at least this fraction of the chosen face's area to count as ambiguous
    multi_face_min_score: float = 0.5  # second face must have at least this det_score to count as ambiguous

    # Reference fetch hardening (image_utils.fetch_image_from_url) -- see
    # README "Security Considerations" for the recommended production
    # values and the residual risks (DNS-rebinding TOCTOU, resolver time).
    fetch_connect_timeout: float = 5  # seconds, per hop
    fetch_read_timeout: float = 10  # seconds, per hop
    # Seconds, whole fetch (every hop, every read). Enforced by both an
    # in-band check and a threading.Timer watchdog that force-closes the
    # socket if it fires -- see fetch_image_from_url's docstring. Actual
    # guarantee: the fetch finishes within this many seconds of being
    # called, plus the watchdog's own (sub-second) scheduling slop -- not
    # "one read timeout", which only bounds the in-band check and misses a
    # drip slower than fetch_read_timeout but never idle long enough to trip
    # it on its own.
    fetch_total_timeout: float = 20
    fetch_max_redirects: int = 3  # at most this many redirect hops followed
    # Empty = any host allowed (still subject to the always-blocked IP
    # classes below). Non-empty = exact, case-insensitive host match (a
    # trailing "." is normalised away first); any other host (including a
    # redirect target) is REFERENCE_URL_NOT_ALLOWED. Recommended in
    # production: set to the calling application's own reference-photo host(s).
    reference_url_allowed_hosts: list[str] = []
    # Link-local/multicast/unspecified/reserved (e.g. 169.254.169.254, the
    # AWS/GCP-style cloud-metadata address) are always blocked regardless of
    # this flag, plus a short explicit list of other known metadata
    # endpoints that don't fall into any of those address classes
    # (100.100.100.200, fd00:ec2::254 -- see image_utils._ALWAYS_BLOCKED_LITERALS).
    # True (default) also allows any address that isn't globally routable
    # (not ip.is_global: private/loopback ranges, CGNAT 100.64.0.0/10, IPv6
    # ULA fc00::/7, 6to4/Teredo) -- blocking them by default risks locking
    # every enrolled person out if the calling application's reference-photo
    # host happens to resolve to one of those. Recommended in production:
    # false, once the allowlist above is set to the real host(s).
    reference_url_allow_private: bool = True

    # Concurrency Settings -- see concurrency.InferenceGate. Inference
    # (model.analyze) runs off the event loop behind a bounded queue so one
    # slow request can't stall every other connection; a request that can't
    # get a slot within inference_queue_timeout gets a fast 503 SERVICE_BUSY
    # instead of piling up indefinitely.
    max_concurrent_inference: int = Field(1, ge=1)  # concurrent model.analyze calls allowed
    inference_queue_timeout: float = 20.0  # seconds to wait for a free inference slot
    busy_retry_after_seconds: int = 5  # Retry-After header value on 503 SERVICE_BUSY

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set ONNX Runtime providers based on device
        if self.providers is None:
            if self.device == "cuda":
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.providers = ["CPUExecutionProvider"]

    @property
    def effective_enhance_mode(self) -> str:
        """`enhance_mode` if set; else derived from the deprecated `enhance_image`.

        `enhance_image is True` -> "always" (today's default deploy behaviour);
        `enhance_image is False` -> "off"; unset -> "detect_fallback".
        """
        if self.enhance_mode is not None:
            return self.enhance_mode
        if self.enhance_image is True:
            return "always"
        if self.enhance_image is False:
            return "off"
        return "detect_fallback"


# Global settings instance
settings = Settings()
