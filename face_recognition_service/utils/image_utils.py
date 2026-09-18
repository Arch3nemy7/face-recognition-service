"""Image processing utilities for face recognition."""

import base64
import io
import math
from typing import Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests
from PIL import Image

from ..config import settings
from ..schemas.api_schemas import ErrorCode


def _redact_url(url: str) -> str:
    """Strip the query string and fragment from a URL before it's logged or
    echoed back in an error.

    The reference photo URL (image1 on /compare-photos) is often a signed or
    presigned link (e.g. cloud storage or a CDN) with its auth embedded in
    the query string; the raw URL must never reach a client-facing error or
    a log line.
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""

    def __init__(
        self, message: str, error_code: str, image: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        # Which photo (reference/selfie) this error is about; set by the
        # endpoint that knows the role, not by this generic utility.
        self.image = image
        super().__init__(self.message)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to a numpy array.

    Args:
        base64_string: Base64-encoded image string (with or without data URI prefix)

    Returns:
        numpy array in BGR format (OpenCV format)

    Raises:
        ImageProcessingError: If decoding fails or image is invalid
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string and base64_string.startswith("data:"):
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Check size limit
        if len(image_bytes) > settings.max_image_size:
            raise ImageProcessingError(
                f"Image size ({len(image_bytes)} bytes) exceeds maximum allowed "
                f"({settings.max_image_size} bytes)",
                ErrorCode.IMAGE_TOO_LARGE
            )

        # Convert bytes to numpy array
        return load_image_from_bytes(image_bytes)

    except base64.binascii.Error as e:
        raise ImageProcessingError(
            f"Invalid base64 encoding: {str(e)}",
            ErrorCode.INVALID_IMAGE
        )
    except Exception as e:
        if isinstance(e, ImageProcessingError):
            raise
        raise ImageProcessingError(
            f"Failed to decode image: {str(e)}",
            ErrorCode.INVALID_IMAGE
        )


def fetch_image_from_url(url: str, timeout: int = 30) -> np.ndarray:
    """
    Fetch an image from a URL and convert to numpy array.

    Args:
        url: Image URL to fetch
        timeout: Request timeout in seconds (default: 30)

    Returns:
        numpy array in BGR format (OpenCV format)

    Raises:
        ImageProcessingError: If fetching fails or image is invalid
    """
    try:
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            raise ImageProcessingError(
                f"Invalid URL format. URL must start with http:// or https://",
                ErrorCode.INVALID_IMAGE
            )

        # Prepare headers to mimic a browser request
        # Many CDN services block requests without proper User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        # Fetch image from URL with proper headers
        response = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        response.raise_for_status()

        # Check Content-Type header (be more lenient for some CDNs)
        content_type = response.headers.get('Content-Type', '')
        # Some CDNs don't return proper Content-Type, so we'll be lenient
        # and rely on image loading validation instead

        # Read image bytes
        image_bytes = response.content

        # Check size limit
        if len(image_bytes) > settings.max_image_size:
            raise ImageProcessingError(
                f"Image size ({len(image_bytes)} bytes) exceeds maximum allowed "
                f"({settings.max_image_size} bytes)",
                ErrorCode.IMAGE_TOO_LARGE
            )

        # Convert bytes to numpy array
        return load_image_from_bytes(image_bytes)

    except requests.exceptions.HTTPError as e:
        response = e.response
        status_code = response.status_code if response is not None else None
        if status_code is not None and 400 <= status_code < 500:
            # The reference link itself is bad -- 404 gone, 403 forbidden,
            # 410 removed, etc. The photo's owner needs a fresh photo/link,
            # so this is a REFERENCE_UNAVAILABLE the caller can tag
            # "reference".
            raise ImageProcessingError(
                f"Reference URL returned {status_code}: {_redact_url(url)}",
                ErrorCode.REFERENCE_UNAVAILABLE
            )
        # A 5xx (or an HTTPError with no response at all) is the far end's
        # own outage, not evidence this particular link is broken -- treat
        # it like the fetch being unreachable (see SERVICE_UNAVAILABLE below),
        # not like a bad photo.
        raise ImageProcessingError(
            f"Reference URL fetch failed ({_redact_url(url)}): "
            f"HTTP {status_code}",
            ErrorCode.SERVICE_UNAVAILABLE
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        # Timeout, DNS failure, refused connection, and TLS/SSL errors
        # (requests.exceptions.SSLError subclasses ConnectionError) all mean
        # the face service's own outbound path is down, not that this
        # specific reference photo is bad -- so this must never read as "the
        # reference photo is broken" (image stays null; see
        # SERVICE_UNAVAILABLE's status mapping in main.py, which routes it
        # like a real outage).
        raise ImageProcessingError(
            f"Reference URL unreachable: {_redact_url(url)}",
            ErrorCode.SERVICE_UNAVAILABLE
        )
    except requests.exceptions.RequestException as e:
        # Any other requests-level failure (too many redirects, malformed
        # URL post-parse, etc.) gets the same "our fetch path failed"
        # treatment as the two branches above, not a photo-content code.
        # Uses the exception type rather than str(e): requests embeds the
        # full requested URL (query string and all) in most of its
        # exception messages, which would leak a signed URL's auth right
        # back into this "dev message".
        raise ImageProcessingError(
            f"Failed to fetch image from URL ({_redact_url(url)}): "
            f"{type(e).__name__}",
            ErrorCode.SERVICE_UNAVAILABLE
        )
    except Exception as e:
        if isinstance(e, ImageProcessingError):
            raise
        raise ImageProcessingError(
            f"Failed to load image from URL: {str(e)}",
            ErrorCode.INVALID_IMAGE
        )


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from bytes to a numpy array.

    Args:
        image_bytes: Raw image bytes

    Returns:
        numpy array in BGR format (OpenCV format)

    Raises:
        ImageProcessingError: If loading fails or format is unsupported
    """
    try:
        # Try to open with PIL first (better format support)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Validate image format
        if pil_image.format:
            format_lower = pil_image.format.lower()
            if format_lower not in settings.allowed_image_formats:
                raise ImageProcessingError(
                    f"Unsupported image format: {pil_image.format}. "
                    f"Allowed formats: {settings.allowed_image_formats}",
                    ErrorCode.UNSUPPORTED_FORMAT
                )

        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert PIL image to numpy array (RGB)
        image_array = np.array(pil_image)

        # Convert RGB to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return image_bgr

    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            f"Failed to load image from bytes: {str(e)}",
            ErrorCode.INVALID_IMAGE
        )


def validate_image(image: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image array is suitable for face detection.

    Args:
        image: Image as numpy array

    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"

    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"

    if image.size == 0:
        return False, "Image is empty"

    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image shape: {image.shape}. Expected 2D or 3D array"

    # Check if image has valid dimensions
    if len(image.shape) == 3:
        height, width, channels = image.shape
        if channels not in [1, 3, 4]:
            return False, f"Invalid number of channels: {channels}. Expected 1, 3, or 4"
    else:
        height, width = image.shape

    # Check minimum dimensions
    min_size = 32
    if height < min_size or width < min_size:
        return False, f"Image too small: {width}x{height}. Minimum size: {min_size}x{min_size}"

    # Check maximum dimensions (prevent memory issues)
    max_size = 8192
    if height > max_size or width > max_size:
        return False, f"Image too large: {width}x{height}. Maximum size: {max_size}x{max_size}"

    return True, None


def _enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE and auto-gamma correction to improve face recognition accuracy
    under varying lighting conditions.

    Args:
        image: BGR image as numpy array

    Returns:
        Enhanced BGR image
    """
    # --- Auto gamma correction ---
    # Measures mean brightness and corrects toward a neutral 128
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness > 0:
        gamma = math.log(128) / math.log(mean_brightness + 1)
        gamma = max(0.5, min(gamma, 2.5))  # clamp to safe range
        lut = np.array(
            [min(255, int(((i / 255.0) ** (1.0 / gamma)) * 255)) for i in range(256)],
            dtype=np.uint8,
        )
        image = cv2.LUT(image, lut)

    # --- CLAHE on the L channel of LAB ---
    # Improves local contrast without blowing out highlights
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for face detection.

    Args:
        image: Image as numpy array in BGR format

    Returns:
        Preprocessed image in BGR format

    Raises:
        ImageProcessingError: If preprocessing fails
    """
    try:
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise ImageProcessingError(error_msg, ErrorCode.INVALID_IMAGE)

        # Ensure image is in BGR format (3 channels)
        if len(image.shape) == 2:
            # Grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # BGRA to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if settings.enhance_image:
            image = _enhance_image(image)

        return image

    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            f"Failed to preprocess image: {str(e)}",
            ErrorCode.PROCESSING_ERROR
        )


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    Encode a numpy array image to base64 string.

    Args:
        image: Image as numpy array (BGR format)
        format: Image format for encoding (JPEG, PNG, etc.)

    Returns:
        Base64-encoded image string with data URI prefix

    Raises:
        ImageProcessingError: If encoding fails
    """
    try:
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)

        # Encode to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Add data URI prefix
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"

    except Exception as e:
        raise ImageProcessingError(
            f"Failed to encode image to base64: {str(e)}",
            ErrorCode.PROCESSING_ERROR
        )
