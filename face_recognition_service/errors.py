"""Base exception for every face-service error, and the FastAPI handlers that
turn it (and every other kind of failure) into one response envelope.

`FaceModelError` (models/face_model.py) and `ImageProcessingError`
(utils/image_utils.py) both subclass `FaceServiceError` so the single
`@app.exception_handler(FaceServiceError)` in main.py covers both without two
near-identical handlers.
"""

from typing import Optional


class FaceServiceError(Exception):
    """Base exception for every face-service error.

    Attributes:
        message: Client-facing error text (never leaks internals -- see
            main.py's generic-500 handling for the unexpected-exception case).
        error_code: One of schemas.api_schemas.ErrorCode.
        image: Which photo ("reference"/"selfie") this error is about, when
            applicable; set by the endpoint that knows the role, not by the
            code that raises the error.
        headers: Extra HTTP headers to send with the response (e.g.
            WWW-Authenticate), when applicable.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        image: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.image = image
        self.headers = headers
        super().__init__(self.message)
