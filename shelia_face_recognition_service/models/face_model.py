"""Face recognition model loading and inference using InsightFace."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from ..config import settings
from ..schemas.api_schemas import ErrorCode

logger = logging.getLogger(__name__)


class FaceModelError(Exception):
    """Exception raised for face model errors."""

    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class FaceRecognitionModel:
    """Face recognition model wrapper using InsightFace."""

    def __init__(self):
        """Initialize the face recognition model."""
        self.model: Optional[FaceAnalysis] = None
        self.model_name: str = settings.model_name
        self.device: str = settings.device
        self.detection_threshold: float = settings.detection_threshold
        self.embedding_size: int = settings.embedding_size

    def load(self) -> None:
        """
        Load the face recognition model.

        This should be called once at application startup.

        Raises:
            FaceModelError: If model loading fails
        """
        try:
            logger.info(f"Loading face recognition model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            logger.info(f"ONNX Runtime providers: {settings.providers}")

            # Initialize FaceAnalysis
            self.model = FaceAnalysis(
                name=self.model_name,
                providers=settings.providers,
            )

            # Prepare model with detection size
            # ctx_id: -1 for CPU, 0 for GPU
            ctx_id = -1 if self.device == "cpu" else 0
            det_size = (640, 640)  # Detection input size

            self.model.prepare(
                ctx_id=ctx_id,
                det_size=det_size,
                det_thresh=self.detection_threshold
            )

            logger.info("Face recognition model loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load face recognition model: {str(e)}"
            logger.error(error_msg)
            raise FaceModelError(error_msg, ErrorCode.MODEL_NOT_LOADED)

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def get_embedding(
        self,
        image: np.ndarray,
        return_detection_info: bool = True
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Extract face embedding from an image.

        Args:
            image: Input image as numpy array (BGR format)
            return_detection_info: Whether to return detection score

        Returns:
            Tuple of (embedding, detection_score)
            - embedding: 512-dimensional face embedding as numpy array
            - detection_score: Confidence score of face detection (0-1), or None

        Raises:
            FaceModelError: If model is not loaded, no face detected, or multiple faces detected
        """
        if not self.is_loaded():
            raise FaceModelError(
                "Face recognition model is not loaded",
                ErrorCode.MODEL_NOT_LOADED
            )

        try:
            # Detect faces
            faces = self.model.get(image)

            # Check number of faces detected
            if len(faces) == 0:
                raise FaceModelError(
                    "No face detected in the image",
                    ErrorCode.NO_FACE_DETECTED
                )

            if len(faces) > 1:
                raise FaceModelError(
                    f"Multiple faces detected ({len(faces)}). "
                    "Please provide an image with a single face.",
                    ErrorCode.MULTIPLE_FACES_DETECTED
                )

            # Get the face
            face = faces[0]

            # Extract embedding
            embedding = face.embedding

            # Validate embedding
            if embedding is None or len(embedding) != self.embedding_size:
                raise FaceModelError(
                    f"Invalid embedding extracted. Expected size {self.embedding_size}, "
                    f"got {len(embedding) if embedding is not None else 'None'}",
                    ErrorCode.INVALID_EMBEDDING
                )

            # Get detection score if available
            detection_score = None
            if return_detection_info and hasattr(face, 'det_score'):
                detection_score = float(face.det_score)

            return embedding, detection_score

        except FaceModelError:
            raise
        except Exception as e:
            logger.error(f"Error during embedding extraction: {str(e)}")
            raise FaceModelError(
                f"Failed to extract embedding: {str(e)}",
                ErrorCode.PROCESSING_ERROR
            )

    def detect_faces(self, image: np.ndarray) -> int:
        """
        Detect the number of faces in an image without extracting embeddings.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Number of faces detected

        Raises:
            FaceModelError: If model is not loaded or detection fails
        """
        if not self.is_loaded():
            raise FaceModelError(
                "Face recognition model is not loaded",
                ErrorCode.MODEL_NOT_LOADED
            )

        try:
            faces = self.model.get(image)
            return len(faces)

        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            raise FaceModelError(
                f"Failed to detect faces: {str(e)}",
                ErrorCode.PROCESSING_ERROR
            )

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self.model_name,
            "embedding_size": self.embedding_size,
            "backend": "insightface",
            "device": self.device,
            "detection_threshold": self.detection_threshold,
            "loaded": self.is_loaded(),
        }


# Global model instance (singleton pattern)
_model_instance: Optional[FaceRecognitionModel] = None


def get_model() -> FaceRecognitionModel:
    """
    Get the global face recognition model instance.

    Returns:
        FaceRecognitionModel instance

    Raises:
        FaceModelError: If model is not initialized
    """
    global _model_instance

    if _model_instance is None:
        raise FaceModelError(
            "Face recognition model not initialized. Call initialize_model() first.",
            ErrorCode.MODEL_NOT_LOADED
        )

    return _model_instance


def initialize_model() -> None:
    """
    Initialize the global face recognition model.

    This should be called once at application startup.

    Raises:
        FaceModelError: If model loading fails
    """
    global _model_instance

    logger.info("Initializing face recognition model...")
    _model_instance = FaceRecognitionModel()
    _model_instance.load()
    logger.info("Face recognition model initialized successfully")


def cleanup_model() -> None:
    """
    Cleanup the global face recognition model.

    This should be called at application shutdown.
    """
    global _model_instance

    if _model_instance is not None:
        logger.info("Cleaning up face recognition model...")
        _model_instance = None
        logger.info("Face recognition model cleaned up")
