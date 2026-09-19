"""Face recognition model loading and inference using InsightFace."""

import logging
import math
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
from insightface.app.common import Face
from insightface.utils import face_align
from insightface.utils.storage import ensure_available

from ..config import settings
from ..errors import FaceServiceError
from ..schemas.api_schemas import ErrorCode
from ..utils.image_utils import _enhance_image
from .loader import load_pack, resolve_pack_dir, session_options

logger = logging.getLogger(__name__)


class FaceModelError(FaceServiceError):
    """Exception raised for face model errors."""

    def __init__(self, message: str, error_code: str, image: Optional[str] = None):
        # Which photo (reference/selfie) this error is about; set by the
        # endpoint that knows the role, not by the model itself.
        super().__init__(message, error_code, image=image)


@dataclass(frozen=True)
class FaceQuality:
    """Quality metrics for the chosen face, reported regardless of whether any
    gate is enabled so real distributions can calibrate future thresholds.

    Landmark-based fields are `None` when the face has no `kps` (5-point:
    left eye, right eye, nose, left mouth corner, right mouth corner).
    """

    face_size_px: float  # shorter side of the bbox
    interocular_px: Optional[float]  # distance between the two eye landmarks
    roll_deg: Optional[float]  # angle of the eye line, degrees (atan2(dy, dx))
    yaw_proxy: Optional[float]  # (nose_x - eye_mid_x) / interocular_px; ~0 frontal
    blur_variance: Optional[float]  # variance of the Laplacian on the aligned 112x112 grayscale crop
    embedding_norm: float  # L2 norm of the raw embedding, before normalisation
    faces_considered: int  # faces passing min_face_quality
    second_face_ratio: float  # area of the next-largest qualifying face / chosen face area; 0.0 if none


@dataclass(frozen=True, eq=False)
class FaceResult:
    """The face chosen in an image and what the pipeline knows about it."""

    embedding: np.ndarray  # unit-normalised float32
    embedding_norm: float  # L2 norm before normalisation; low values often mean a poor face
    det_score: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels
    kps: Optional[np.ndarray]  # 5-point landmarks in image pixels
    faces_detected: int  # detections above the detector threshold in the pass that produced this result (the padded retry, if it ran)
    detected_with_padding: bool = False  # True if the first pass found nothing and a padded retry recovered a face
    enhanced_for_detection: bool = False  # True if the face was found on the enhanced copy (always: whenever found at all)
    quality: Optional[FaceQuality] = None


class FaceRecognitionModel:
    """Face recognition model wrapper using InsightFace."""

    def __init__(self):
        """Initialize the face recognition model."""
        self.model: Optional[object] = None
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
            logger.info(f"Device: {self.device}; ONNX Runtime providers: {settings.providers}")
            # INSIGHTFACE_HOME is passed explicitly so the model is found where the image baked it.
            root = os.path.expanduser(os.environ.get('INSIGHTFACE_HOME', '~/.insightface'))
            try:
                pack_dir = resolve_pack_dir(root, self.model_name)
            except FileNotFoundError:
                ensure_available('models', self.model_name, root=root)  # download + unzip, as FaceAnalysis did
                pack_dir = resolve_pack_dir(root, self.model_name)
            self.model = load_pack(
                pack_dir,
                name=self.model_name,
                providers=settings.providers,
                options=session_options(settings.ort_intra_op_threads, settings.ort_inter_op_threads),
                det_size=settings.det_size,
                det_thresh=self.detection_threshold,
            )
            logger.info("Face recognition model loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load face recognition model: {str(e)}"
            logger.error(error_msg)
            raise FaceModelError(error_msg, ErrorCode.MODEL_NOT_LOADED) from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    @staticmethod
    def _area(bbox) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    @classmethod
    def _select_best_face(cls, faces):
        """Select dominant face by det_score × bounding-box area."""
        return max(faces, key=lambda f: f.det_score * cls._area(f.bbox))

    @staticmethod
    def _landmark_metrics(kps) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """interocular_px, roll_deg, yaw_proxy from 5-point landmarks (left eye,
        right eye, nose, left mouth, right mouth), or (None, None, None) without kps."""
        if kps is None:
            return None, None, None
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        dx, dy = float(right_eye[0]) - float(left_eye[0]), float(right_eye[1]) - float(left_eye[1])
        interocular = math.hypot(dx, dy)
        roll = math.degrees(math.atan2(dy, dx))
        if interocular > 0:
            eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
            yaw = (float(nose[0]) - eye_mid_x) / interocular
        else:
            yaw = None
        return interocular, roll, yaw

    @staticmethod
    def _blur_variance(embed_image: np.ndarray, kps) -> Optional[float]:
        """Variance of the Laplacian on the grayscale, aligned 112x112 crop; None without kps."""
        if kps is None:
            return None
        crop = face_align.norm_crop(embed_image, kps, image_size=112)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _reject_if(condition: bool, message: str) -> None:
        if condition:
            raise FaceModelError(message, ErrorCode.FACE_LOW_QUALITY)

    @classmethod
    def _check_multi_face(cls, chosen, others: list) -> None:
        """Raise MULTIPLE_FACES_DETECTED when policy is `reject_ambiguous` and
        another face is close in score and size to the chosen one.

        `others` is every face the detector found in this pass (not just
        those meeting `min_face_quality`), so a bystander with a det_score
        >= `multi_face_min_score` but below `min_face_quality` still counts
        here -- otherwise `multi_face_min_score` could never fire below
        `min_face_quality` and would be dead configuration. This is
        deliberately a different population than `second_face_ratio` on
        `FaceQuality`, which is computed only among faces that passed
        `min_face_quality` (the ones actually "considered").
        """
        if settings.multi_face_policy != "reject_ambiguous":
            return
        chosen_area = cls._area(chosen.bbox)
        for f in others:
            if f is chosen:
                continue
            area_ratio = cls._area(f.bbox) / chosen_area if chosen_area > 0 else 0.0
            if f.det_score >= settings.multi_face_min_score and area_ratio >= settings.multi_face_area_ratio:
                raise FaceModelError(
                    "Multiple faces of similar prominence were detected "
                    f"(second face: det_score {f.det_score:.2f}, area ratio {area_ratio:.2f}). "
                    "Please retake the photo with only one person in frame.",
                    ErrorCode.MULTIPLE_FACES_DETECTED,
                )

    def _detect(self, image: np.ndarray) -> list:
        """Run the detector only: every face above the detection threshold."""
        bboxes, kpss = self.model.det_model.detect(image, max_num=0, metric="default")
        return [
            Face(
                bbox=bboxes[i, 0:4],
                kps=None if kpss is None else kpss[i],
                det_score=float(bboxes[i, 4]),
            )
            for i in range(bboxes.shape[0])
        ]

    def _embed(self, image: np.ndarray, face) -> np.ndarray:
        """Run recognition on one face: aligned 112x112 crop from its landmarks -> raw embedding."""
        return self.model.rec_model.get(image, face)

    def _detect_with_pad_retry(self, image: np.ndarray) -> tuple[list, np.ndarray, int]:
        """Detect on `image`; if nothing is found and padding retry is enabled, pad and
        retry once. Returns (faces, image detection ran on last, pad applied to that image)."""
        faces = self._detect(image)
        pad = 0
        if not faces and settings.pad_retry_ratio > 0:
            # A face that fills the whole frame (e.g. a tightly cropped
            # ID-card-style reference photo) gives the detector no context
            # and is often missed.
            pad = int(round(settings.pad_retry_ratio * max(image.shape[:2])))
            image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = self._detect(image)
        return faces, image, pad

    def _find_faces(self, image: np.ndarray) -> tuple[list, np.ndarray, int, bool]:
        """Detect faces per `settings.effective_enhance_mode`.

        Returns (faces, image to embed from, pad applied to that image,
        whether the face was found on the enhanced copy).

        - "always": detect and embed on the enhanced image; padding retry
          pads the enhanced image (bit-identical to today's ENHANCE_IMAGE=true).
        - "off": detect and embed on the original; padding retry pads the original.
        - "detect_fallback": detect on the original; if no detection there
          meets `min_face_quality` (not merely "if nothing is found at
          all" -- a face detected but too dark/low-score to pass the
          quality gate must still get the enhancement safety net, or a dark
          reference photo could be rejected under `detect_fallback` where
          `always` would have accepted it), detect on the enhanced copy;
          if that also has no valid detection and padding retry is
          enabled, pad the *original* and detect; if still nothing valid,
          detect on the *padded enhanced* copy as a last resort (this is
          the same final step `always` gets from its own padding retry, so
          a tightly-cropped *and* dark image doesn't fail here where
          `always` would succeed). Embedding always comes from the
          original (or padded original) pixels -- the gamma LUT and CLAHE
          don't move pixels, so landmarks found on the enhanced copy are
          still valid there.

          If no attempt ever finds a valid detection, the faces from every
          attempt are pooled together so the caller sees the same error it
          would today: NO_FACE_DETECTED if nothing was ever detected, or
          FACE_LOW_QUALITY reporting the best `det_score` seen across every
          attempt otherwise.
        """
        mode = settings.effective_enhance_mode

        if mode == "always":
            faces, embed_image, pad = self._detect_with_pad_retry(_enhance_image(image))
            return faces, embed_image, pad, bool(faces)

        if mode == "off":
            faces, embed_image, pad = self._detect_with_pad_retry(image)
            return faces, embed_image, pad, False

        # detect_fallback
        def _has_valid_face(faces: list) -> bool:
            return any(f.det_score >= settings.min_face_quality for f in faces)

        all_faces: list = []

        faces = self._detect(image)
        all_faces += faces
        if _has_valid_face(faces):
            return faces, image, 0, False

        enhanced = _enhance_image(image)
        faces = self._detect(enhanced)
        all_faces += faces
        if _has_valid_face(faces):
            return faces, image, 0, True

        if settings.pad_retry_ratio <= 0:
            return all_faces, image, 0, False

        pad = int(round(settings.pad_retry_ratio * max(image.shape[:2])))
        padded_original = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = self._detect(padded_original)
        all_faces += faces
        if _has_valid_face(faces):
            return faces, padded_original, pad, False

        padded_enhanced = cv2.copyMakeBorder(enhanced, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = self._detect(padded_enhanced)
        all_faces += faces
        if _has_valid_face(faces):
            return faces, padded_original, pad, True

        return all_faces, padded_original, pad, False

    def analyze(self, image: np.ndarray, role: Literal["selfie", "reference"] = "selfie") -> FaceResult:
        """
        Detect faces, keep those meeting the quality threshold, pick the dominant
        one and embed only that face.

        Quality metrics are always computed and reported on the result. Quality
        gates and the multi-face policy are opt-in (every limit defaults to 0 =
        off) and, even when enabled, only apply to `role="selfie"` unless
        `settings.quality_gates_apply_to_reference` is set -- a rejected
        reference photo is a serious problem for the enrolled person, so it
        never happens by surprise.

        Raises:
            FaceModelError: MODEL_NOT_LOADED, NO_FACE_DETECTED, FACE_LOW_QUALITY,
                MULTIPLE_FACES_DETECTED, INVALID_EMBEDDING or PROCESSING_ERROR
        """
        if not self.is_loaded():
            raise FaceModelError("Face recognition model is not loaded", ErrorCode.MODEL_NOT_LOADED)
        try:
            faces, image, pad, enhanced_for_detection = self._find_faces(image)
            valid_faces = [f for f in faces if f.det_score >= settings.min_face_quality]
            if not valid_faces:
                if not faces:
                    # Nothing detected at all -- distinct from "found a face
                    # but it's too blurry/dark/small/far" (FACE_LOW_QUALITY
                    # below), since the two need different user guidance.
                    raise FaceModelError(
                        "No face was detected in the image. "
                        "Please use a clearer, well-lit, close-up photo.",
                        ErrorCode.NO_FACE_DETECTED
                    )
                best_score = max(f.det_score for f in faces)
                raise FaceModelError(
                    f"Face(s) detected but none met the quality threshold "
                    f"({settings.min_face_quality}). Best score was {best_score:.2f}. "
                    "Please use a clearer, well-lit, close-up photo.",
                    ErrorCode.FACE_LOW_QUALITY
                )

            # Select the dominant face (highest det_score × face area) and
            # embed only that one -- recognition is expensive, and every
            # other detected box (including detector junk) never needs it.
            face = self._select_best_face(valid_faces)
            gates_apply = role == "selfie" or settings.quality_gates_apply_to_reference

            if gates_apply:
                # Deliberately `faces` (every detection this pass), not
                # `valid_faces` -- see _check_multi_face's docstring.
                self._check_multi_face(face, faces)

            size = min(
                max(0.0, float(face.bbox[2]) - float(face.bbox[0])),
                max(0.0, float(face.bbox[3]) - float(face.bbox[1])),
            )
            if gates_apply and settings.min_face_size_px > 0:
                self._reject_if(
                    size < settings.min_face_size_px,
                    f"Face too small: {size:.0f} px < {settings.min_face_size_px:.0f} px minimum",
                )

            interocular, roll, yaw = self._landmark_metrics(face.kps)

            # A metric that is None (no landmarks) always skips its gate rather
            # than rejecting -- missing data must never create a new
            # rejection the caller can't act on.
            if gates_apply and settings.min_interocular_px > 0:
                self._reject_if(
                    interocular is not None and interocular < settings.min_interocular_px,
                    f"Interocular distance too small: {(interocular or 0.0):.1f} px "
                    f"< {settings.min_interocular_px:.1f} px minimum",
                )
            if gates_apply and settings.max_abs_roll_deg > 0:
                # `roll is not None and ...`: None (no landmarks) skips this
                # gate rather than rejecting.
                self._reject_if(
                    roll is not None and abs(roll) > settings.max_abs_roll_deg,
                    f"Roll too large: {abs(roll or 0.0):.1f} deg > {settings.max_abs_roll_deg:.1f} deg maximum",
                )
            if gates_apply and settings.max_abs_yaw_proxy > 0:
                # Same None-skips-the-gate rule as roll above.
                self._reject_if(
                    yaw is not None and abs(yaw) > settings.max_abs_yaw_proxy,
                    f"Yaw too large: {abs(yaw or 0.0):.2f} > {settings.max_abs_yaw_proxy:.2f} maximum",
                )

            blur = self._blur_variance(image, face.kps)
            if gates_apply and settings.min_blur_variance > 0:
                self._reject_if(
                    blur is not None and blur < settings.min_blur_variance,
                    f"Face too blurry: variance {(blur or 0.0):.1f} < {settings.min_blur_variance:.1f} minimum",
                )

            raw = np.asarray(self._embed(image, face), dtype=np.float32).ravel()
            if raw.size != self.embedding_size:
                raise FaceModelError(
                    f"Invalid embedding extracted. Expected size {self.embedding_size}, got {raw.size}",
                    ErrorCode.INVALID_EMBEDDING
                )
            norm = float(np.linalg.norm(raw))
            if not np.isfinite(norm) or norm == 0.0:
                raise FaceModelError(
                    "Invalid embedding extracted (zero or non-finite norm)",
                    ErrorCode.INVALID_EMBEDDING
                )
            if gates_apply and settings.min_embedding_norm > 0:
                self._reject_if(
                    norm < settings.min_embedding_norm,
                    f"Embedding norm too low: {norm:.2f} < {settings.min_embedding_norm:.2f} minimum",
                )

            second_face_ratio = 0.0
            chosen_area = self._area(face.bbox)
            other_areas = [self._area(f.bbox) for f in valid_faces if f is not face]
            if other_areas and chosen_area > 0:
                second_face_ratio = max(other_areas) / chosen_area

            quality = FaceQuality(
                face_size_px=size,
                interocular_px=interocular,
                roll_deg=roll,
                yaw_proxy=yaw,
                blur_variance=blur,
                embedding_norm=norm,
                faces_considered=len(valid_faces),
                second_face_ratio=second_face_ratio,
            )

            return FaceResult(
                embedding=raw / norm,
                embedding_norm=norm,
                det_score=float(face.det_score),
                bbox=tuple(float(v) - pad for v in face.bbox),
                kps=None if face.kps is None else np.asarray(face.kps, dtype=np.float32) - pad,
                faces_detected=len(faces),
                detected_with_padding=pad > 0,
                enhanced_for_detection=enhanced_for_detection,
                quality=quality,
            )
        except FaceModelError:
            raise
        except Exception as e:
            logger.error(f"Error during face analysis: {str(e)}")
            raise FaceModelError(
                f"Failed to extract embedding: {str(e)}",
                ErrorCode.PROCESSING_ERROR
            ) from e

    def get_embedding(
        self,
        image: np.ndarray,
        return_detection_info: bool = True,
        role: Literal["selfie", "reference"] = "selfie",
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Unit-normalised embedding of the dominant face, and its detection score.

        Thin wrapper over analyze(); kept because the API and its tests call it.

        Raises:
            FaceModelError: If the model is not loaded (MODEL_NOT_LOADED), no face
                was found (NO_FACE_DETECTED), a face was found but below the quality
                threshold (FACE_LOW_QUALITY), multiple faces of similar prominence
                were found (MULTIPLE_FACES_DETECTED, opt-in only), the extracted
                embedding is invalid (INVALID_EMBEDDING), or another failure
                occurred while processing the image (PROCESSING_ERROR)
        """
        result = self.analyze(image, role=role)
        return result.embedding, (result.det_score if return_detection_info else None)

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
