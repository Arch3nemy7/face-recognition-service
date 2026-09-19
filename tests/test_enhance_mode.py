"""ENHANCE_MODE: enhancement as a detection fallback, not an always-on filter.

Precedence between the new `enhance_mode` and the deprecated `enhance_image`,
and how each mode drives detection/embedding in `FaceRecognitionModel.analyze`.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_recognition_service.config import Settings
from face_recognition_service.models import face_model as face_model_module
from face_recognition_service.models.face_model import FaceRecognitionModel

IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)


def _settings(enhance_mode, enhance_image) -> Settings:
    return Settings(_env_file=None, api_token="x", enhance_mode=enhance_mode, enhance_image=enhance_image)


@pytest.mark.parametrize(
    "enhance_mode, enhance_image, expected",
    [
        (None, None, "detect_fallback"),
        (None, True, "always"),
        (None, False, "off"),
        ("off", True, "off"),
        ("always", False, "always"),
        ("detect_fallback", True, "detect_fallback"),
    ],
)
def test_effective_enhance_mode_precedence(enhance_mode, enhance_image, expected) -> None:
    assert _settings(enhance_mode, enhance_image).effective_enhance_mode == expected


def _face(score: float, box: tuple[float, float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(det_score=score, bbox=np.array(box, dtype=np.float32), kps=np.zeros((5, 2)))


def _model():
    model = FaceRecognitionModel()
    model.model = SimpleNamespace()
    detect_calls: list[np.ndarray] = []
    embed_calls: list[np.ndarray] = []

    def _embed(image, face):
        embed_calls.append(image)
        return np.full(512, 2.0)

    model._embed = _embed
    return model, detect_calls, embed_calls


def test_always_mode_detects_and_embeds_on_the_enhanced_image(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "always")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()

    def _detect(image):
        detect_calls.append(image)
        return [_face(0.9, (0, 0, 50, 50))]

    model._detect = _detect
    result = model.analyze(IMAGE)

    assert len(detect_calls) == 1
    assert np.array_equal(detect_calls[0], IMAGE + 1)
    assert np.array_equal(embed_calls[0], IMAGE + 1)
    assert result.enhanced_for_detection is True


def test_always_mode_pads_the_enhanced_image_when_nothing_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production runs `always`; its padding path must behave like the old
    unconditional-enhance-then-pad-retry code path, including coordinate mapping."""
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "always")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(settings, "pad_retry_ratio", 0.5)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()
    pad = int(round(0.5 * max(IMAGE.shape[:2])))
    face = _face(0.9, (60.0, 60.0, 160.0, 160.0))  # coordinates in the padded, enhanced image
    face.kps = np.full((5, 2), 100.0)

    def _detect(image):
        detect_calls.append(image)
        return [] if len(detect_calls) == 1 else [face]

    model._detect = _detect
    result = model.analyze(IMAGE)

    expected_padded_enhanced = cv2.copyMakeBorder(
        IMAGE + 1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    assert len(detect_calls) == 2
    assert np.array_equal(detect_calls[0], IMAGE + 1)
    assert np.array_equal(detect_calls[1], expected_padded_enhanced)
    assert np.array_equal(embed_calls[0], expected_padded_enhanced)
    assert result.detected_with_padding is True
    assert result.bbox == (60.0 - pad, 60.0 - pad, 160.0 - pad, 160.0 - pad)
    assert result.kps.tolist() == np.full((5, 2), 100.0 - pad).tolist()


def test_off_mode_detects_and_embeds_on_the_original(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "off")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()

    def _detect(image):
        detect_calls.append(image)
        return [_face(0.9, (0, 0, 50, 50))]

    model._detect = _detect
    result = model.analyze(IMAGE)

    assert len(detect_calls) == 1
    assert np.array_equal(detect_calls[0], IMAGE)
    assert np.array_equal(embed_calls[0], IMAGE)
    assert result.enhanced_for_detection is False


def test_detect_fallback_face_found_on_original(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "detect_fallback")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()

    def _detect(image):
        detect_calls.append(image)
        return [_face(0.9, (0, 0, 50, 50))]

    model._detect = _detect
    result = model.analyze(IMAGE)

    assert len(detect_calls) == 1
    assert np.array_equal(detect_calls[0], IMAGE)
    assert np.array_equal(embed_calls[0], IMAGE)
    assert result.enhanced_for_detection is False


def test_detect_fallback_falls_back_to_enhanced_but_embeds_original(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "detect_fallback")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()

    def _detect(image):
        detect_calls.append(image)
        if len(detect_calls) == 1:
            return []
        return [_face(0.9, (0, 0, 50, 50))]

    model._detect = _detect
    result = model.analyze(IMAGE)

    assert len(detect_calls) == 2
    assert np.array_equal(detect_calls[0], IMAGE)
    assert np.array_equal(detect_calls[1], IMAGE + 1)
    assert np.array_equal(embed_calls[0], IMAGE)  # embedded from the original, not the enhanced copy
    assert result.enhanced_for_detection is True


def test_detect_fallback_pads_the_original_when_neither_finds_a_face(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "detect_fallback")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(settings, "pad_retry_ratio", 0.5)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()
    pad = int(round(0.5 * max(IMAGE.shape[:2])))
    face = _face(0.9, (60.0, 60.0, 110.0, 110.0))  # coordinates in the padded original
    face.kps = np.full((5, 2), 90.0)

    def _detect(image):
        detect_calls.append(image)
        if len(detect_calls) < 3:
            return []
        return [face]

    model._detect = _detect
    result = model.analyze(IMAGE)

    assert len(detect_calls) == 3
    assert detect_calls[0].shape == IMAGE.shape
    assert detect_calls[1].shape == IMAGE.shape  # enhanced copy: same dims as original
    padded_shape = (IMAGE.shape[0] + 2 * pad, IMAGE.shape[1] + 2 * pad, 3)
    assert detect_calls[2].shape == padded_shape
    # the interior is the original (all zeros), not the enhanced copy (IMAGE + 1 == all ones)
    assert np.array_equal(detect_calls[2][pad:-pad, pad:-pad], IMAGE)
    assert embed_calls[0].shape == padded_shape  # embedded from the padded original
    assert np.array_equal(embed_calls[0][pad:-pad, pad:-pad], IMAGE)
    assert result.enhanced_for_detection is False
    assert result.bbox == (60.0 - pad, 60.0 - pad, 110.0 - pad, 110.0 - pad)
    assert result.kps.tolist() == np.full((5, 2), 90.0 - pad).tolist()


def test_detect_fallback_pads_the_enhanced_image_as_a_last_resort(monkeypatch: pytest.MonkeyPatch) -> None:
    """A face that's both tightly cropped and dark: original detect fails,
    unpadded enhanced detect fails, padded-original detect fails, and only
    padded-enhanced finds it. Embedding must still come from the padded
    ORIGINAL pixels, with the same pad used for the padded-enhanced detect."""
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "detect_fallback")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(settings, "pad_retry_ratio", 0.5)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()
    pad = int(round(0.5 * max(IMAGE.shape[:2])))
    face = _face(0.9, (60.0, 60.0, 160.0, 160.0))  # coordinates in the padded image
    face.kps = np.full((5, 2), 100.0)

    def _detect(image):
        detect_calls.append(image)
        return [] if len(detect_calls) < 4 else [face]

    model._detect = _detect
    result = model.analyze(IMAGE)

    expected_padded_original = cv2.copyMakeBorder(
        IMAGE, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    expected_padded_enhanced = cv2.copyMakeBorder(
        IMAGE + 1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    assert len(detect_calls) == 4
    assert np.array_equal(detect_calls[0], IMAGE)  # original
    assert np.array_equal(detect_calls[1], IMAGE + 1)  # unpadded enhanced
    assert np.array_equal(detect_calls[2], expected_padded_original)  # padded original
    assert np.array_equal(detect_calls[3], expected_padded_enhanced)  # padded enhanced (last resort)
    assert np.array_equal(embed_calls[0], expected_padded_original)  # embedded from padded ORIGINAL
    assert result.enhanced_for_detection is True
    assert result.detected_with_padding is True
    assert result.bbox == (60.0 - pad, 60.0 - pad, 160.0 - pad, 160.0 - pad)
    assert result.kps.tolist() == np.full((5, 2), 100.0 - pad).tolist()


def test_detect_fallback_falls_back_when_original_face_is_below_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dark selfie: the original pass finds a face, but its det_score
    (0.6) is below min_face_quality (0.7, the default) -- that must still
    trigger the enhanced-copy fallback, not be accepted (or rejected)
    immediately. The enhanced pass finds the same face at 0.8, which is
    valid, so the request succeeds (embedding from the original pixels)."""
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "detect_fallback")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()
    low_quality = _face(0.6, (0, 0, 50, 50))
    high_quality = _face(0.8, (0, 0, 50, 50))

    def _detect(image):
        detect_calls.append(image)
        return [low_quality] if len(detect_calls) == 1 else [high_quality]

    model._detect = _detect
    result = model.analyze(IMAGE)

    assert len(detect_calls) == 2
    assert np.array_equal(detect_calls[0], IMAGE)
    assert np.array_equal(detect_calls[1], IMAGE + 1)
    assert np.array_equal(embed_calls[0], IMAGE)  # embedded from the original, not the enhanced copy
    assert result.det_score == pytest.approx(0.8)
    assert result.enhanced_for_detection is True


def test_detect_fallback_reports_best_score_across_attempts_when_never_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither the original (0.6) nor the enhanced (0.65) pass clears
    min_face_quality (0.7): same FACE_LOW_QUALITY semantics as today, with
    the best score reported across every attempt (0.65, not 0.6)."""
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "detect_fallback")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(settings, "pad_retry_ratio", 0.0)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()

    def _detect(image):
        detect_calls.append(image)
        if len(detect_calls) == 1:
            return [_face(0.6, (0, 0, 50, 50))]
        return [_face(0.65, (0, 0, 50, 50))]

    model._detect = _detect
    with pytest.raises(face_model_module.FaceModelError) as exc_info:
        model.analyze(IMAGE)

    assert len(detect_calls) == 2
    assert exc_info.value.error_code == face_model_module.ErrorCode.FACE_LOW_QUALITY
    assert "0.65" in exc_info.value.message
    assert not embed_calls


def test_detect_fallback_no_detections_anywhere_is_no_face_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "detect_fallback")
    monkeypatch.setattr(settings, "enhance_image", None)
    monkeypatch.setattr(settings, "pad_retry_ratio", 0.0)
    monkeypatch.setattr(face_model_module, "_enhance_image", lambda img: img + 1)

    model, detect_calls, embed_calls = _model()
    model._detect = lambda image: []

    with pytest.raises(face_model_module.FaceModelError) as exc_info:
        model.analyze(IMAGE)

    assert exc_info.value.error_code == face_model_module.ErrorCode.NO_FACE_DETECTED
    assert not embed_calls


def test_preprocess_image_no_longer_enhances(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings
    from face_recognition_service.utils.image_utils import preprocess_image

    monkeypatch.setattr(settings, "enhance_image", True)
    monkeypatch.setattr(settings, "enhance_mode", None)

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    result = preprocess_image(image.copy())
    assert np.array_equal(result, image)
