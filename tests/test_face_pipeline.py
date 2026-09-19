"""analyze(): detect -> quality filter -> select -> embed only the chosen face -> unit vector."""

from types import SimpleNamespace

import numpy as np
import pytest

from face_recognition_service.models.face_model import (
    FaceModelError,
    FaceRecognitionModel,
    FaceResult,
)
from face_recognition_service.schemas.api_schemas import ErrorCode

IMAGE = np.zeros((200, 200, 3), dtype=np.uint8)


def _face(score: float, box: tuple[float, float, float, float], fill: float = 2.0) -> SimpleNamespace:
    return SimpleNamespace(
        det_score=score, bbox=np.array(box, dtype=np.float32), kps=np.zeros((5, 2)), embedding=np.full(512, fill)
    )


def _model(faces: list, embed=None) -> tuple[FaceRecognitionModel, list]:
    model = FaceRecognitionModel()
    model.model = SimpleNamespace()
    calls: list = []

    def _embed(image, face):
        calls.append(face)
        return face.embedding if embed is None else embed(image, face)

    model._detect = lambda image: faces
    model._embed = _embed
    return model, calls


def test_returns_unit_embedding_and_raw_norm() -> None:
    model, _ = _model([_face(0.9, (0, 0, 50, 50))])
    result = model.analyze(IMAGE)
    assert isinstance(result, FaceResult)
    assert float(np.linalg.norm(result.embedding)) == pytest.approx(1.0)
    assert result.embedding.dtype == np.float32
    assert result.embedding_norm == pytest.approx(2.0 * np.sqrt(512))
    assert result.det_score == pytest.approx(0.9)
    assert result.bbox == (0.0, 0.0, 50.0, 50.0)


def test_recognition_runs_only_on_the_selected_face() -> None:
    small, big, junk = _face(0.95, (0, 0, 20, 20)), _face(0.9, (0, 0, 100, 100)), _face(0.3, (0, 0, 150, 150))
    model, calls = _model([small, big, junk])
    result = model.analyze(IMAGE)
    assert calls == [big]  # det_score x area among faces meeting min_face_quality (0.7)
    assert result.faces_detected == 3


def test_no_face_and_low_quality_keep_their_codes_and_message() -> None:
    with pytest.raises(FaceModelError) as none:
        _model([])[0].analyze(IMAGE)
    assert none.value.error_code == ErrorCode.NO_FACE_DETECTED
    with pytest.raises(FaceModelError) as low:
        _model([_face(0.55, (0, 0, 50, 50))])[0].analyze(IMAGE)
    assert low.value.error_code == ErrorCode.FACE_LOW_QUALITY
    assert "0.55" in low.value.message


@pytest.mark.parametrize("vector", [np.zeros(512), np.full(512, np.nan), np.ones(128)])
def test_degenerate_embeddings_are_invalid(vector: np.ndarray) -> None:
    model, _ = _model([_face(0.9, (0, 0, 50, 50))], embed=lambda image, face: vector)
    with pytest.raises(FaceModelError) as exc_info:
        model.analyze(IMAGE)
    assert exc_info.value.error_code == ErrorCode.INVALID_EMBEDDING


def test_unexpected_errors_become_processing_error() -> None:
    model = FaceRecognitionModel()
    model.model = SimpleNamespace()

    def _boom(image):
        raise RuntimeError("onnx exploded")

    model._detect = _boom
    with pytest.raises(FaceModelError) as exc_info:
        model.analyze(IMAGE)
    assert exc_info.value.error_code == ErrorCode.PROCESSING_ERROR


def test_not_loaded() -> None:
    with pytest.raises(FaceModelError) as exc_info:
        FaceRecognitionModel().analyze(IMAGE)
    assert exc_info.value.error_code == ErrorCode.MODEL_NOT_LOADED


def test_detect_converts_detector_output() -> None:
    model = FaceRecognitionModel()
    boxes = np.array([[1, 2, 30, 40, 0.88], [5, 5, 9, 9, 0.6]], dtype=np.float32)
    kpss = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
    captured = {}

    def _detect(image, max_num, metric):
        captured.update(max_num=max_num, metric=metric)
        return boxes, kpss

    model.model = SimpleNamespace(det_model=SimpleNamespace(detect=_detect))
    faces = model._detect(IMAGE)
    assert captured == {"max_num": 0, "metric": "default"}
    assert len(faces) == 2
    assert faces[0].det_score == pytest.approx(0.88)
    assert faces[0].bbox.tolist() == pytest.approx([1, 2, 30, 40])
    assert faces[1].kps.shape == (5, 2)


def test_get_embedding_wraps_analyze() -> None:
    model, _ = _model([_face(0.9, (0, 0, 50, 50))])
    embedding, score = model.get_embedding(IMAGE)
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0)
    assert score == pytest.approx(0.9)
    assert model.get_embedding(IMAGE, return_detection_info=False)[1] is None


def _padding_model(retry_faces: list) -> tuple[FaceRecognitionModel, list]:
    """Detector finds nothing on the first call and `retry_faces` on the padded retry."""
    model = FaceRecognitionModel()
    model.model = SimpleNamespace()
    seen: list = []

    def _detect(image):
        seen.append(image.shape)
        return [] if len(seen) == 1 else retry_faces

    def _embed(image, face):
        seen.append(("embed", image.shape))
        return face.embedding

    model._detect = _detect
    model._embed = _embed
    return model, seen


def test_padding_retry_finds_tight_crops_and_maps_coordinates_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings

    # "off": isolates the padding-retry mechanic from the enhance_mode="detect_fallback"
    # default, which would otherwise run an extra enhanced-copy detect attempt in between.
    monkeypatch.setattr(settings, "enhance_mode", "off")
    monkeypatch.setattr(settings, "pad_retry_ratio", 0.5)
    face = _face(0.9, (60, 60, 160, 160))  # coordinates in the padded image
    face.kps = np.full((5, 2), 100.0)
    model, seen = _padding_model([face])
    result = model.analyze(np.zeros((112, 112, 3), dtype=np.uint8))
    pad = 56
    assert seen[0] == (112, 112, 3)
    assert seen[1] == (112 + 2 * pad, 112 + 2 * pad, 3)
    assert seen[2] == ("embed", (224, 224, 3))  # embedded from the padded image
    assert result.detected_with_padding is True
    assert result.bbox == (60.0 - pad, 60.0 - pad, 160.0 - pad, 160.0 - pad)
    assert result.kps.tolist() == np.full((5, 2), 100.0 - pad).tolist()


def test_padding_retry_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_recognition_service.config import settings

    monkeypatch.setattr(settings, "enhance_mode", "off")
    monkeypatch.setattr(settings, "pad_retry_ratio", 0.0)
    model, seen = _padding_model([_face(0.9, (0, 0, 50, 50))])
    with pytest.raises(FaceModelError) as exc_info:
        model.analyze(IMAGE)
    assert exc_info.value.error_code == ErrorCode.NO_FACE_DETECTED
    assert len(seen) == 1


def test_no_retry_when_the_first_pass_found_faces() -> None:
    model, _ = _model([_face(0.9, (0, 0, 50, 50))])
    assert model.analyze(IMAGE).detected_with_padding is False
