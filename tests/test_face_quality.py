"""Face quality metrics, optional gates (off by default), opt-in multi-face policy."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_recognition_service.config import settings
from face_recognition_service.models.face_model import (
    FaceModelError,
    FaceRecognitionModel,
)
from face_recognition_service.schemas.api_schemas import ErrorCode

IMAGE = np.zeros((200, 200, 3), dtype=np.uint8)


def _face(
    score: float,
    box: tuple[float, float, float, float],
    kps=None,
    fill: float = 2.0,
) -> SimpleNamespace:
    if kps is None:
        kps = np.array([[40, 50], [60, 50], [50, 60], [45, 75], [55, 75]], dtype=np.float32)
    return SimpleNamespace(
        det_score=score,
        bbox=np.array(box, dtype=np.float32),
        kps=None if kps is None else np.asarray(kps, dtype=np.float32),
        embedding=np.full(512, fill),
    )


def _model(faces: list) -> FaceRecognitionModel:
    model = FaceRecognitionModel()
    model.model = SimpleNamespace()
    model._detect = lambda image: faces
    model._embed = lambda image, face: face.embedding
    return model


# --- Metric math -------------------------------------------------------


def test_interocular_roll_yaw_size_and_embedding_norm() -> None:
    kps = [[40, 50], [60, 50], [50, 60], [45, 75], [55, 75]]
    face = _face(0.9, (0, 0, 80, 100), kps=kps, fill=2.0)
    result = _model([face]).analyze(IMAGE)
    q = result.quality
    assert q.interocular_px == pytest.approx(20.0)
    assert q.roll_deg == pytest.approx(0.0)
    assert q.yaw_proxy == pytest.approx(0.0)
    assert q.face_size_px == pytest.approx(80.0)
    assert q.embedding_norm == pytest.approx(2.0 * np.sqrt(512))
    assert result.embedding_norm == q.embedding_norm


def test_roll_for_tilted_eyes() -> None:
    kps = [[40, 50], [60, 60], [50, 60], [45, 75], [55, 75]]
    face = _face(0.9, (0, 0, 80, 100), kps=kps)
    result = _model([face]).analyze(IMAGE)
    assert result.quality.roll_deg == pytest.approx(26.565, abs=0.01)


def test_yaw_for_offset_nose() -> None:
    kps = [[40, 50], [60, 50], [55, 60], [45, 75], [55, 75]]
    face = _face(0.9, (0, 0, 80, 100), kps=kps)
    result = _model([face]).analyze(IMAGE)
    assert result.quality.yaw_proxy == pytest.approx(0.25)


def test_landmark_fields_none_without_kps() -> None:
    face = _face(0.9, (0, 0, 80, 100))
    face.kps = None
    result = _model([face]).analyze(IMAGE)
    q = result.quality
    assert q.interocular_px is None
    assert q.roll_deg is None
    assert q.yaw_proxy is None
    assert q.blur_variance is None
    assert q.face_size_px == pytest.approx(80.0)
    assert q.embedding_norm == pytest.approx(2.0 * np.sqrt(512))


# --- Blur ---------------------------------------------------------------


def test_blur_variance_higher_for_sharp_image() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3)
    kps = [[70, 80], [130, 80], [100, 110], [80, 140], [120, 140]]

    sharp_result = _model([_face(0.9, (30, 30, 170, 170), kps=kps)]).analyze(sharp)
    blurred_result = _model([_face(0.9, (30, 30, 170, 170), kps=kps)]).analyze(blurred)

    assert sharp_result.quality.blur_variance > blurred_result.quality.blur_variance


# --- Defaults / second_face_ratio ---------------------------------------


def test_defaults_never_reject_even_a_tiny_face() -> None:
    face = _face(0.9, (0, 0, 10, 10))
    result = _model([face]).analyze(IMAGE)
    assert result.quality.face_size_px == pytest.approx(10.0)


def test_second_face_ratio_with_two_qualifying_faces() -> None:
    big = _face(0.9, (0, 0, 10, 10))  # area 100, score*area 90 -> chosen
    small = _face(0.8, (0, 0, 8, 8))  # area 64, score*area 51.2
    result = _model([big, small]).analyze(IMAGE)
    assert result.quality.faces_considered == 2
    assert result.quality.second_face_ratio == pytest.approx(64.0 / 100.0)


def test_second_face_ratio_zero_with_one_face() -> None:
    face = _face(0.9, (0, 0, 10, 10))
    result = _model([face]).analyze(IMAGE)
    assert result.quality.second_face_ratio == pytest.approx(0.0)


# --- Gates: each rejects when enabled, passes at exactly the limit ------


def test_min_face_size_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "min_face_size_px", 60.0)
    face = _face(0.9, (0, 0, 38, 100))  # shorter side 38
    with pytest.raises(FaceModelError) as exc:
        _model([face]).analyze(IMAGE)
    assert exc.value.error_code == ErrorCode.FACE_LOW_QUALITY
    assert "Face too small" in exc.value.message
    assert "38" in exc.value.message and "60" in exc.value.message

    ok_face = _face(0.9, (0, 0, 60, 100))  # exactly at the limit -> passes
    result = _model([ok_face]).analyze(IMAGE)
    assert result.quality.face_size_px == pytest.approx(60.0)


def test_min_interocular_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "min_interocular_px", 20.0)
    kps = [[40, 50], [55, 50], [47.5, 60], [45, 75], [55, 75]]  # interocular 15
    face = _face(0.9, (0, 0, 80, 100), kps=kps)
    with pytest.raises(FaceModelError) as exc:
        _model([face]).analyze(IMAGE)
    assert exc.value.error_code == ErrorCode.FACE_LOW_QUALITY
    assert "nterocular" in exc.value.message

    ok_kps = [[40, 50], [60, 50], [50, 60], [45, 75], [55, 75]]  # interocular exactly 20
    ok_face = _face(0.9, (0, 0, 80, 100), kps=ok_kps)
    result = _model([ok_face]).analyze(IMAGE)
    assert result.quality.interocular_px == pytest.approx(20.0)


def test_max_roll_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    ok_kps = [[40, 50], [60, 60], [50, 60], [45, 75], [55, 75]]  # roll ~= 26.565
    baseline_roll = _model([_face(0.9, (0, 0, 80, 100), kps=ok_kps)]).analyze(IMAGE).quality.roll_deg

    monkeypatch.setattr(settings, "max_abs_roll_deg", baseline_roll)
    steep_kps = [[40, 50], [60, 61], [50, 60], [45, 75], [55, 75]]
    face = _face(0.9, (0, 0, 80, 100), kps=steep_kps)
    with pytest.raises(FaceModelError) as exc:
        _model([face]).analyze(IMAGE)
    assert exc.value.error_code == ErrorCode.FACE_LOW_QUALITY
    assert "oll" in exc.value.message

    ok_face = _face(0.9, (0, 0, 80, 100), kps=ok_kps)  # exactly at the limit -> passes
    result = _model([ok_face]).analyze(IMAGE)
    assert result.quality.roll_deg == pytest.approx(26.565, abs=0.01)


def test_max_yaw_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_abs_yaw_proxy", 0.25)
    steep_kps = [[40, 50], [60, 50], [57, 60], [45, 75], [55, 75]]  # yaw 0.35
    face = _face(0.9, (0, 0, 80, 100), kps=steep_kps)
    with pytest.raises(FaceModelError) as exc:
        _model([face]).analyze(IMAGE)
    assert exc.value.error_code == ErrorCode.FACE_LOW_QUALITY
    assert "aw" in exc.value.message

    ok_kps = [[40, 50], [60, 50], [55, 60], [45, 75], [55, 75]]  # yaw exactly 0.25
    ok_face = _face(0.9, (0, 0, 80, 100), kps=ok_kps)
    result = _model([ok_face]).analyze(IMAGE)
    assert result.quality.yaw_proxy == pytest.approx(0.25)


def test_min_blur_variance_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8).astype(np.uint8)
    kps = [[70, 80], [130, 80], [100, 110], [80, 140], [120, 140]]
    face = _face(0.9, (30, 30, 170, 170), kps=kps)
    baseline = _model([face]).analyze(sharp).quality.blur_variance

    monkeypatch.setattr(settings, "min_blur_variance", baseline + 1000.0)
    with pytest.raises(FaceModelError) as exc:
        _model([_face(0.9, (30, 30, 170, 170), kps=kps)]).analyze(sharp)
    assert exc.value.error_code == ErrorCode.FACE_LOW_QUALITY
    assert "lurry" in exc.value.message

    monkeypatch.setattr(settings, "min_blur_variance", baseline)  # exactly at limit -> passes
    result = _model([_face(0.9, (30, 30, 170, 170), kps=kps)]).analyze(sharp)
    assert result.quality.blur_variance == pytest.approx(baseline)


def test_min_embedding_norm_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_norm = _model([_face(0.9, (0, 0, 80, 100), fill=2.0)]).analyze(IMAGE).quality.embedding_norm

    monkeypatch.setattr(settings, "min_embedding_norm", baseline_norm)
    face = _face(0.9, (0, 0, 80, 100), fill=1.0)  # norm sqrt(512) < limit
    with pytest.raises(FaceModelError) as exc:
        _model([face]).analyze(IMAGE)
    assert exc.value.error_code == ErrorCode.FACE_LOW_QUALITY
    assert "mbedding" in exc.value.message

    ok_face = _face(0.9, (0, 0, 80, 100), fill=2.0)  # norm exactly at limit -> passes
    result = _model([ok_face]).analyze(IMAGE)
    assert result.quality.embedding_norm == pytest.approx(2.0 * np.sqrt(512))


# --- Role: gates never apply to the reference unless opted in -----------


def test_gate_skipped_for_reference_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "min_face_size_px", 60.0)
    face = _face(0.9, (0, 0, 38, 100))
    result = _model([face]).analyze(IMAGE, role="reference")
    assert result.quality.face_size_px == pytest.approx(38.0)


def test_gate_applies_to_reference_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "min_face_size_px", 60.0)
    monkeypatch.setattr(settings, "quality_gates_apply_to_reference", True)
    face = _face(0.9, (0, 0, 38, 100))
    with pytest.raises(FaceModelError) as exc:
        _model([face]).analyze(IMAGE, role="reference")
    assert exc.value.error_code == ErrorCode.FACE_LOW_QUALITY


# --- Multi-face policy ----------------------------------------------------


def test_reject_ambiguous_rejects_two_similar_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "multi_face_policy", "reject_ambiguous")
    chosen = _face(0.9, (0, 0, 100, 100))  # area 10000
    other = _face(0.8, (0, 0, 90, 90))  # area 8100, ratio 0.81 >= 0.5, score 0.8 >= 0.5
    with pytest.raises(FaceModelError) as exc:
        _model([chosen, other]).analyze(IMAGE)
    assert exc.value.error_code == ErrorCode.MULTIPLE_FACES_DETECTED


def test_reject_ambiguous_accepts_low_score_second_face(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "multi_face_policy", "reject_ambiguous")
    chosen = _face(0.9, (0, 0, 100, 100))
    other = _face(0.4, (0, 0, 90, 90))  # below multi_face_min_score (0.5)
    result = _model([chosen, other]).analyze(IMAGE)
    assert result.det_score == pytest.approx(0.9)


def test_reject_ambiguous_accepts_small_second_face(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "multi_face_policy", "reject_ambiguous")
    chosen = _face(0.9, (0, 0, 100, 100))  # area 10000
    other = _face(0.8, (0, 0, 50, 50))  # area 2500, ratio 0.25 < 0.5
    result = _model([chosen, other]).analyze(IMAGE)
    assert result.det_score == pytest.approx(0.9)


def test_largest_policy_never_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "multi_face_policy", "largest")
    chosen = _face(0.9, (0, 0, 100, 100))
    other = _face(0.85, (0, 0, 95, 95))
    result = _model([chosen, other]).analyze(IMAGE)
    assert result.det_score == pytest.approx(0.9)


def test_reject_ambiguous_bystander_below_min_face_quality_still_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """multi_face_min_score (default 0.5) must be reachable even though it sits
    below min_face_quality (0.7 pinned in conftest): a bystander with a
    det_score in [0.5, 0.7) never enters `valid_faces` and so would never be
    checked if _check_multi_face only saw faces that passed min_face_quality
    -- that would make multi_face_min_score dead configuration."""
    monkeypatch.setattr(settings, "multi_face_policy", "reject_ambiguous")
    chosen = _face(0.9, (0, 0, 100, 100))  # area 10000, passes min_face_quality
    bystander = _face(0.6, (0, 0, 95, 95))  # area 9025, ratio 0.9025 -- 0.6 is >= multi_face_min_score (0.5) but < min_face_quality (0.7)
    with pytest.raises(FaceModelError) as exc:
        _model([chosen, bystander]).analyze(IMAGE)
    assert exc.value.error_code == ErrorCode.MULTIPLE_FACES_DETECTED
    assert "0.60" in exc.value.message
    assert "0.90" in exc.value.message or "0.9" in exc.value.message


def test_multi_face_policy_respects_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "multi_face_policy", "reject_ambiguous")
    chosen = _face(0.9, (0, 0, 100, 100))
    other = _face(0.8, (0, 0, 90, 90))
    result = _model([chosen, other]).analyze(IMAGE, role="reference")
    assert result.det_score == pytest.approx(0.9)
