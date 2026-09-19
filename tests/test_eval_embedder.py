"""The embedder runs the service's decode -> preprocess -> model path, with a cache."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.embedder import (
    PRESETS,
    QUALITY_FIELDS,
    EmbeddingResult,
    EvalConfig,
    apply_config,
    embed_images,
    embed_one,
    make_config,
    pipeline_fingerprint,
)
from face_recognition_service.config import settings
from face_recognition_service.models.face_model import (
    FaceModelError,
    FaceQuality,
    FaceResult,
)
from face_recognition_service.schemas.api_schemas import ErrorCode

CONFIG_FIELDS = ("model_name", "detection_threshold", "min_face_quality", "enhance_mode")


@pytest.fixture(autouse=True)
def _restore_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # embed_images mutates the global settings; register originals so monkeypatch restores them.
    for field in CONFIG_FIELDS:
        monkeypatch.setattr(settings, field, getattr(settings, field))


def _png(path: Path, value: int) -> Path:
    ok, buffer = cv2.imencode(".png", np.full((64, 64, 3), value, dtype=np.uint8))
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def _fake_quality(**overrides) -> FaceQuality:
    fields = {
        "face_size_px": 100.0,
        "interocular_px": 40.0,
        "roll_deg": 1.5,
        "yaw_proxy": 0.05,
        "blur_variance": 120.0,
        "embedding_norm": 5.0,
        "faces_considered": 1,
        "second_face_ratio": 0.0,
    }
    fields.update(overrides)
    return FaceQuality(**fields)


class _FakeModel:
    """Black images have no face; anything else embeds to [3, 4, 0, ...]."""

    def __init__(self, quality: FaceQuality | None = None) -> None:
        self.calls = 0
        self.roles: list[str] = []
        self.quality = quality if quality is not None else _fake_quality()

    def analyze(self, image: np.ndarray, role: str = "selfie") -> FaceResult:
        self.calls += 1
        self.roles.append(role)
        if image.mean() < 1:
            raise FaceModelError("no face", ErrorCode.NO_FACE_DETECTED)
        vector = np.zeros(512, dtype=np.float32)
        vector[:2] = [3.0, 4.0]
        return FaceResult(
            embedding=vector,
            embedding_norm=5.0,
            det_score=0.9,
            bbox=(0.0, 0.0, 100.0, 100.0),
            kps=None,
            faces_detected=1,
            quality=self.quality,
        )


def test_presets_and_fingerprint() -> None:
    assert set(PRESETS) >= {"code-defaults", "prod-like", "prod-fallback", "prod-off", "no-enhance"}
    prod = make_config("prod-like", "antelopev2")
    assert (prod.model_name, prod.detection_threshold, prod.min_face_quality) == ("antelopev2", 0.1, 0.1)
    assert prod.enhance_mode == "always"
    assert make_config("prod-fallback", "antelopev2").enhance_mode == "detect_fallback"
    assert make_config("prod-off", "antelopev2").enhance_mode == "off"
    assert EvalConfig("a").fingerprint() == EvalConfig("b").fingerprint()  # name is a label only
    assert EvalConfig("a").fingerprint() != EvalConfig("a", enhance_mode="off").fingerprint()


def test_apply_config_rejects_an_invalid_enhance_mode() -> None:
    with pytest.raises(ValueError, match="detect-fallback"):
        apply_config(EvalConfig("t", enhance_mode="detect-fallback"))


def test_embeds_through_service_path_and_normalises(tmp_path: Path) -> None:
    images = {"face": _png(tmp_path / "face.png", 200), "blank": _png(tmp_path / "blank.png", 0)}
    model = _FakeModel()
    config = EvalConfig("t", enhance_mode="off")
    results = embed_images(images, config, cache_dir=None, model_factory=lambda cfg: model)
    face = results["face"]
    assert face.error_code is None and face.det_score == pytest.approx(0.9)
    assert face.embedding[:2].tolist() == pytest.approx([0.6, 0.8])
    assert results["blank"] == EmbeddingResult(None, None, ErrorCode.NO_FACE_DETECTED)
    assert settings.enhance_mode == "off"  # config applied to the settings the service reads


def test_cache_skips_recomputation_per_config(tmp_path: Path) -> None:
    images = {"face": _png(tmp_path / "face.png", 200), "blank": _png(tmp_path / "blank.png", 0)}
    cache_dir = tmp_path / "cache"
    # enhancement off: CLAHE lifts a pure-black frame to mean ~11, which would hide the fake model's "no face" signal
    config = EvalConfig("t", enhance_mode="off")
    first = embed_images(images, config, cache_dir=cache_dir, model_factory=lambda cfg: _FakeModel())

    def _must_not_load(cfg: EvalConfig) -> _FakeModel:
        raise AssertionError("cache hit expected; model must not be loaded")

    second = embed_images(images, config, cache_dir=cache_dir, model_factory=_must_not_load)
    assert second["face"].embedding.tolist() == pytest.approx(first["face"].embedding.tolist())
    assert second["blank"].error_code == ErrorCode.NO_FACE_DETECTED

    other = EvalConfig("t", detection_threshold=0.3, enhance_mode="off")
    model = _FakeModel()
    embed_images(images, other, cache_dir=cache_dir, model_factory=lambda cfg: model)
    assert model.calls == 2  # a different config never reuses another config's cache


def test_cache_is_keyed_by_content_not_name(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    config = EvalConfig("t", enhance_mode="off")
    embed_images({"x": _png(tmp_path / "x.png", 200)}, config, cache_dir=cache_dir, model_factory=lambda c: _FakeModel())
    renamed = {"y": _png(tmp_path / "y.png", 200)}  # same bytes, new key and path

    def _must_not_load(cfg: EvalConfig) -> _FakeModel:
        raise AssertionError("same content must hit the cache")

    assert embed_images(renamed, config, cache_dir=cache_dir, model_factory=_must_not_load)["y"].error_code is None


def test_cache_keys_the_same_bytes_separately_per_role(tmp_path: Path) -> None:
    # Role can change the result (quality gates / multi-face policy apply to role="selfie"
    # always, and to role="reference" too under quality_gates_apply_to_reference) -- so the
    # same image bytes embedded as "reference" and as "selfie" must not share a cache entry.
    path = _png(tmp_path / "face.png", 200)
    cache_dir = tmp_path / "cache"
    config = EvalConfig("t", enhance_mode="off")
    reference_model = _FakeModel(quality=_fake_quality(face_size_px=111.0))
    embed_images(
        {"alice/reference": path}, config, cache_dir=cache_dir,
        model_factory=lambda cfg: reference_model, roles={"alice/reference": "reference"},
    )
    assert reference_model.calls == 1

    selfie_model = _FakeModel(quality=_fake_quality(face_size_px=222.0))
    result = embed_images(
        {"alice/probe": path}, config, cache_dir=cache_dir,
        model_factory=lambda cfg: selfie_model, roles=None,  # defaults to "selfie"
    )["alice/probe"]
    assert selfie_model.calls == 1  # a cache hit here (instead of a second model call) would be the bug
    assert result.quality["face_size_px"] == 222.0

    # Re-embedding the reference key still hits its own role-specific cache entry.
    def _must_not_load(cfg: EvalConfig) -> _FakeModel:
        raise AssertionError("same bytes+role must hit the cache")

    cached = embed_images(
        {"alice/reference": path}, config, cache_dir=cache_dir,
        model_factory=_must_not_load, roles={"alice/reference": "reference"},
    )["alice/reference"]
    assert cached.quality["face_size_px"] == 111.0


def test_pipeline_fingerprint_has_the_three_keys() -> None:
    fp = pipeline_fingerprint("buffalo_l")
    assert set(fp) == {"service_code", "packages", "model_files"}
    assert all(isinstance(v, str) for v in fp.values())


def test_pipeline_fingerprint_reacts_to_model_file_size_and_insightface_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    (home_a / "models" / "fake_model").mkdir(parents=True)
    (home_b / "models" / "fake_model").mkdir(parents=True)
    (home_a / "models" / "fake_model" / "w.onnx").write_bytes(b"x" * 10)
    (home_b / "models" / "fake_model" / "w.onnx").write_bytes(b"x" * 10)

    monkeypatch.setenv("INSIGHTFACE_HOME", str(home_a))
    fp_a = pipeline_fingerprint("fake_model")
    fp_a_again = pipeline_fingerprint("fake_model")
    assert fp_a["model_files"] == fp_a_again["model_files"]  # deterministic for the same directory

    # Growing the model file changes model_files but nothing else.
    (home_a / "models" / "fake_model" / "w.onnx").write_bytes(b"x" * 999)
    fp_a_grown = pipeline_fingerprint("fake_model")
    assert fp_a_grown["model_files"] != fp_a["model_files"]
    assert fp_a_grown["service_code"] == fp_a["service_code"]
    assert fp_a_grown["packages"] == fp_a["packages"]

    # Same file name and size, but a different INSIGHTFACE_HOME: the nested-model-file case below
    # exercises the "one nested level" lookup, and confirms INSIGHTFACE_HOME is read at call time.
    monkeypatch.setenv("INSIGHTFACE_HOME", str(home_b))
    fp_b = pipeline_fingerprint("fake_model")
    assert fp_b["model_files"] == fp_a["model_files"]  # home_b still has the original 10-byte file

    (home_b / "models" / "fake_model" / "fake_model").mkdir()
    (home_b / "models" / "fake_model" / "fake_model" / "nested.onnx").write_bytes(b"y" * 5)
    fp_b_nested = pipeline_fingerprint("fake_model")
    assert fp_b_nested["model_files"] != fp_b["model_files"]  # the nested model dir is picked up too

    # An INSIGHTFACE_HOME with no model files at all fingerprints to an empty model_files string.
    monkeypatch.setenv("INSIGHTFACE_HOME", str(tmp_path / "empty_home"))
    assert pipeline_fingerprint("fake_model")["model_files"] == ""


def test_cache_hits_with_same_config_and_pipeline(tmp_path: Path) -> None:
    images = {"face": _png(tmp_path / "face.png", 200)}
    cache_dir = tmp_path / "cache"
    config = EvalConfig("t", enhance_mode="off")
    embed_images(images, config, cache_dir=cache_dir, model_factory=lambda cfg: _FakeModel())

    def _must_not_load(cfg: EvalConfig) -> _FakeModel:
        raise AssertionError("same config and pipeline must hit the cache")

    result = embed_images(images, config, cache_dir=cache_dir, model_factory=_must_not_load)
    assert result["face"].error_code is None


def test_embed_one_records_quality_from_analyze() -> None:
    model = _FakeModel()
    ok, buffer = cv2.imencode(".png", np.full((64, 64, 3), 200, dtype=np.uint8))
    assert ok
    result = embed_one(model, buffer.tobytes())
    assert result.quality == {
        "face_size_px": 100.0,
        "interocular_px": 40.0,
        "roll_deg": 1.5,
        "yaw_proxy": 0.05,
        "blur_variance": 120.0,
        "embedding_norm": 5.0,
        "faces_considered": 1,
        "second_face_ratio": 0.0,
    }
    assert set(result.quality) == set(QUALITY_FIELDS)


def test_embed_one_quality_is_none_on_failure() -> None:
    model = _FakeModel()
    ok, buffer = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    result = embed_one(model, buffer.tobytes())
    assert result.embedding is None and result.quality is None


def test_reference_keys_get_role_reference_others_selfie(tmp_path: Path) -> None:
    images = {
        "alice/reference": _png(tmp_path / "ref.png", 200),
        "alice/probe": _png(tmp_path / "probe.png", 200),
    }
    model = _FakeModel()
    config = EvalConfig("t", enhance_mode="off")
    embed_images(
        images,
        config,
        cache_dir=None,
        model_factory=lambda cfg: model,
        roles={"alice/reference": "reference"},
    )
    # embed_images iterates `images` in insertion order, so this lines up 1:1 with model.roles.
    assert dict(zip(list(images), model.roles, strict=True)) == {
        "alice/reference": "reference",
        "alice/probe": "selfie",
    }


def test_cache_round_trips_quality(tmp_path: Path) -> None:
    images = {"face": _png(tmp_path / "face.png", 200)}
    cache_dir = tmp_path / "cache"
    config = EvalConfig("t", enhance_mode="off")
    first = embed_images(images, config, cache_dir=cache_dir, model_factory=lambda cfg: _FakeModel())

    def _must_not_load(cfg: EvalConfig) -> _FakeModel:
        raise AssertionError("cache hit expected; model must not be loaded")

    second = embed_images(images, config, cache_dir=cache_dir, model_factory=_must_not_load)
    assert second["face"].quality == first["face"].quality
    assert second["face"].quality is not None


def test_cache_tolerates_a_file_missing_the_quality_arrays(tmp_path: Path) -> None:
    # Simulates an older cache written before quality was persisted: the cache file exists
    # (so the pipeline/config fingerprint still matches) but lacks the quality_* arrays.
    images = {"face": _png(tmp_path / "face.png", 200)}
    cache_dir = tmp_path / "cache"
    config = EvalConfig("t", enhance_mode="off")
    embed_images(images, config, cache_dir=cache_dir, model_factory=lambda cfg: _FakeModel())
    cache_files = list(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1
    with np.load(cache_files[0], allow_pickle=False) as data:
        old_style = {k: data[k] for k in ("digests", "embeddings", "ok", "det_scores", "error_codes")}
    np.savez_compressed(cache_files[0], **old_style)

    def _must_not_load(cfg: EvalConfig) -> _FakeModel:
        raise AssertionError("same config and pipeline must still hit the cache for the embedding itself")

    result = embed_images(images, config, cache_dir=cache_dir, model_factory=_must_not_load)
    assert result["face"].error_code is None
    assert result["face"].quality is None  # not recorded in the old-style file; treated as unknown, not a crash


def test_cache_misses_when_pipeline_fingerprint_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = {"face": _png(tmp_path / "face.png", 200)}
    cache_dir = tmp_path / "cache"
    config = EvalConfig("t", enhance_mode="off")
    model = _FakeModel()
    embed_images(images, config, cache_dir=cache_dir, model_factory=lambda cfg: model)
    assert model.calls == 1

    monkeypatch.setattr(
        "evaluation.embedder.pipeline_fingerprint",
        lambda model_name: {"service_code": "different", "packages": "different", "model_files": "different"},
    )
    other_model = _FakeModel()
    embed_images(images, config, cache_dir=cache_dir, model_factory=lambda cfg: other_model)
    assert other_model.calls == 1  # a changed pipeline fingerprint must not reuse the old cache file
