"""Loader: pack resolution, session options, and only detection + recognition get sessions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import face_recognition_service.models.loader as loader
from face_recognition_service.models import face_model as face_model_module
from face_recognition_service.models.face_model import (
    FaceModelError,
    FaceRecognitionModel,
)
from face_recognition_service.schemas.api_schemas import ErrorCode


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"onnx")
    return path


class TestResolvePackDir:
    def test_flat(self, tmp_path: Path) -> None:
        _touch(tmp_path / "models" / "buffalo_l" / "det_10g.onnx")
        assert loader.resolve_pack_dir(tmp_path, "buffalo_l") == tmp_path / "models" / "buffalo_l"

    def test_nested_upstream_zip_layout(self, tmp_path: Path) -> None:
        _touch(tmp_path / "models" / "antelopev2" / "antelopev2" / "glintr100.onnx")
        assert loader.resolve_pack_dir(tmp_path, "antelopev2") == tmp_path / "models" / "antelopev2" / "antelopev2"

    def test_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            loader.resolve_pack_dir(tmp_path, "antelopev2")

    def test_expands_tilde(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        _touch(tmp_path / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx")
        assert loader.resolve_pack_dir("~/.insightface", "buffalo_l") == tmp_path / ".insightface" / "models" / "buffalo_l"


def test_session_options_threads() -> None:
    default = loader.session_options(0, 0)
    tuned = loader.session_options(2, 1)
    assert tuned.intra_op_num_threads == 2 and tuned.inter_op_num_threads == 1
    assert default.intra_op_num_threads == 0 and default.inter_op_num_threads == 0


class _FakeRouter:
    built: list[str] = []

    def __init__(self, onnx_file: str) -> None:
        self.onnx_file = onnx_file

    def get_model(self, **kwargs):
        _FakeRouter.built.append(Path(self.onnx_file).name)
        name = Path(self.onnx_file).name
        task = "detection" if name.startswith(("det", "scrfd")) else "recognition" if name.startswith(("w600k", "glint")) else "landmark_3d_68"
        prepared: dict = {}
        return SimpleNamespace(
            taskname=task, kwargs=kwargs, prepared=prepared, prepare=lambda ctx_id, **kw: prepared.update(ctx_id=ctx_id, **kw)
        )


@pytest.fixture
def fake_router(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRouter]:
    _FakeRouter.built = []
    monkeypatch.setattr(loader, "ModelRouter", _FakeRouter)
    return _FakeRouter


def test_known_pack_builds_only_detector_and_recognizer(tmp_path: Path, fake_router) -> None:
    for name in ["scrfd_10g_bnkps.onnx", "glintr100.onnx", "1k3d68.onnx", "2d106det.onnx", "genderage.onnx"]:
        _touch(tmp_path / name)
    options = loader.session_options(2, 1)
    pack = loader.load_pack(tmp_path, name="antelopev2", providers=["CPUExecutionProvider"], options=options, det_size=640, det_thresh=0.5)
    assert sorted(fake_router.built) == ["glintr100.onnx", "scrfd_10g_bnkps.onnx"]
    assert pack.det_model.kwargs == {"providers": ["CPUExecutionProvider"], "sess_options": options}
    assert pack.det_model.prepared == {"ctx_id": 0, "input_size": (640, 640), "det_thresh": 0.5}
    assert pack.rec_model.taskname == "recognition"


def test_unknown_pack_routes_every_file(tmp_path: Path, fake_router) -> None:
    for name in ["det_custom.onnx", "w600k_custom.onnx", "1k3d68.onnx"]:
        _touch(tmp_path / name)
    pack = loader.load_pack(tmp_path, name="custom", providers=["CPUExecutionProvider"], options=loader.session_options(0, 0), det_size=320, det_thresh=0.3)
    assert len(fake_router.built) == 3
    assert pack.det_model.prepared["input_size"] == (320, 320)


def test_missing_recognizer_raises(tmp_path: Path, fake_router) -> None:
    _touch(tmp_path / "det_10g.onnx")
    with pytest.raises(FileNotFoundError):
        loader.load_pack(tmp_path, name="custom", providers=["CPUExecutionProvider"], options=loader.session_options(0, 0), det_size=640, det_thresh=0.5)


def test_model_load_wires_settings_and_reports_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}
    monkeypatch.setenv("INSIGHTFACE_HOME", str(tmp_path))
    monkeypatch.setattr(face_model_module, "resolve_pack_dir", lambda root, name: Path(root) / name)
    monkeypatch.setattr(face_model_module, "load_pack", lambda pack_dir, **kw: captured.update(pack_dir=pack_dir, **kw) or "PACK")
    model = FaceRecognitionModel()
    model.load()
    assert model.model == "PACK"
    assert captured["pack_dir"] == tmp_path / model.model_name
    assert captured["det_thresh"] == model.detection_threshold
    assert captured["det_size"] == 640

    def _fail(pack_dir, **kw):
        raise FileNotFoundError("no models")

    monkeypatch.setattr(face_model_module, "load_pack", _fail)
    with pytest.raises(FaceModelError) as exc_info:
        FaceRecognitionModel().load()
    assert exc_info.value.error_code == ErrorCode.MODEL_NOT_LOADED


def test_model_load_expands_tilde_in_insightface_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A literal '~' in INSIGHTFACE_HOME must be expanded before it reaches resolve_pack_dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("INSIGHTFACE_HOME", "~/.insightface")
    captured: dict = {}
    monkeypatch.setattr(face_model_module, "load_pack", lambda pack_dir, **kw: captured.update(pack_dir=pack_dir, **kw) or "PACK")

    model = FaceRecognitionModel()
    _touch(tmp_path / ".insightface" / "models" / model.model_name / "det_10g.onnx")
    model.load()

    assert model.model == "PACK"
    assert captured["pack_dir"] == tmp_path / ".insightface" / "models" / model.model_name


def test_model_load_downloads_pack_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INSIGHTFACE_HOME", str(tmp_path))
    ensure_available_calls: list[tuple] = []

    def _fake_ensure_available(*args, **kwargs):
        ensure_available_calls.append((args, kwargs))
        # Simulate the upstream antelopev2 zip's nested <name>/<name> layout.
        model = FaceRecognitionModel()
        _touch(tmp_path / "models" / model.model_name / model.model_name / "det_10g.onnx")

    monkeypatch.setattr(face_model_module, "ensure_available", _fake_ensure_available)
    captured: dict = {}
    monkeypatch.setattr(face_model_module, "load_pack", lambda pack_dir, **kw: captured.update(pack_dir=pack_dir, **kw) or "PACK")

    model = FaceRecognitionModel()
    model.load()

    assert model.model == "PACK"
    assert captured["pack_dir"] == tmp_path / "models" / model.model_name / model.model_name
    assert ensure_available_calls == [(("models", model.model_name), {"root": str(tmp_path)})]
