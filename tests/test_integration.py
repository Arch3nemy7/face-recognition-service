"""End-to-end checks against the real InsightFace model.

Uses the sample photos bundled inside the insightface package, so no face
images are committed to this repo. Skipped automatically when the configured
model pack has no .onnx files (e.g. in CI).
"""

import base64
import os
from collections.abc import Generator
from pathlib import Path

import cv2
import insightface
import numpy as np
import pytest
from fastapi.testclient import TestClient

import face_recognition_service.main as main_module
from face_recognition_service.config import settings
from tests.conftest import TEST_API_TOKEN

SAMPLES = Path(insightface.__file__).parent / "data" / "images"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_TOKEN}"}

# Two different people in t1.jpg, as (x1, y1, x2, y2) boxes. The first is
# the face the service selects as dominant on the full photo.
DOMINANT_FACE = (466, 269, 573, 415)
OTHER_FACE = (904, 62, 1014, 205)


def _model_files_present() -> bool:
    root = Path(os.environ.get("INSIGHTFACE_HOME", Path.home() / ".insightface"))
    return any((root / "models" / settings.model_name).glob("*.onnx"))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _model_files_present(), reason=f"{settings.model_name} model files not found"),
]


@pytest.fixture(scope="module")
def real_client() -> Generator[TestClient, None, None]:
    with TestClient(main_module.app) as client:
        yield client


def _group_photo() -> np.ndarray:
    image = cv2.imread(str(SAMPLES / "t1.jpg"))
    assert image is not None
    return image


def _jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _crop_around(image: np.ndarray, box: tuple[int, int, int, int], margin: float = 1.0) -> bytes:
    """JPEG of `box` grown by `margin` x its own size on every side."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    top, bottom = max(0, int(y1 - margin * h)), min(image.shape[0], int(y2 + margin * h))
    left, right = max(0, int(x1 - margin * w)), min(image.shape[1], int(x2 + margin * w))
    return _jpeg(image[top:bottom, left:right])


def _compare_upload(client: TestClient, reference: bytes, selfie: bytes):
    return client.post(
        "/api/v1/compare-photos-upload",
        files={
            "image1": ("reference.jpg", reference, "image/jpeg"),
            "image2": ("selfie.jpg", selfie, "image/jpeg"),
        },
        headers=AUTH_HEADERS,
    )


def test_health_reports_loaded_model(real_client: TestClient) -> None:
    body = real_client.get("/api/v1/health").json()
    assert body == {"status": "healthy", "model_loaded": True, "model_name": settings.model_name}


def test_embed_group_photo_returns_512_dim_vector(real_client: TestClient) -> None:
    image_b64 = base64.b64encode((SAMPLES / "t1.jpg").read_bytes()).decode()
    response = real_client.post("/api/v1/embed", json={"image": image_b64}, headers=AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert len(body["embedding"]) == 512
    assert body["detection_score"] >= settings.min_face_quality


def test_same_person_matches(real_client: TestClient) -> None:
    # t1.jpg's six faces are near-equal in size, so "dominant" (the largest
    # detected face) is close to a tie; a future Phase 2 multi-face policy
    # (e.g. reject_ambiguous) or other pipeline changes may pick a different
    # face here and require revisiting this test.
    group = _group_photo()
    response = _compare_upload(real_client, _jpeg(group), _crop_around(group, DOMINANT_FACE))
    assert response.status_code == 200
    body = response.json()
    assert body["match"] is True
    assert body["distance"] < 0.2


def test_different_people_do_not_match(real_client: TestClient) -> None:
    group = _group_photo()
    response = _compare_upload(
        real_client, _crop_around(group, DOMINANT_FACE), _crop_around(group, OTHER_FACE)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match"] is False
    assert body["distance"] > 0.5


def test_compare_photos_upload_reports_plausible_quality(real_client: TestClient) -> None:
    """/compare-photos-upload's optional quality fields, computed against a
    real detection/landmark pass, must carry plausible numbers -- not just
    the right shape -- so a mocked-contract test alone couldn't catch a
    metric wired to the wrong field."""
    group = _group_photo()
    response = _compare_upload(real_client, _jpeg(group), _crop_around(group, DOMINANT_FACE))
    assert response.status_code == 200
    body = response.json()
    for key in ("image1_quality", "image2_quality"):
        quality = body[key]
        assert quality is not None
        assert quality["face_size_px"] > 0
        assert quality["interocular_px"] > 0


def test_compare_photos_route_matches_same_person(
    real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers /api/v1/compare-photos, the route the production backend
    actually calls (image1 = reference URL, image2 = uploaded selfie,
    distance_metric always "cosine"). fetch_image_from_url is monkeypatched
    so no real network call is made; it must accept exactly one positional
    arg (the URL) to match the real signature main.py calls it with.
    """
    monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: _group_photo())

    response = real_client.post(
        "/api/v1/compare-photos",
        data={"image1": "https://reference.example.com/ref.jpg", "distance_metric": "cosine"},
        files={"image2": ("selfie.jpg", _crop_around(_group_photo(), DOMINANT_FACE), "image/jpeg")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["match"] is True
    assert body["distance"] < 0.2


def test_tight_face_crop_is_detected_via_padding(real_client: TestClient) -> None:
    """A 112x112 pre-aligned crop has no context for the detector; the padding retry recovers it."""
    tom = (SAMPLES / "Tom_Hanks_54745.png").read_bytes()
    response = _compare_upload(real_client, tom, tom)
    assert response.status_code == 200
    assert response.json()["match"] is True


def test_embed_returns_a_unit_vector(real_client: TestClient) -> None:
    image_b64 = base64.b64encode((SAMPLES / "t1.jpg").read_bytes()).decode()
    body = real_client.post("/api/v1/embed", json={"image": image_b64}, headers=AUTH_HEADERS).json()
    assert float(np.linalg.norm(body["embedding"])) == pytest.approx(1.0, abs=1e-4)


def _onnx_dir_for(pack_name: str) -> Path | None:
    """Directory actually holding `<pack_name>`'s .onnx files -- flat or the nested
    `models/<pack_name>/<pack_name>` layout the upstream antelopev2 zip extracts to."""
    for home in {os.environ.get("INSIGHTFACE_HOME", ""), str(Path.home() / ".insightface")}:
        if not home:
            continue
        root = Path(home).expanduser()
        flat = root / "models" / pack_name
        if any(flat.glob("*.onnx")):
            return flat
        nested = flat / pack_name
        if any(nested.glob("*.onnx")):
            return nested
    return None


@pytest.mark.parametrize("pack_name", ["buffalo_l", "antelopev2"])
def test_loader_embedding_matches_face_analysis(
    pack_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The new loader must give the same embedding InsightFace's FaceAnalysis gave (normalised),
    for every model pack we ship, whether its .onnx files sit flat or nested (antelopev2)."""
    from insightface.app import FaceAnalysis

    from face_recognition_service.models.face_model import FaceRecognitionModel
    from face_recognition_service.utils.image_utils import (
        load_image_from_bytes,
        preprocess_image,
    )

    onnx_dir = _onnx_dir_for(pack_name)
    if onnx_dir is None:
        pytest.skip(f"{pack_name} model files not found")

    image = preprocess_image(load_image_from_bytes((SAMPLES / "t1.jpg").read_bytes()))

    monkeypatch.setattr(settings, "model_name", pack_name)
    model = FaceRecognitionModel()  # model_name is read at construction
    model.load()
    ours = model.analyze(image)

    if onnx_dir.name == pack_name and onnx_dir.parent.name == pack_name:
        # Nested layout (e.g. antelopev2): FaceAnalysis(name=...) looks for
        # `root/models/<name>/*.onnx` directly and won't find files nested one
        # level deeper, so build a `root` whose `models/<pack_name>/` holds
        # symlinks to the real .onnx files.
        flat_root = tmp_path / "insightface"
        flat_pack_dir = flat_root / "models" / pack_name
        flat_pack_dir.mkdir(parents=True)
        for onnx_file in onnx_dir.glob("*.onnx"):
            (flat_pack_dir / onnx_file.name).symlink_to(onnx_file)
        root = str(flat_root)
    else:
        root = str(onnx_dir.parent.parent)  # onnx_dir is root/models/<pack_name>

    reference_app = FaceAnalysis(name=pack_name, root=root, providers=["CPUExecutionProvider"])
    reference_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=settings.detection_threshold)
    faces = reference_app.get(image)
    theirs = max(faces, key=lambda f: f.det_score * (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    expected = theirs.embedding / np.linalg.norm(theirs.embedding)
    cosine = float(ours.embedding @ expected)
    print(f"\n{pack_name}: parity cosine = {cosine:.6f}")
    assert cosine > 0.9999
