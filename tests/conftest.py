"""Shared pytest setup: hermetic settings plus the fake-model API client.

`Settings()` is built when `face_recognition_service` is first imported and
reads a local `.env`. That file is gitignored; on a developer machine it
holds the real API token and deliberately permissive ML thresholds (e.g.
MIN_FACE_QUALITY=0.1), so letting it leak in would both expose the
production token to the test process and make results machine-dependent.
Environment variables take precedence over `.env` in pydantic-settings, so
every setting a test is known to observe is forced here, before any test
module imports the package. `tests/test_settings_isolation.py` guards that
these pins track the `Settings` code defaults, so a future default change
can't silently drift from what the suite exercises.
"""

import ipaddress
import os

TEST_API_TOKEN = "test-token-for-pytest-only"

# Defined as a module-level constant (rather than inlined into the
# `os.environ.update` call below) so `test_settings_isolation.py` can import
# it directly and diff it against `Settings` field defaults, instead of
# duplicating these values.
PINNED_ENV = {
    "API_TOKEN": TEST_API_TOKEN,
    # buffalo_l ships as a flat directory; the loader also resolves the
    # nested <name>/<name> layout the upstream antelopev2 zip extracts to.
    "MODEL_NAME": "buffalo_l",
    "DETECTION_THRESHOLD": "0.5",
    "MIN_FACE_QUALITY": "0.7",
    "EMBEDDING_SIZE": "512",
    "COSINE_MATCH_THRESHOLD": "0.5",
    "ENHANCE_MODE": "detect_fallback",
    "MIN_FACE_SIZE_PX": "0",
    "MIN_INTEROCULAR_PX": "0",
    "MIN_BLUR_VARIANCE": "0",
    "MAX_ABS_YAW_PROXY": "0",
    "MAX_ABS_ROLL_DEG": "0",
    "MIN_EMBEDDING_NORM": "0",
    "QUALITY_GATES_APPLY_TO_REFERENCE": "false",
    "MULTI_FACE_POLICY": "largest",
    "MULTI_FACE_AREA_RATIO": "0.5",
    "MULTI_FACE_MIN_SCORE": "0.5",
    # Pinned because a developer's local .env may enable CORS; the CORS tests
    # assert the code default (off) and middleware is wired at import time.
    "CORS_ENABLED": "false",
    "DEVICE": "cpu",
    "DET_SIZE": "640",
    "PAD_RETRY_RATIO": "0.5",
    "ORT_INTRA_OP_THREADS": "0",
    "ORT_INTER_OP_THREADS": "0",
    "MAX_IMAGE_SIZE": str(10 * 1024 * 1024),
    "MAX_IMAGE_PIXELS": "50000000",
    "MAX_IMAGE_SIDE": "2048",
    "DEBUG": "false",
    "LOG_LEVEL": "info",
}

os.environ.update(PINNED_ENV)

from collections.abc import Generator  # noqa: E402
from typing import Optional  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import face_recognition_service.main as main_module  # noqa: E402
from face_recognition_service.models import (  # noqa: E402
    face_model as face_model_module,
)
from face_recognition_service.models.face_model import (  # noqa: E402
    FaceQuality,
    FaceResult,
)
from face_recognition_service.utils import (  # noqa: E402
    image_utils as image_utils_module,
)


@pytest.fixture(autouse=True)
def _stub_resolve_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test offline by default: `_resolve_host` (a real DNS
    lookup via socket.getaddrinfo) always resolves to a fixed public IP
    instead of touching the network. Tests exercising the real resolver
    (tests/test_fetch_hardening.py's gaierror case) call
    `monkeypatch.undo()` to remove this before patching `socket.getaddrinfo`
    directly.
    """
    monkeypatch.setattr(
        image_utils_module,
        "_resolve_host",
        lambda host, **kwargs: [ipaddress.ip_address("93.184.216.34")],
    )


def _face_result(
    embedding: np.ndarray,
    det_score: float,
    quality: Optional[FaceQuality] = None,
) -> FaceResult:
    """A real `FaceResult` for `fake_model.analyze` mocks.

    Contract tests set `fake_model.analyze.return_value`/`side_effect` to
    this rather than a bare tuple or a MagicMock: main.py builds its
    response from real `FaceResult`/`FaceQuality` attributes, so a forgotten
    mock (bare `MagicMock()`) would make pydantic raise on serialisation --
    turning an intended 400 test into a silent 500. Building a real
    dataclass here makes that failure loud (an AttributeError/TypeError at
    mock-setup time) instead.
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    return FaceResult(
        embedding=embedding,
        embedding_norm=float(np.linalg.norm(embedding)),
        det_score=det_score,
        bbox=(0.0, 0.0, 100.0, 100.0),
        kps=None,
        faces_detected=1,
        quality=quality,
    )


@pytest.fixture
def fake_model() -> MagicMock:
    """Stand-in for the FaceRecognitionModel singleton every endpoint resolves."""
    model = MagicMock()
    model.is_loaded.return_value = True
    model.model_name = "fake"
    model.get_model_info.return_value = {
        "name": "fake",
        "embedding_size": 512,
        "backend": "insightface",
        "device": "cpu",
    }
    return model


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock) -> Generator[TestClient, None, None]:
    """TestClient whose lifespan installs `fake_model` instead of loading InsightFace."""

    def _fake_initialize_model() -> None:
        face_model_module._model_instance = fake_model

    monkeypatch.setattr(main_module, "initialize_model", _fake_initialize_model)
    monkeypatch.setattr(main_module, "cleanup_model", lambda: None)
    # Every scenario controls fetch/decode directly; preprocessing is not
    # what these suites test, so make it a no-op by default.
    monkeypatch.setattr(main_module, "preprocess_image", lambda image: image)

    with TestClient(main_module.app) as test_client:
        yield test_client

    face_model_module._model_instance = None
