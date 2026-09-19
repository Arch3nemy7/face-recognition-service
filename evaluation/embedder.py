"""Embed dataset images through the service's own decode -> preprocess -> model path.

The service reads its tuning from the module-level `settings` object, so an
evaluation config is applied by setting those attributes before the model is
built (model_name and detection_threshold are read at construction;
min_face_quality and enhance_mode on every call). This makes evaluation
numbers describe exactly what production code does with the same values.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import cv2
import insightface
import numpy as np
import onnxruntime
import PIL

from face_recognition_service.config import settings
from face_recognition_service.models import face_model as _face_model_module
from face_recognition_service.models.face_model import (
    FaceModelError,
    FaceRecognitionModel,
    FaceResult,
)
from face_recognition_service.utils import embedding_utils as _embedding_utils_module
from face_recognition_service.utils import image_utils as _image_utils_module
from face_recognition_service.utils.image_utils import (
    ImageProcessingError,
    load_image_from_bytes,
    preprocess_image,
)

# Directories whose *.py files are part of the pipeline that produces an
# embedding: the face model wrapper and the decode/preprocess/embedding-math
# helpers it calls. Hashing every *.py file under these two directories --
# not just the three files named in the design -- means a future file such as
# a models/loader.py is picked up automatically, without editing this list.
_PACKAGE_ROOT = Path(_face_model_module.__file__).resolve().parent.parent
_SERVICE_CODE_DIRS = (
    Path(_face_model_module.__file__).resolve().parent,  # face_recognition_service/models
    Path(_image_utils_module.__file__).resolve().parent,  # face_recognition_service/utils (also embedding_utils)
)
assert Path(_embedding_utils_module.__file__).resolve().parent in _SERVICE_CODE_DIRS

EnhanceMode = Literal["off", "detect_fallback", "always"]
_ENHANCE_MODES: tuple[str, ...] = ("off", "detect_fallback", "always")

Role = Literal["selfie", "reference"]

# FaceQuality's fields, in declaration order -- also the cache's per-field array names
# (as "quality_<field>") and the keys of EmbeddingResult.quality.
QUALITY_FIELDS: tuple[str, ...] = (
    "face_size_px",
    "interocular_px",
    "roll_deg",
    "yaw_proxy",
    "blur_variance",
    "embedding_norm",
    "faces_considered",
    "second_face_ratio",
)


@dataclass(frozen=True)
class EvalConfig:
    name: str
    model_name: str = "buffalo_l"
    detection_threshold: float = 0.5
    min_face_quality: float = 0.7
    enhance_mode: EnhanceMode = "detect_fallback"

    def fingerprint(self) -> str:
        fields = {k: v for k, v in asdict(self).items() if k != "name"}
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


PRESETS: dict[str, dict] = {
    # config.py defaults
    "code-defaults": {"detection_threshold": 0.5, "min_face_quality": 0.7, "enhance_mode": "detect_fallback"},
    # what the live deployment's .env sets today (quality gate effectively off), enhancement always on
    "prod-like": {"detection_threshold": 0.1, "min_face_quality": 0.1, "enhance_mode": "always"},
    # live deployment's thresholds, but enhancement only as a detection fallback
    "prod-fallback": {"detection_threshold": 0.1, "min_face_quality": 0.1, "enhance_mode": "detect_fallback"},
    # live deployment's thresholds, enhancement fully off
    "prod-off": {"detection_threshold": 0.1, "min_face_quality": 0.1, "enhance_mode": "off"},
    # code defaults without CLAHE/gamma, to measure what enhancement buys
    "no-enhance": {"detection_threshold": 0.5, "min_face_quality": 0.7, "enhance_mode": "off"},
}


def make_config(preset: str, model_name: str) -> EvalConfig:
    return EvalConfig(name=preset, model_name=model_name, **PRESETS[preset])


def _hash_service_code() -> str:
    """sha256 over every *.py file under the models/ and utils/ packages, salted with each file's relative name."""
    files: list[Path] = []
    for directory in _SERVICE_CODE_DIRS:
        files.extend(directory.glob("*.py"))
    hasher = hashlib.sha256()
    for path in sorted(set(files)):
        hasher.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _package_versions() -> str:
    versions = {
        "insightface": getattr(insightface, "__version__", "?"),
        "onnxruntime": getattr(onnxruntime, "__version__", "?"),
        "cv2": cv2.__version__,
        "PIL": getattr(PIL, "__version__", "?"),
        "numpy": np.__version__,
    }
    return "|".join(f"{name}={version}" for name, version in versions.items())


def _model_files(model_name: str) -> str:
    home = os.environ.get("INSIGHTFACE_HOME") or os.path.expanduser("~/.insightface")
    base = Path(home) / "models" / model_name
    candidates = list(base.glob("*.onnx")) + list((base / model_name).glob("*.onnx"))
    entries = sorted(f"{path.name}:{path.stat().st_size}" for path in candidates if path.is_file())
    return "|".join(entries)


def pipeline_fingerprint(model_name: str) -> dict[str, str]:
    """Everything besides EvalConfig that can change what an embedding for `model_name` means.

    Covers the service code that turns bytes into an embedding, the
    versions of the libraries that code depends on, and the actual model
    weight files on disk (size, since InsightFace model packs aren't
    versioned) -- so a code change, a dependency upgrade, or swapping model
    files in place all show up here even though none of them touch
    EvalConfig.
    """
    return {
        "service_code": _hash_service_code(),
        "packages": _package_versions(),
        "model_files": _model_files(model_name),
    }


@dataclass(frozen=True, eq=False)
class EmbeddingResult:
    embedding: np.ndarray | None  # unit-normalised float32; None when the image failed
    det_score: float | None
    error_code: str | None
    quality: dict[str, float | None] | None = None  # FaceQuality fields as a dict; None when the image failed

    # A generated __eq__ would compare numpy arrays with `==` (ambiguous truth value).
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingResult):
            return NotImplemented
        same_vector = (self.embedding is None and other.embedding is None) or (
            self.embedding is not None and other.embedding is not None and np.array_equal(self.embedding, other.embedding)
        )
        return (
            same_vector
            and self.det_score == other.det_score
            and self.error_code == other.error_code
            and self.quality == other.quality
        )

    __hash__ = None  # holds a numpy array; not hashable


class _Model(Protocol):
    def analyze(self, image: np.ndarray, role: Role = "selfie") -> FaceResult: ...


ModelFactory = Callable[[EvalConfig], _Model]


def apply_config(config: EvalConfig) -> None:
    if config.enhance_mode not in _ENHANCE_MODES:
        # A frozen dataclass field annotated Literal[...] isn't runtime-checked by
        # itself, so a typo (e.g. "detect-fallback") would otherwise silently fall
        # through Settings.effective_enhance_mode to "detect_fallback" rather than
        # failing loudly -- catch it here instead.
        raise ValueError(f"invalid enhance_mode {config.enhance_mode!r}; expected one of {_ENHANCE_MODES}")
    settings.model_name = config.model_name
    settings.detection_threshold = config.detection_threshold
    settings.min_face_quality = config.min_face_quality
    settings.enhance_mode = config.enhance_mode
    settings.enhance_image = None  # enhance_mode is authoritative once set


def load_service_model(config: EvalConfig) -> FaceRecognitionModel:
    apply_config(config)
    model = FaceRecognitionModel()
    model.load()
    return model


def embed_one(model: _Model, image_bytes: bytes, role: Role = "selfie") -> EmbeddingResult:
    try:
        image = preprocess_image(load_image_from_bytes(image_bytes))
        result = model.analyze(image, role=role)
    except (FaceModelError, ImageProcessingError) as exc:
        return EmbeddingResult(None, None, exc.error_code, None)
    vector = np.asarray(result.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        return EmbeddingResult(None, None, "INVALID_EMBEDDING", None)
    det_score = result.det_score
    quality = asdict(result.quality) if result.quality is not None else None
    return EmbeddingResult(vector / norm, None if det_score is None else float(det_score), None, quality)


class EmbeddingCache:
    """Per-(config, pipeline) .npz of results keyed by the sha256 of each image's bytes.

    The cache file name covers both EvalConfig.fingerprint() and
    pipeline_fingerprint(config.model_name) (service code, package versions,
    model files -- see that function), so a pipeline change that isn't
    reflected in EvalConfig (e.g. an edit to embedding_utils.py, an
    onnxruntime upgrade, or replacing the .onnx weights on disk) invalidates
    the cache automatically instead of silently reusing stale embeddings.

    Holds biometric templates: it lives in a gitignored directory and should
    be deleted when an evaluation is finished.
    """

    def __init__(self, cache_dir: Path, config: EvalConfig) -> None:
        pipeline = pipeline_fingerprint(config.model_name)
        pipeline_fp = hashlib.sha256(json.dumps(pipeline, sort_keys=True).encode()).hexdigest()[:16]
        self.path = Path(cache_dir) / f"{config.model_name}-{config.fingerprint()}-{pipeline_fp}.npz"
        self._entries: dict[str, EmbeddingResult] = {}
        if self.path.exists():
            self._load()

    def get(self, digest: str) -> EmbeddingResult | None:
        return self._entries.get(digest)

    def put(self, digest: str, result: EmbeddingResult) -> None:
        self._entries[digest] = result

    def _load(self) -> None:
        with np.load(self.path, allow_pickle=False) as data:
            # Older caches (or a partially-written .npz) may lack the quality arrays
            # entirely; treat that as "no quality recorded" for every entry rather
            # than failing to load the file at all.
            has_quality = "quality_present" in data.files and all(f"quality_{f}" in data.files for f in QUALITY_FIELDS)
            quality_present = data["quality_present"] if has_quality else None
            quality_cols = {field: data[f"quality_{field}"] for field in QUALITY_FIELDS} if has_quality else None
            for idx, (digest, vector, ok, score, code) in enumerate(
                zip(data["digests"], data["embeddings"], data["ok"], data["det_scores"], data["error_codes"], strict=True)
            ):
                quality: dict[str, float | None] | None = None
                if has_quality and bool(quality_present[idx]):
                    quality = {
                        field: (None if np.isnan(quality_cols[field][idx]) else float(quality_cols[field][idx]))
                        for field in QUALITY_FIELDS
                    }
                    if quality["faces_considered"] is not None:
                        quality["faces_considered"] = int(quality["faces_considered"])
                self._entries[str(digest)] = EmbeddingResult(
                    vector.astype(np.float32) if ok else None,
                    None if np.isnan(score) else float(score),
                    str(code) or None,
                    quality,
                )

    def save(self) -> None:
        digests = list(self._entries)
        results = [self._entries[d] for d in digests]
        dim = settings.embedding_size
        embeddings = np.zeros((len(results), dim), dtype=np.float32)
        for row, result in enumerate(results):
            if result.embedding is not None:
                embeddings[row] = result.embedding
        quality_present = np.array([r.quality is not None for r in results], dtype=bool)
        quality_columns = {
            field: np.array(
                [
                    np.nan if r.quality is None or r.quality.get(field) is None else float(r.quality[field])
                    for r in results
                ],
                dtype=np.float64,
            )
            for field in QUALITY_FIELDS
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.stem + ".partial.npz")
        np.savez_compressed(
            partial,
            digests=np.array(digests, dtype=str),
            embeddings=embeddings,
            ok=np.array([r.embedding is not None for r in results], dtype=bool),
            det_scores=np.array([np.nan if r.det_score is None else r.det_score for r in results], dtype=np.float64),
            error_codes=np.array([r.error_code or "" for r in results], dtype=str),
            quality_present=quality_present,
            **{f"quality_{field}": column for field, column in quality_columns.items()},
        )
        os.replace(partial, self.path)


def embed_images(
    images: Mapping[str, Path],
    config: EvalConfig,
    *,
    cache_dir: Path | None,
    model_factory: ModelFactory = load_service_model,
    progress: Callable[[int, int], None] | None = None,
    roles: Mapping[str, Role] | None = None,
) -> dict[str, EmbeddingResult]:
    apply_config(config)  # preprocess_image and analyze read settings per call
    cache = EmbeddingCache(cache_dir, config) if cache_dir is not None else None
    results: dict[str, EmbeddingResult] = {}
    pending: list[tuple[str, str, str, bytes]] = []
    for key, path in images.items():
        data = Path(path).read_bytes()
        role: Role = (roles or {}).get(key, "selfie")
        # Role is part of the cache key -- not just the image bytes -- because it can change the
        # result: quality gates and the multi-face policy apply to role="selfie" always, and to
        # role="reference" too when settings.quality_gates_apply_to_reference is set, so the same
        # bytes embedded under different roles are not guaranteed to produce the same result.
        digest = hashlib.sha256(data + b"|" + role.encode()).hexdigest()
        cached = cache.get(digest) if cache is not None else None
        if cached is not None:
            results[key] = cached
        else:
            pending.append((key, role, digest, data))
    if pending:
        model = model_factory(config)
        for done, (key, role, digest, data) in enumerate(pending, start=1):
            result = embed_one(model, data, role=role)
            results[key] = result
            if cache is not None:
                cache.put(digest, result)
                if done % 500 == 0:
                    cache.save()
            if progress is not None:
                progress(done, len(pending))
        if cache is not None:
            cache.save()
    return results
