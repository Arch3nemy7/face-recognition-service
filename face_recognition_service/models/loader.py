"""Load exactly the two ONNX models the service uses: the face detector and the recognizer.

InsightFace's FaceAnalysis builds a session for every .onnx file in a pack
(the 3D/2D landmark and gender/age models are ~200 MB the service never uses),
cannot pass ONNX Runtime SessionOptions (so the thread pool is sized from host
cores rather than the container's CPU quota), and does not look inside the
nested directory the upstream antelopev2 zip extracts to. This loader does all
three, using InsightFace's own model classes so pre/post-processing is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnxruntime
from insightface.model_zoo.model_zoo import ModelRouter

# Detector and recognizer file names for the packs we ship or test with.
# Unknown packs fall back to routing every file and picking by task.
KNOWN_PACK_FILES: dict[str, tuple[str, str]] = {
    "antelopev2": ("scrfd_10g_bnkps.onnx", "glintr100.onnx"),
    "buffalo_l": ("det_10g.onnx", "w600k_r50.onnx"),
}


@dataclass(frozen=True)
class ModelPack:
    det_model: object  # insightface RetinaFace (SCRFD weights): .detect(image, max_num, metric)
    rec_model: object  # insightface ArcFaceONNX: .get(image, face) -> embedding


def resolve_pack_dir(root: Path, name: str) -> Path:
    """`<root>/models/<name>`, or the nested `<root>/models/<name>/<name>` the upstream zip creates."""
    pack = Path(root).expanduser() / "models" / name
    if any(pack.glob("*.onnx")):
        return pack
    nested = pack / name
    if any(nested.glob("*.onnx")):
        return nested
    raise FileNotFoundError(f"no .onnx files for model pack {name!r} under {pack}")


def session_options(intra_op_threads: int, inter_op_threads: int) -> onnxruntime.SessionOptions:
    """ONNX Runtime session options; 0 keeps ONNX Runtime's default for that pool."""
    options = onnxruntime.SessionOptions()
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        options.inter_op_num_threads = inter_op_threads
    return options


def load_pack(
    pack_dir: Path,
    *,
    name: str,
    providers: list[str],
    options: onnxruntime.SessionOptions,
    det_size: int,
    det_thresh: float,
) -> ModelPack:
    onnxruntime.set_default_logger_severity(3)  # same quiet ORT logging FaceAnalysis used
    pack_dir = Path(pack_dir)
    known = KNOWN_PACK_FILES.get(name)
    if known and all((pack_dir / file).is_file() for file in known):
        candidates = [pack_dir / file for file in known]
    else:
        candidates = sorted(pack_dir.glob("*.onnx"))

    found: dict[str, object] = {}
    for onnx_file in candidates:
        model = ModelRouter(str(onnx_file)).get_model(providers=providers, sess_options=options)
        task = getattr(model, "taskname", None)
        if task in ("detection", "recognition") and task not in found:
            found[task] = model
    missing = {"detection", "recognition"} - found.keys()
    if missing:
        raise FileNotFoundError(f"model pack {name!r} in {pack_dir} has no {', '.join(sorted(missing))} model")

    # ctx_id=0 skips InsightFace's CPU path, which would rebuild the session
    # (set_providers) and is unnecessary because providers are passed explicitly.
    found["detection"].prepare(0, input_size=(det_size, det_size), det_thresh=det_thresh)
    found["recognition"].prepare(0)
    return ModelPack(det_model=found["detection"], rec_model=found["recognition"])
