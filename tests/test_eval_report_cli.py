"""Report numbers and the CLI (offline pieces; the real-model run is an integration test)."""

import json
from pathlib import Path

import cv2
import insightface
import numpy as np
import pytest

from evaluation import report as report_module
from evaluation.__main__ import main, parse_dataset_spec
from evaluation.embedder import EmbeddingResult, EvalConfig
from evaluation.metrics import equal_error_rate
from evaluation.report import build_report, render_markdown
from evaluation.scoring import PairSet


def _vec(*head: float) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[: len(head)] = head
    return v / np.linalg.norm(v)


def _embeddings() -> dict[str, EmbeddingResult]:
    return {
        "a1": EmbeddingResult(_vec(1, 0), 0.9, None),
        "a2": EmbeddingResult(_vec(1, 0.1), 0.9, None),  # ~0.995 to a1
        "b1": EmbeddingResult(_vec(0, 1), 0.9, None),  # 0 to a1
        "dead": EmbeddingResult(None, None, "NO_FACE_DETECTED"),
    }


def test_build_report_counts_and_service_threshold() -> None:
    pairs = PairSet(genuine=(("a1", "a2"), ("a1", "dead")), impostor=(("a1", "b1"), ("a2", "b1")))
    report = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=_embeddings(), pairs=pairs, service_threshold=0.5
    )
    assert report["images"] == {"total": 4, "embedded": 3, "failed_by_code": {"NO_FACE_DETECTED": 1}}
    assert report["pairs"] == {
        "genuine_total": 2, "genuine_scored": 1, "genuine_fta": 1,
        "impostor_total": 2, "impostor_scored": 2, "impostor_fta": 0,
    }
    service = report["service_threshold"]
    assert service["far"] == 0.0 and service["frr"] == 0.0
    assert service["frr_including_fta"] == pytest.approx(0.5)  # the dead genuine pair is a reject to the user
    assert report["threshold_at_far"]["0.01"] is None  # 2 impostors cannot support a 1% claim
    json.dumps(report)  # must be serialisable


def test_build_report_with_no_scorable_genuine() -> None:
    pairs = PairSet(genuine=(("a1", "dead"),), impostor=(("a1", "b1"),))
    report = build_report(dataset_name="t", config=EvalConfig("t"), embeddings=_embeddings(), pairs=pairs, service_threshold=0.5)
    assert report["genuine_scores"] is None and report["eer"] is None and report["service_threshold"] is None
    assert report["eer_including_fta"] is None
    assert report["threshold_at_far_including_fta"] == {"0.01": None, "0.001": None, "0.0001": None}


def test_run_metadata_is_recorded_as_given() -> None:
    pairs = PairSet(genuine=(("a1", "a2"),), impostor=(("a1", "b1"),))
    run = {"dataset_spec": "lfw", "seed": 0, "max_impostors": 100, "augment_impostors": True,
           "service_cosine_threshold": 0.5, "pipeline": {"service_code": "x", "packages": "y", "model_files": "z"}}
    report = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=_embeddings(), pairs=pairs,
        service_threshold=0.5, run=run,
    )
    assert report["run"] == run
    json.dumps(report)


def test_run_defaults_to_none() -> None:
    pairs = PairSet(genuine=(("a1", "a2"),), impostor=(("a1", "b1"),))
    report = build_report(dataset_name="toy", config=EvalConfig("t"), embeddings=_embeddings(), pairs=pairs, service_threshold=0.5)
    assert report["run"] is None


def test_eer_including_fta_is_worse_than_scored_only_eer_with_extra_failed_genuine_pairs() -> None:
    # a1/a2 and b1 all embed and separate cleanly (scores 0.995 genuine vs 0.0 impostor -> EER 0).
    # "dead" never embeds; make many genuine pairs fail to acquire so the FTA-inclusive EER is worse
    # than the scored-only EER, which ignores those failures entirely.
    embeddings = _embeddings()
    genuine_pairs = (("a1", "a2"),) + tuple(("a1", "dead") for _ in range(20))
    pairs = PairSet(genuine=genuine_pairs, impostor=(("a1", "b1"), ("a2", "b1")))
    report = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=embeddings, pairs=pairs, service_threshold=0.5
    )
    assert report["eer"]["rate"] == 0.0  # only the one scorable genuine pair counts here
    assert report["eer_including_fta"]["rate"] > report["eer"]["rate"]
    json.dumps(report)


def test_failed_impostor_pairs_do_not_raise_far() -> None:
    # A failed impostor pair is treated as a correct rejection (a stranger who never even got
    # scored), not as an accept. With 400 such pairs added, the FTA-inclusive impostor pool clears
    # the rule-of-three bar for FAR=0.01 (needs >=300) while the scored-only pool (2 pairs) does not,
    # and because the added "impostors" are certain rejects, the resulting threshold is generous (at
    # or below the sentinel), not stricter -- failed impostors must never make FAR look harder to hit.
    embeddings = _embeddings()
    impostor_pairs = (("a1", "b1"), ("a2", "b1")) + tuple(("a1", "dead") for _ in range(400))
    pairs = PairSet(genuine=(("a1", "a2"),), impostor=impostor_pairs)
    report = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=embeddings, pairs=pairs, service_threshold=0.5
    )
    assert report["threshold_at_far"]["0.01"] is None  # only 2 scored impostors: below the rule-of-three bar
    fta_point = report["threshold_at_far_including_fta"]["0.01"]
    assert fta_point is not None  # 402 impostor pairs (incl. FTA) clear it
    assert fta_point["similarity_threshold"] <= -1.0  # honestly reflects that it fell to the FTA sentinel
    assert fta_point["frr"] == pytest.approx(0.0)  # a near-everything-accepting threshold rejects no genuine pair
    json.dumps(report)


def test_pct_does_not_round_tiny_nonzero_far_to_zero() -> None:
    # 2 false accepts out of 1,000,000 impostor pairs (a real measured FAR from
    # docs/evaluation.md) is 2e-6 -- "{:.2f}%" alone rounds this to "0.00%",
    # indistinguishable from an exactly-zero FAR. It must render distinctly and
    # keep at least 3 significant figures.
    text = report_module._pct(2e-6)
    assert text != "0.00%"
    assert text == "2.00e-04%"
    assert report_module._pct(0.0) == "0.00%"  # an exact zero still prints as zero


def test_render_markdown_has_one_row_per_report() -> None:
    pairs = PairSet(genuine=(("a1", "a2"),), impostor=(("a1", "b1"),))
    report = build_report(dataset_name="toy", config=EvalConfig("cfg"), embeddings=_embeddings(), pairs=pairs, service_threshold=0.5)
    table = render_markdown([report, report])
    assert table.count("| toy | cfg |") == 2
    assert "n/a" in table  # thresholds at FAR are unsupported with one impostor


def test_render_markdown_shows_fta_sentinel_as_dominated_not_a_number() -> None:
    # Force the FTA sentinel (-2.0) into both the EER-incl-FTA threshold and a sim-thr-@-FAR
    # threshold, as happens when FTA pairs alone decide a rate -- the sentinel must never be
    # printed as if it were a real similarity, in either place.
    pairs = PairSet(genuine=(("a1", "a2"),), impostor=(("a1", "b1"),))
    report = build_report(dataset_name="toy", config=EvalConfig("cfg"), embeddings=_embeddings(), pairs=pairs, service_threshold=0.5)
    report["eer_including_fta"] = {"rate": 0.5, "similarity_threshold": -2.0}
    report["threshold_at_far"]["0.001"] = {
        "similarity_threshold": -2.0, "cosine_distance_threshold": 3.0, "frr": 0.0,
    }
    table = render_markdown([report])
    assert "n/a (FTA-dominated)" in table
    assert "-2.000" not in table and "-2.0" not in table


def test_build_report_quality_and_bootstrap_default_to_none_without_data() -> None:
    pairs = PairSet(genuine=(("a1", "a2"),), impostor=(("a1", "b1"),))
    report = build_report(dataset_name="toy", config=EvalConfig("t"), embeddings=_embeddings(), pairs=pairs, service_threshold=0.5)
    assert report["quality"] is None  # _embeddings() carries no quality data
    assert report["bootstrap"] is None  # bootstrap=0 by default


def test_quality_report_summarises_per_role_ignoring_none_fields() -> None:
    def _q(**overrides) -> dict:
        fields = {
            "face_size_px": 100.0, "interocular_px": 40.0, "roll_deg": 1.0, "yaw_proxy": 0.02,
            "blur_variance": 200.0, "embedding_norm": 5.0, "faces_considered": 1, "second_face_ratio": 0.0,
        }
        fields.update(overrides)
        return fields

    embeddings = {
        "alice/reference": EmbeddingResult(_vec(1, 0), 0.9, None, _q(face_size_px=120.0)),
        "alice/probe": EmbeddingResult(_vec(1, 0.1), 0.9, None, _q(face_size_px=80.0, interocular_px=None, roll_deg=None)),
        "bob/probe": EmbeddingResult(_vec(0, 1), 0.9, None, _q(face_size_px=90.0, interocular_px=38.0)),
        "dead": EmbeddingResult(None, None, "NO_FACE_DETECTED"),  # no embedding: excluded entirely
    }
    pairs = PairSet(genuine=(("alice/reference", "alice/probe"),), impostor=(("alice/reference", "bob/probe"),))
    roles = {"alice/reference": "reference"}  # everything else defaults to "selfie"
    report = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=embeddings, pairs=pairs,
        service_threshold=0.5, roles=roles,
    )
    quality = report["quality"]
    assert set(quality) == {"reference", "selfie"}
    assert quality["reference"]["face_size_px"] == {"n": 1, "p01": 120.0, "p05": 120.0, "p50": 120.0, "p95": 120.0, "p99": 120.0}
    # selfie bucket combines alice/probe and bob/probe.
    assert quality["selfie"]["face_size_px"]["n"] == 2
    # alice/probe's interocular_px is None, so only bob/probe contributes to that field.
    assert quality["selfie"]["interocular_px"] == {"n": 1, "p01": 38.0, "p05": 38.0, "p50": 38.0, "p95": 38.0, "p99": 38.0}
    json.dumps(report)


def test_resample_metrics_matches_metrics_module_exactly() -> None:
    # _resample_metrics's eer_including_fta.rate must equal metrics.equal_error_rate exactly --
    # not approximately -- for any genuine/impostor arrays a resample could produce, including
    # ties (real here: an identity drawn more than once in a resample repeats its exact scores)
    # and the reviewer's realistic well-separated regime, where an earlier grid-based
    # approximation was found to be ~12x biased (see docs/evaluation.md and the commit fixing this).
    rng = np.random.default_rng(0)
    mismatches = []
    scenarios = []
    # Small random arrays, some drawn from a tiny pool to force exact ties.
    for trial in range(300):
        m = int(rng.integers(1, 30))
        n = int(rng.integers(1, 200))
        if trial % 3 == 0:
            pool = rng.uniform(-1, 1, size=int(rng.integers(2, 8)))
            g = rng.choice(pool, size=m)
            imp = rng.choice(pool, size=n)
        else:
            g = rng.uniform(-1, 1, size=m)
            imp = rng.uniform(-1, 1, size=n)
        scenarios.append((g, imp))
    # The reviewer's realistic well-separated regime (genuine ~N(0.7, 0.1), impostor
    # ~N(0.05, 0.08)), scaled down from 300 identities / ~1M pairs for test speed.
    for _ in range(30):
        m = int(rng.integers(20, 60))
        n = int(rng.integers(1000, 6000))
        g = np.clip(rng.normal(0.7, 0.1, m), -1, 1)
        imp = np.clip(rng.normal(0.05, 0.08, n), -1, 1)
        scenarios.append((g, imp))

    for g, imp in scenarios:
        exact_rate, _ = equal_error_rate(g, imp)
        fast_rate, _far, _frr, _thresholds = report_module._resample_metrics(g, imp, 0.5)
        if not np.isclose(fast_rate, exact_rate, atol=1e-12):
            mismatches.append((exact_rate, fast_rate, g.size, imp.size))
    assert not mismatches, mismatches[:5]


def test_bootstrap_ci_brackets_the_exact_point_estimate_on_realistic_well_separated_data() -> None:
    # The reviewer's reproduction (scaled down): well-separated genuine/impostor distributions,
    # where a coarse fixed-grid EER approximation was found to be badly biased (exact EER 8.0e-6,
    # grid EER 1.0e-4 -- 12.6x too high -- with a 95% CI that excluded the exact point estimate).
    # This checks the current (exact) bootstrap CI actually brackets the exact point estimate.
    rng = np.random.default_rng(7)
    people = [f"person{i}" for i in range(50)]
    identities: dict[str, str] = {}
    embeddings: dict[str, EmbeddingResult] = {}
    genuine_pairs = []
    impostor_pairs = []
    for person in people:
        ref, probe = f"{person}/reference", f"{person}/probe"
        identities[ref] = identities[probe] = person
        # Cosine similarity is bounded to [-1, 1]; use scalar "scores" stashed as a 1-D
        # embedding so cosine(ref, probe) reproduces a chosen similarity exactly (dot product
        # of unit vectors [x, sqrt(1-x^2)]).
        g_sim = float(np.clip(rng.normal(0.7, 0.1), -0.99, 0.99))
        embeddings[ref] = EmbeddingResult(np.array([1.0, 0.0], dtype=np.float32), 0.9, None)
        embeddings[probe] = EmbeddingResult(
            np.array([g_sim, np.sqrt(max(0.0, 1 - g_sim**2))], dtype=np.float32), 0.9, None
        )
        genuine_pairs.append((ref, probe))
        for j in range(1000):  # 50 * 1000 = 50,000 impostor pairs
            stranger = f"{person}/stranger{j}"
            identities[stranger] = "public"
            i_sim = float(np.clip(rng.normal(0.05, 0.08), -0.99, 0.99))
            embeddings[stranger] = EmbeddingResult(
                np.array([i_sim, np.sqrt(max(0.0, 1 - i_sim**2))], dtype=np.float32), 0.9, None
            )
            impostor_pairs.append((ref, stranger))
    pairs = PairSet(genuine=tuple(genuine_pairs), impostor=tuple(impostor_pairs))

    report = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=embeddings, pairs=pairs,
        service_threshold=0.5, identities=identities, bootstrap=100, seed=0,
    )
    point_estimate = report["eer_including_fta"]["rate"]
    ci = report["bootstrap"]["ci"]["eer_including_fta.rate"]
    assert ci["resamples_used"] > 0
    assert ci["low"] <= point_estimate <= ci["high"]


def test_bootstrap_ci_is_deterministic_and_brackets_a_known_point_estimate() -> None:
    # 4 identities, each with a reference and a probe that embed to the *same* vector
    # (genuine similarity exactly 1.0), and cross-identity impostor pairs on orthogonal
    # basis vectors (impostor similarity exactly 0.0). EER-including-FTA is 0.0 in every
    # possible resample, since genuine/impostor scores never overlap regardless of which
    # identities get drawn -- an exact, not approximate, bracket check.
    people = ["alice", "bob", "carol", "dave"]
    identities: dict[str, str] = {}
    embeddings: dict[str, EmbeddingResult] = {}
    genuine_pairs = []
    for i, person in enumerate(people):
        ref, probe = f"{person}/reference", f"{person}/probe"
        identities[ref] = identities[probe] = person
        vec = np.zeros(512, dtype=np.float32)
        vec[i] = 1.0
        embeddings[ref] = EmbeddingResult(vec, 0.9, None)
        embeddings[probe] = EmbeddingResult(vec, 0.9, None)
        genuine_pairs.append((ref, probe))
    impostor_pairs = [
        (f"{a}/reference", f"{b}/probe") for a in people for b in people if a != b
    ]
    pairs = PairSet(genuine=tuple(genuine_pairs), impostor=tuple(impostor_pairs))

    report = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=embeddings, pairs=pairs,
        service_threshold=0.5, identities=identities, bootstrap=50, seed=0,
    )
    assert report["eer_including_fta"]["rate"] == pytest.approx(0.0)
    bootstrap = report["bootstrap"]
    assert bootstrap["resamples_requested"] == 50
    assert bootstrap["identities"] == 4
    ci = bootstrap["ci"]
    for key in ("eer_including_fta.rate", "service_threshold.far", "service_threshold.frr_including_fta"):
        assert ci[key]["resamples_used"] == 50  # every identity has both a genuine and an impostor pair
        assert ci[key]["low"] == pytest.approx(0.0, abs=1e-9)
        assert ci[key]["high"] == pytest.approx(0.0, abs=1e-9)
    # Only 3 impostor pairs per identity: far below the rule-of-three bar (>=3000) for FAR 1e-3/1e-4.
    assert ci["threshold_at_far_including_fta.0.001.similarity_threshold"] == {"low": None, "high": None, "resamples_used": 0}
    assert ci["threshold_at_far_including_fta.0.0001.similarity_threshold"] == {"low": None, "high": None, "resamples_used": 0}

    repeat = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=embeddings, pairs=pairs,
        service_threshold=0.5, identities=identities, bootstrap=50, seed=0,
    )
    assert repeat["bootstrap"] == bootstrap  # deterministic for a fixed seed

    different_seed = build_report(
        dataset_name="toy", config=EvalConfig("t"), embeddings=embeddings, pairs=pairs,
        service_threshold=0.5, identities=identities, bootstrap=50, seed=1,
    )
    assert different_seed["bootstrap"]["seed"] == 1


def test_bootstrap_threshold_at_far_gets_a_ci_once_enough_impostors_clear_the_rule_of_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `_bootstrap_report` resamples pair *positions*, not vectors, so the score math itself
    # can be stubbed out with deterministic arrays -- this isolates the resample/grouping/CI
    # logic from cosine-similarity computation (already covered by the test above and by
    # evaluation/tests for score_pairs_full's underlying chunked cosine math).
    people = [f"person{i}" for i in range(5)]
    identities: dict[str, str] = {}
    genuine_pairs = []
    impostor_pairs = []
    for person in people:
        ref, probe = f"{person}/reference", f"{person}/probe"
        identities[ref] = identities[probe] = person
        genuine_pairs.append((ref, probe))
        for j in range(700):  # 5 * 700 = 3500 >= rule-of-three minimum (3000) for FAR 1e-3
            stranger = f"{person}/stranger{j}"
            identities[stranger] = "public"  # never drawn as a left key; identity is irrelevant
            impostor_pairs.append((ref, stranger))
    pairs = PairSet(genuine=tuple(genuine_pairs), impostor=tuple(impostor_pairs))

    genuine_scores = np.full(len(genuine_pairs), 0.9, dtype=np.float64)
    impostor_scores = np.linspace(0.0, 0.999, len(impostor_pairs))

    def _fake_score_pairs_full(pair_list, vectors, fta_value):
        return genuine_scores if pair_list is pairs.genuine else impostor_scores

    monkeypatch.setattr(report_module, "score_pairs_full", _fake_score_pairs_full)

    result = report_module._bootstrap_report(
        pairs=pairs, vectors={}, identities=identities, service_threshold=0.5, bootstrap=30, seed=0,
    )
    assert result["identities"] == len(people) + 1  # 5 people + the shared "public" bucket
    ci = result["ci"]
    assert ci["threshold_at_far_including_fta.0.001.similarity_threshold"]["resamples_used"] > 0
    assert ci["service_threshold.far"]["resamples_used"] > 0

    repeat = report_module._bootstrap_report(
        pairs=pairs, vectors={}, identities=identities, service_threshold=0.5, bootstrap=30, seed=0,
    )
    assert repeat == result


def test_render_markdown_includes_a_bootstrap_table_when_present() -> None:
    pairs = PairSet(genuine=(("a1", "a2"),), impostor=(("a1", "b1"),))
    report = build_report(dataset_name="toy", config=EvalConfig("cfg"), embeddings=_embeddings(), pairs=pairs, service_threshold=0.5)
    report["bootstrap"] = {
        "resamples_requested": 10,
        "identities": 3,
        "seed": 0,
        "ci": {"eer_including_fta.rate": {"low": 0.01, "high": 0.05, "resamples_used": 9}},
    }
    table = render_markdown([report])
    assert "Bootstrap CIs" in table
    assert "eer_including_fta.rate" in table
    assert "9/10" in table


def test_cli_bootstrap_and_role_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    vectors = {"alice": (1.0, 0.0), "bob": (0.0, 1.0)}
    for person in vectors:
        (root / person).mkdir(parents=True)
        (root / person / "reference.jpg").write_bytes(f"{person}-ref".encode())
        (root / person / "probe.jpg").write_bytes(f"{person}-probe".encode())

    seen_roles = {}

    def fake_embed_images(images, config, *, cache_dir, model_factory=None, progress=None, roles=None):
        seen_roles.update(roles or {})
        quality = {
            "face_size_px": 100.0, "interocular_px": 40.0, "roll_deg": 0.0, "yaw_proxy": 0.0,
            "blur_variance": 200.0, "embedding_norm": 5.0, "faces_considered": 1, "second_face_ratio": 0.0,
        }
        return {key: EmbeddingResult(_vec(*vectors[key.split("/")[0]]), 0.9, None, quality) for key in images}

    monkeypatch.setattr("evaluation.__main__.embed_images", fake_embed_images)
    out = tmp_path / "out"
    assert main(["--dataset", f"folder:{root}", "--out", str(out), "--bootstrap", "5", "--seed", "3", "--no-cache"]) == 0

    assert seen_roles == {"alice/reference.jpg": "reference", "bob/reference.jpg": "reference"}
    report = json.loads((out / "report.json").read_text())[0]
    assert report["bootstrap"]["resamples_requested"] == 5
    assert report["bootstrap"]["seed"] == 3
    assert report["bootstrap"]["identities"] == 2
    assert set(report["quality"]) == {"reference", "selfie"}


def test_cli_bootstrap_defaults_to_zero_and_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    (root / "alice").mkdir(parents=True)
    (root / "alice" / "reference.jpg").write_bytes(b"ref")
    (root / "alice" / "probe.jpg").write_bytes(b"probe")

    def fake_embed_images(images, config, *, cache_dir, model_factory=None, progress=None, roles=None):
        return {key: EmbeddingResult(_vec(1, 0), 0.9, None) for key in images}

    monkeypatch.setattr("evaluation.__main__.embed_images", fake_embed_images)
    out = tmp_path / "out"
    assert main(["--dataset", f"folder:{root}", "--out", str(out), "--no-cache"]) == 0
    report = json.loads((out / "report.json").read_text())[0]
    assert report["bootstrap"] is None


def test_parse_dataset_spec(tmp_path: Path) -> None:
    person = tmp_path / "alice"
    person.mkdir()
    (person / "reference.jpg").write_bytes(b"x")
    assert parse_dataset_spec(f"folder:{tmp_path}").images
    with pytest.raises(SystemExit):
        parse_dataset_spec("s3:bucket")


SAMPLES = Path(insightface.__file__).parent / "data" / "images"


def _crop(image: np.ndarray, box: tuple[int, int, int, int], margin: float) -> bytes:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    top, bottom = max(0, int(y1 - margin * h)), min(image.shape[0], int(y2 + margin * h))
    left, right = max(0, int(x1 - margin * w)), min(image.shape[1], int(x2 + margin * w))
    ok, buffer = cv2.imencode(".jpg", image[top:bottom, left:right])
    assert ok
    return buffer.tobytes()


@pytest.mark.integration
def test_cli_end_to_end_on_real_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path.home() / ".insightface" / "models" / "buffalo_l"
    if not any(root.glob("*.onnx")):
        pytest.skip("buffalo_l model files not found")
    from face_recognition_service.config import settings

    for field in ("model_name", "detection_threshold", "min_face_quality", "enhance_mode", "enhance_image"):
        monkeypatch.setattr(settings, field, getattr(settings, field))
    group = cv2.imread(str(SAMPLES / "t1.jpg"))
    data = tmp_path / "data"
    for person, box in {"alice": (466, 269, 573, 415), "bob": (904, 62, 1014, 205)}.items():
        (data / person).mkdir(parents=True)
        (data / person / "reference.jpg").write_bytes(_crop(group, box, 1.0))
        (data / person / "probe.jpg").write_bytes(_crop(group, box, 0.6))
    out = tmp_path / "out"
    assert main(["--dataset", f"folder:{data}", "--out", str(out), "--no-cache"]) == 0
    report = json.loads((out / "report.json").read_text())[0]
    assert report["pairs"]["genuine_scored"] == 2 and report["pairs"]["impostor_scored"] == 2
    assert report["service_threshold"]["far"] == 0.0 and report["service_threshold"]["frr"] == 0.0
    assert (out / "report.md").read_text().startswith("|")
