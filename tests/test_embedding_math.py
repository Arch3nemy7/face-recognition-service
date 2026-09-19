"""Distance math: scale-invariant, metrics agree, batch == loop, and the API reports the threshold."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import face_recognition_service.main as main_module
from face_recognition_service.schemas.api_schemas import ReferenceEmbedding
from face_recognition_service.utils.embedding_utils import (
    batch_calculate_distances,
    calculate_distance,
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    euclidean_threshold_for,
    find_best_match,
    match_threshold,
)
from tests.conftest import TEST_API_TOKEN, _face_result

AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_TOKEN}"}
RNG = np.random.default_rng(42)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_euclidean_ignores_embedding_magnitude() -> None:
    v, w = RNG.standard_normal(512), RNG.standard_normal(512)
    assert euclidean_distance(v, 25.0 * v) == pytest.approx(0.0, abs=1e-9)
    assert euclidean_distance(3.0 * v, 17.0 * w) == pytest.approx(euclidean_distance(v, w))


def test_euclidean_and_cosine_are_the_same_geometry() -> None:
    for _ in range(50):
        v, w = RNG.standard_normal(512), RNG.standard_normal(512)
        assert euclidean_distance(v, w) ** 2 == pytest.approx(2.0 * cosine_distance(v, w))


def test_raw_arcface_scale_genuine_pair_now_matches_under_euclidean() -> None:
    anchor = RNG.standard_normal(512)
    same_person = 25.0 * _unit(anchor + 0.3 * RNG.standard_normal(512))  # raw norm like real ArcFace output
    distance = calculate_distance(25.0 * _unit(anchor), same_person, metric="euclidean")
    assert distance < match_threshold("euclidean", 0.5)


def test_metrics_make_identical_decisions() -> None:
    cosine_thr = 0.5
    for noise in np.linspace(0.0, 3.0, 400):
        a = RNG.standard_normal(512)
        b = a + noise * RNG.standard_normal(512)
        by_cosine = cosine_distance(a, b) < match_threshold("cosine", cosine_thr)
        by_euclid = euclidean_distance(a, b) < match_threshold("euclidean", cosine_thr)
        assert by_cosine == by_euclid


def test_threshold_helpers() -> None:
    assert euclidean_threshold_for(0.5) == pytest.approx(1.0)
    assert match_threshold("cosine", 0.4) == 0.4
    assert match_threshold("euclidean", 0.4) == pytest.approx(math.sqrt(0.8))
    with pytest.raises(ValueError):
        match_threshold("manhattan", 0.5)


def test_cosine_similarity_range() -> None:
    v = RNG.standard_normal(512)
    assert cosine_similarity(v, 4 * v) == pytest.approx(1.0)
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_batch_matches_loop_with_rows_of_different_scale(metric: str) -> None:
    query = RNG.standard_normal(512) * 20
    refs = RNG.standard_normal((6, 512)) * np.array([[1], [5], [10], [0.1], [30], [2]])
    batched = batch_calculate_distances(query, refs, metric=metric)
    looped = [calculate_distance(query, ref, metric=metric) for ref in refs]
    assert batched.tolist() == pytest.approx(looped)


def test_find_best_match_sorts_ascending_and_picks_the_closest() -> None:
    query = RNG.standard_normal(512)
    refs = [
        ReferenceEmbedding(id="far", embedding=(-query).tolist()),
        ReferenceEmbedding(id="near", embedding=(query * 7 + 0.01 * RNG.standard_normal(512)).tolist()),
        ReferenceEmbedding(id="mid", embedding=RNG.standard_normal(512).tolist()),
    ]
    matches, best = find_best_match(query.tolist(), refs, metric="cosine")
    assert best.id == "near"
    assert [m.distance for m in matches] == sorted(m.distance for m in matches)


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_compare_photos_reports_threshold_and_cosine_similarity(
    client: TestClient, fake_model, monkeypatch: pytest.MonkeyPatch, metric: str
) -> None:
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(main_module, "fetch_image_from_url", lambda url: dummy)
    monkeypatch.setattr(main_module, "decode_image_bytes", lambda data: dummy)
    reference = _unit(RNG.standard_normal(512))
    selfie = _unit(reference + 0.02 * RNG.standard_normal(512))  # noise norm ~0.45 vs unit reference -> cosine ~0.9
    fake_model.analyze.side_effect = [
        _face_result(reference * 20, 0.9),
        _face_result(selfie * 30, 0.8),
    ]
    response = client.post(
        "/api/v1/compare-photos",
        data={"image1": "https://example.com/ref.jpg", "distance_metric": metric},
        files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == pytest.approx(match_threshold(metric, 0.5))
    assert body["cosine_similarity"] == pytest.approx(float(reference @ selfie), abs=1e-5)
    assert body["match"] is (body["distance"] < body["threshold"])
    assert body["match"] is True
