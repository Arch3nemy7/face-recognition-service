"""Datasets and pair construction: no model, no network."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from evaluation.datasets import (
    Dataset,
    dataset_from_lfw_home,
    load_folder,
    load_pairs_csv,
    parse_lfw_pairs,
)
from evaluation.scoring import build_pairs, score_pairs


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _ds(identity: dict[str, str], references: frozenset[str] | set[str] = frozenset()) -> Dataset:
    return Dataset(
        name="t",
        images={k: Path(k) for k in identity},
        identity=identity,
        references=frozenset(references),
    )


class TestLoadFolder:
    def test_identities_references_and_hidden_files(self, tmp_path: Path) -> None:
        _touch(tmp_path / "alice" / "reference.jpg")
        _touch(tmp_path / "alice" / "probe-1.jpg")
        _touch(tmp_path / "alice" / ".DS_Store")
        _touch(tmp_path / "alice" / "notes.txt")
        _touch(tmp_path / "bob" / "Reference.PNG")
        ds = load_folder(tmp_path)
        assert set(ds.images) == {"alice/reference.jpg", "alice/probe-1.jpg", "bob/Reference.PNG"}
        assert ds.identity["alice/probe-1.jpg"] == "alice"
        assert ds.references == {"alice/reference.jpg", "bob/Reference.PNG"}
        assert ds.name == tmp_path.name

    def test_empty_folder_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_folder(tmp_path)


class TestLoadPairsCsv:
    def test_reads_pairs_relative_to_csv(self, tmp_path: Path) -> None:
        _touch(tmp_path / "a.jpg")
        _touch(tmp_path / "b.jpg")
        _touch(tmp_path / "c.jpg")
        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text("image_a,image_b,same\na.jpg,b.jpg,1\na.jpg,c.jpg,false\n")
        ds = load_pairs_csv(csv_path)
        assert ds.genuine_pairs == (("a.jpg", "b.jpg"),)
        assert ds.impostor_pairs == (("a.jpg", "c.jpg"),)
        assert ds.images["c.jpg"] == tmp_path / "c.jpg"

    def test_missing_column_and_missing_file_raise(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("image_a,image_b\na.jpg,b.jpg\n")
        with pytest.raises(ValueError):
            load_pairs_csv(bad)
        missing = tmp_path / "missing.csv"
        missing.write_text("image_a,image_b,same\nnope.jpg,nada.jpg,1\n")
        with pytest.raises(FileNotFoundError):
            load_pairs_csv(missing)


class TestLfw:
    PAIRS = "2\t1\nAaron_Peirsol\t1\t2\nAJ_Cook\t1\tAbel_Pacheco\t3\n"

    def test_parse_pairs(self) -> None:
        genuine, impostor = parse_lfw_pairs(self.PAIRS)
        assert genuine == [("Aaron_Peirsol/Aaron_Peirsol_0001.jpg", "Aaron_Peirsol/Aaron_Peirsol_0002.jpg")]
        assert impostor == [("AJ_Cook/AJ_Cook_0001.jpg", "Abel_Pacheco/Abel_Pacheco_0003.jpg")]

    def test_malformed_line_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_lfw_pairs("1\nonly_two fields\n")

    def test_dataset_from_lfw_home(self, tmp_path: Path) -> None:
        (tmp_path / "pairs.txt").write_text(self.PAIRS)
        for key in [
            "Aaron_Peirsol/Aaron_Peirsol_0001.jpg",
            "Aaron_Peirsol/Aaron_Peirsol_0002.jpg",
            "AJ_Cook/AJ_Cook_0001.jpg",
            "Abel_Pacheco/Abel_Pacheco_0003.jpg",
        ]:
            _touch(tmp_path / "lfw_funneled" / key)
        ds = dataset_from_lfw_home(tmp_path)
        assert ds.name == "lfw"
        assert len(ds.genuine_pairs) == 1 and len(ds.impostor_pairs) == 1
        assert ds.identity["AJ_Cook/AJ_Cook_0001.jpg"] == "AJ_Cook"


class TestBuildPairs:
    def test_reference_mode(self) -> None:
        ds = _ds(
            {"a/ref": "a", "a/p1": "a", "a/p2": "a", "b/ref": "b", "b/p1": "b"},
            references={"a/ref", "b/ref"},
        )
        pairs = build_pairs(ds, max_impostors=100)
        assert set(pairs.genuine) == {("a/ref", "a/p1"), ("a/ref", "a/p2"), ("b/ref", "b/p1")}
        # someone else's probe against my reference
        assert set(pairs.impostor) == {("a/ref", "b/p1"), ("b/ref", "a/p1"), ("b/ref", "a/p2")}

    def test_all_pairs_mode_without_references(self) -> None:
        ds = _ds({"a/1": "a", "a/2": "a", "b/1": "b"})
        pairs = build_pairs(ds, max_impostors=100)
        assert pairs.genuine == (("a/1", "a/2"),)
        assert {frozenset(p) for p in pairs.impostor} == {frozenset({"a/1", "b/1"}), frozenset({"a/2", "b/1"})}

    def test_sampling_is_capped_unique_cross_identity_and_seeded(self) -> None:
        identity = {f"p{i}/{j}": f"p{i}" for i in range(30) for j in range(5)}
        ds = _ds(identity)
        first = build_pairs(ds, max_impostors=500, seed=7)
        again = build_pairs(ds, max_impostors=500, seed=7)
        other = build_pairs(ds, max_impostors=500, seed=8)
        assert len(first.impostor) == 500
        assert len({frozenset(p) for p in first.impostor}) == 500
        assert all(identity[a] != identity[b] for a, b in first.impostor)
        assert first.impostor == again.impostor
        assert first.impostor != other.impostor

    def test_explicit_pairs_win_unless_augmented(self) -> None:
        ds = Dataset(
            name="t",
            images={"a/1": Path("a/1"), "a/2": Path("a/2"), "b/1": Path("b/1")},
            identity={"a/1": "a", "a/2": "a", "b/1": "b"},
            genuine_pairs=(("a/1", "a/2"),),
            impostor_pairs=(("a/1", "b/1"),),
        )
        assert build_pairs(ds, max_impostors=100).impostor == (("a/1", "b/1"),)
        augmented = build_pairs(ds, max_impostors=100, augment_impostors=True)
        assert {frozenset(p) for p in augmented.impostor} == {frozenset({"a/1", "b/1"}), frozenset({"a/2", "b/1"})}

    def test_no_identities_cannot_derive_impostors(self) -> None:
        ds = Dataset(name="t", images={"x": Path("x")}, identity={}, genuine_pairs=(), impostor_pairs=None)
        with pytest.raises(ValueError):
            build_pairs(ds, max_impostors=10)

    def test_mixed_reference_and_referenceless_identities_raise(self) -> None:
        # "a" has a reference photo; "b" does not -- derived genuine pairs for
        # "b" would be probe-vs-probe while "a"'s impostor pairs would be
        # reference-vs-probe, mixing two different pairing modes.
        ds = _ds(
            {"a/ref": "a", "a/p1": "a", "b/p1": "b", "b/p2": "b"},
            references={"a/ref"},
        )
        with pytest.raises(ValueError, match="reference"):
            build_pairs(ds, max_impostors=100)

    def test_mixed_references_ok_with_explicit_genuine_pairs(self) -> None:
        # Explicit genuine pairs make the pairing unambiguous even though
        # only some identities have a reference photo.
        ds = Dataset(
            name="t",
            images={"a/ref": Path("a/ref"), "a/p1": Path("a/p1"), "b/p1": Path("b/p1"), "b/p2": Path("b/p2")},
            identity={"a/ref": "a", "a/p1": "a", "b/p1": "b", "b/p2": "b"},
            references=frozenset({"a/ref"}),
            genuine_pairs=(("a/ref", "a/p1"), ("b/p1", "b/p2")),
        )
        pairs = build_pairs(ds, max_impostors=100)
        assert set(pairs.genuine) == {("a/ref", "a/p1"), ("b/p1", "b/p2")}

    def test_explicit_impostors_always_kept_and_derived_pairs_capped_and_deduped(self) -> None:
        identity = {f"p{i}/{j}": f"p{i}" for i in range(30) for j in range(5)}
        images = {k: Path(k) for k in identity}
        explicit = tuple((f"p{i}/0", f"p{i + 1}/0") for i in range(0, 20, 2))
        ds = Dataset(
            name="t",
            images=images,
            identity=identity,
            genuine_pairs=(),
            impostor_pairs=explicit,
        )
        pairs = build_pairs(ds, max_impostors=50, augment_impostors=True)
        explicit_set = {frozenset(p) for p in explicit}
        result_set = [frozenset(p) for p in pairs.impostor]
        assert explicit_set.issubset(set(result_set))
        derived_count = len(pairs.impostor) - len(explicit)
        assert derived_count <= 50
        assert len(result_set) == len(set(result_set))  # no duplicates, including reversed-order ones

    def test_reference_mode_sampling_at_scale(self) -> None:
        identity = {}
        references = set()
        for i in range(40):
            identity[f"p{i}/reference"] = f"p{i}"
            references.add(f"p{i}/reference")
            for j in range(5):
                identity[f"p{i}/probe{j}"] = f"p{i}"
        ds = _ds(identity, references=references)
        pairs = build_pairs(ds, max_impostors=300, seed=3)
        again = build_pairs(ds, max_impostors=300, seed=3)
        assert len(pairs.impostor) == 300
        for a, b in pairs.impostor:
            assert a in references
            assert b not in references
            assert identity[a] != identity[b]
        assert pairs.impostor == again.impostor


class TestScorePairs:
    def test_cosine_similarity_and_failures(self) -> None:
        vectors = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([2.0, 0.0]),  # not unit length: must still score 1.0
            "c": np.array([0.0, 1.0]),
            "dead": None,
        }
        scores, failed = score_pairs([("a", "b"), ("a", "c"), ("a", "dead"), ("a", "missing")], vectors, chunk=1)
        assert scores.tolist() == pytest.approx([1.0, 0.0])
        assert failed == 2

    def test_all_failed_returns_empty(self) -> None:
        scores, failed = score_pairs([("x", "y")], {})
        assert scores.size == 0 and failed == 1

    def test_zero_norm_and_non_finite_vectors_count_as_failed(self) -> None:
        vectors = {
            "a": np.array([1.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "n": np.array([np.nan, 1.0]),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scores, failed = score_pairs([("a", "z"), ("a", "n")], vectors)
        assert scores.size == 0
        assert failed == 2
