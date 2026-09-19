"""Evaluation datasets: which images exist, whose they are, and which pairs to score."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REFERENCE_STEM = "reference"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Pair = tuple[str, str]


@dataclass(frozen=True)
class Dataset:
    name: str
    images: dict[str, Path]
    identity: dict[str, str]
    references: frozenset[str] = frozenset()
    genuine_pairs: tuple[Pair, ...] | None = None
    impostor_pairs: tuple[Pair, ...] | None = None


def load_folder(root: Path, name: str | None = None) -> Dataset:
    """`root/<identity>/<image>`; an image whose stem is `reference` is that identity's reference photo."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    images: dict[str, Path] = {}
    identity: dict[str, str] = {}
    references: set[str] = set()
    for person in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for file in sorted(person.iterdir()):
            if not file.is_file() or file.name.startswith(".") or file.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = f"{person.name}/{file.name}"
            images[key] = file
            identity[key] = person.name
            if file.stem.lower() == REFERENCE_STEM:
                references.add(key)
    if not images:
        raise ValueError(f"no images found under {root}")
    return Dataset(name=name or root.name, images=images, identity=identity, references=frozenset(references))


def _parse_same(value: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "same"}:
        return True
    if normalised in {"0", "false", "no", "different"}:
        return False
    raise ValueError(f"cannot read 'same' value {value!r}")


def load_pairs_csv(csv_path: Path, name: str | None = None) -> Dataset:
    """Explicit pairs: columns image_a,image_b,same; paths relative to the CSV's directory."""
    csv_path = Path(csv_path)
    base = csv_path.parent
    images: dict[str, Path] = {}
    genuine: list[Pair] = []
    impostor: list[Pair] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"image_a", "image_b", "same"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            a, b = row["image_a"].strip(), row["image_b"].strip()
            for key in (a, b):
                images.setdefault(key, base / key)
            (genuine if _parse_same(row["same"]) else impostor).append((a, b))
    absent = [key for key, path in images.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} image(s) listed in {csv_path} do not exist, e.g. {absent[0]}")
    return Dataset(
        name=name or csv_path.stem,
        images=images,
        identity={},
        genuine_pairs=tuple(genuine),
        impostor_pairs=tuple(impostor),
    )


def _lfw_key(name: str, index: int) -> str:
    return f"{name}/{name}_{index:04d}.jpg"


def parse_lfw_pairs(text: str) -> tuple[list[Pair], list[Pair]]:
    """LFW pairs.txt: a header line, then `name i j` (same) or `name1 i name2 j` (different)."""
    genuine: list[Pair] = []
    impostor: list[Pair] = []
    for line in text.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 3:
            name, i, j = parts
            genuine.append((_lfw_key(name, int(i)), _lfw_key(name, int(j))))
        elif len(parts) == 4:
            name_a, i, name_b, j = parts
            impostor.append((_lfw_key(name_a, int(i)), _lfw_key(name_b, int(j))))
        else:
            raise ValueError(f"unexpected LFW pairs line: {line!r}")
    return genuine, impostor


def dataset_from_lfw_home(lfw_home: Path, pairs_file: str = "pairs.txt") -> Dataset:
    lfw_home = Path(lfw_home)
    genuine, impostor = parse_lfw_pairs((lfw_home / pairs_file).read_text())
    images_root = lfw_home / "lfw_funneled"
    keys = sorted({key for pair in genuine + impostor for key in pair})
    images = {key: images_root / key for key in keys}
    absent = [key for key, path in images.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} LFW image(s) missing under {images_root}, e.g. {absent[0]}")
    return Dataset(
        name="lfw",
        images=images,
        identity={key: key.split("/", 1)[0] for key in keys},
        genuine_pairs=tuple(genuine),
        impostor_pairs=tuple(impostor),
    )


def load_lfw(data_home: Path | None = None) -> Dataset:
    """LFW 10-fold pairs (3,000 same / 3,000 different), read as the original JPEG files.

    scikit-learn is used only to download and unpack the archive; its own
    loader rescales and crops images, which would bypass the service's
    decode path, so the files on disk are what gets evaluated.
    """
    from sklearn.datasets import fetch_lfw_pairs, get_data_home

    lfw_home = Path(get_data_home(data_home)) / "lfw_home"
    if not (lfw_home / "lfw_funneled").is_dir() or not (lfw_home / "pairs.txt").is_file():
        fetch_lfw_pairs(subset="10_folds", data_home=data_home, download_if_missing=True, resize=0.25)
    return dataset_from_lfw_home(lfw_home)
