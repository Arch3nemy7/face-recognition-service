"""python -m evaluation -- measure face-verification accuracy on a labelled dataset.

Examples:
  python -m evaluation --dataset lfw --preset code-defaults --preset prod-like --augment-impostors
  python -m evaluation --dataset folder:eval_data/verified --model antelopev2 --insightface-home /path
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Settings() requires an API token at import time; the harness never serves HTTP.
os.environ.setdefault("API_TOKEN", "evaluation-harness-not-a-server")

from face_recognition_service.config import Settings  # noqa: E402

from .datasets import Dataset, load_folder, load_lfw, load_pairs_csv  # noqa: E402
from .embedder import (  # noqa: E402
    PRESETS,
    embed_images,
    make_config,
    pipeline_fingerprint,
)
from .report import build_report, render_markdown  # noqa: E402
from .scoring import build_pairs  # noqa: E402

# The code default, not whatever a local .env sets, so reports are comparable.
DEFAULT_COSINE_THRESHOLD = Settings.model_fields["cosine_match_threshold"].default


def parse_dataset_spec(spec: str) -> Dataset:
    kind, _, value = spec.partition(":")
    if kind == "folder" and value:
        return load_folder(Path(value))
    if kind == "pairs" and value:
        return load_pairs_csv(Path(value))
    if kind == "lfw":
        return load_lfw(Path(value) if value else None)
    raise SystemExit(f"unknown dataset spec {spec!r}; use folder:PATH, pairs:CSV, lfw or lfw:DATA_HOME")


def _progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"  embedded {done}/{total}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", required=True, help="folder:PATH, pairs:CSV, lfw or lfw:DATA_HOME")
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS), help="repeatable; default code-defaults")
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--insightface-home", help="directory containing models/<model>/*.onnx")
    parser.add_argument("--service-cosine-threshold", type=float, default=DEFAULT_COSINE_THRESHOLD)
    parser.add_argument("--max-impostors", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment-impostors", action="store_true", help="add sampled cross-identity pairs to explicit ones")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="identity-level bootstrap resamples for confidence intervals (0 = off; see docs/evaluation.md §2)",
    )
    parser.add_argument("--out", default="eval_out")
    parser.add_argument("--cache-dir", default=".eval_cache")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    if args.insightface_home:
        os.environ["INSIGHTFACE_HOME"] = args.insightface_home  # read by the model loader
    service_threshold = 1.0 - args.service_cosine_threshold
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    reports = []
    for spec in args.dataset:
        dataset = parse_dataset_spec(spec)
        pairs = build_pairs(
            dataset, max_impostors=args.max_impostors, seed=args.seed, augment_impostors=args.augment_impostors
        )
        print(
            f"{dataset.name}: {len(dataset.images)} images, {len(pairs.genuine)} genuine / "
            f"{len(pairs.impostor)} impostor pairs",
            file=sys.stderr,
        )
        roles = {key: "reference" for key in dataset.references}
        for preset in args.preset or ["code-defaults"]:
            config = make_config(preset, args.model)
            print(f" config {preset} ({args.model})", file=sys.stderr)
            embeddings = embed_images(dataset.images, config, cache_dir=cache_dir, progress=_progress, roles=roles)
            run = {
                "dataset_spec": spec,
                "seed": args.seed,
                "max_impostors": args.max_impostors,
                "augment_impostors": args.augment_impostors,
                "service_cosine_threshold": args.service_cosine_threshold,
                "pipeline": pipeline_fingerprint(config.model_name),
            }
            reports.append(
                build_report(
                    dataset_name=dataset.name,
                    config=config,
                    embeddings=embeddings,
                    pairs=pairs,
                    service_threshold=service_threshold,
                    run=run,
                    roles=roles,
                    identities=dataset.identity,
                    bootstrap=args.bootstrap,
                    seed=args.seed,
                )
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(reports, indent=2))
    markdown = render_markdown(reports)
    (out / "report.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
