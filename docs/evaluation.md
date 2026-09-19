# Accuracy evaluation

## 1. Purpose

Any change to detection, enhancement or the match threshold should be judged
against measured numbers, not intuition. This harness (`evaluation/`) runs
the service's own decode → preprocess → embed → compare pipeline over a
labelled dataset and reports the accuracy numbers that matter for a
face-verification deployment: EER, FAR/FRR at the threshold the service
actually uses, and the threshold needed to hit a target false-accept rate.

The harness ships with public LFW results only (§4 below) as a worked
example of its output. **It ships no results measured on real people.** A
deployment must run it against its own data before trusting any threshold or
enhancement decision — see §5.

## 2. What it measures

- **FAR / FRR at a threshold** — false-accept rate (impostor pairs scored
  above the threshold) and false-reject rate (genuine pairs scored at or
  below it), computed over pairs that produced a similarity score at all
  (`evaluation/metrics.py`'s `far_frr`).
- **EER** — the equal-error-rate point: the threshold where FAR and FRR are
  closest, searched over every observed score (`equal_error_rate`).
- **Threshold at a target FAR** — the similarity threshold that would hold
  the false-accept rate at a chosen target (`threshold_at_far`), and the FRR
  paid for it. By the rule of three, reporting a rate down to `FAR` needs at
  least `ceil(3 / FAR)` impostor pairs (3,000 for FAR 1e-3, 30,000 for
  1e-4) before any number is reported at all — `min_impostors_for`.
- **Failure to acquire (FTA)** — an image that never produced an embedding at
  all (detection failure, quality gate, decode error). A genuine pair with an
  FTA image is a missed match; an impostor pair with one is a (correct)
  refusal.
- **FTA-inclusive rates (ISO/IEC 19795-1 "generalised" convention)** — EER,
  FAR/FRR and threshold-at-FAR recomputed with every FTA pair counted as a
  rejection at every threshold, using a finite sentinel score
  (`FTA_SENTINEL = -2.0`, strictly below any real cosine similarity, which
  lies in `[-1, 1]`) rather than special-casing the metrics math
  (`evaluation/report.py`). This makes runs with different FTA counts
  comparable to each other, which the scored-only numbers are not — two
  presets that fail to acquire different numbers of images are otherwise
  compared on different, non-comparable subsets. When the sentinel alone
  decides a reported threshold (the reported similarity threshold is at or
  below `-1`), the report prints `n/a (FTA-dominated)` instead of a number.
- **Identity-level bootstrap confidence intervals** (`--bootstrap N`) — 95%
  CIs on `eer_including_fta.rate`, `service_threshold.far`,
  `service_threshold.frr_including_fta`, and `threshold_at_far_including_fta`
  at the two lowest FAR targets, built by resampling *identities* (not
  individual pairs) with replacement `N` times and recomputing every metric
  on each resample. Identity-level resampling matters because many pairs
  share an image (one reference photo behind many probe pairs, say); pair-
  level resampling would treat those as independent draws when they aren't,
  understating the true uncertainty. This makes the interval conservative
  (usually wider) rather than an exact i.i.d. frequentist CI — it estimates
  how much the number would move on a different, similarly-sized population,
  not a claim about a single "true" population rate.
- **Exact per-resample EER** — every value in the bootstrap block is exact,
  not an approximation: a resample's `eer_including_fta.rate` equals what
  `metrics.equal_error_rate` would compute directly on that resample's
  genuine/impostor arrays, and the FAR/FRR/threshold values equal their
  respective `metrics` functions on the same resample
  (`evaluation/report.py`'s `_resample_metrics`, checked by a randomised
  property test in `tests/test_eval_report_cli.py`). An early version of this
  feature used a fixed quantile grid to approximate the EER for speed; on a
  well-separated genuine/impostor regime that approximation overstated the
  EER by an order of magnitude, so it was replaced with an exact method
  before use.

## 3. Running it

```bash
python -m evaluation --dataset SPEC [--dataset SPEC ...] \
  [--preset NAME ...] [--model NAME] [--insightface-home DIR] \
  [--service-cosine-threshold 0.5] [--max-impostors 200000] [--seed 0] \
  [--augment-impostors] [--bootstrap 0] [--out eval_out] [--cache-dir .eval_cache | --no-cache]
```

### Dataset specs (`--dataset`, repeatable)

- `folder:PATH` — `PATH/<identity>/<images>`. See
  [`docs/evaluation-data.md`](evaluation-data.md) for the exact layout.
- `pairs:CSV` — explicit pairs, columns `image_a,image_b,same`; see
  [`docs/evaluation-data.md`](evaluation-data.md).
- `lfw` or `lfw:DATA_HOME` — Labeled Faces in the Wild, downloaded once via
  scikit-learn (`DATA_HOME` defaults to `~/scikit_learn_data`) and read
  through the service's own JPEG-decode path.

### Presets (`--preset`, repeatable; default `code-defaults`)

Enhancement (CLAHE + auto-gamma) is controlled by `ENHANCE_MODE`
(`face_recognition_service/config.py`'s `enhance_mode`, with the deprecated
`ENHANCE_IMAGE`/`enhance_image` boolean still honoured as a fallback — see
`Settings.effective_enhance_mode`): `always` detects and embeds on the
enhanced image; `off` never enhances; `detect_fallback` detects on the
original first and only falls back to the enhanced copy when nothing is
found there, always embedding from the original pixels.

| preset | detection threshold | min face quality | enhance_mode |
|---|---|---|---|
| `code-defaults` | 0.5 | 0.7 | `detect_fallback` |
| `prod-like` | 0.1 | 0.1 | `always` |
| `prod-fallback` | 0.1 | 0.1 | `detect_fallback` |
| `prod-off` | 0.1 | 0.1 | `off` |
| `no-enhance` | 0.5 | 0.7 | `off` |

`code-defaults` mirrors `face_recognition_service/config.py`'s defaults. The
`prod-*` presets hold the looser detection/quality thresholds some
deployments choose to run with, at each of the three enhancement modes, so
you can measure what changing just the enhancement mode costs or buys at
those thresholds. `no-enhance` isolates what CLAHE/gamma enhancement buys or
costs relative to `code-defaults`.

### Other flags

- `--model` — InsightFace model pack name (`buffalo_l`, `antelopev2`, ...).
- `--insightface-home DIR` — point at a directory whose `models/<model>/`
  holds the `.onnx` files directly (not nested), for model packs that don't
  extract flat by default.

  **antelopev2 note:** the upstream antelopev2 zip extracts nested, as
  `models/antelopev2/antelopev2/*.onnx`, but the service looks for
  `models/antelopev2/*.onnx`. The Docker image already flattens it; locally,
  point `--insightface-home` at a directory whose `models/antelopev2/`
  contains the `.onnx` files directly, e.g.:

  ```bash
  H=/tmp/insightface-flat
  mkdir -p "$H/models/antelopev2"
  ln -sf ~/.insightface/models/antelopev2/antelopev2/*.onnx "$H/models/antelopev2/"
  python -m evaluation --dataset lfw --model antelopev2 --insightface-home "$H"
  ```
- `--service-cosine-threshold` — defaults to the code default in
  `config.py` (cosine distance 0.5), not whatever a local `.env` happens to
  set, so runs stay comparable across machines. If the deployment you care
  about sets `COSINE_MATCH_THRESHOLD` to something else, pass that value here
  so the "FAR/FRR @ service" row reflects what that deployment actually does.
  The service accepts a pair when `cosine_distance < service-cosine-threshold`,
  i.e. `similarity > 1 - service-cosine-threshold`; the harness reports
  against that same similarity cut.
- `--bootstrap N` (default `0`, off) — see §2 above for what it computes.
  Draws `N` resamples of identities with replacement, using
  `numpy.random.default_rng(--seed)` — deterministic for a fixed `--seed`.
  `report.json`'s `bootstrap` block holds, per metric, the 2.5th/97.5th
  percentile across resamples where it could be computed (`{low, high}`) and
  `resamples_used` (how many of the `N` resamples actually produced a value).
  `report.md` gets one extra small table per report when `--bootstrap` is
  set.
- `--max-impostors` / `--seed` — cross-identity impostor pairs are sampled up
  to this cap, with this seed, for reproducibility. Pairs explicitly given by
  the dataset (LFW's official pairs, a `pairs:` CSV) are always kept in full;
  `--augment-impostors` additionally samples cross-identity pairs for
  datasets that only define identities (`folder:`, and LFW beyond its
  official pairs).
- `--out` — output directory for `report.json` and `report.md` (default
  `eval_out`, gitignored — see §5).
- `--cache-dir` / `--no-cache` — embeddings are cached per
  `(model, EvalConfig fingerprint, pipeline fingerprint)` in
  `.eval_cache/<model>-<config-fp>-<pipeline-fp>.npz`, keyed by each image's
  sha256. The pipeline fingerprint covers the service code that produces an
  embedding, the versions of the libraries it depends on, and the actual
  `.onnx` model files on disk — so a code change, a dependency upgrade, or
  swapping model files invalidates the cache automatically. The cache has no
  lock: don't run two evaluations against the same `--cache-dir` at once.

### The `python -m evaluation` CWD caveat

If you run `python -m evaluation` with `PYTHONPATH` set instead of
`pip install -e .`, run it from a directory that doesn't itself contain the
repo's code: `python -m` puts the current directory first on `sys.path`, and
a same-named local module or package there would shadow the intended one.

## 4. Results: LFW (public-dataset baseline)

**These numbers are from Labeled Faces in the Wild, a public benchmark — not
from any real deployment.** LFW is celebrities photographed in varied, often
flattering, in-the-wild conditions: different pose, lighting and camera
quality than most real verification workloads (kiosk or phone selfies
against an enrollment photo, say). Use LFW to compare presets and models to
each other; do not use it to set a production threshold — see §5.

Dataset: LFW 10-fold verification pairs — 3,000 genuine pairs, 3,000 official
impostor pairs (always kept), plus ~1,000,000 sampled cross-identity impostor
pairs added via `--augment-impostors --max-impostors 1000000`. Model:
`antelopev2`. Bootstrapped over 4,281 identities, 200 resamples, `--seed 0`.

Reproduce with:

```bash
python -m evaluation --dataset lfw --augment-impostors --max-impostors 1000000 \
  --preset prod-like --preset prod-fallback --preset prod-off \
  --model antelopev2 --bootstrap 200
```

| dataset | config | model | images failed | genuine scored | EER (sim thr) | EER incl. FTA | FAR @ service | FRR @ service | FRR incl. FTA | sim thr @ FAR 1e-3 (FRR) | sim thr @ FAR 1e-4 (FRR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lfw | prod-like | antelopev2 | 0/7701 | 3000/3000 | 1.43% (0.157) | 1.43% (0.157) | 2.99e-04% (3/1,002,899) | 6.10% | 6.10% | 0.227 (1.53%) | 0.282 (1.60%) |
| lfw | prod-fallback | antelopev2 | 0/7701 | 3000/3000 | 1.33% (0.149) | 1.33% (0.149) | 1.99e-04% (2/1,002,899) | 3.73% | 3.73% | 0.217 (1.43%) | 0.272 (1.50%) |
| lfw | prod-off | antelopev2 | 0/7701 | 3000/3000 | 1.33% (0.149) | 1.33% (0.149) | 1.99e-04% (2/1,002,899) | 3.73% | 3.73% | 0.217 (1.43%) | 0.272 (1.50%) |

FAR @ service is shown as false accepts / impostor pairs, with the
percentage rounded to 3 significant figures rather than 2 decimal places —
at this rate (order 1e-6), rounding to "0.00%" would read as zero when it
isn't.

Bootstrap 95% CIs (4,281 identities, 200/200 resamples used, seed 0):

| config | metric | 95% CI low | 95% CI high |
|---|---|---|---|
| prod-like | `eer_including_fta.rate` | 1.03% | 1.96% |
| prod-like | `service_threshold.far` | 0 | 7.85e-04% |
| prod-like | `service_threshold.frr_including_fta` | 5.07% | 7.13% |
| prod-fallback | `eer_including_fta.rate` | 0.98% | 1.90% |
| prod-fallback | `service_threshold.far` | 0 | 6.12e-04% |
| prod-fallback | `service_threshold.frr_including_fta` | 3.00% | 4.52% |
| prod-off | `eer_including_fta.rate` | 0.98% | 1.90% |
| prod-off | `service_threshold.far` | 0 | 6.12e-04% |
| prod-off | `service_threshold.frr_including_fta` | 3.00% | 4.52% |

FAR bounds are the unrounded values from the run's `report.json` (the
`report.md` renderer rounds anything below 0.005% to "0.00%"; these bounds
are printed here from the underlying JSON instead so a non-zero rate never
reads as zero).

On this public dataset, `prod-like`'s FRR-incl.-FTA CI (5.07–7.13%) and
`prod-fallback`/`prod-off`'s (3.00–4.52%) are disjoint: always-on enhancement
of the embedded image (not just detection) measurably hurt recognition on
LFW. `prod-fallback` and `prod-off` are identical to the digit on every LFW
column here — nothing in this LFW run distinguishes a conditional detection
fallback from no enhancement at all. **Whether the same holds on your own
data is a separate, required measurement — see §5.**

## 5. Method notes (apply regardless of dataset)

- **Calibrate on your own data.** The threshold and preset that look best on
  LFW are not necessarily what your deployment should run: §4's LFW numbers
  exist to demonstrate the harness and to compare presets/models to each
  other on a public, reproducible dataset, not to set a production
  threshold. Real impostor pairs from your own population (people
  photographed by the same camera/kiosk, under the same lighting, who
  resemble your reference photos more than a random LFW pair does) can score
  substantially higher than random LFW impostor pairs — so a threshold tuned
  against LFW alone can be unsafe against real impostors from your
  population. Run the harness against your own labelled dataset
  (`docs/evaluation-data.md`) before trusting any threshold number.
- **Judge enhancement (and any detection/preprocessing change) on FRR
  including FTA, not FRR alone.** FRR at the service threshold is computed
  only over images that produced an embedding at all; comparing two presets
  with different detection/quality gates on that number alone hides
  differences in how many images each one fails to acquire in the first
  place. FRR incl. FTA (§2) counts every failure to acquire as a rejection,
  so it is what an end user actually experiences and is the fair way to
  compare two configurations that reject a different number of images
  outright.
- **Quality gates ship off by default, and should stay off until measured.**
  This template's `MIN_*`/`MAX_ABS_*` quality-gate settings default to `0`
  (off) and `QUALITY_GATES_APPLY_TO_REFERENCE=false`. Before turning any of
  them on, run the harness against your own genuine population and look at
  `report.json`'s `quality` block (percentile summaries of `face_size_px`,
  `interocular_px`, `roll_deg`, `yaw_proxy`, `blur_variance`,
  `embedding_norm`, `faces_considered`, `second_face_ratio` — see
  `face_recognition_service/models/face_model.py`'s `FaceQuality`) to see
  what your own accepted images actually look like. A gate set without that
  measurement risks rejecting legitimate images your population already
  produces.
- **Survivor bias in any real-world genuine set.** If you build a genuine
  dataset from images a running system already accepted (rather than a
  controlled, independent capture), that dataset is biased toward what the
  *current* model already likes — an image the current model rejects, and
  that nobody separately confirms as genuine, is never in it. Any FRR
  measured against such a dataset is a lower bound on the real reject rate,
  not an independent estimate, and any comparison between a survivor-biased
  real dataset and a non-biased one (like LFW) is biased in the survivor-
  biased dataset's favour.
- **Random-pair FAR understates targeted impersonation, and there is no
  liveness check.** Every FAR number this harness produces — on LFW or on
  your own data — comes from randomly paired strangers (a "zero-effort"
  impostor: someone who happens to be compared and isn't who the reference
  says). It does not model a person who deliberately tries to pass as a
  specific target (a look-alike, a close relative, or someone who studies
  that target's photo and prepares an attack). Treat any FAR measured this
  way as a floor on the real risk from a deliberate impersonation attempt,
  not an estimate of it — and note that this service performs no liveness
  check, so a sufficiently good printed photo or screen replay of the target
  is not ruled out by anything this harness measures.

## 6. Data handling

`eval_data/`, `eval_out/` and `.eval_cache/` are gitignored. They hold face
images, embeddings (biometric templates) and comparison reports — treat them
like any other biometric data:

- Never commit them, attach them to a PR, or paste their contents into chat,
  tickets or logs.
- Keep them on the machine that generated them; if you must move them, copy
  over an authenticated channel and delete the source copy afterwards.
- Delete `eval_data/`, `eval_out/` and `.eval_cache/` when you're done with a
  run. They regenerate cheaply from LFW or from a fresh export of your own
  data.
- If you pass a custom `--out` or `--cache-dir` outside the repo, make sure
  that location is also private and, if it's anywhere near a git repo,
  gitignored — the harness only protects the default paths.

See [`docs/evaluation-data.md`](evaluation-data.md) for how to build your own
labelled dataset, including how to pseudonymise identities before anything
touches `eval_data/`.
