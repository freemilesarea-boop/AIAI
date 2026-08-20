# Dataset Curation

Phase 23 answers *is this file usable*. This layer answers the harder
question: **given a large eligible dataset, is it a good dataset to
train on?**

It consumes the canonical manifest and nothing else — no rescanning, no
re-decoding, no audio touched. It runs in seconds on metadata alone.

```
PHASE 23 MANIFEST
        ↓
PROFILE                 distributions by count and by duration
        ↓
MISSINGNESS             every share states its denominator
        ↓
CONCENTRATION           top-k, HHI, effective count, entropy
        ↓
TARGET PROFILE          declared intent, versioned and hashed
        ↓
GAPS / OVERREPRESENTATION
        ↓
CURATION SCORING        explicit weighted components, stored per track
        ↓
DOWNSAMPLING / WEIGHTS  never duplication
        ↓
CURATED MANIFEST        derived; the original is untouched
        ↓
CURATION LOCK
        ↓
TRAINING INPUT
```

---

## 1. Three commitments

**Every share states its denominator.** A genre distribution built from
10% coverage is a statement about that 10%. The accessor returns an
`Observation` carrying whether a value is *known*, which makes "unknown"
impossible to drop silently, and unknowns are never a category.

**A gap requires a declared target.** Without one, "12% Korean" is a
measurement, not a deficiency. The default profile declares almost
nothing — it detects domination only — because music datasets are not
meant to be uniform and a neutral profile pushing toward equality would
be a strong opinion wearing a neutral name.

**Rights are a hard gate, never a weight.** Curation runs strictly after
eligibility. Provenance is absent from the scoring components entirely,
so there is no score at which unknown rights become trainable.

---

## 2. Architecture

`packages/dataset/src/luber_dataset/factory/intelligence/`

| Module | Responsibility |
|---|---|
| `schemas.py` | `TrackView` accessor, `Observation`, actions, findings |
| `distributions.py` | categorical and numeric distributions, bucketing, completeness |
| `concentration.py` | HHI, entropy, effective count, long tail, family pressure |
| `profile.py` | one pass producing every distribution for a population |
| `targets.py` | target profiles, validation, built-ins |
| `findings.py` | gaps, overrepresentation, domination |
| `scoring.py` | weighted, auditable per-track score |
| `selection.py` | the gate pipeline and caps |
| `sampling.py` | bounded weights |
| `curation.py` | orchestration |
| `reports.py` | artifacts, lock, verification |
| `human_report.py` | the markdown a person reads |
| `drift.py` | comparison between two datasets |

---

## 3. Input contract

Input is `dataset_manifest.jsonl` and is **immutable**. Every run records
the manifest SHA-256, the dataset lock digest when present, the factory
schema version and the factory version.

**Fields the manifest does not provide first-class**, discovered by
auditing a real one rather than assuming its shape:

- `artist`, `album`, `genre`, `subgenre`, `mood` live inside
  `metadata.sidecar` (operator-declared) or `metadata.embedded_tags`
  (container tags, unverified). A sidecar outranks a tag, and which
  source supplied a value is recorded.
- `evaluation_only` **does not exist in Phase 23 at all**. It is
  supplied by configuration — a file of track ids — never inferred.

Tempo and key are gated on confidence (default 0.55). Phase 23 reports a
tempo for material with no pulse; the fixture used to audit the schema
came back at 50.17 BPM with 0.70 confidence. A low-confidence value is
counted separately as `low_confidence`, never as known.

---

## 4. Distributions and missingness

Each categorical distribution reports, per category: count, share by
count, hours, share by duration. And for the dimension: `known_count`,
`unknown_count`, `low_confidence_count`, `coverage`, and a
`source_breakdown`.

Duration weighting matters and routinely disagrees with counting — a
hundred thirty-second sketches and ten six-minute pieces are ten-to-one
by count and roughly equal by exposure.

The completeness scorecard covers artist, album, genre, subgenre, mood,
language, vocal class, bpm, key and mode.

---

## 5. Concentration metrics

Reported together because no single one is sufficient: top-1 misses a
three-way split, HHI misses a long tail behind a moderate head, entropy
is insensitive to *which* category dominates.

The load-bearing measure is **effective category count** (1/HHI): how
many categories the dataset behaves as though it has. A corpus naming
seven artists and behaving like 2.2 will teach the model 2.2 artists.

Codes: `ONE_ARTIST_DOMINATES`, `ONE_ALBUM_DOMINATES`,
`ONE_SOURCE_DOMINATES`, `ONE_GENRE_DOMINATES`, `ONE_LANGUAGE_DOMINATES`,
`DUPLICATE_FAMILY_DOMINATES`, `LOW_EFFECTIVE_ARTIST_COUNT`,
`EXCESSIVE_SYNTHETIC_SHARE`.

`top1_label` and `top1_share` are always drawn from the *same* ordering.
An earlier version took the label from the count ranking and the share
from the duration ranking, which could name one category and quote
another's number.

**Duplicate family pressure** counts solo tracks as families of one —
excluding them would make a deduplicated corpus appear to have no
families at all and flatter every measurement built on it.

---

## 6. Target profiles

Versioned (`luber-target-profile/1`), hashable, and validated.

```json
{
  "name": "KOREAN_POP_CAPPED",
  "schema_version": "luber-target-profile/1",
  "shares": {
    "language": {"ko": {"min": 0.30, "target": 0.45, "max": 0.60}},
    "vocal_class": {"VOCAL": {"min": 0.60, "max": 0.90}}
  },
  "selection": {"max_tracks_per_artist": 4}
}
```

Built in: `NEUTRAL` (default), `KOREAN_POP`, `GLOBAL_POP`,
`INSTRUMENTAL`. Constrainable dimensions are a closed set — a profile
targeting something the manifest cannot measure is refused rather than
silently ignored.

`min_coverage_to_evaluate` (default 0.60) means a target on a
thinly-labelled dimension produces one `NOT_ASSESSABLE` finding
explaining why, instead of a confident gap computed from twenty tracks.

**Future profiles**, listed rather than half-implemented, with what each
is waiting on: `VOCAL_QUALITY` (needs per-track vocal annotation),
`KOREAN_LYRICS` (needs verified lyrics; no ASR exists),
`MODERN_NON_TROT` (nothing in the manifest distinguishes the styles).

---

## 7. Curation scoring

Five explicit weighted components, summing to 1.0, stored per track:

| Component | Weight | Meaning |
|---|---|---|
| `quality` | 0.30 | tier A 1.0, B 0.8, C 0.5 |
| `coverage_contribution` | 0.30 | how much it fills a thin region |
| `metadata_completeness` | 0.15 | how well described it is |
| `source_diversity` | 0.15 | lower if from a dominant source |
| `duplicate_pressure` | 0.10 | 1/family size |

Coverage is weighted above quality deliberately, so **a rare Tier B
track can beat a redundant Tier A one** — a dataset of uniformly
excellent duplicates is worse than a varied one of merely good tracks.

Coverage averages over the dimensions a track is *known* on, not over
all of them. Counting unknowns as zero sounds conservative and is not:
it divides the one real signal by five and erases the term on any
sparsely-labelled corpus. Measured on nine redundant Tier A tracks and
one rare Tier B, dilution made the rare track **lose** 0.504 to 0.516 —
the opposite of what the weight exists for. Unknowns are skipped rather
than rewarded, so being unlabelled never becomes valuable.

Rights appear nowhere in this table, and that is the point.

---

## 8. Selection pipeline

Fixed order; each stage can only remove.

1. **Rights and policy** — hard gate, nothing later reopens it
2. **Evaluation protection** — `evaluation_only` withheld
3. **Split respect** — only TRAIN is a candidate
4. **Duplicate family caps** — default 1 record per family
5. **Concentration caps** — artist, album, artist-hours

Within each cap the *lowest-scoring* members are dropped, so what
survives is the best of an overrepresented region. Ties break on track
id, so selection is deterministic without any random number generator.

A track whose group key is unknown is never capped — grouping unknowns
would treat "nobody knows the artist" as an artist.

Actions: `KEEP`, `KEEP_PRIORITY`, `DOWNSAMPLE`, `HOLDOUT`, `REVIEW`,
`EXCLUDE_POLICY`, `EXCLUDE_DUPLICATE_PRESSURE`. **No `DELETE`** — the
factory never removes source audio, and a vocabulary that could express
it would eventually be used.

Caps are **opt-in**. The default profiles set none beyond the duplicate
family cap, because a cap is a curation control rather than an automatic
truth.

---

## 9. Sampling weights

Rare material is weighted, **never duplicated**. Copying a rare track
twelve times is not more data; it is the same forty seconds twelve times
an epoch, and the model memorises it while the loss curve looks fine.

- Bounded to `[0.25, 4.0]`, configurable, and the range must contain 1.0
- Only for TRAIN; validation and test receive none, so evaluation stays
  a stable, comparable measurement
- A weight is the factor that would bring a category to its declared
  minimum, before the cap — so it is explainable as "this region is at
  12% and was asked for 30%"
- A track with no targeted, known dimension gets exactly 1.0

---

## 10. Rights hard gate

Barred outright, with the reason recorded:

- `commercial_training_allowed` FALSE or UNKNOWN
- `rights_status` RESTRICTED
- any `hard_blocks` (self-model output, unlawful acquisition)
- `training_eligible` false

Read from Phase 23's verdict rather than recomputed — a second
implementation of the rights rule is a second answer, and the two would
eventually disagree. Regression tests assert that no target profile,
however aggressive, can admit barred material.

---

## 11. Evaluation protection

`--evaluation-only <file>` names track ids that must never enter
training, one per line. Applied before anything else can weigh in.
Benchmark material — the P20 set above all — belongs here. Those assets
are not modified by any part of this layer.

---

## 12. Artifacts

Written to `--curation-output`; the Phase 23 manifest is never touched.

| File | Contents |
|---|---|
| `curated_manifest.jsonl` | original record + action, score, components, reasons, weight |
| `curation_summary.json` | machine summary with before/after distributions |
| `curation_report.md` | the human report |
| `dataset_wishlist.json` | what more material would help |
| `prioritized_review_queue.jsonl` | Phase 23's items, ordered |
| `training_sampling_weights.jsonl` | per-track weights |
| `curation_lock.json` | the frozen decision |

The human report answers, in order: the top 10 risks, what dominates,
what is missing, what is uncertain, **what cannot be assessed and why**,
what to add, what to reduce. The blind-spot section is given equal
weight — a confident report that omits its own limits is worse than
none.

The wishlist derives hours from the declared range: hours required to
reach a minimum with the rest held constant. Where that is not derivable
it says so rather than guessing.

Review prioritisation puts rights first, because resolving rights is the
only review that can *add* training hours; within a reason, the item
unblocking the most hours comes first. Human decisions are never
overridden.

---

## 13. Curation lock

```bash
python -m luber_dataset.factory curate --manifest … --curation-id ds-2026-08-20
```

Records `curation_id`, `created_at`, engine and schema versions, and
digests of: source manifest, source dataset lock, target profile,
config, curated manifest, sampling weights, plus selected counts and a
distribution digest.

The curated-manifest digest is **canonical** — over record content, not
the file — so a lock survives a reformat and cannot survive a changed
decision.

`verify-curation` recomputes the curation and compares. Recomputing
rather than re-reading is the point: a check that compared the curated
file against itself would pass after the source manifest changed
underneath it. It also asserts weights are bounded and no
evaluation-only track entered the selection. Non-zero exit on failure.

A training run should cite **both** `dataset_lock` and `curation_lock`.

---

## 14. Drift comparison

```bash
python -m luber_dataset.factory compare --manifest-a A.jsonl --manifest-b B.jsonl \
    --curation-output ./diff
```

Writes `curation_diff.json` and `curation_diff.md`. Movements below 2%
are treated as noise.

Direction is reported rather than judged, with one exception:
**effective category count falling is called out as a collapse**,
because that is a regression under every training objective.

---

## 15. CLI

| Command | Purpose |
|---|---|
| `profile` | describe a dataset's distributions |
| `curate` | plan a training selection |
| `compare` | drift between two manifests |
| `report` | render the human report |
| `verify-curation` | check a curation against its lock |

Shared options: `--manifest`, `--profile`, `--profile-file`,
`--evaluation-only`, `--min-music-confidence`, `--seed`,
`--curation-output`. `curate` adds `--max-sampling-weight`,
`--curation-id` and `--dry-run`.

`--dry-run` reports the plan and writes no curated artifacts at all.

---

## 16. Determinism and performance

Same manifest, config, profile and seed produce byte-identical curated
manifests and sampling weights, identical scores and identical selected
ids. Record order does not affect the result. Timestamps live outside
every canonical digest.

Measured on a synthetic 10,000-record manifest (26 MB): **2.77 s wall,
347 MB peak RSS** for a full profile plus curation. No all-pairs
comparison happens here — near-duplicate resolution is Phase 23's job
and is not repeated.
