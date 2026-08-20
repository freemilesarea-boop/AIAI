# Dataset Factory

Deterministic, auditable, resumable dataset preparation for LUBER
training. It exists so that GPU time is spent training rather than
scanning files, chasing duplicates, discovering that a file will not
decode, or arguing about which version of a dataset a run used.

It requires no GPU and runs entirely locally.

```
RAW MUSIC FOLDER
        ↓
SCAN                    recursive, read-only, content-addressed
        ↓
HASH                    SHA-256 is the canonical identity
        ↓
DECODE VALIDATION       every file proven decodable before anything expensive
        ↓
DEDUP                   exact bytes, exact audio, near audio → review
        ↓
TECHNICAL ANALYSIS      reuses the Phase 22 finishing analyser
        ↓
QC                      flags → score → tier, all configurable
        ↓
MUSICAL/TEXT ANALYSIS   tempo and key measured; structure and ASR are not
        ↓
PROVENANCE GATE         UNKNOWN never becomes TRUE
        ↓
ELIGIBILITY             four separate questions, every refusal recorded
        ↓
DETERMINISTIC SPLIT     grouped so duplicates and albums cannot leak
        ↓
CANONICAL MANIFEST      dataset_manifest.jsonl, versioned
        ↓
REVIEW                  what a human still has to decide
        ↓
DATASET FREEZE          dataset_lock.json
        ↓
TRAINING EXPORT         train / validation / test
```

---

## 1. Three properties everything else follows from

**Source audio is read-only.** Nothing writes, renames, converts, tags
or deletes anything under the scan root. Every run re-hashes its sources
afterwards and reports a non-zero exit code if a single digest moved.

**Nothing is fabricated.** Where the repository has no detector, the
record says so and carries the reason. Tempo and key are computed
because they can be; vocal class, language, song structure and
transcripts are not, and are null with an explanation rather than
guessed.

**UNKNOWN never becomes TRUE.** Unknown provenance can be analysed and
cannot enter a training export without an explicit, recorded override —
and no override at all clears a hard block.

---

## 2. Architecture

Built inside `packages/dataset` as `luber_dataset.factory`, reusing the
existing `luber_dataset.rights` gate and `luber_dataset.discovery`
heuristics rather than duplicating them.

| Module | Responsibility |
|---|---|
| `config.py` | every threshold, and the hash that keys the cache |
| `scanner.py` | recursive read-only discovery, content identity, immutability check |
| `decoder.py` | ffprobe + full ffmpeg decode; VALID / PARTIAL / INVALID / UNSUPPORTED |
| `audio_analysis.py` | maps `luber_audio_finishing.analyze_audio`; adds rolloff and HF cutoff |
| `musical.py` | tempo and key estimation; structure explicitly unavailable |
| `dedup.py` | fingerprint, exact and near duplicate detection |
| `quality.py` | flags, score, tier |
| `metadata.py` | sidecar validation and embedded tags |
| `provenance.py` | the rights gate |
| `classification.py` | vocals, language, lyrics — mostly saying "unknown" |
| `splitting.py` | eligibility and leak-free deterministic splits |
| `schemas.py` | the versioned manifest record |
| `cache.py` | per-stage resumable analysis cache |
| `pipeline.py` | orchestration and bounded CPU parallelism |
| `manifest.py` | writers, freeze, lock verification |
| `export.py` | training manifests |
| `cli.py` | `python -m luber_dataset.factory` |

Technical measurement is **delegated**, not reimplemented.
`analyze_audio` is what the Phase 22 finishing engine decides from; a
second loudness or stereo implementation here would produce numbers that
disagree with it in ways nobody could adjudicate.

---

## 3. CLI

```bash
python -m luber_dataset.factory build \
    --input  ~/Music/library \
    --output ./dataset-build \
    --workers 8
```

| Command | Purpose |
|---|---|
| `build` | scan, analyse, write the manifest and its companions |
| `freeze` | write `dataset_lock.json` for an approved build |
| `export` | write `train/validation/test.jsonl` from the manifest |
| `verify` | check a build still matches its lock |

Build options: `--workers` (0 chooses `cpu_count - 2`), `--seed`,
`--quality-config`, `--min-tier`, `--include-rights-unknown`,
`--no-resume`, `--force-reanalyze`, `--max-files`, `--dry-run`.

`--dry-run` scans and reports without writing a manifest — for first
contact with an unfamiliar library.

**Exit codes.** `build` returns non-zero if any source hash changed or
any split group leaked. Both are defects, and automation has to be able
to notice them.

---

## 4. Manifest schema

`dataset_manifest.jsonl`, one canonical track per line, `schema_version:
"luber-dataset-factory/1"`.

```json
{
  "schema_version": "luber-dataset-factory/1",
  "track_id": "trk_a822416126ff2b6f",
  "source":      { "source_path", "source_filename", "source_extension",
                   "source_size_bytes", "source_mtime", "sha256" },
  "audio":       { "status", "decode_error", "duration_seconds",
                   "sample_rate", "channels", "bit_depth", "codec", "container" },
  "analysis":    { "peak_dbfs", "true_peak_dbtp", "integrated_lufs",
                   "loudness_range_lu", "rms_dbfs", "crest_factor_db",
                   "dc_offset", "silence_ratio", "clipping_sample_ratio",
                   "spectral_centroid_hz", "spectral_rolloff_hz",
                   "low_energy_ratio", "high_energy_ratio", "stereo_width",
                   "phase_correlation", "dynamic_range_proxy_db",
                   "high_frequency_cutoff_hz", "unavailable" },
  "music":       { "bpm", "bpm_confidence", "key", "key_confidence", "mode",
                   "estimated_downbeat_seconds", "estimated_structure",
                   "structure_status", "unavailable" },
  "vocals":      { "vocal_class", "vocal_confidence", "vocal_source",
                   "vocal_gender", "centre_dominance_db", "reason" },
  "text":        { "lyrics", "lyrics_source", "lyrics_confidence",
                   "transcript", "transcript_source", "notes" },
  "quality":     { "quality_flags", "quality_score", "quality_tier", "reasons" },
  "provenance":  { "source_type", "source_reference", "rights_status",
                   "license", "commercial_training_allowed",
                   "provenance_notes", "field_sources", "hard_blocks",
                   "training_permitted" },
  "dedup":       { "canonical_track_id", "duplicate_of", "duplicate_group_id",
                   "duplicate_type", "similarity_score", "dedup_decision",
                   "all_source_paths", "fingerprint" },
  "eligibility": { "analysis_eligible", "training_eligible",
                   "validation_eligible", "test_eligible", "eligibility_reasons" },
  "metadata":    { "language", "embedded_tags", "sidecar", "sidecar_error" },
  "split": "TRAIN"
}
```

Companion files: `dataset_summary.json`, `dataset_rejections.jsonl`,
`dataset_duplicates.jsonl`, `dataset_review_queue.jsonl`.

**`unavailable`** maps a metric name to *why* it is null. A null meaning
"an MP3 has no bit depth" and a null meaning "the meter did not run" are
different facts, and a bare null makes them look identical.

---

## 5. Quality tiers

| Tier | Meaning |
|---|---|
| `A` | preferred training material |
| `B` | acceptable |
| `C` | usable with caution — appears in the review queue |
| `REJECT` | excluded from training |

A flag is a **finding**, not a verdict. Only `DECODE_ERROR`, `CORRUPT`
and `TOO_SHORT` disqualify by default; everything else costs score.
Plenty of excellent recordings are mono, and a pipeline that rejects on
every flag discards most of a real library.

Flags: `TOO_SHORT`, `TOO_LONG`, `DECODE_ERROR`, `CLIPPING`,
`EXTREME_LOUDNESS`, `EXTREME_SILENCE`, `LOW_SAMPLE_RATE`, `MONO`,
`PHASE_RISK`, `DC_OFFSET`, `LOW_DYNAMIC_RANGE`, `SUSPICIOUS_BANDWIDTH`,
`CORRUPT`, `NEAR_DUPLICATE`, `UNMEASURED`.

Every threshold lives in `QualityThresholds` and can be overridden with
`--quality-config file.json`. An unrecognised key in that file is an
error, not a no-op: a misspelled threshold that silently keeps its
default is a change the operator believes they made and did not.

---

## 6. Dedup rules

| Type | Meaning | Action |
|---|---|---|
| `EXACT_FILE` | identical bytes | one track, all paths retained |
| `EXACT_AUDIO` | similarity ≥ 0.995 | merged automatically |
| `NEAR_AUDIO` | similarity ≥ 0.85 | **`REVIEW_REQUIRED`** — never merged |
| `NONE` | — | kept |

The fingerprint is a Philips-style robust hash reduced to its most
codec-stable component: per 100 ms frame, one bit per adjacent band pair
recording which carried more energy.

Thresholds are set from measurement rather than intuition. Re-encoding
three signals five ways each and comparing against the sources:

| Encoding | Similarity to source |
|---|---|
| lossless (FLAC) | 1.000 |
| MP3 320k | 0.988 – 0.991 |
| MP3 192k | 0.949 – 0.981 |
| MP3 128k | 0.917 – 0.956 |
| AAC 128k | 0.879 – 0.946 |
| **unrelated pairs** | **0.386 – 0.900** |

**Those ranges overlap.** An AAC-128 copy scored 0.879 while two
genuinely different tracks scored 0.900, so no single threshold
separates same from different once lossy encoding is involved. That is
why only a lossless-grade match merges automatically and every lossy
re-encode goes to a person. Reviewing a false alarm costs a minute;
merging two different songs costs one of them permanently.

A pair whose durations differ by more than 2 s is never considered,
however similar the fingerprints — and a *missing* duration does not
clear that gate, because an unknown is not a match.

---

## 7. Rights and provenance

`rights_status`: `VERIFIED`, `USER_OWNED`, `LICENSED`, `PUBLIC_DOMAIN`,
`UNKNOWN`, `RESTRICTED`.
`commercial_training_allowed`: `TRUE`, `FALSE`, `UNKNOWN`.

Training requires permission `TRUE` **and** a permissive status. Both
come only from an operator sidecar.

- A folder name grants nothing. `originals/`, `제작음원/` and
  `public_domain/` are filing decisions, recorded as `INFERRED`.
- Permission claimed without a supporting status is **downgraded to
  UNKNOWN**, with the downgrade written into `provenance_notes`.
- `commercial_training_allowed` is tri-state, not Python truthiness —
  `bool("false")` is `True`, and that must never grant a right.

**Hard blocks** — self-model output and unlawful acquisition — are found
before anything else and cannot be cleared by any sidecar, any
configuration flag, or any export policy.

### Sidecars

`track.wav` → `track.json`:

```json
{
  "title": "…", "artist": "…", "album": "…",
  "genre": "…", "subgenre": "…", "mood": "…",
  "language": "ko",
  "lyrics": "[Verse]\n…",
  "vocal_type": "female",
  "source": "own studio recording",
  "source_type": "USER_ORIGINAL",
  "rights_status": "USER_OWNED",
  "license": "…",
  "commercial_training_allowed": "true",
  "notes": "…"
}
```

Unknown fields are rejected outright — a typo in
`commercial_training_allowed` that silently does nothing is the worst
possible failure for the one field that governs whether audio may be
trained on.

**A sidecar belongs to a path, not to audio.** When the same bytes exist
in two folders and only one carries a sidecar, the factory checks every
path sharing an identity and adopts the declaration it finds. This was
found by running it: the first integration run turned three
explicitly-declared tracks into `RIGHTS_UNKNOWN` because a copy sorted
first.

---

## 8. What is deliberately not inferred

| Field | Status | Why |
|---|---|---|
| `estimated_structure` | always `UNAVAILABLE` | no trained segmenter; a heuristic would be wrong often enough to poison what it labelled |
| `transcript` | always null | no ASR configured; a machine guess in a human-labelled field is worse than a gap |
| `vocal_class` | `UNCERTAIN` unless declared | no validated detector, and a wrong label is indistinguishable from a right one in a manifest |
| `vocal_gender` | only ever the operator's word | a filename is not a voice, and pitch does not determine gender |
| `language` | `unknown` unless declared | no detector; a folder name says what it was filed under, not what is sung |
| `estimated_downbeat_seconds` | always null | tempo recovers the beat period, not its phase |

Language *is* read from supplied lyrics by writing system — a fact about
the text, never a guess about the audio.

### What is measured

**Tempo** — onset-envelope autocorrelation with a log-normal prior
centred at 120 BPM to resolve octave ambiguity, and parabolic
interpolation for sub-lag resolution. Verified against synthetic click
tracks: 72–175 BPM recovered within ~1 BPM. Without the prior it
reported 120 BPM as 60; without the interpolation it was off by up to
3 BPM from integer-lag quantisation.

**Key** — chroma correlated against the Krumhansl-Kessler profiles.
Confidence is the margin over the runner-up, so a track that fits two
keys equally well reports neither.

---

## 9. Splits and leakage

Default 90 / 5 / 5, configurable, deterministic from a seed.

Tracks are **grouped** before being split, and the group is what gets
assigned:

1. duplicate group, if any
2. artist + album
3. artist
4. containing directory
5. the track's own id

Assignment hashes the group key with the seed rather than shuffling a
list, so the result does not depend on input order and **adding a track
cannot move an unrelated track between splits**.

If the same song reaches train and test, the test score measures
memorisation and reports it as generalisation. The number comes out
*better*, which is why this bug survives — nobody investigates a good
result. `verify_no_leakage` therefore runs on every build and the CLI
exits non-zero if it finds anything.

---

## 10. Resume and cache

Keyed by `sha256 + stage algorithm version + configuration digest`,
**per stage**.

- Content identity means a renamed or moved file keeps its cached
  analysis, and a file whose bytes changed cannot reuse one.
- Per-stage keys mean changing a quality threshold does not invalidate
  decode results, and changing the decoder does not discard every
  fingerprint. A single global key would make every tuning pass a full
  re-analysis, and people respond to that by not tuning.
- Writes are atomic; a corrupt cache is discarded and recomputed, never
  trusted.
- Entries for files no longer present are pruned.

Measured on the integration fixture: second run 87.5% hit rate, byte-identical manifest.

---

## 11. Freeze

```bash
python -m luber_dataset.factory freeze --output ./dataset-build --dataset-id ds-2026-08-20
```

Writes `dataset_lock.json` with `dataset_id`, `created_at`,
`schema_version`, `factory_version`, `configuration_hash`,
`manifest_sha256`, `track_count`, `total_duration_seconds`,
`split_counts` and `source_identity_digest`.

The manifest digest is taken over the **canonical** form of each record
— timestamps and mtimes excluded — so two builds of unchanged audio
produce the same lock. `source_identity_digest` is separate and covers
the audio itself: a threshold tweak moves the first and must not move
the second.

Freezing is a separate step from building on purpose. A dataset is
frozen when a person decides it is ready, not automatically because a
run finished.

`verify` re-checks a build against its lock and exits non-zero on any
difference.

---

## 12. Training export

```bash
python -m luber_dataset.factory export --output ./dataset-build --export-dir ./train-manifests
```

Consumes canonical records; never rescans, never re-analyses, never
touches the source tree. Excluded by default:

- quality tier `REJECT`
- `commercial_training_allowed` FALSE
- `commercial_training_allowed` UNKNOWN
- duplicates awaiting review

`--include-rights-unknown` and `--include-review-required` relax the
last two. Nothing relaxes a hard block.

**To train on rights-unknown material you must pass
`--include-rights-unknown` at `build` time as well as at `export`.** The
build decides the split; a track excluded there has no split for the
exporter to place it in. This is deliberate friction on the one decision
that cannot be undone.

---

## 13. Review workflow

`dataset_review_queue.jsonl`, one item per decision, each with
`track_id`, `reason`, `detail`, `source_path`, `recommended_action` and
the relevant metrics.

Reasons: `NEAR_DUPLICATE`, `RIGHTS_UNKNOWN`, `VOCAL_CLASS_UNCERTAIN`,
`LANGUAGE_UNCERTAIN`, `QUALITY_BORDERLINE`, `METADATA_CONFLICT`.

Typical loop: build → read the queue → add or correct sidecars → build
again (cached, so it is quick) → freeze → export.

Hard-blocked tracks do **not** appear in the queue. There is nothing to
decide.

---

## 14. Parallelism

Bounded pool, default `cpu_count - 2` — a machine pinned at 100% for six
hours is unusable, and the operator is normally sitting at it.

Everything expensive and per-file runs in workers; everything needing
the whole corpus (dedup, grouping, splitting) runs afterwards in the
parent, single-threaded and in sorted order. That division is what makes
output deterministic despite the parallelism, and serial and parallel
runs are asserted to produce identical manifests.

A worker failure costs one track. Anything raised is caught, recorded
against that file, and the run continues — a single unreadable file in a
library of forty thousand must not end a six-hour job.
