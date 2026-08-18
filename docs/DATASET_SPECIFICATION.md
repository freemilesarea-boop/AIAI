# Dataset specification

What may become training data, and what must not. The schema is
`luber_schemas.dataset`; the tool is `scripts/dataset/ingest.py`. This
document explains the decisions those encode.

The organising principle is that **nothing becomes trainable by
default**. Rights start at `UNKNOWN`, tier starts at `REJECT`, split
starts at `EVALUATION_ONLY`, and each has to be raised deliberately by a
person. Training on material the project has no right to, or on the
benchmark it is scored against, are both mistakes that cannot be undone
after the fact — the model is already trained and the score already
meaningless.

---

## 1. What the dataset is for

Not "more music". The dataset targets the specific weaknesses the
baseline exposes:

| Weakness | What the data must supply |
|---|---|
| Trot-like vocal delivery | Contemporary Korean vocals with restrained, modern phrasing |
| Korean phrase omission | Korean vocal tracks with verbatim, verified lyrics |
| Korean pronunciation | Clear Korean diction across tempos and densities |
| Synthetic instrument timbre | Real recorded instruments, well captured |
| Weak production polish | Competently mixed material, not loudness-maximised |
| Narrow stereo, flat depth | Material with genuine stereo staging |
| Arrangement weakness | Complete songs with real section structure |

A track that adds none of these adds nothing, regardless of how good it
is.

## 2. Rights

Seven categories, of which four are trainable:

| Category | Trainable | Meaning |
|---|---|---|
| `OWNED` | yes | Produced by this project or its owner |
| `LICENSED_FOR_TRAINING` | yes | A licence that explicitly permits training |
| `PUBLIC_DOMAIN` | yes | Public domain in the relevant jurisdiction |
| `AI_GENERATED_ALLOWED` | yes | Machine-generated, terms permit training |
| `REFERENCE_ONLY` | **no** | Analysis and comparison only |
| `UNKNOWN` | **no** | Provenance not established — the default |
| `DO_NOT_TRAIN` | **no** | Positively excluded |

Rules that hold without exception:

- **Provenance is never inferred from a path.** A folder named
  `licensed` proves nothing. Rights are supplied explicitly, per scan,
  by the operator.
- **`UNKNOWN` never becomes trainable through inaction.** There is no
  code path that promotes it and no flag that defaults it.
- **`LICENSED_FOR_TRAINING` requires a `rights_note`** naming where the
  permission comes from. "We assumed it was fine" is not a licence.
- **`REFERENCE_ONLY` stays separate from trainable material.** Reference
  audio the user supplied for analysis must never migrate into a
  training manifest, and the schema refuses it.

## 3. Quality tiers

Independent of rights. Material can be perfectly licensed and still too
poor to train on.

- **GOLD** — manually approved. Clean audio, coherent arrangement, good
  mix, accurate metadata, verbatim lyrics, strong vocal where present.
- **SILVER** — useful, with weaker or partial annotation.
- **REJECT** — not for training: low quality, corrupt, duplicate,
  artefact-heavy, or held for evaluation.

Everything arrives as `REJECT` and is promoted after review.

## 4. Splits and leakage

`TRAIN`, `VALIDATION`, `TEST`, `EVALUATION_ONLY`. Only `TRAIN` is
consumed by a training run.

The Phase 20 benchmark — every case in `BENCHMARK_P20.json`, and the
Korean stress set in particular — is `EVALUATION_ONLY` permanently. A
benchmark the model trained on measures the model's memory.

Leakage is detected **by content, not by filename**: the manifest
refuses to validate if one `sha256` or `pcm_sha256` appears under two
splits. The same recording under two names is the classic way a
benchmark quietly stops meaning anything, and a filename check would not
catch it.

## 5. Duplicates

Implemented today:

- **Exact file duplicates** — sha256 of the bytes.
- **Re-encoded / re-containered duplicates** — sha256 of the decoded
  PCM, for WAV.

**Not implemented:** perceptual or fingerprint matching. A trimmed,
resampled or lossily re-encoded near-duplicate will *not* be detected.
The ingestion tool prints this limitation on every run rather than
letting a clean duplicate report imply a guarantee it cannot make.

Adding fingerprinting would mean a new heavyweight dependency, and it is
not justified until the corpus is large enough for near-duplicates to
plausibly exist.

## 6. AI-generated material

Permitted where its terms allow, and tracked separately —
`DatasetItem.is_synthetic` and `DatasetManifest.synthetic_fraction()`.

It is genuinely useful for genre coverage, structure examples and
metadata-rich items, because the metadata is exact rather than inferred.

The risks are real and are the reason for the accounting:

- **Model collapse.** A generative model trained on its own output
  narrows toward its own priors.
- **Artefact reinforcement.** Synthetic timbre is one of the reported
  problems. Training on synthetic timbre teaches more of it.
- **Vocal-style reinforcement.** The trot-like delivery is *in* the
  current output. Feeding that output back is the most direct way to
  entrench precisely the failure this phase is trying to remove.

The consequence: LUBER-generated audio must not be a significant share of
any run intended to fix vocal style, and the share is a number the
manifest reports rather than something discovered afterwards.

## 7. Korean-specific metadata

Recorded: `language`, `lyrics` (verbatim), `lyrics_source`,
`instrumental`, `genre_tags`, `bpm`.

Not recorded, deliberately: phoneme alignments. The project has no
aligner and no verified alignment data. Inventing alignments would
poison the exact capability they are meant to support. The schema leaves
room for them; nothing fabricates them.

Lyrics must be verbatim and their source recorded. A Korean training
item whose lyrics are approximate teaches the model to approximate.

## 8. Audio standardisation

Constrained by the engine, not by preference. `preprocess_vae.py` sets
`TARGET_SR = 48000`, so:

- **Sample rate** — 48 kHz.
- **Channels** — stereo preserved; mono kept mono, never faked wide.
- **Format** — decode-validated on ingest; anything that will not decode
  is reported invalid rather than skipped silently.
- **Checks** — DC offset, clipping, duration sanity.
- **Silence** — leading and trailing silence trimmed conservatively;
  interior silence left alone, because it is often musical.

**Loudness is not normalised.** Mastering the corpus to one target would
teach the model that all music sits at one loudness and would flatten the
dynamics the baseline shows the model currently *has* (median crest
factor 16.2 dB — a genuine strength worth not destroying).

## 9. Phase 14 finishing must not touch training data

Finishing is a delivery step. Running it across the corpus would train
the model on LUBER's own EQ decisions and teach it to produce
already-finished audio, after which finishing would be applied a second
time at delivery.

Any preprocessing EQ, if ever needed, is a separate decision with its own
justification. It is not the finishing pipeline.

## 10. Ingestion

`scripts/dataset/ingest.py`:

- Scans **only an explicitly named directory**. No default, no home-folder
  sweep.
- **Never mutates the source.** Files are opened read-only and hashed.
- **Dry run by default.** `--write` is required to produce a manifest.
- Reports: discovered, valid, invalid, duplicates, rights-unknown,
  eligible, quarantined.
- Emits every item as `REJECT` / `EVALUATION_ONLY` — a candidate, never a
  training set.

Verified against the project's own audio (91 files): all valid, zero
duplicates, all 91 correctly quarantined as `UNKNOWN`.

## 11. What is never committed

Audio, derived features, model weights, and any manifest containing an
absolute path. The schema rejects absolute and root-escaping paths, so a
manifest carrying somebody's home directory cannot be produced in the
first place.
