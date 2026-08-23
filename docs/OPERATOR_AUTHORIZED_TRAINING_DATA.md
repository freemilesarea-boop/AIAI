# Operator-authorised training data

Phase 35B is the first time LUBER trained on real music. This document
records what was authorised, on what evidence, what was built from it,
and — just as importantly — what is still not established.

## What the operator authorised

The operator explicitly authorised the contents of a single directory
on their own machine, referred to here as `~/Desktop/LUBER_TRAINING_DATA`,
for use as training material for the LUBER generative music model.

The authorisation is recorded exactly as it was given:

```
authorization_source: OPERATOR_EXPLICIT_AUTHORIZATION
authorization_scope:  <the authorised root>/**
```

It covers that directory and nothing else. No other location on the
machine was scanned, no external or downloaded source was treated as
covered, and nothing outside the scope entered the library. The
ingestion path enforces this: it walks the named root and records the
scope on every track it produces.

## What the authorisation is, and is not

It is one person's decision about a directory they control. That is a
real basis and it is recorded as one. It is also the weakest basis in
the rights model, and the code says so rather than dressing it up:

- `RightsBasis.OPERATOR_AUTHORIZED_SCOPE` is its own value, separate
  from `ORIGINAL_WORK`, `LICENSED` and the rest.
- `RightsStatus.OPERATOR_AUTHORIZED` is its own value in the dataset
  factory, separate from `VERIFIED` and `USER_OWNED`.
- `lyrics_rights_confirmed` and `performer_rights_confirmed` are
  **false** on every record. Nobody produced a publisher clearance or a
  performer agreement, so nothing claims one exists.
- `origin_type` is `UNKNOWN`. A directory of files does not say whether
  a human or a machine made them.

**No contract, licence, ownership document, publisher clearance or
performer agreement was produced or verified for this material.** Any
future claim of that kind needs separate evidence; it cannot be derived
from this record, and the record is shaped so that reading it carelessly
still does not produce one.

The one rule the rights model has always had survives intact: a file
path is not a licence. `validate_rights` refuses an
`OPERATOR_AUTHORIZED_SCOPE` record that does not name a source, a scope
and a date — a folder name wearing a basis is still refused.

## What was found in the authorised root

Read-only inventory of the authorised directory:

| | |
|---|---|
| files (excluding hidden) | 230 |
| WAV | 129 |
| FLAC | 100 |
| MP3 | 1 |
| operator groups | `POP`, `Lofi`, `기성 음원` |

Every one of the 129 WAV files validated: 48 000 Hz, 2 channels,
16-bit, header and bounded decode probe. Total WAV duration 7.06 hours,
from 81.6 s to 377.5 s per track.

Content identity was computed for all 230 audio files by SHA-256. One
byte-identical pair was found — the same audio filed under both `Lofi`
and `POP` — and it enters the library once.

**Evaluation separation: zero collisions.** No authorised file's digest
appears in the benchmark reference audio, the evaluation registry, the
training registry or the data root.

### The `기성 음원` group

That folder name means "released commercial music", and it holds the
100 FLAC files. Phase 35's audit had already classified this material as
`COMMERCIAL_REFERENCE` — reference-only, not trainable. The directory
authorisation does not silently overturn that: the group is outside the
WAV pilot target and **no track from it entered the training library**.
Whether it may ever be trained on is a separate question, needing
separate evidence, and it has not been answered.

## What was built

Two artifacts, kept apart on purpose.

**The authorised library.** All 128 unique valid WAVs, ingested through
the existing `scripts/dataset/ingest_pilot.py` in its operator-authorised
mode. Track ids are the first 16 hex of the audio digest, so nothing in
a report or a manifest carries an operator's filename. Every track
records its `source_group`. Content hash
`c2561442c8d29fef3a846b4fca43181faff4f714cd2d510301352724050ac56f`
over 128 tracks, 420.7 minutes.

**The pilot subset.** Four tracks — two `Lofi`, two `POP`, 12.4 minutes
— chosen by `luber_dataset.select_pilot_subset`, which is a pure
function of the manifest: tracks keyed by audio digest, drawn
round-robin across source groups. The same library and size always give
the same four tracks, on any machine.

The subset was copied — never moved, renamed or transcoded — into a
LUBER-managed staging directory, where the operator sidecars the dataset
factory needs were written beside the copies. Every source file was
re-hashed after staging and every hash was unchanged.

The factory then built, froze and curated the subset, and all five
Phase 25 gates passed on it:

```
dataset_lock       LUBER_AUTHORIZED_PILOT_V1 matches its lock (4 tracks)
curation_lock      LUBER_AUTHORIZED_PILOT_CURATION_V1 derives from this dataset
rights             all 4 selected tracks carry explicit training permission
evaluation_leakage no evaluation-only or held-out material (checked by id and digest)
self_generated     no self-model output in the selection
```

An incidental cross-check worth recording: the subset digest computed by
the selector and the `source_identity_digest` computed independently by
the factory are byte-identical
(`4191518eeb02ce6e361fb3756161df8887f53f9c09b62ad327dd3935a2c983bb`).
Two code paths that share no logic agree on which four tracks these are.

## Where the artifacts live

Nothing derived from this material is in Git. Source audio, staged
copies, manifests carrying machine-local paths, preprocessed tensors,
checkpoints and LoRA weights all live under gitignored roots. What is
committed is code, tests and this document.

The source path map (`*.paths.json`) is machine-local by construction
and says so in its own header: the manifests themselves carry digests
and counts, never filesystem paths.

## What the pilot then showed

The four staged tracks were preprocessed by ACE-Step itself into real
latents — every tensor finite, latent lengths 3360 / 4296 / 4470 / 6000
frames — and a bounded LoRA pilot ran on them under the Phase 35 hard
cap of 48 optimizer steps. It completed with `VALID_SIGNAL` on
`REAL_OPERATOR_AUTHORIZED` material. The numbers are in
`docs/REAL_LORA_PILOT.md` §15b.

What that establishes: this authorised material moves through ingestion,
gating, preprocessing, training, checkpointing and resume coherently.

What it does not establish: anything about convergence, musical quality,
or whether the adapter is good. The resulting weights are stamped
`EXPERIMENTAL`, `NON_PRODUCTION`, `NEVER_AUTO_PROMOTE` and are not in
Git.

## Still not established

- No contract, licence, ownership document, publisher clearance or
  performer agreement exists for this material.
- The `기성 음원` group (100 FLAC, previously classified
  `COMMERCIAL_REFERENCE`) has not been cleared for training and did not
  enter the library.
- Nothing was verified about who created the audio: `origin_type` is
  `UNKNOWN` on every record.
- 124 of the 128 library tracks have never been preprocessed or trained
  on. The pilot used four.

## A rename, and why

Phase 35B recorded the pilot's dataset kind as `REAL_RIGHTS_CLEARED`.
That name claims more than this material has. What clears it is an
operator's authorisation of a directory — no ownership document, no
licence, no publisher clearance, no performer agreement — and a reader
seeing "rights cleared" would reasonably infer all four.

Phase 36 renamed the value to **`REAL_OPERATOR_AUTHORIZED`**. The old
spelling is still read, never written: records that already exist say
what they said, and the API, the console and `DatasetKind.is_real`
accept both. No rights gate was weakened by the rename — the gates
still require exactly what they required, and the value only describes
their outcome more accurately.
