# Phase 35 — real training data readiness audit

**Question:** is there any real music this project is permitted to train
on today?

**Answer: no.** Zero tracks are `ELIGIBLE_FOR_TRAINING`.

This audit changes no rights metadata, moves no file, and acquires
nothing. It reports what is on this machine and what the project's own
policy says about it. Sources are described by group and count;
filenames, lyrics and absolute paths are deliberately absent — see
`docs/DATASET_POLICY.md` and Phase 35's privacy rule.

---

## 1. Classification

| Group | Count | Classification | Why |
|---|---|---|---|
| Imported from a commercial generation service | ~120 | **INELIGIBLE** | `docs/DATASET_POLICY.md` prohibits training on that service's output. Sidecars carry that service's track URLs and CDN image URLs, so the provenance is unambiguous |
| Commercial reference recordings (`COMMERCIAL_REFERENCE` in the discovery summary) | 100 | **EVALUATION_ONLY** | indexed as reference material; no licence, no training grant. The project's own summary already records them as reference-only |
| Operator-produced / released tracks | 3–7 | **UNVERIFIED_RIGHTS** | plausibly the strongest candidates, but **no rights record of any kind exists for them**. Policy: "A file with no license metadata **fails** dataset validation" |
| LUBER's own generated audio (`data/audio`, `data/raw-model-output`) | ~170 files | **INELIGIBLE** | this project's own model output. Phase 25's `self_generated_gate` blocks `SELF_MODEL_OUTPUT` by default, and the policy forbids training a model on its own generations |
| `data/reference` uploads | 4 | **INELIGIBLE** | user-supplied reference audio uploaded for *generation*. Uploading a reference for one purpose is not a training licence |
| `ops-fixture/builds/…` | 4 records | **INELIGIBLE** | a synthetic console fixture. Track ids `trk-0001`…, artist "Operator", and source paths under `/library/` that do not exist on any machine |

**ELIGIBLE_FOR_TRAINING: 0.**

## 2. The rights position, from the project's own records

`data/rights_approval_summary.json` (Phase 7, operational, gitignored):

```
candidates_awaiting_decision : 131
reference_only_indexed       : 100
every group: "group_decision": null
```

Every candidate is **awaiting a human decision**. None has been granted.

A search of this machine for any `training_rights_status` record — the
field `scripts/dataset/ingest_pilot.py` requires, with
`origin_type`, `basis`, `rights_holder`, `document_reference`,
`audio_use_confirmed`, `lyrics_rights_confirmed`,
`performer_rights_confirmed` and `commercial_training_allowed` —
returns **nothing**. No track anywhere carries one.

## 3. `LUBER_TRAINSET_PILOT_V1` does not exist on disk

`data/pilot_manifest_summary.json` records a pilot set:

```
dataset_id            LUBER_TRAINSET_PILOT_V1
track_count           10
total_duration        1670.68 s
content_hash          d14b6911…
lyrics_available      0
```

`infra/gpu/PHASE7_GPU_LAUNCH_PLAN.md` cites the same hash. But
`data/trainset/pilot_manifest.json` — the path
`scripts/dataset/ingest_pilot.py` writes — **is not present**, and
neither is the audio it indexed. Only the summary survives.

So even the historical pilot set cannot be re-verified: its manifest is
gone, its digests cannot be recomputed, and a summary is not a manifest.
Under Phase 35's immutability rule that set would need rebuilding from
source material and re-approving from scratch regardless.

Note also `lyrics_available: 0`. The trainer conditions on lyrics, and
`trainer_adapter._lyrics` deliberately emits an empty string rather than
guessing — a set with no lyrics is trainable but teaches nothing about
vocal phrasing, which is the stated purpose of the first experiment.

## 4. What is missing, exactly

To lift `BLOCKED_DATASET`, **one** of these has to exist:

1. **Operator-original material with a rights record.** The 3–7
   produced/released tracks, each with the sidecar
   `scripts/dataset/ingest_pilot.py` already specifies —
   `training_rights_status: CONFIRMED`, a basis, a rights holder, a
   document reference, and the three confirmation booleans. The audio is
   already here; only the authorisation is absent, and only a human can
   supply it.
2. **A dataset licensed for commercial ML training** (policy source C),
   with its licence document referenced per track.
3. **Contracted material** (policy source B) from composers, producers
   or vocalists, with the contract referenced.

The minimum for a signal pilot is small — see
`docs/REAL_LORA_PILOT.md` §3 — but it is not zero, and it cannot be
synthesised.

## 5. What was explicitly not done

- No audio was acquired, downloaded, scraped or moved.
- No rights metadata was written, edited or inferred.
- The generation-service material was **not** used, and is not a
  borderline case: the policy names it directly.
- Synthetic fixtures were **not** substituted for real data. They
  validate the pilot's mechanics and cannot satisfy real-data
  acceptance; Phase 35's own rules say so and the runner enforces it.

## 6. Consequence for Phase 35

The pilot infrastructure is built, tested and documented; the real pilot
is **not run**. Status: `PARTIAL / BLOCKED_DATASET`.

The next action is a human one: authorise a small rights-cleared set.
It is not another infrastructure phase.
