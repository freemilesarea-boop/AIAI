# Dataset Pipeline (Phase 6/10 design)

No training happens before the MVP ships. This document fixes the
design so the platform is provenance-first from day one.

## Ingestion

```
RAW AUDIO → LICENSE VALIDATION → SHA256 → DUPLICATE CHECK →
AUDIO VALIDATION → NORMALIZATION → METADATA EXTRACTION →
ANNOTATION → QUALITY SCORE → TRAINING DATASET
```

License validation is the gate: files without a license record fail.

## Tables

`dataset_assets`: id, storage_key, sha256, duration, sample_rate,
channels, source_type, source_url, creator, owner, license_id,
training_allowed, commercial_training_allowed, created_at.

`dataset_licenses`: id, license_name, licensor, license_document,
training_allowed, commercial_training_allowed,
derivative_model_allowed, output_commercialization_allowed,
effective_date, expiration_date, notes.

## Deduplication

- Exact: SHA256.
- Near-duplicate: audio fingerprint / embedding similarity, so the same
  recording encoded as WAV/MP3/AAC is still caught.

## Splits & leakage

Train/validation/test splits must not share: the same song in different
encodings/variations, and are checked for artist- and source-level
leakage.

## Extracted metadata

duration, sample rate, channels, BPM, key, mode, genre, mood,
instrumentation, vocal presence, language, lyrics, section structure,
loudness, peak.

## Audit

A dataset audit report generator (Phase 6) proves every training file's
eligibility chain. No Suno catalog scraper will ever be built
(`DATASET_POLICY.md`).
