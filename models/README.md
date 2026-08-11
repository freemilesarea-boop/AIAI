# Models

Model weights are **never** committed to this repository.

This directory documents which models the platform can run and where
their weights come from. Weights are downloaded at deploy time onto GPU
workers (Phase 2+).

## Planned engines

| Provider key | Engine | Status |
|---|---|---|
| `mock` | MockGenerationProvider (CI fixture WAV) | Phase 1 |
| `ace_step` | ACE-Step 1.5 (open-weight, self-hosted) | Phase 2 |
| future | Stable Audio family / licensed / custom | Later |

All engines plug in behind the `MusicGenerationProvider` interface in
`packages/generation-client`. Business logic never imports an engine
directly.

## Policy

- No Suno API usage, reverse engineering, output scraping, or
  credential/token workarounds — see `docs/DATASET_POLICY.md`.
- Every deployed model version is registered and recorded on each
  generation (`model_name`, `model_version`) for reproducibility.
