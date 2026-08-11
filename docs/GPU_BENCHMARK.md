# GPU Benchmark (Phase 9 design)

## Benchmark set

A fixed set of ≥100 prompts across genres: KPOP, POP, ROCK, JAZZ, LOFI,
HIPHOP, R&B, BALLAD, EDM, ACOUSTIC. Each model version runs the same
prompts with the same seeds.

## Measurements

- Generation latency (wall clock) and real-time factor.
- VRAM peak / GPU utilization.
- Queue waiting time.
- Failure rate by error code.
- `gpu_seconds` per generation (cost accounting).

## Quality evaluation

Automated: audio corruption, silence, clipping, duration accuracy,
prompt adherence, lyrics adherence, vocal presence, audio quality
score, duplicate similarity.

Plus a human listening test set. Models are never promoted on loss
curves alone, and a LoRA/fine-tune is not deployed unless benchmarks
demonstrate improvement over base.

## Hardware targets

RTX 4090 24GB, RTX 5090 32GB, L40S, A100, H100. Apple Silicon (MPS) is
a development convenience, not a benchmark platform.
