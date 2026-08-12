# Music Quality Benchmark

Measures what the generation engine actually produces musically. This is
an evaluation instrument, not a product feature — nothing here is
reachable from the app.

**Audio is never committed.** Only prompts, manifests, code, rubric,
metadata results, scores, and reports live in git. Generated audio stays
under `data/` (gitignored) and is addressed by benchmark id.

## Layout

```
prompts/     versioned prompt dataset (BENCHMARK_V1.json)
configs/     reserved for alternate configuration matrices
manifests/   which prompts to run, at which duration/seed
results/     JSONL: one line per generation (metadata + metrics)
listening/   rubric, human scores, A/B results
reports/     generated baseline reports
scripts/     the toolkit (importable as `bench`)
tests/       run by the repo's normal pytest suite
```

## Rules

1. **No mock provider.** Benchmark results come from real inference or
   they do not exist.
2. **Prompts are frozen.** Never edit a prompt because a score was
   disappointing. Changing any prompt requires a new
   `benchmark_version`; only equal versions are compared directly.
3. **Technical ≠ musical.** `metrics` (peak, silence, duration flags)
   and rubric scores are separate namespaces. A clean waveform is not a
   good song.
4. **Blind by default.** The listening tool hides model, configuration,
   and seed until a score is saved.
5. **Reference audio is user-supplied only.** The importer has no
   network capability and requires full provenance metadata.

## Running

```bash
# 1. Generate (resumable; skips completed units)
uv run python scripts/benchmark/run_music_quality_benchmark.py \
    --manifest benchmarks/music_quality/manifests/pilot_baseline.json --resume

# 2. Listen and score (blind; developer tool, localhost only)
uv run python scripts/benchmark/listening_tool.py \
    --results benchmarks/music_quality/results/pilot_baseline_p5_v1.jsonl
#    open http://127.0.0.1:8777/        scoring
#    open http://127.0.0.1:8777/ab      blind A/B (needs --ab RUN_A RUN_B)

# 3. Report
uv run python scripts/benchmark/generate_quality_report.py \
    --results benchmarks/music_quality/results/pilot_baseline_p5_v1.jsonl
```

The runner requires the LUBER API, worker, Redis, Postgres, and the
ACE-Step server to be running — it drives the same path a user does.

It refuses to start a generation when free disk is below its safety
margin (`--min-free-disk-gb`, default 6 GB).

## Current baseline

`LUBER_BASELINE_P5_V1` — see [BASELINE.md](BASELINE.md) and
[reports/PHASE5_BASELINE_REPORT.md](reports/PHASE5_BASELINE_REPORT.md).
