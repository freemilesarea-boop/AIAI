# Inference Quality Control

*Phase 29. What gets measured, what gets rejected, what gets retried, and
what this system deliberately refuses to have an opinion about.*

---

## The line this phase does not cross

This engine improves **delivered reliability**. It does not judge whether
a song is good.

That distinction is not modesty, it is the design. A number that claimed
to measure musical quality would be compared, ranked on, optimised
against and eventually trusted — and it would be wrong, because nothing
in this repository can hear. So there is no musical quality score, no
vocal naturalness score, no melody score, no commercial-readiness score,
and the one number that does exist is called
`technical_selection_score` in every place it appears, including the
persisted trace.

Where the repository has no validated detector, the answer recorded is
that it has none. `CONTROL_VOCAL_UNKNOWN` is a real finding with
`not_measurable: true`, and it never rejects anything.

## The path a generation now takes

```
request
  → candidate generation          (one call; the default is one candidate)
  → technical QC                  (measure, then check)
  → classification                (ELIGIBLE or REJECTED, on facts)
  → adaptive retry decision       (is another inference justified?)
  → optional further candidates   (bounded by a hard budget)
  → selection                     (rank only what passed the gate)
  → Phase 22 finishing            (unchanged)
  → delivery
```

The healthy path costs **one provider call**. A first candidate with
nothing critical is selected immediately: no second candidate is
generated to compare against it, because the comparison could only say
which of the two is less broken and neither is.

## Eligibility and ranking are separate, and never merge

A candidate with a critical finding **cannot be delivered**, whatever
else is true about it. That is a gate, not a low score.

The alternative — a single scoring function where invalid audio simply
scores badly — would put a broken file into the same ordering as a
working one and rely on arithmetic to keep it from winning. Arithmetic is
exactly the thing that can be wrong. A rejected candidate is never
scored at all: `technical_selection_score` is `None` and
`score_components` is empty, so there is no number for a later
comparison to pick up.

## What is measured

Everything below comes from Phase 22's analyser (`analyze_audio`) or
Phase 23's estimators, with one addition.

| Property | Source |
| --- | --- |
| decode, duration, sample rate, channels | Phase 22 technical block |
| peak, true peak, clipped sample ratio, DC offset | Phase 22 |
| stereo width, correlation, channel balance | Phase 22 |
| band energy shares, spectral rolloff, slope | Phase 22 frequency block |
| harshness and sibilance proxies | Phase 22 risk block |
| tempo and key, with confidences | Phase 23 estimators |
| **positional collapse** | **new in Phase 29** (`collapse.py`) |

The collapse detector is new because Phase 22 has none. A whole-file
silence *ratio* cannot tell a song that fades out over its last six
seconds from a song that stops dead after twelve of forty — averaged
over the file they can look identical. So the measurement is positional:
it finds where content ends, measures the silence after it, and requires
at least eight seconds of trailing nothing before it will say anything.

## What rejects, and what only gets recorded

Rejection is reserved for things that are facts rather than degrees.

**Rejects (CRITICAL):** `INVALID_AUDIO`, `NON_FINITE_SAMPLES`,
`DURATION_SHORT`, `DURATION_LONG`, `SILENT_OUTPUT`, `NEAR_SILENT`,
`EARLY_COLLAPSE`, `SEVERE_CLIPPING`, `DC_OFFSET` (severe),
`PHASE_UNSAFE`, `CHANNEL_IMBALANCE` (severe), `SPECTRAL_COLLAPSE`,
`PROVIDER_TIMEOUT`, `PROVIDER_ERROR`, `PROVIDER_MISCONFIGURED`.

**Records only:** `EXCESSIVE_SILENCE`, `PEAK_OVERSHOOT`, `NARROW_STEREO`,
`LOW_END_PHASE_RISK`, `HIGH_HARSHNESS_PROXY`, `HIGH_SIBILANCE_PROXY`,
`CONTROL_BPM_MISMATCH`, `CONTROL_KEY_MISMATCH`, `CONTROL_VOCAL_UNKNOWN`,
`CONTROL_NOT_MEASURABLE`.

The second list matters more than the first. A dark master, a narrow
mix, a long fade, a mono file and a track right at the ceiling are all
production decisions somebody made on purpose, and all of them look like
defects to a naive threshold. Rejecting them would not protect users
from bad generations; it would destroy good ones and charge for the
retry. Several of them are things **Phase 22 exists to repair** — and
rejecting a candidate for a defect the next stage fixes would spend an
inference to avoid a problem that was about to be solved.

There is no finding for a bright mix, and none for a dark one.

## Where the thresholds come from

Not one was chosen to make a test pass. Each is either taken from a part
of this project that already decided the question, or derived from the
corpus, and `thresholds.py` records the provenance beside every number.

Four are copied rather than imported — Phase 26's evaluation suite and
the Phase 5 benchmark hold numbers this engine needs, and importing
either would drag the training registry or a benchmark harness into the
runtime generation path. `tests/test_thresholds.py` asserts every copy
still agrees with its original, so a drift is a test failure rather than
a silent divergence. That guard is the only reason the copy is
acceptable.

### The spectral rule, and why the first version was wrong

The first version of `SPECTRAL_COLLAPSE` used spectral rolloff: below
2 kHz, the top of the spectrum is empty. Measuring the corpus killed it.

| Metric, 97 raw tracks | min | p05 | median | p95 | max |
| --- | --- | --- | --- | --- | --- |
| Rolloff 85% (Hz) | 352 | 668 | 3727 | 8543 | 9480 |
| Spectral slope (dB/oct) | -11.20 | -9.83 | -6.20 | -4.86 | -3.51 |
| Largest single band share | 0.279 | 0.297 | 0.383 | 0.642 | 0.805 |

Real, deliverable, bass-heavy songs live at a 352 Hz rolloff — exactly
where a "collapsed" file would. A rolloff threshold low enough to spare
them would be too low to catch anything, and the slope conjunction that
propped the rule up had only 0.8 dB of margin against the corpus
minimum.

What does separate the two is **concentration**: a mix, however dark,
occupies several bands, and a degenerate tone occupies one. The corpus
tops out at 0.805; a synthetic single-tone fixture reaches 0.967.
`SPECTRAL_CONCENTRATION_SHARE` is 0.90 — set clear of the corpus rather
than midway between the two, because the cost of being wrong in one
direction is a discarded song and in the other is a defect Phase 22
would have had to handle anyway.

## Retry

The planner is deliberately reluctant. Every retry costs a full
generation, and four rules are applied in order:

1. **Budget first.** Nothing is planned that cannot be paid for, so the
   trace never shows a retry that did not happen.
2. **Non-retryable failures stop immediately.** A misconfigured provider
   answers the same way every time; retrying spends an inference to
   reproduce an error.
3. **The same critical failure twice is a pattern.** A defect the model
   reproduces deterministically will not be fixed by the third attempt.
4. **The change is a seed, and only a seed.** No prompt rewrite, no
   duration adjustment, no provider parameter. A retry that quietly
   altered the request would answer a different question and call it a
   recovery, and the user would have no way to know their song came from
   a prompt they did not write.

A transport failure — the request never produced audio — is retried with
the *identical* request, seed included. Changing the seed there would
silently turn a delivery failure into a different song.

Seeds for later attempts are derived by hashing `(base_seed,
attempt_index, request_digest)`, so attempt *n* of a request always gets
the same seed and an operator can reproduce a run. A request with no
seed keeps no seed on every attempt: inventing one would take away the
provider's own randomisation and make every retry identical, which is
the opposite of what a retry is for.

### Policies

| Profile | Candidates | Retries | Hard call ceiling |
| --- | --- | --- | --- |
| `STRICT_REPRODUCIBLE` | 1 | 0 | 1 |
| `CONSERVATIVE` | 1 | 1 | 2 |
| `STANDARD` *(default)* | 1 | 2 | 3 |
| `EXPERIMENTAL_MULTI_CANDIDATE` | 3 | 1 | 4 |

Every profile carries a hard provider-call ceiling that nothing can
exceed. Covers, extends and replace-range edits run under a
single-attempt policy: an edit's value is what it preserves, and a
second attempt with a different seed may preserve differently. They
still get QC and a trace; they simply run once.

There is **no "best effort" switch**. When the budget runs out the
generation fails, because every candidate that *could* have been
delivered was already eligible and would already have been selected. The
only thing such a switch could add is the delivery of a candidate this
engine measured and rejected, which is the outcome the phase exists to
prevent.

## Selection

Ranking runs only on candidates that passed the gate, and the order is
deterministic and total:

1. fewest MAJOR findings
2. closest control adherence
3. closest duration
4. fewest MINOR findings
5. highest technical selection score
6. attempt order — the earlier attempt wins a tie, which is also the
   cheapest outcome

The trace records **which axis actually separated** two candidates,
rather than gesturing at a score nobody can decompose. The same
candidates always produce the same winner: a selection an operator
cannot reproduce is a selection they cannot audit.

## The trace

Every generation that ran QC stores one on `generations.inference_qc_trace`.
It is written **as the run proceeds**, not assembled at the end, so a
crash between the provider returning and QC finishing still leaves the
record that the call was made — which is what lets a resumed job reuse
the audio instead of buying it again.

Two things it deliberately does not contain:

- **No prompt, no lyrics, no reference audio.** The request is
  identified by a digest, precisely so the record can be read and moved
  without handling the text. This is Phase 29's privacy rule, and
  `tests/test_trace.py` and the service integration tests both assert
  it against the bytes that actually land.
- **No local paths.** Candidate audio lives in a worker's directory and
  is gone by the time anybody reads the trace. The SHA-256 is recorded
  instead.

Failures are recorded, never hidden. A rejected attempt stays in the
trace with its findings and the reason the next attempt happened.

## Candidate audio

Candidates live in a `CandidateWorkspace` under a configured root —
**not** a `tempfile` directory, because `tempfile` cleans up when the
process exits and that is exactly the case resume has to survive.

- Files are named by attempt index, not candidate id: resume knows which
  attempt it wants, not the id a dead process minted.
- Reuse verifies the hash. A file that survived a crash may have
  survived it half-written, and a half-written file is worse than no
  file because it looks like one.
- The workspace is removed on a terminal outcome and scoped to one
  generation, so a stale directory can never be read as another run's
  candidate.

**No candidate is ever uploaded except the winner.** A rejected
candidate cannot reach a library.

## Cost

QC costs roughly **0.9% of the audio's duration**, measured on the real
corpus with tempo estimation enabled:

| Track length | QC time |
| --- | --- |
| 30 s | 0.32 s |
| 180 s | 1.52-1.60 s |
| 240 s | 2.15 s |

Across all 97 corpus tracks the median is 0.24 s. Against generation
times measured in minutes this is not a meaningful addition, and the
measurement is cached by `(sha256, finishing version, engine version)`
so the winner is not re-measured before delivery.

## The dry run

`packages/inference-qc` ships a CLI that runs QC over audio that already
exists and generates nothing:

```bash
uv run python -m luber_inference_qc analyze data/raw-model-output
uv run python -m luber_inference_qc analyze song.wav --duration 180 --bpm 120
uv run python -m luber_inference_qc explain trace.json
uv run python -m luber_inference_qc explain trace.json --summary
```

`analyze` exits non-zero when more than half a corpus is rejected,
because that means the thresholds are wrong rather than that the songs
are. Against the 97-track raw corpus the current thresholds produce
**97 eligible, 0 rejected, 0 critical findings**.

## Configuration

| Setting | Default | Effect |
| --- | --- | --- |
| `inference_qc_enabled` | `true` | `false` restores the pre-Phase-29 path exactly: one call, no measurement, no retry |
| `inference_qc_policy` | `STANDARD` | Any profile name from the table above |
| `candidate_workspace_dir` | `data/generation-candidates` | Where candidate audio lives between generation and delivery |

The switch is a real bypass rather than a quieter version of the same
loop, and there is a test asserting exactly that — an operator turning it
off during an incident should get what they expect.

## Errors a user can see

| Code | Meaning |
| --- | --- |
| `QUALITY_CHECK_FAILED` | Every candidate was measured and rejected, and nothing further would have helped |
| `QUALITY_RETRY_EXHAUSTED` | Every attempt the budget allowed was used and all were rejected |

A provider that never produced audio does **not** get either of these.
It failed to answer, not a quality check, and it keeps its own error
code — reporting a timeout as `QUALITY_CHECK_FAILED` would tell an
operator the model is producing bad songs when the truth is that it is
unreachable.

Internal retry mechanics are not exposed to users. A customer has no
business knowing there were two attempts; an operator has every business
knowing it, and reads it from the trace.

## What this phase did not build

**A vocal presence detector.** The check is behind an injectable
`VocalPresenceDetector` protocol, and the default implementation
honestly returns `UNKNOWN` at zero confidence. Phase 23 already refused
to build a vocal classifier and the reason applies with more force here.
When a validated detector exists it can be injected without touching a
threshold.

**A genre or style detector.** None exists to wire in.

**Lyric completeness or structure adherence.** Not measurable in this
repository, and recorded as `CONTROL_NOT_MEASURABLE` rather than guessed.

**An operator console panel.** Deferred, with reasons — see
[GENERATION_RELIABILITY_RUNBOOK.md](GENERATION_RELIABILITY_RUNBOOK.md).

---

*See also:* [PHASE29_INFERENCE_QC_AUDIT.md](PHASE29_INFERENCE_QC_AUDIT.md)
for the pre-implementation audit,
[GENERATION_PIPELINE.md](GENERATION_PIPELINE.md) for the lifecycle this
sits inside, and
[AUDIO_FINISHING_ARCHITECTURE_AUDIT.md](AUDIO_FINISHING_ARCHITECTURE_AUDIT.md)
for the Phase 22 stage that runs after selection.
