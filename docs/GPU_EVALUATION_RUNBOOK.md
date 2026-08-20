# GPU Evaluation Runbook

What the operator does after a training run produces checkpoints, on a
machine that can actually generate audio.

Companion to [`GPU_TRAINING_DAY_RUNBOOK.md`](GPU_TRAINING_DAY_RUNBOOK.md).
That one ends with checkpoints on disk. This one starts there, and ends
with a recorded verdict about whether any of them is better.

No provider is named and no credentials appear anywhere in this
document.

---

## Before you rent anything

Everything except generation runs locally, and every step that can fail
cheaply should fail before the meter starts.

1. **Confirm the benchmark is intact.**

   ```bash
   python -m luber_evaluation --repository . suite list
   ```

   `P20_FULL` must report `available: true`. If it does not, the frozen
   benchmark has changed and nothing measured against it would be
   comparable to anything measured before.

2. **Confirm the candidate is real.**

   ```bash
   python -m luber_evaluation --registry ./training-registry \
     run create --candidate-id cand_… --suite SMOKE
   ```

   This refuses a MOCK artifact and a checkpoint that is not `READY`.
   Both refusals are much cheaper here than after an hour of generation.

3. **Rehearse the whole pipeline with no model at all.**

   ```bash
   python -m luber_evaluation --registry ./training-registry \
     run start --evaluation-id eval_… --backend synthetic \
     --baseline-profile baseline.json --candidate-profile candidate.json
   python -m luber_evaluation --registry ./training-registry qualify --evaluation-id eval_…
   ```

   The synthetic backend produces metric values and no audio. It
   exercises the suite, lifecycle, aggregation, comparison, gates,
   ranking and report end to end. Its results are stamped `SIMULATED`
   and are evidence about the tooling, never about a model.

4. **Know what the experiment claimed.** The hypothesis is carried on
   the evaluation from `run create`. If it is about vocal naturalness,
   Korean pronunciation or anything else only a listener can judge, the
   outcome will be `HUMAN_REVIEW_REQUIRED` no matter how clean the
   audio is. That is the correct outcome, and it is worth knowing
   before renting a GPU rather than after.

---

## 1. Provision the host

Same checklist as training:
[`GPU_PROVIDER_CHECKLIST.md`](GPU_PROVIDER_CHECKLIST.md).

Evaluation is inference, not training, so the memory requirement is
lower than a training run's — but **no VRAM figure is stated here**,
because none has been measured on NVIDIA hardware by this project.
Measure it on the first run and record what you observe.

Two full passes of the P20 suite are required, one per side. Budget for
`28 cases × seeds × 2` generations.

---

## 2. The one thing that ruins an evaluation

**An ACE-Step server hosts one model at a time.** Nothing in a
generation request says which weights answered it.

If the baseline and the candidate are generated against the same server
without swapping the adapter in between, the result is a comparison of a
model against itself — and it looks exactly like an honest evaluation
where nothing improved. There is no way to detect this after the fact
from the audio.

Two defences, and use both:

- **Run two servers**, one per side, on different ports, and pass
  `--baseline-url` and `--candidate-url`. Each backend is told which
  model id it serves and refuses any case for a different one.
- **Or render in two passes** into two directories, then evaluate with
  `--backend rendered`. Same guard applies, and the passes cannot
  interleave.

Never point both flags at one URL. Nothing in the tooling can save you
from that, because from its side the two requests are identical.

---

## 3. Generate against two servers

```bash
python -m luber_evaluation \
  --registry ./training-registry --repository . \
  run create --candidate-id cand_… --suite P20_FULL --policy NEUTRAL_CONSERVATIVE

python -m luber_evaluation \
  --registry ./training-registry --repository . \
  run start --evaluation-id eval_… --backend ace-step \
  --baseline-url  http://127.0.0.1:8001 \
  --candidate-url http://127.0.0.1:8002 \
  --api-key-ref ACE_STEP_EVAL_KEY
```

`--api-key-ref` names a secret. It is never a secret value. Nothing in
this package stores a key, and the recorded backend config holds only
the reference name.

A case that fails is recorded as a failure and the run continues. One
failed generation is a measurement — it is what `generation_failure_rate`
counts — and aborting would discard every case already done.

---

## 3b. Or evaluate audio rendered earlier

Render each side into its own directory, named
`<case_id>__seed<seed>.wav`:

```bash
python -m luber_evaluation --registry ./training-registry --repository . \
  run start --evaluation-id eval_… --backend rendered \
  --baseline-audio  ./renders/baseline \
  --candidate-audio ./renders/candidate
```

A missing render is recorded as a failure for that case. Nothing is
substituted from another seed or the other side.

The requested duration is never taken on trust: every file is measured,
and a render short of what the case asked for shows up in
`wrong_duration_rate`.

---

## 4. Qualify

```bash
python -m luber_evaluation --registry ./training-registry --repository . \
  qualify --evaluation-id eval_… --hypothesis-metric <metric>
```

Pass `--hypothesis-metric` when the experiment's claim maps onto a
metric. Omit it when the claim is about something no metric measures —
the hypothesis text alone will drive the decision to
`HUMAN_REVIEW_REQUIRED` or `BLOCKED`, which is the honest result.

Outcomes:

| Outcome | Meaning | What to do |
|---|---|---|
| `QUALIFIED` | Safe, no intolerable regression, claim supported | Promotion review |
| `REJECTED` | Evidence shows a failure | Read the failed gates; do not re-run hoping for a different number |
| `BLOCKED` | The evidence is incomplete or the inputs are compromised | Fix what is missing and evaluate again |
| `HUMAN_REVIEW_REQUIRED` | Technically clean; the actual claim was never examined | Section 5 |

A verdict is written once and never edited. Re-deciding means a new
evaluation, so that the audit log never describes a history that no
longer exists.

---

## 5. When human review is required

```bash
python -m luber_evaluation --registry ./training-registry \
  human-package --evaluation-id eval_…
```

Two files are written. **Send the package. Never send the mapping.** The
mapping is what makes the review blind; it is stored separately so that
sharing the package cannot accidentally share the answer.

Responses come back as JSONL and are appended, never overwritten — a
second listener is additional evidence, not a correction of the first:

```bash
python -m luber_evaluation --registry ./training-registry \
  human-record --evaluation-id eval_… --responses responses.jsonl
```

A synthetic run has no audio, so it cannot be packaged for review. The
command refuses rather than inventing files.

Note that P20 has **no human baseline scores**. Until it does, an A/B
review against the current production model is the only human evidence
available, and it is a comparison rather than an absolute score.

---

## 6. Pick a checkpoint

```bash
python -m luber_evaluation --registry ./training-registry \
  checkpoint rank --run-id run_… --target-metric <metric>
```

Every checkpoint with an evaluation is ranked; every checkpoint without
one is listed as unranked with the reason. **Checkpoints are never
ordered by step or training loss.** The command exits non-zero when
nothing in the run has been evaluated, rather than producing an order
that would look authoritative.

---

## 7. Verify before you trust it

```bash
python -m luber_evaluation --registry ./training-registry --repository . \
  verify --evaluation-id eval_…
```

Recomputes rather than re-reads: suite digest from the suite object,
policy digest from the policy, and the SHA-256 of every sample from the
bytes on disk. Catches an edited suite, a swapped WAV, a missing
artifact, a decision citing a different policy, and a candidate that
turns out to reference a MOCK checkpoint.

Run this before quoting a result to anyone.

---

## 8. Promotion review

```bash
python -m luber_evaluation --registry ./training-registry \
  promote --evaluation-id eval_… --decision APPROVE_FOR_STAGING \
  --by <operator> --rationale "<why>"
```

`APPROVE_FOR_STAGING` is refused unless the candidate is `QUALIFIED`.
The review exists to add operator judgement *on top of* evidence, not
to substitute for it. `HOLD` and `REJECT` are always available.

**Approval for staging is not production activation.** Nothing in this
package serves a model to anyone.

---

## 9. Shutting down

Copy the artifact directory off the host before terminating it —
`evaluation.json`, `suite.json`, `policy.json`, `metrics.jsonl`,
`comparisons.json`, `qualification.json`, `report.md`, `samples.jsonl`.

Generated audio is **not** committed to git. Keep it only if a human
review is outstanding; every sample's digest is recorded either way, so
audio that is discarded and later re-rendered can be checked against
what was measured.

---

## 10. Record what you actually observed

The project has no NVIDIA measurements of its own. After the first real
run, write down and commit:

- generations per hour, per side, at the suite's duration mix
- peak VRAM during inference
- wall-clock for one full P20 pass
- any case that failed, and why

Real numbers replace the absence of numbers. Do not fill this section
with estimates.
