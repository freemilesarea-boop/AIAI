# Model evaluation and promotion policy

When a trained model is allowed to replace the one in production, and
what evidence is required. Written before any candidate exists, because a
promotion rule invented while looking at a result is not a rule.

---

## 1. Training loss is not evidence of quality

A candidate does not get promoted because its loss went down. Loss says
the model fits the data it was given; it says nothing about whether the
music is better, and on a small adapter run it can improve while the
output gets worse in ways the objective is blind to.

Promotion requires all four of:

1. **Technical safety** — no regression in the measurable properties.
2. **Objective baseline comparison** — the frozen benchmark, RAW output.
3. **Blind human preference** — against the same baseline.
4. **No material regression** in the areas the change was not about.

## 2. The comparison must be like-for-like

- **RAW model master only.** Never the Phase 14 finished master. Two
  models compared through the finishing pipeline are partly a comparison
  of the equaliser.
- **Same benchmark version.** `BENCHMARK_P20.json`, unedited. A changed
  prompt suite makes the comparison meaningless, which is why the file
  carries a hash in the baseline manifest.
- **Same rubric and anchors.** `RUBRIC_P20.md`, unchanged. Re-anchoring
  between baseline and candidate is the easiest way to manufacture an
  improvement, and it is forbidden.
- **Blind.** The listener sees the benchmark id, the prompt and the
  expected lyrics. Not the model, the checkpoint or the run.

## 3. Thresholds

**Not set yet, deliberately.** The baseline's human scores do not exist,
and a numeric target chosen before the baseline is known is a number
picked to be beatable.

What is fixed now is the *shape* of the rule:

- The candidate must win **overall blind preference** against the
  baseline across the benchmark.
- It must not materially regress **Korean lyric completeness**,
  **structure/long-form coherence**, or **technical safety**.
- An improvement concentrated in the dimension the experiment targeted,
  with everything else flat, is a success. An improvement in the target
  bought by a regression elsewhere is not.

Exact numbers are set once, immediately after the baseline listening
pass completes, and are then frozen for that benchmark version.

## 4. Technical safety gate

Objective, automatic, and evaluated before a human listens. A candidate
fails outright if, against the baseline distribution, it:

- clips, or produces true peak above the delivery ceiling;
- produces a materially higher silence ratio (early fade / collapse);
- collapses stereo correlation toward mono beyond the baseline range;
- fails to produce the requested duration;
- produces output the finishing pipeline's validator rejects.

These need no listener and no debate. A candidate that fails here is
`REJECTED` without occupying anyone's afternoon.

## 5. Human scores are never estimated

A dimension not scored stays unscored. A partially completed listening
pass is reported as partial, with the count.

No average is computed over invented values, no "approximately" score is
carried forward as if measured, and no overall quality figure is quoted
for a model whose listening pass has not happened. The prior subjective
figure of roughly 2/10 is a recollection of earlier listening, not a
measurement against this baseline, and it is not used as one.

## 6. The trap this policy exists to avoid

Phase 14 finishing genuinely improves delivered audio. It would be easy —
and completely wrong — to show a "quality improvement" that is finishing
doing more work, or a benchmark quietly edited toward what the new model
happens to do well, or an anchor shifted by one point.

Each of those produces a real-looking number and a model that is not
better. The countermeasures are the boring ones: RAW only, hashed
benchmark, frozen anchors, blind listening, and thresholds set before the
candidate exists rather than after.

## 7. Promotion

An `ACCEPTED` candidate becomes the new `BASELINE` only as a deliberate
act, with:

- a new baseline manifest (new id, new frozen date, new engine and model
  identity);
- the previous baseline retained, so old comparisons stay interpretable;
- the benchmark version unchanged if possible — changing the model and
  the benchmark in one step means neither result can be attributed.

The production provider configuration changes only after that.
