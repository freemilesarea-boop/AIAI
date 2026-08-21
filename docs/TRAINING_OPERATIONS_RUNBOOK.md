# Training operations runbook

The sequence an operator follows to get a training run onto a rented GPU
and a verdict back out, and which of the thirteen steps happen in the
console and which happen in the CLI.

The split is not arbitrary. The console holds no remote credentials — see
`docs/TRAINING_CONSOLE.md` §1 — so anything that reaches a machine LUBER
does not own runs from a terminal, and everything that reads, decides or
records runs from the browser.

Nothing here has been done for real. No GPU has been rented, no dataset
transferred and no model trained; every figure this project holds about
NVIDIA hardware is still `null`. This is the procedure, written before it
is needed rather than reconstructed afterwards.

---

## Before anything

- A **model baseline** registered (`luber-training baseline register`).
- A **dataset build** with a `dataset_lock.json` (Phase 23).
- A **curation build** with a `curation_lock.json` recording the dataset
  lock it was computed from (Phase 24).
- A **worker** registered *from the machine itself*, so its capabilities
  come from a probe rather than an assertion.
- A clean working tree. A run from an unidentified revision cannot be
  reproduced, and preflight refuses one.

---

## 1. Check the worker — console

`/ops/training/workers`

Read three columns, not one:

- **Class.** `GPU_TRAINING_READY` is granted by a probe demonstrating
  CUDA through torch on that machine. Nothing else grants it.
- **Liveness.** Derived from the last heartbeat. A record that says
  ONLINE proves only that something wrote ONLINE; a worker that has not
  spoken for eight minutes is STALE whatever the record says.
- **The unmeasured values.** Open the worker. A machine with a list of
  UNKNOWN capabilities has not been probed, and scheduling onto it means
  finding out by renting it.

If nothing is registered:

```bash
# on the worker
python -m luber_training.remote init --name <name> --base /opt/luber
# from the control plane
luber-training remote worker register --transport ssh --host <host> …
```

The classification comes from the probe. There is no flag that asserts
it.

## 2. Create or select an experiment — console

`/ops/training/experiments`

An experiment is a hypothesis and outlives the runs that test it. A
failed run does not disprove it. Write the hypothesis as something that
could be checked: qualification later asks whether a candidate addressed
its own claim, and a vague hypothesis is one nothing can answer.

Creating one starts nothing.

## 3. Create a run — console

`/ops/training/runs/new`

Select an experiment, a dataset build, a curation build, a preset and a
backend. Every digest on the resulting run is read from the locks
themselves; nothing is typed. A curation built from a different dataset
lock is refused before the run exists, so a mismatch does not become a
FAILED run in the experiment's history.

This writes a DRAFT. Nothing is transferred and nothing starts.

## 4. Validate — console

The run's page → **Validate**.

Runs every Phase 25 gate against the files **as they are now**: dataset
lock, curation lock, rights, evaluation leakage, self-generated audio.
On success the run reaches QUEUED and the plan is compiled and hashed.
The gate report is recorded either way, so a rights refusal survives for
the next operator instead of living in whoever's terminal.

If a gate fails there is no override anywhere. Resolve the cause,
re-curate, and create a new run.

Control-plane preflight runs here too, against the assigned worker. Read
the unknowns: they are not a pass.

## 5. Stage — CLI

```bash
luber-training remote run stage --run-id <id> \
  --dataset-build <dir> --curation-build <dir> --audio-root <dir> …
```

Builds the artifact set and re-runs the rights and leakage gates on the
records that are about to become the manifest — not because they ran
badly at validation, but because time passes and a curation can be
regenerated between Monday and Friday. There is no override flag in that
module.

## 6. Remote preflight — CLI, read in the console

Dispatch runs preflight on the worker before launching. Before
confirming, check: worker identity, protocol version, code revision,
environment, dataset digests, disk, and the plan hash. The console shows
the recorded report at the run's **Remote preflight** panel when a local
worker transport is configured.

An UNKNOWN required check blocks just as a failed one does. The run
needs the thing, and nobody established it is there.

## 7. Dispatch — CLI

```bash
luber-training remote run dispatch --run-id <id> --worker-id <id> …
```

Dispatch reconciles unconditionally first and launches only when the
worker positively says there is nothing there. A launch whose reply was
lost may have started a trainer.

The console's Dispatch button is for the **dry-run** backend only, and
its confirmation says so: it trains nothing, produces metrics marked
SIMULATED and a checkpoint of kind MOCK that can never be evaluated.

## 8. Observe — console

`/ops/training/runs/[id]` is where the run lives from here.

- **Lifecycle** — the Phase 25 state machine, including the endings this
  run did not take.
- **Remote state** — the worker's own view, kept separate. If it implies
  a different status than the registry holds, the panel says so and asks
  you to reconcile.
- **Progress** — latest step, epoch, elapsed, loss, learning rate,
  latest checkpoint. No ETA: the trainer measures epochs and records no
  step total.
- **Metrics** — only what exists. Simulated series stay labelled.
- **Logs** — incremental, redacted server-side, stdout and stderr apart.
- **Heartbeat** — the friendly age with the exact instant beside it.

Polling stops on a terminal run and on a hidden tab.

## 9. If the worker is lost — console, then CLI

The run goes **LOST**, not FAILED. Contact stopped; nothing established
that the trainer did.

Do not retry. The console does not offer the button, and the endpoint
refuses: launching a second trainer against one checkpoint directory
produces artifacts that are individually well-formed and jointly
worthless.

**Reconcile first.** With a local worker transport, the console's
*Reconcile remote state* button runs the real Phase 27 reconciliation.
Otherwise:

```bash
luber-training remote run reconcile --run-id <id> …
```

It changes nothing on the worker and may be repeated safely. Read the
outcome literally:

| Outcome | Meaning | Next |
|---|---|---|
| `RUNNING_RECOVERED` | The trainer is still going | Keep watching. The registry stays LOST — the state machine has no path back. |
| `COMPLETED_RECOVERED` | It finished | Collect the checkpoint. |
| `FAILED_RECOVERED` | It failed | Read the failure code; create a retry run. |
| `CANCELLED_RECOVERED` | It was cancelled | Nothing to recover. |
| `UNKNOWN` | The worker answered and cannot say | Do not launch. Investigate on the box. |
| `UNREACHABLE` | The worker did not answer | Do not launch. The trainer may be running. |
| `NOT_PRESENT` | No record of the run | Safe to launch. The only outcome that is. |

## 10. Collect the checkpoint — CLI

```bash
luber-training remote run collect --run-id <id> …
```

Bytes are verified against the digest they were sent with, moved
atomically, and re-hashed locally before the Phase 25 registry hears
about them. A checkpoint reaches READY by having been validated, hashed
and moved — not by being a directory that might be one.

## 11. Evaluate — CLI

A READY checkpoint containing real weights may be nominated as a
candidate; a MOCK artifact can never be. Then run the Phase 26
evaluation against an explicit baseline. It needs a GPU.

The console shows the result, the comparison, and the regressions.

## 12. Read the qualification — console

`/ops/training/evaluations/[id]`

Four outcomes, and none of them is a score:

- **QUALIFIED** — every hard gate passed, nothing regressed beyond
  tolerance, and the hypothesis was addressed.
- **REJECTED** — something regressed. The failed gates are named.
- **BLOCKED** — the evidence is missing or the checkpoint could not be
  evaluated. Not a failure of the model; a failure to have looked
  properly.
- **HUMAN_REVIEW_REQUIRED** — the claim is about something no technical
  metric measures. A real outcome, not a gap.

Read the failed gates, not the headline. A borderline pass and a
catastrophic regression are different facts.

## 13. Decide — CLI, shown in the console

Record a promotion review: `APPROVE_FOR_STAGING`, `REJECT` or `HOLD`.

Approval is for **staging**. Activating a model in production is a
runtime deployment decision made elsewhere, and nothing in this console
does it.

---

## Cancelling

| Backend | What the console's Cancel does |
|---|---|
| `dry-run` | Cancels it. Metrics, logs and finished checkpoints are kept. |
| `remote-gpu` | Records the request in the audit log and says the run is unchanged. |

For a remote run, deliver the signal and then establish what happened:

```bash
luber-training remote run cancel --run-id <id> …
luber-training remote run reconcile --run-id <id> …
```

The run stays as it was until a worker confirms it stopped. A run shown
CANCELLED is a GPU an operator believes they have stopped paying for.

## Retrying

There is no button that re-runs a failed run. **Create retry run**
writes a new DRAFT citing its parent, with the same experiment, dataset
reference and configuration. The original is not edited — the third
attempt can be traced back to the first without the first being
rewritten.

To change a setting, change it on the new run. That is a deliberate act,
not a side effect of retrying.

## Rehearsing without a GPU

```bash
uv run python scripts/development/seed_operator_fixture.py --root ./ops-fixture
```

Point the API at what it writes and every screen has something real to
show, all of it simulated and none of it a measurement of anything.
