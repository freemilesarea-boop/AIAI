# Training Orchestration

Everything needed to launch training on a rented GPU **before** one is
rented, so that GPU day is provisioning and a launch rather than
architecture work.

This layer performs no training. It decides what may be trained, on what
data, with what configuration, and records what happened. Nothing here
imports torch, CUDA, or a provider SDK.

```
DATASET LOCK + CURATION LOCK
            ↓
EXPERIMENT                a hypothesis, outliving its runs
            ↓
TRAINING RUN              one concrete attempt
            ↓
PREFLIGHT                 gates + code version + worker + disk
            ↓
TRAINING PLAN             immutable, hashed
            ↓
WORKER MATCH              reported capabilities, never assumed
            ↓
EXECUTION BACKEND         dry-run today; remote GPU contract ready
            ↓
TRAINER                   ACE-Step `train.py fixed`
            ↓
METRICS + CHECKPOINTS     append-only; atomically finalised
            ↓
EVALUATION CANDIDATE      a request for evidence, not a claim
            ↓
PHASE 26 EVALUATION
            ↓
PROMOTION DECISION
```

**Phase 25 improves reproducibility, safety, operability and future GPU
efficiency. It does not improve vocal quality, instrument quality,
melody, Korean pronunciation, trot bias or music quality in any way.**

---

## 1. Architecture

`packages/training/src/luber_training/`

| Module | Responsibility |
|---|---|
| `ids.py` | prefixed collision-resistant identities |
| `config.py` | `TrainingConfig`, validation, presets |
| `entities.py` | entities and state machines |
| `registry.py` | crash-safe locked filesystem registry, audit log |
| `gates.py` | dataset, curation, rights, leakage, self-generated |
| `plan.py` | immutable plan, environment and code-version locks |
| `backends.py` | backend abstraction, dry-run, remote contract |
| `metrics.py` | metric events, checkpoint finalisation, retention |
| `trainer_adapter.py` | manifest→trainer dataset; plan→argv |
| `probe.py` | worker capability probe |
| `orchestrator.py` | the service layer |
| `cli.py` | operator command line |

Orchestration never depends on a GPU vendor. Training *intent* and
*execution* are separated so that choosing a provider later is a
configuration decision, not a rewrite.

---

## 2. Entities and state machines

`ModelBaseline`, `Experiment`, `TrainingRun`, `TrainingConfig`,
`TrainingDatasetRef`, `TrainingWorker`, `Checkpoint`, `MetricEvent`,
`ArtifactRef`, `EvaluationCandidate`, `PromotionDecision`.

**Run states**

```
DRAFT → VALIDATING → QUEUED → STARTING → RUNNING → COMPLETED
                  ↘        ↘          ↘        ↘ FAILED / CANCELLED / LOST
```

There is **no PAUSED state**: the installed trainer cannot pause, and a
state nothing can enter would misrepresent what the system can do.

`LOST` is deliberately distinct from `FAILED`. When a worker stops
reporting we know we lost contact; we do **not** know training stopped,
and a remote process may still be running. Saying `FAILED` would assert
something nobody checked.

**Experiment status** is derived from its runs and never rewrites them.
A failed run does not fail the experiment — a hypothesis is not disproved
by a crashed process, and marking it so would push people toward editing
history to make an experiment look clean.

Terminal states are terminal. **A retry is a new run** citing
`parent_run_id`; nothing is overwritten.

---

## 3. Dataset and curation gates

Before `QUEUED`, five gates run in order. All are hard.

1. **Dataset lock** — recomputes Phase 23's *canonical* manifest digest
   and source identity digest. A reformat passes; a changed record does
   not.
2. **Curation lock** — recomputes the curated-manifest digest **and**
   checks that the curation cites *this* dataset lock. That linkage
   catches the mistake that is otherwise invisible: a curation computed
   against last week's manifest, internally consistent, paired with
   today's dataset.
3. **Rights**
4. **Evaluation leakage**
5. **Self-generated data**

If either lock fails, the remaining three are reported as **not
evaluated** rather than run — answering a rights question from a file
nobody verified would be answering the right question from the wrong
data.

Failure sets the run to `FAILED` with the gate's own code, so the reason
survives in the registry rather than only in a terminal.

---

## 4. Rights gate

Barred outright: `commercial_training_allowed` FALSE or UNKNOWN,
`rights_status` RESTRICTED, any `hard_blocks`, `training_eligible` false.

**There is no `--ignore-rights`, no `--force`, and no override parameter
anywhere in the module.** The function's signature takes the records and
nothing else, and a test asserts that.

The gate re-derives from each record's own `provenance` block rather than
trusting `training_eligible`. Phase 23 has an `include_rights_unknown`
build option and Phase 24 an export policy — both legitimate for
inventory and analysis, either capable of marking an unknown-rights track
eligible. Production training must not be able to reach that path by
accident, so provenance itself is the authority here.

---

## 5. Evaluation leakage gate

Training may not contain P20 benchmark material, frozen evaluation
material, `evaluation_only` tracks, or anything Phase 23 placed in
VALIDATION or TEST.

Checked on **track id and content digest — never filenames**. A
benchmark track copied under a different name is the same audio and the
same leak; a filename check would miss it, and the benchmark would
quietly stop measuring generalisation while the numbers looked fine.

`evaluation_only` does not exist as a Phase 23 field. It is supplied by
configuration — a file of track ids — and never inferred.

---

## 6. Self-generated data policy

`ALLOW_SELF_GENERATED` defaults to **false**. Training a model on its own
generations teaches it its own artifacts, and the Phase 5 human verdict
rated that output 2/10.

A record whose provenance cannot distinguish origin **blocks** rather
than being assumed human — unknown provenance is precisely how
self-generated audio gets in unnoticed. The allow-flag admits *known*
self-model output; it does not clear indeterminate provenance.

Third-party `AI_GENERATED` audio with cleared rights is legitimate and is
not blocked; it is counted in the synthetic share.

---

## 7. Training configuration

Every field mirrors a flag in the installed parser. See
`docs/TRAINING_CAPABILITY_AUDIT.md`.

**Absent on purpose**, because the trainer has no such flag:
`max_steps`, `validation_interval`, `checkpoint_interval` (steps). The
checkpoint field is named `checkpoint_every_epochs` because
`--save-every` counts epochs.

**`FULL` fine-tune is not offered.** The installed trainer has no entry
point for it, and a `training_strategy: FULL` that silently trained an
adapter would be a lie in the run record.

Unknown keys are a validation failure, not a shrug. Configs hash
canonically: same config, same digest.

**Presets** — `SMOKE`, `LORA_SMALL`, `LORA_STANDARD`,
`LORA_HIGH_QUALITY` — are named for intent. Upstream ships `vram_8gb`
and `vram_24gb_plus`; those names promise memory behaviour nothing here
has measured, so they are not reused and **no VRAM figure appears in any
LUBER preset**.

---

## 8. Training plan

Compiled once, hashed, frozen. Any change to model, data, config or code
means a **new run**.

The plan contains **no secrets** — only `secret_refs`, names a backend
resolves out of band — and **no machine-specific paths as identity**. It
names dataset and curation identities and digests; paths are placeholders
(`${LUBER_DATASET_DIR}`) the backend substitutes on the worker. That is
what lets a plan compiled on a Mac execute on a rented Linux host.

The hash excludes `compiled_at` and `plan_id`: two plans that would train
identically must hash identically.

Since Phase 32 the plan also records the **compute device** it was
compiled for (`requirements.execution_device`: `CUDA`, `MPS` or `CPU`),
and the schema version moved to `luber-training-plan/2`. It is inside
the hash on purpose — MPS and CUDA do not train identically, so a plan
naming one is not the same plan as one naming the other. Left `None`, a
plan behaves exactly as it did before the field existed. What stays
*out* of the hash is every hardware measurement: free disk, torch patch
version and GPU utilisation describe a machine at a moment, not the
training being requested. See `docs/HARDWARE_EXECUTION_COMPATIBILITY.md`.

---

## 9. Execution backends

```python
validate_environment() prepare_run() start() status()
cancel() collect_metrics() collect_checkpoints() cleanup()
```

**`LocalDryRunBackend`** walks the real lifecycle without training. Its
metrics are all marked `SIMULATED`, and it emits **no `train_loss`** —
there is no honest number to put there, and a simulated loss would
eventually be plotted beside a real one. It produces **no checkpoint at
all**. Where a test needs an artifact it registers one of kind `MOCK`,
which is a distinct kind rather than a flag, so no query for a real
checkpoint can return it. A `MOCK` artifact can never become an
evaluation candidate and can never be resumed from.

A dry run does **not** launder a capability mismatch: if the plan
requires CUDA, the same capability check runs as for a real backend.
Otherwise "it passed on dry-run" would become evidence that a
development Mac could take a GPU job.

**`RemoteGpuBackend`** is a contract with no implementation. Phase 25
connects to nothing; every execution method raises. Only
`validate_environment` works, because capability matching is
provider-independent arithmetic. It is deliberately provider-neutral —
RunPod, Vast.ai, Lambda, AWS and a bare SSH box differ in provisioning
and billing, not in the seven verbs orchestration needs.

---

## 10. Workers and capability matching

Facts come from a probe, never from assumption. `None` means unmeasured;
it never means zero, and it is never defaulted.

**An unmeasured capability does not pass.** A worker that has never
reported CUDA cannot satisfy a CUDA requirement by virtue of nobody
having checked. The local Mac registers as `DEVELOPMENT_ONLY` and is
refused for any CUDA plan.

`minimum_vram_mb` is `UNKNOWN_REQUIREMENT` — no VRAM figure has been
measured for any LUBER configuration on NVIDIA hardware. It is reported
as unknown, not silently passed.

`max_concurrent_runs` defaults to 1. Multiple GPUs on a host means one
bigger run, not several models.

### Probe

```bash
python -m luber_training worker probe --output worker_capabilities.json
```

Captures GPU model, VRAM, driver, CUDA, torch, BF16 support, CPU, RAM,
disk and Python. `nvidia-smi` is **only invoked when it is on PATH** —
running it on a Mac produces a confusing error and no information.
torch is the authority on `cuda_available`, since a driver can exist
while torch was built without CUDA.

---

## 11. Environment and code-version locks

`environment_lock.json` per run: Python, platform, torch, CUDA, PEFT,
transformers, ACE-Step commit, LUBER commit and dirty flag, dependency
lock digest. Versions and commits only — no environment variable values,
so nothing here can leak a credential.

**Production training requires a clean repository.** "Commit abc123 plus
whatever was in the editor" is not a revision anyone can reproduce, so
this is a gate rather than a warning. `--allow-dirty` exists for
dry-run experimentation and is recorded in the preflight.

---

## 12. Trainer adapter

The canonical manifest is **not** reshaped to suit ACE-Step. The adapter
reads curated records and emits the `dataset.json` the trainer's
preprocessing expects, leaving both sides free to change.

The trainer's loader fills in defaults for missing fields — a caption
from the filename, and `"[Instrumental]"` lyrics. Those defaults are why
every field is written explicitly: a vocal track silently labelled
instrumental is a training-data error invisible in a loss curve. A track
whose vocal class is UNCERTAIN gets an empty lyrics string, not the
instrumental marker.

Captions invent nothing. Tempo and key are forwarded only when Phase 23
recorded confidence; a track with no known metadata gets
`"unlabelled music"`.

### Command compilation

Plan → **argv list**, never a shell string, and never executed in this
phase. Every emitted flag is checked against the installed parser by a
test that reads `args.py` at test time.

`subprocess` without `shell=True` makes shell injection structural
rather than something to remember: an experiment name of
`x"; rm -rf ~` is one inert argv element. Control characters are
refused outright, since they corrupt logs and any future shell
transport. `display()` quotes with `shlex.quote` for **reading only** —
nothing executes that string.

---

## 13. Metrics

Append-only JSONL. No Prometheus, MLflow or W&B: a run writes a few
thousand numbers and an operator reads them afterwards. Appending line
by line also means a run killed halfway keeps everything it had emitted.

Every event carries a `source` — `TRAINER`, `WORKER_TELEMETRY`,
`ORCHESTRATOR`, `SIMULATED` — so a real number and a simulated one are
never indistinguishable in storage.

Resource metrics (GPU utilisation, VRAM, power, CPU, RAM, disk, step
time) come from worker telemetry. No GPU means null, not zero.

---

## 14. Checkpoint lifecycle

Statuses: `WRITING` → `READY` / `CORRUPT` / `REJECTED` / `ARCHIVED`.
Kinds: `ADAPTER` (what the trainer produces), `FULL_MODEL` (nothing
produces one today), `MOCK`.

**Atomic finalisation.** Write to a staging path, validate the required
adapter files, hash the tree, then `os.replace` into place. An
interrupted write leaves a staging directory and **no registry entry** —
never a `READY` checkpoint. A destination is never overwritten, and a
crash leaves diagnostic evidence rather than a directory that might be a
model.

`finalize_checkpoint_record` is the only path that sets `READY`.

**Retention** produces a *plan*, and executing it requires
`confirm=True`. Phase 25 deletes no model files. Every kept checkpoint
records which rule kept it.

---

## 15. Resume and lineage

`resume_from_checkpoint` validates that the checkpoint is `READY`, is
real weights rather than a `MOCK`, and came from a run on the **same base
model**. Resuming across model families produces an adapter shaped for
weights it was never trained against.

Lineage is `parent_run_id` + `resume_from_checkpoint_id`. Failed runs are
never overwritten.

---

## 16. Cancellation and idempotency

Cancellation requests graceful termination and **preserves metrics, logs
and completed checkpoints**. A cancelled run is part of an experiment's
history, not an embarrassment to erase.

Starting the same run twice starts one trainer. The guard is the state
machine rather than a flag: `QUEUED` is the only state a start may leave,
so the second call finds the run already past it and returns.

---

## 17. Failure taxonomy

`DATASET_LOCK_INVALID`, `CURATION_LOCK_INVALID`, `RIGHTS_GATE_FAILED`,
`EVALUATION_LEAKAGE`, `SELF_GENERATED_BLOCKED`, `ENVIRONMENT_INVALID`,
`INSUFFICIENT_HARDWARE`, `CODE_VERSION_DIRTY`, `WORKER_LOST`,
`TRAINER_CRASH`, `OOM`, `CHECKPOINT_WRITE_FAILED`,
`CANCELLED_BY_OPERATOR`, `UNKNOWN`.

Closed on purpose. Parsing arbitrary exception text into ever more
specific codes produces fiction; the raw sanitised diagnostic is stored
separately and the code stays honest, including `UNKNOWN`. Diagnostics
matching a credential pattern are redacted before storage.

---

## 18. Evaluation candidates and promotion

A `COMPLETED` run produces checkpoints. **Nothing is promoted
automatically.** A checkpoint may become an `EvaluationCandidate`
(`PENDING_EVALUATION`), which is a request for evidence and carries no
quality claim.

Model stages exist — `BASELINE`, `EXPERIMENT`, `CANDIDATE`, `ACCEPTED`,
`PRODUCTION`, `REJECTED`, `ARCHIVED` — and **nothing in Phase 25 moves
anything to `PRODUCTION`**. Current ACE-Step remains the production
baseline. Training completion is not model success.

---

## 19. Registry

Filesystem JSON, versioned, one directory per entity kind. Not a service:
this is operator infrastructure for one project.

- **Atomic writes** — temp file, flush, fsync, `os.replace`
- **Exclusive `flock`**, reentrant within a process, so two CLI processes
  cannot assign the same identity or interleave a read-modify-write
- **Append-only audit log** — `BASELINE_REGISTERED`,
  `EXPERIMENT_CREATED`, `RUN_CREATED`, `RUN_VALIDATED`, `RUN_BLOCKED`,
  `RUN_QUEUED`, `RUN_STARTED`, `RUN_COMPLETED`, `RUN_FAILED`,
  `RUN_CANCELLED`, `RUN_LOST`, `CHECKPOINT_REGISTERED`,
  `CHECKPOINT_FINALIZED`, `CANDIDATE_CREATED`, `WORKER_REGISTERED`. No
  secrets; a torn final line is skipped rather than breaking the read.

Ids are `mdl_`, `exp_`, `run_`, `ckpt_`, `wrk_`, `cand_`, `plan_` plus 64
bits of entropy. Identity is never derived from a mutable name, and an id
containing a path separator is refused at the registry boundary.

**Immutability**: once a run has started, `base_model_id`, `dataset_ref`,
`config`, `training_plan_sha256`, `experiment_id` and
`resume_from_checkpoint_id` cannot change. A field still unset may be
filled in once — the plan hash is assigned after `QUEUED` — but a
recorded value may never be edited.

---

## 20. Artifact layout

```
training_runs/<experiment_id>/<run_id>/
    plan.json
    environment_lock.json
    run.json
    metrics.jsonl
    checkpoints/
    logs/
    artifacts/
```

Gitignored. These are operator state, often large, and sometimes
describe a private library.

---

## 21. Reproducibility bundle

`run bundle` returns references and digests for: model baseline, dataset
lock, curation lock, training config, training plan, repository commit,
environment lock, worker capabilities, checkpoints, lineage and the full
audit trail for that run. References rather than copies — it is meant to
be readable in a year and to point at the artifacts that explain what
happened.

---

## 22. Cost accounting

Schema only. `provider`, `instance_type`, `hourly_rate`, `currency`,
`gpu_seconds`, `wall_seconds`, `estimated_cost`, `actual_cost`. Phase 25
fetches no live prices and hardcodes none.

---

## 23. Security boundary

Training orchestration is **operator-only**. There is no HTTP surface, no
user account, no role. An ordinary LUBER account cannot launch training,
cancel a run, reach a training dataset or download a checkpoint, because
none of those paths exist outside the CLI.

A weak role check bolted onto the consumer API would be worse than the
absence of one: it would imply a boundary a bug could cross. When a
Training Console is eventually built, its endpoints stay separate from
the consumer generation API and are not exposed publicly.

**Secrets**: no entity, plan, checkpoint record, log or audit event ever
stores a secret value. Only reference names — `ssh_key_ref`,
`credential_ref`, `secret_refs`. `RemoteGpuBackend`'s constructor takes
references; if it ever takes a key, the boundary has been broken.

---

## 24. CLI

```bash
python -m luber_training --registry ./training-registry <command>
```

`baseline register|list` · `experiment create|list` ·
`worker probe|register|list` ·
`run create|validate|start|status|cancel|list|bundle|command` ·
`checkpoint list` · `candidate create` · `presets` · `verify`

`run start --backend dry-run` exercises the dataset gate, curation gate,
rights gate, leakage gate, plan compilation, the run lifecycle, metrics
and completion. No training.

`run command` shows the compiled trainer invocation for inspection and
never executes it.
