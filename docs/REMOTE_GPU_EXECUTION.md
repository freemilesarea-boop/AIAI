# Remote GPU execution

How a training run reaches a machine LUBER does not own, and how what it
produces comes back trustworthy.

Phase 25 decides *what* may be trained. This is the bridge to *where*.
It sits between `luber_training`'s orchestration and a rented NVIDIA
box, and it is deliberately provider-neutral: no module here names
RunPod, Vast, Lambda or AWS, and choosing one is a configuration
decision rather than a code change.

---

> **Phase 32.** Which *location and device* a workload goes to is
> decided before this document applies — see
> `docs/HARDWARE_EXECUTION_COMPATIBILITY.md`. Placement selects
> `REMOTE + CUDA`; everything below is how that gets executed, and
> none of it changed. For preparing the host itself, see
> `docs/NVIDIA_TRAINING_WORKER_RUNBOOK.md`.

## 1. Two roles, and a boundary of authority

The split is not about code layout. It is about who is allowed to decide
things.

**The control plane** runs on the operator's machine. It owns every
judgement: which experiment is worth running, which dataset is
permitted, whether rights are valid, whether a checkpoint deserves
evaluation. It creates runs, compiles plans, runs gates, stages
artifacts, dispatches, tracks status, registers checkpoints.

**The worker** runs on the GPU host. It owns no judgement at all. It
reports what it is, receives approved artifacts, verifies they are what
was sent, runs the exact command it was given, and reports truthfully.

The worker cannot decide that a dataset is acceptable, because it is
never asked. That is what makes a misconfigured or compromised worker
able to waste time and unable to make a decision nobody sanctioned.

```
CONTROL PLANE                              REMOTE WORKER
─────────────                              ─────────────
experiment / run / plan
gates (rights, leakage, locks)
stage  ──────── artifacts + digests ──────▶ receive, record manifest
                                            preflight: verify everything
dispatch ──────── launch ─────────────────▶ run the given argv, detached
        ◀─────── heartbeat, status ───────  report state
        ◀─────── logs (cursor) ───────────  incremental
        ◀─────── metrics (dedup) ─────────  incremental
                                            write checkpoints
        ◀─────── result manifest ─────────  digests of everything
collect ◀─────── checkpoint bytes ────────  verified, atomic
register checkpoint (READY)
        └──────▶ Phase 26 evaluation
```

---

## 2. Protocol

`luber-remote/1`. Every interaction carries it, and a version this build
does not recognise is refused rather than attempted. A worker running
older code that silently ignored a field it did not understand would
produce a run whose configuration nobody could reconstruct.

Every worker reply is one JSON envelope on stdout:

```json
{"protocol_version": "luber-remote/1", "ok": true, "command": "status",
 "worker_id": "wrk_…", "timestamp": "…", "payload": {…}, "error": null}
```

Diagnostics go to stderr, so stdout is always parseable.

### Why no daemon

The worker is a set of short-lived commands invoked over SSH, not a
service. No port to open, no service manager to configure, no process to
supervise, nothing listening on a rented box. State lives in files under
the run root, so any invocation reconstructs what it needs by reading
them, and the trainer — launched detached, in its own session — outlives
the SSH connection that started it.

An HTTP daemon would buy live streaming and cost a listening port,
authentication, a supervisor and an upgrade path. Everything Phase 27
needs (heartbeat, status, launch, cancel, artifact inspection, log and
metric polling) works over command invocations, so the daemon is not
built. If push-based streaming becomes necessary, it can be added
without changing the boundary.

---

## 3. Worker identity

A hostname is not an identity. Providers call every box `gpu-01`, DHCP
reassigns names, and a reprovisioned instance keeps the name and loses
everything else.

So a worker mints a `worker_id` once, writes it into its own root, and
carries a **host fingerprint** — a digest over stable machine facts
including GPU UUIDs, which are burned into the hardware. Same id, same
fingerprint means the same machine rebooted. Same id, different
fingerprint means it was rebuilt, and that is reported rather than
silently accepted.

Separately there is a **capability signature**: a digest over everything
that decides whether a plan can run here — GPU model, count, VRAM,
driver, CUDA, torch, Python. It deliberately excludes utilisation, free
memory, temperature and the probe timestamp, because an identity that
changed every probe could never answer the question it exists for.

---

## 4. Capability probe

`python -m luber_training.remote probe` measures the machine and invents
nothing. Every field is a measurement or `null`, and `null` means nobody
could look — never zero.

GPU facts come from `nvidia-smi --query-gpu … --format=csv,noheader`,
never from parsing the human-readable table. The parser survives the
tool being absent, exiting non-zero, returning nothing, returning
`[N/A]` in any field, an old driver returning fewer columns than asked,
and multiple cards.

**torch is the authority on CUDA**, not the presence of a driver. A
driver can be installed while torch was built without CUDA, and training
runs through torch.

Classification:

| Class | Requires |
|---|---|
| `CUDA_TRAINING` | torch demonstrates CUDA **and** a GPU is visible |
| `CUDA_EVALUATION` | reserved; a host that can infer but not train |
| `DEVELOPMENT_ONLY` | everything else, including every Mac |
| `UNAVAILABLE` | the machine could not be probed |

There is no flag anywhere that asserts a classification. A machine
becomes `CUDA_TRAINING` by demonstrating CUDA on itself.

---

## 5. Heartbeat and liveness

The worker writes a heartbeat carrying its state, active run, health,
free disk and GPU telemetry where available. The control plane derives
liveness from timestamps:

| State | Default | Meaning |
|---|---|---|
| `ONLINE` | < 300 s | answering |
| `STALE` | 300–900 s | late; networks blink |
| `OFFLINE` | ≥ 900 s | presumed unreachable |
| `UNKNOWN` | never seen | nobody has heard from it |

Deliberately patient, and configurable. **An OFFLINE worker does not
make its run FAILED — it makes it LOST.** We know we cannot see the
trainer; we do not know it stopped. Failing a run over a missed poll
wastes a rented GPU for nothing, and dispatching it elsewhere puts two
trainers on one checkpoint directory.

---

## 6. Run lease

One run, one worker, one trainer. A lease binds `run_id` to `worker_id`
and to a **plan hash**, and the plan hash is the load-bearing part:

- same run, same worker, same plan → returns the existing execution;
- same run, *different* worker → refused, or two machines train it;
- same run, *different* plan → an integrity violation, because a run id
  must mean exactly one training configuration.

That third case is what makes redispatch after an edited config an error
rather than a silent second attempt.

---

## 7. Artifacts

A `RemoteArtifactManifest` lists every file the run needs remotely, each
with a role, a relative target path and a SHA-256. Content addressing
buys four things at once: transfer skips what is already there, an
interrupted transfer resumes by comparing digests, corruption is
detectable rather than merely unlikely, and the whole set has a
deterministic identity.

No absolute paths appear in the recorded manifest. A manifest built on a
Mac is replayable on Linux because it never said where anything was.

### What is sent

Only the **selected, authorised, training-split** audio from the
approved curated manifest, plus the plan, the environment lock, the
locks and the trainer dataset. Never the source library.

### Staging

`remote_staging/<run_id>/` is derived entirely from immutable inputs, so
staging the same run twice produces the same tree and the same digest.
`staging_manifest_sha256` is computed over the manifest digest and the
plan digest rather than over the directory, so two builds on two
machines at two times hash identically.

---

## 8. The gates, run again before transfer

This is the last point at which forbidden data can be stopped. After
staging, bytes leave for a machine somebody else owns and there is no
recalling them.

So `build_staging` re-runs Phase 25's **rights gate** and **leakage
gate** on the records that are about to become the manifest —
immediately before it, not at run creation. Time passes between
validation and dispatch: a curated manifest can be regenerated, a lock
replaced, a run validated on Monday dispatched on Friday. Re-reading the
file that is about to be transferred is the only check that describes
what will actually be sent.

Both are hard failures that abort before the first file is opened, and
the staging directory is not created until they pass — so a blocked
dispatch leaves nothing behind that a later transfer could pick up.

**There is no override flag in this module.** Adding one would defeat
every gate upstream of it.

A third check runs at the same time: every audio file is hashed and
compared to the digest the curated manifest recorded. A file that
changed since curation is not the dataset that was approved.

---

## 9. Transport

`ArtifactTransport` is the interface: probe, exists, stat, upload,
download, list_files, remove_temp. Three rules every implementation
honours.

**Nothing is finished until it is verified.** Files are written to a
`.luber-partial` name, hashed where they landed, and only then moved
into place. A partially transferred file never occupies the path a
reader would look at — and the reader is a trainer that would happily
start on a truncated dataset.

**The move is atomic.** `os.replace` locally, `mv` remotely. Not
copy-then-delete.

**Resume is by content.** Before sending, the transport asks what the
destination already has and compares digests. Matching files are
skipped; differing files are re-sent whole.

> Resume is **file-granular**. `scp` cannot resume a byte offset safely,
> and this build does not claim to. `supports_byte_range_resume` is
> `False` on every transport here, stated rather than implied, so nobody
> relies on it for a 40 GB dataset.

| Transport | Use |
|---|---|
| `LocalArtifactTransport` | a second machine simulated by a second directory. Real writes, real hashing, real atomic renames, real partial files. Used by every lifecycle test. |
| `SshArtifactTransport` | `ssh` + `scp`, provider-neutral, verified on the far side. |

### Remote content cache

Immutable artifacts — dataset audio, code bundles — are stored on the
worker under their own digest, so a second experiment on the same corpus
transfers nothing. **A cache hit is verified before it is used**: the
file is hashed, and one that fails is deleted rather than served. The
filename is a claim; the digest is the fact.

The cache assumes a single trusted operator domain — one person's runs
on one rented box. It is not a shared cache and must not become one
without a rights review: content addressing alone would happily serve
one tenant's audio to another.

---

## 10. SSH security

- `StrictHostKeyChecking=yes`. Not `accept-new`, which trusts on first
  use — an operator dispatching a job is in no position to notice that
  the host they enrolled last week has a different key today.
- `BatchMode=yes`, `PasswordAuthentication=no`,
  `KbdInteractiveAuthentication=no`. A misconfiguration fails rather
  than silently falling back to a prompt nobody is there to answer.
- **No password ever appears on a command line.** Authentication is a
  key file, used by path, resolved by reference at the moment of use.
- First-contact enrolment is a deliberate one-time operator step,
  documented in [`FIRST_REMOTE_GPU_CONNECTION.md`](FIRST_REMOTE_GPU_CONNECTION.md).

### Injection

`ssh` concatenates its remote arguments and hands them to a login shell.
So every remote argument is quoted with `shlex.quote`, **and** every
identifier that reaches it is validated against a narrow pattern first.
Both, because they fail differently: quoting stops a path becoming
syntax, validation stops a perfectly-quoted path being
`../../../etc/passwd`. A run id containing `; rm -rf ~` is refused
outright, not escaped.

Locally, every subprocess is an argv list with `shell=False`.

### Secrets

Configuration holds `ssh_key_ref`, `known_hosts_ref`,
`provider_token_ref` — **names, never values**. A `SecretResolver`
turns a name into a value at the moment of use:

- `EnvironmentSecretResolver` reads `LUBER_SECRET_<NAME>`. The prefix is
  deliberate: without it, `resolve("PATH")` would succeed.
- `FileSecretResolver` reads a directory, **checks the file's
  permissions** (a key readable by other accounts is refused, matching
  ssh's own behaviour), and refuses outright to be pointed at a
  directory inside the repository.
- `NullSecretResolver` is the default and resolves nothing, so a
  component that unexpectedly asks for a credential fails loudly.

Private keys are used by path and never copied into staging. Resolved
values are registered and scrubbed from any text on its way into a log,
an error or a registry record; fields whose *name* looks like a secret
are redacted whatever they hold.

---

## 11. Paths

A plan names **logical roots** — `code_root`, `data_root`, `run_root`,
`checkpoint_root`, `cache_root` — and each worker's registration says
where those live on that machine. There is no universal filesystem
layout, and a plan that claimed to know one would not be portable.

Every relative path crossing the boundary is validated: absolute paths,
parent traversal, backslashes, drive letters, null bytes, reserved
device names, trailing spaces and dots, and over-long components are all
**refused rather than sanitised**. Sanitising invites the question of
whether the sanitiser is complete.

The string check is not the only one. `resolve_within` also resolves the
real path and proves containment, because a well-formed relative path
can still land on a symlink pointing elsewhere.

### Run layout

```
runs/<run_id>/
    plan.json  artifact_manifest.json  environment_lock.json
    lease.json  status.json  remote_preflight.json  remote_result.json
    trainer/  dataset/  logs/  metrics/  checkpoints/  output/  temp/
```

Fixed, so a reconnecting control plane and a restarted worker look in
the same places without negotiating. All writes are constrained to the
run root.

---

## 12. Preflight

A misconfiguration caught here costs seconds; the same one caught an
hour into a rented GPU costs the hour. So preflight checks everything
knowable without training: protocol, plan hash, manifest hash, every
required artifact rehashed **on the worker**, code revision, Python,
torch, ACE-Step commit, PEFT, CUDA, GPU count, VRAM, precision, disk,
checkpoint-directory writability, dataset presence, and whether the
trainer entry point exists.

Three statuses, and the third is the important one:

- `PASS` — established.
- `FAIL` — definitively wrong.
- `UNKNOWN` — **not a soft pass.** A required capability nobody could
  measure is treated as unsatisfied. Treating "nobody measured VRAM" as
  a tick is how a run lands on hardware that cannot hold it.

Overall: `PASS`, `FAIL` (something is wrong), or `BLOCKED` (something
could not be established). Both stop the run; they differ in what the
operator should do next.

Dependency comparison is graded rather than absolute — `REQUIRED`
(torch, ACE-Step commit, code revision), `COMPATIBLE` (Python, PEFT),
`INFORMATIONAL` (everything else). Demanding byte-identical environments
would block every real deployment; demanding nothing would let a run
train against a torch it was never tested with.

**No trainer starts unless preflight passes.**

### Code revision

The worker's LUBER commit must match the dispatch's. Rsyncing a dirty
working tree and calling it reproducible is exactly what this refuses;
the alternative is an immutable source bundle identified by digest.
`--allow-code-mismatch` exists for infrastructure smoke tests and
downgrades the check to informational, visibly, in the report.

---

## 13. Execution

The trainer command is compiled by **Phase 25's compiler**, not rebuilt
here. A second implementation would drift, and then the command an
operator reviewed on the control plane would not be the command that
ran. The worker's only contribution is substituting its own directories
for the plan's logical placeholders.

- **Own process group and session.** Cancellation signals the group, so
  dataloader workers stop too; the new session detaches the trainer from
  the SSH channel, so closing the connection does not kill a job three
  hours in.
- **The group is proven before it is used.** `setsid` runs in the child
  after the fork, so reading the group immediately can still see the
  *launcher's* group — recording that would be dangerous, because the
  launcher exits and its group id may be reused. The group is recorded
  only once observably the child's own, re-verified before every signal,
  and where the two disagree only the single pid is signalled.
- **Logs go to files, never a pipe.** The launching process exits
  immediately and cannot drain one. stdout and stderr are separate,
  appended, so a multi-hour log never occupies memory and a restarted
  worker keeps what came before.
- **No training timeout.** Connection, transfer, preflight, launch
  confirmation and heartbeat each have their own; training itself has
  none, because a multi-hour job killed by a global deadline is the most
  expensive possible bug.

### Launch idempotency

`start` returns the existing state if the recorded pid is alive. That is
what makes the ambiguous-launch case safe: a control plane that never
received the acknowledgement can call again and find the first launch's
trainer.

### Exit classification

Conservative on purpose.

| Evidence | Code |
|---|---|
| `CUDA out of memory`, `torch.cuda.OutOfMemoryError`, `CUBLAS_STATUS_ALLOC_FAILED` | `OOM` |
| `No space left on device`, `ENOSPC`, `errno 28` | `CHECKPOINT_WRITE_FAILED` |
| non-zero exit, no recognised signature | `TRAINER_CRASH` |
| killed by a signal, no CUDA OOM message | `TRAINER_CRASH`, explicitly **not** OOM |
| process gone, exit status never collected | state FAILED, code **unset**, detail says UNKNOWN |

A SIGKILL is what the kernel OOM killer *and* `kill -9` produce, and a
run mislabelled OOM sends the next experiment chasing a memory problem
that never existed.

The last row matters: a worker invocation that did not launch the
trainer cannot know how it ended. It says so rather than reporting
COMPLETED, which would be the single most dangerous thing it could do.

---

## 14. Status mapping

| Worker state | Run status |
|---|---|
| `IDLE` / `RECEIVING` / `PREFLIGHT` / `READY` | `QUEUED` |
| `STARTING` | `STARTING` |
| `RUNNING` | `RUNNING` |
| `CANCELLING` | `RUNNING` — requested is not done |
| `CANCELLED` | `CANCELLED` |
| `COMPLETED` | `COMPLETED` |
| `FAILED` | `FAILED` |
| `LOST` (control-plane derived) | `LOST` |
| anything unrecognised | `LOST` |

Two enums rather than one, because the worker's view of a process and
the control plane's record of a run can legitimately disagree — most
importantly when contact is lost. Phase 25's `FailureCode` taxonomy is
reused unchanged; a second one would give the same problem two names.

---

## 15. Logs and metrics

Both poll from a cursor. Re-downloading a 5 GB log every thirty seconds
would cost more bandwidth than the training data did.

**Logs**: a byte offset, returned with every response. An offset past
the end means the file was rotated; the read restarts and says so.

**Metrics**: a line cursor *and* a per-event identity —
`(run_id, step, metric_name, source)`. The cursor makes the common case
cheap; the identity makes correctness independent of it, so a file
replayed from the start — which is what a resumed trainer does — cannot
duplicate anything. The timestamp is deliberately not part of identity:
two reports of step 140's loss are one measurement.

Phase 25's `MetricEvent` schema is reused. Nothing parses a metric out
of console text when the trainer writes a structured file, and a metric
that is unavailable is absent rather than fabricated.

---

## 16. Checkpoints

**A remote checkpoint is not a ready checkpoint.** It has been written on
a machine the control plane cannot see, and the bytes have not crossed
the network. So it gets its own state:

```
WRITING → (validate, hash) → READY_REMOTE → (transfer, re-hash) → Phase 25 READY
```

Discovery is by contract, not by glob. Trainers write optimiser state,
temp files and sample audio into their checkpoint directories;
registering every directory found there would eventually register
something that is not a model. A candidate must look like what
`save_adapter_flat` writes, and one that does not is reported
`REJECTED` **with its reasons** rather than skipped silently.

### Collection

Files land in a `.collecting` directory; the destination comes into
existence only via a rename, after the whole-tree digest matches the one
the worker reported. The digest algorithm is byte-for-byte the same on
both sides, so the comparison is between two independent measurements.

On mismatch: the collection fails, the local staging directory is kept
as evidence and as the basis for a resume, **and the remote copy is left
untouched** — it is the known-good one, and deleting it because the
transfer went wrong would destroy the only intact artifact.

`ArtifactLocation` distinguishes `LOCAL`, `REMOTE_ONLY` and
`OBJECT_STORE`, so physical location is separate from identity. Phase 27
implements `LOCAL` collection; the vocabulary is complete so a later
phase adds a backend rather than a concept.

### Retention

Nothing remote is deleted automatically. A rented instance can be
terminated at any moment and take its disk with it, so the remote copy
is a second copy for exactly as long as the machine exists — and that
window closes without warning. `plan_remote_retention` recommends;
deletion is an operator decision.

**Cleanup never removes evidence.** Temporary files and partial
transfers go; logs, metrics, checkpoints and the result manifest stay,
whatever the outcome. A failed run's logs are the only thing that
explains it.

---

## 17. Failure and recovery

The rule throughout: **never resolve ambiguity by launching something.**

`reconcile` asks the worker what is actually happening. It changes
nothing, is safe to call repeatedly, and returns one of:

| Outcome | Meaning | Safe to launch |
|---|---|---|
| `NOT_PRESENT` | the worker has no record of this run | **yes** — the only one |
| `RUNNING_RECOVERED` | a trainer is alive | no |
| `COMPLETED_RECOVERED` / `FAILED_RECOVERED` / `CANCELLED_RECOVERED` | it finished | no |
| `UNKNOWN` | the worker answered and cannot say | no |
| `UNREACHABLE` | the worker did not answer | no |

`dispatch` reconciles unconditionally before it launches. That one
status call is what stands between a lost acknowledgement and two
trainers on one run.

### Retry policy

| Safe to repeat | Not safe |
|---|---|
| `identity`, `probe`, `heartbeat`, `status`, `logs`, `metrics`, `checkpoints`, `result`, `preflight`, `prepare`, `receive` | `start`, `cancel` |

`prepare` is idempotent by construction — the lease check makes a repeat
return the existing state. `start` is not: reconcile first, always.

### Cancellation

SIGTERM to the process group → configurable grace period → SIGKILL only
if it is still there. The grace period is generous because a trainer
mid-checkpoint should be allowed to finish writing.

If the trainer completed successfully before the signal landed and that
was durably recorded, **the run stays COMPLETED**. Overwriting a real
success with CANCELLED would discard a checkpoint that exists and was
paid for.

### Control-plane restart

Everything needed is on the worker and in the registry. A fresh process
reconstructs the picture — lease, remote state, live pid — and restarts
nothing.

### Worker restart

Where the worker process restarts but the trainer survives, liveness is
recovered by pid probe. Where the process state cannot be recovered, the
worker reports UNKNOWN. It never reports COMPLETED on that basis.

---

## 18. Cost and telemetry

Phase 25's cost fields are reused. The result manifest records wall time
derived from recorded timestamps. **`gpu_seconds` is null**: nothing here
samples the device continuously, and a figure derived from wall time
would be wall time wearing a different name.

GPU telemetry — utilisation, memory, temperature, power — is sampled on
demand from `nvidia-smi`, at a configurable interval, and kept entirely
separate from training metrics. No GPU means no readings, not readings
of zero. No provider pricing is ever fetched.

---

## 19. Authorization boundary

Operator-only, local, and there is no second path. No HTTP surface, no
role, no API route. SSH configuration, worker host details, checkpoint
paths and dataset transfer controls exist only in
`python -m luber_training remote` and `python -m luber_training.remote`.
No consumer account can dispatch a job, because no code accepts a
request to.

---

## 20. Commands

**Control plane** — `python -m luber_training remote …`

```
worker register-remote     probe a machine and record what it is
worker verify              confirm it is still that machine
worker heartbeat           liveness, with the policy that judged it

run stage                  assemble the transfer set (gates run here)
run verify-staging         recheck the staged tree before uploading
run dispatch               transfer, preflight, launch — reconciling first
run remote-status          worker state and reconciliation
run remote-logs            incremental, from a cursor
run remote-metrics         incremental, deduplicated into the run's metrics
run remote-cancel          graceful stop
run reconcile              establish what actually happened; idempotent
run collect                bring checkpoints back, verify, register
run verify-remote          the worker still holds what this run believes
run cleanup                remove scratch, never evidence
```

**Worker** — `python -m luber_training.remote …`

```
init  identity  probe  heartbeat  prepare  receive  preflight
start  status  logs  metrics  cancel  checkpoints  result  cleanup
```

---

## 21. Limitations

Recorded so nobody rediscovers them.

- **No real NVIDIA host has been used.** Every capability figure this
  project holds is `null`. VRAM requirements, checkpoint sizes, transfer
  throughput and GPU-hours are all unmeasured, and preflight reports
  them as UNKNOWN rather than guessing.
- **SSH transport is untested against a real host.** Its argv
  construction, quoting, host-key policy and secret handling are
  covered by unit tests; nothing has connected to a remote machine. Real
  SSH acceptance is deferred to the first GPU day.
- **Resume is file-granular.** A 40 GB file interrupted at 39 GB
  restarts from zero.
- **Byte-level integrity only.** A checkpoint that hashes correctly is
  the checkpoint the worker wrote. Nothing here loads it to check it is
  a usable model — that is Phase 26's job, and it needs a GPU.
- **The content cache is single-tenant.** Safe for one operator's runs
  on one machine, and not safe as a shared cache without a rights
  review.
- **`gpu_seconds` is never populated.**
- **No push streaming.** Logs and metrics are polled.
