# First remote GPU connection

The exact sequence for connecting LUBER to an NVIDIA machine for the
first time. Follow it in order; every step is cheap and the ones near
the end are not.

Companion to [`REMOTE_GPU_EXECUTION.md`](REMOTE_GPU_EXECUTION.md), which
explains why the machinery works this way, and to
[`GPU_TRAINING_DAY_RUNBOOK.md`](GPU_TRAINING_DAY_RUNBOOK.md), which
covers training once the infrastructure is proven.

**Nothing real is trained on the first day.** The goal is a smoke run
that proves the pipeline end to end on a tiny approved dataset. A real
experiment starts only after that passes.

No provider is named here and no credentials appear anywhere in this
document.

---

## Before you rent anything

Everything below runs on your own machine and costs nothing. Every
failure found here is a failure not found on a metered box.

1. **Confirm the whole pipeline works without a GPU.**

   ```bash
   uv run pytest packages/training/tests -q
   ```

   The remote lifecycle tests run a real worker in a real subprocess
   with a real detached trainer. If they fail locally, they will fail
   remotely for the same reason and a rented machine will not tell you
   anything new.

2. **Have an approved, frozen dataset.** A Phase 23 build with
   `dataset_lock.json` and a Phase 24 curation with
   `curation_lock.json`. For the smoke run, cut it down to **two or
   three tracks**. Transferring a full corpus to prove that transfer
   works is the expensive way to find out it does not.

3. **Have an evaluation-only list.** The track ids that must never enter
   training — P20 benchmark material above all. Without the file, the
   leakage gate has nothing to protect.

4. **Stage locally and verify.**

   ```bash
   python -m luber_training remote run stage --run-id run_… \
     --dataset-build ./dataset-build --curation-build ./curation-build \
     --audio-root ./library --evaluation-only ./evaluation_only.txt

   python -m luber_training remote run verify-staging --run-id run_… \
     --dataset-build ./dataset-build --curation-build ./curation-build \
     --audio-root ./library --evaluation-only ./evaluation_only.txt
   ```

   The rights and leakage gates run here. If either fails, nothing is
   written and nothing would have left your machine — fix it now.

5. **Commit and push.** The worker will be required to hold the same
   LUBER commit, and a dirty tree cannot be identified later.

---

## 1. Provision the host

Use the checklist in [`GPU_PROVIDER_CHECKLIST.md`](GPU_PROVIDER_CHECKLIST.md)
and the provider-neutral notes in §11 below.

Note the hourly rate, the minimum billing period and the shutdown
behaviour **before** starting the instance. A host that bills while idle
and one that terminates on disconnect fail in opposite directions.

---

## 2. Establish SSH access by hand

Connect once, interactively, as yourself:

```bash
ssh -i ~/.ssh/luber_gpu_key ops@<host>
```

Do this before any LUBER command touches the machine. You are checking
that you can get in, that the user has a home directory, and that the
disk is where the provider said it is.

> Type `! ssh …` in a Claude Code session to run it here and keep the
> output in the conversation.

---

## 3. Pin the host key

This is the step people skip, and it is the one that matters.

```bash
ssh-keyscan -H <host> >> ~/.luber-secrets/gpu-known-hosts
```

Then **verify the fingerprint out of band** — against the provider's
console, not against the machine that just presented it. A key you
accepted because the machine offered it proves nothing.

LUBER's SSH transport uses `StrictHostKeyChecking=yes` and will refuse
an unknown host. There is no flag that disables it, and `accept-new` is
deliberately not used: an operator dispatching a job is in no position
to notice that last week's host now has a different key.

Keep the secrets directory **outside the repository** and lock it down:

```bash
mkdir -p ~/.luber-secrets && chmod 700 ~/.luber-secrets
chmod 600 ~/.luber-secrets/*
```

`FileSecretResolver` refuses a directory inside the working tree, and
refuses a key any other account can read.

---

## 4. Clone the exact LUBER commit

On the host:

```bash
git clone <repo> /opt/luber && cd /opt/luber
git checkout <the commit you pushed in step 5 above>
git rev-parse HEAD          # note this; preflight will compare it
```

Not a branch name. The commit.

---

## 5. Install the environment

```bash
cd /opt/luber
uv sync --all-packages
```

Then confirm torch actually reaches the GPU — this is what decides the
worker's classification, and a driver alone does not:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

If that prints `False`, stop. Nothing downstream will work and preflight
will (correctly) refuse to start a trainer.

---

## 6. Install and pin ACE-Step

```bash
git clone <ace-step> /opt/ace-step && cd /opt/ace-step
git checkout 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
git rev-parse HEAD          # must match the training config's pin
ls train.py                 # preflight requires this to exist
```

---

## 7. Initialise the worker

```bash
cd /opt/luber
uv run python -m luber_training.remote --root /opt/luber/worker init \
  --name gpu-01 --base /workspace/luber \
  --trainer-root /opt/ace-step --repository-root /opt/luber
```

`--base` decides where runs, datasets, checkpoints and the cache live.
Point it at the **large, fast disk**, not the root volume — providers
routinely give you a small root and a big mount, and the run root is
where a dataset and every checkpoint will land.

---

## 8. Probe the worker

```bash
uv run python -m luber_training.remote --root /opt/luber/worker probe
```

Read the output rather than skimming it:

- `classification` must be `CUDA_TRAINING`. `DEVELOPMENT_ONLY` means
  torch did not demonstrate CUDA, whatever `nvidia-smi` says.
- `gpu_model`, `vram_total_mb`, `driver_version`, `cuda_version` should
  be populated. `null` means nobody could measure it, and preflight
  treats an unmeasured requirement as unsatisfied.
- `free_disk_mb` should reflect the big disk. If it reflects the root
  volume, `--base` is wrong — fix it now, not after a 40 GB transfer.
- `unknown[]` lists what could not be established, with reasons.

Note the `capability_signature`. Registration records it, and `worker
verify` compares against it later.

---

## 9. Register the worker

From your machine:

```bash
python -m luber_training remote worker register-remote \
  --transport ssh --host <host> --user ops --worker-root /opt/luber/worker \
  --remote-python "uv run python" \
  --ssh-key-ref luber_gpu_key --known-hosts-ref gpu-known-hosts \
  --secret-dir ~/.luber-secrets
```

The classification comes from the worker's own probe. There is no flag
here that asserts one — a machine becomes `CUDA_TRAINING` by
demonstrating CUDA on itself, and no amount of registration changes
that.

Record the `worker_id`. It was minted on the machine and survives its
reboots.

---

## 10. Verify the worker

```bash
python -m luber_training remote worker verify --worker-id wrk_… \
  --transport ssh --host <host> --user ops --worker-root /opt/luber/worker \
  --remote-python "uv run python" --ssh-key-ref luber_gpu_key \
  --known-hosts-ref gpu-known-hosts --secret-dir ~/.luber-secrets
```

Exits non-zero on a protocol mismatch, a changed worker id, a changed
host fingerprint (the machine was rebuilt) or a changed capability
signature (the hardware or software is not what was registered).

Also check the heartbeat, and the liveness policy that will judge it:

```bash
python -m luber_training remote worker heartbeat --transport ssh …
```

---

## 11. Check the disk before you fill it

The staging output states `transfer_bytes` and a disk requirement. Compare
it with the worker's `free_disk_mb` from step 8.

The requirement **excludes checkpoint size**, and says so: nobody has
measured what a LUBER checkpoint weighs, so it is reported as unknown
rather than folded into a total that would look authoritative. Leave
real headroom.

---

## 12. Dispatch the SMOKE run

Two or three tracks. One epoch. This is infrastructure, not training.

```bash
python -m luber_training remote run dispatch --run-id run_… --worker-id wrk_… \
  --transport ssh --host <host> --user ops \
  --worker-root /opt/luber/worker --remote-run-root /workspace/luber/runs \
  --remote-python "uv run python" \
  --ssh-key-ref luber_gpu_key --known-hosts-ref gpu-known-hosts \
  --secret-dir ~/.luber-secrets
```

What happens, in order: reconcile (nothing is running) → transfer with
per-file digests → the worker records the manifest → preflight rehashes
everything on the far side and checks the environment → launch.

If preflight returns `BLOCKED` or `FAIL`, read `blocking_reasons`. The
common first-day ones:

| Reason | Cause |
|---|---|
| `code_revision` | the host is on a different commit than you pushed |
| `ace_step_commit` | ACE-Step is unpinned or absent |
| `torch_version` | the host's torch differs from the recorded lock |
| `cuda` | torch does not see the GPU |
| `artifact_digests` | a file did not survive transfer — re-dispatch, it resumes |
| `trainer_command` | `train.py` is not where `--trainer-root` says |

No trainer starts until it passes. That is the point.

---

## 13. Watch it

```bash
# incremental; pass the returned next_offset back next time
python -m luber_training remote run remote-logs --run-id run_… --offset 0 …
python -m luber_training remote run remote-metrics --run-id run_… …
python -m luber_training remote run remote-status --run-id run_… …
python -m luber_training remote worker heartbeat …
```

If the connection drops, **do not re-dispatch.** Reconcile:

```bash
python -m luber_training remote run reconcile --run-id run_… --apply …
```

`UNREACHABLE` means the trainer may still be running, and nothing will
be launched for that run until the worker answers. That is deliberate:
a second trainer writing into one checkpoint directory produces
artifacts that are individually well-formed and jointly worthless.

---

## 14. Verify what it produced

```bash
python -m luber_training remote run verify-remote --run-id run_… …
```

Confirms the worker still holds the artifacts this run believes it sent,
with matching digests.

---

## 15. Collect the checkpoint

```bash
python -m luber_training remote run collect --run-id run_… \
  --collect-root ./collected_checkpoints --transport ssh …
```

The checkpoint is transferred, the whole-tree digest is recomputed
locally, and only if it matches what the worker reported does Phase 25's
registry record it as READY.

On mismatch: nothing is registered, the remote copy is left untouched,
and the command can simply be run again — it resumes.

**The remote copy is not deleted.** The instance may be terminated at
any moment and take its disk with it, so the remote copy is a second
copy for exactly as long as the machine exists.

---

## 16. Phase 26 evaluation smoke

```bash
python -m luber_evaluation --repository . suite list
python -m luber_evaluation run create --candidate-id cand_… --suite SMOKE
```

See [`GPU_EVALUATION_RUNBOOK.md`](GPU_EVALUATION_RUNBOOK.md). Note the
hazard it leads with: generating both sides of a comparison against one
ACE-Step server compares a model with itself and looks exactly like an
honest evaluation where nothing improved.

---

## 17. Only now, a real experiment

The smoke run must have produced, in order:

- [ ] a probe reporting `CUDA_TRAINING` with real GPU figures
- [ ] `worker verify` exiting zero
- [ ] staging with both gates passing
- [ ] preflight `PASS`
- [ ] a trainer that ran and exited zero
- [ ] logs and metrics that polled incrementally without duplicates
- [ ] a checkpoint discovered as `READY_REMOTE`
- [ ] a collection whose local digest matched the remote one
- [ ] a Phase 25 registry entry marked READY
- [ ] a Phase 26 evaluation that at least created

Any box unticked is a box to fix before spending money on training.

---

## 18. Record what you actually measured

The project holds **no NVIDIA measurements of its own**. Every VRAM
figure, checkpoint size and throughput number is currently `null`, and
preflight reports them UNKNOWN rather than guessing.

After the smoke run, write down and commit:

- peak VRAM during training, from `nvidia-smi` while it ran
- the size of one checkpoint directory
- transfer throughput, up and down
- wall-clock for one epoch on the smoke dataset
- anything in `unknown[]` that you were able to resolve

Real numbers replace the absence of numbers. Do not fill this in with
estimates — an invented VRAM figure is how a run gets scheduled onto
hardware that cannot hold it, which is the failure this entire package
is built to prevent.

---

## 19. Shutting down

1. `run collect` — checkpoints, verified.
2. Copy the run directory off the host: `remote_result.json`,
   `remote_preflight.json`, `status.json`, `logs/`, `metrics/`.
3. `run cleanup` — removes scratch and partial transfers. It never
   removes logs, metrics, checkpoints or the result manifest.
4. Only then terminate the instance.

Remote checkpoints are lost when the instance goes. That is expected and
is why collection happens first.

---

## 20. What to check when renting a server

Provider-neutral, and no prices are quoted here — they change weekly and
a number in a document is a number that will be wrong.

**GPU**
- model and VRAM per card; count
- CUDA compute capability, and that it is supported by the pinned torch
- whether the card is shared, virtualised or MIG-partitioned

**Software**
- preinstalled CUDA and driver versions
- whether you get root, and whether the image is Ubuntu-like

**Disk**
- size of the root volume *and* of any large mount
- whether storage **persists** across a stop, or only across a reboot
- IOPS, if the dataset is many small files

**Network**
- upload bandwidth (your dataset goes up)
- download bandwidth (checkpoints come back)
- **egress fees** — often the surprise on the bill
- whether the public IP is stable across a restart

**Access**
- SSH with key auth, and whether a jump host is imposed
- whether you can pin a host key at all

**Lifecycle and billing**
- shutdown semantics: does stopping bill, does disconnecting terminate
- minimum billing period
- spot/preemptible behaviour: how much warning before reclamation
- whether an idle instance bills at the running rate
