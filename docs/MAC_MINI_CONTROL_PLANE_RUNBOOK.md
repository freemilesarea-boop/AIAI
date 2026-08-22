# Mac mini control-plane runbook

Setting up an always-on Apple Silicon Mac mini as LUBER's control plane.

Written for hardware that does not exist yet. Everything below is
derived from what this repository actually contains — the services it
defines, the commands it documents, the settings it reads — rather than
from a general idea of how a server is set up. Where a step has not been
performed on real hardware, it says so.

**No secrets appear in this document.** No key, no token, no password,
and no path from anybody's machine.

---

## 1. What the Mac mini is for

| Role | Runs there? | Why |
|---|---|---|
| API (`luber_api`) | yes | FastAPI, no accelerator |
| PostgreSQL | yes | generations, jobs, circuits, observability |
| Redis | yes | the ARQ queue |
| Generation worker | yes | calls ACE-Step over HTTP; no torch in the process |
| Audio worker | yes | ffmpeg and numpy |
| Training orchestration | yes | compiles plans, stages runs, talks to remote workers |
| Dataset preparation | yes | numpy and ffmpeg, no GPU |
| Evaluation | yes | HTTP plus arithmetic |
| Checkpoint collection | yes | file transfer |
| Operator console | yes | non-production deployments only |
| **Light fine-tuning (LoRA)** | **possible** | Apple MPS is verified working; see the caveat in §9 |
| **Heavy training** | **no** | prefers a remote CUDA worker and is refused locally by default |

The Mac mini is a control plane before it is a trainer. Everything in
§7 exists to keep it one.

## 2. Machine bootstrap

```bash
# Command line tools (git, clang)
xcode-select --install

# Homebrew, then the runtimes this repository needs
brew install python@3.12 node pnpm postgresql@15 redis ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
corepack enable pnpm
```

Node ≥ 22 and Python 3.11+ are the documented minimums
(`docs/LOCAL_DEVELOPMENT.md`).

Set the machine to stay awake — a control plane that sleeps stops
answering:

```bash
sudo pmset -a sleep 0 disksleep 0 displaysleep 10
sudo pmset -a autorestart 1     # come back after a power cut
```

## 3. Repository checkout

```bash
git clone <the repository> ~/luber-music-ai
cd ~/luber-music-ai
pnpm install
uv sync --all-packages          # --all-packages, or workspace members are not installed
```

`uv sync` **without** `--all-packages` installs only the root's
dependencies and uninstalls the workspace members. It is the one
foot-gun in the setup.

## 4. PostgreSQL and Redis

```bash
brew services start postgresql@15
createdb luber

uv run alembic -c packages/database/alembic.ini upgrade head
```

Redis: Homebrew's redis formula has shipped a `redis.conf` referencing a
`modules/` directory it does not create, which makes
`brew services start redis` fail. Running it directly avoids the whole
question:

```bash
redis-server --port 6379 --dir ~/luber-data/redis --save '' --daemonize yes
```

## 5. Configuration

Copy `.env.example` to `.env` and fill it in. Never commit it.

The Phase 32 setting worth knowing:

```bash
# The interpreter that has torch. LUBER's own environment does not —
# the control plane never imports it — so without this the compute
# panel honestly reports "CPU only".
TRAINING_PYTHON_EXECUTABLE=/path/to/the/trainer/venv/bin/python
```

Check it took:

```bash
uv run python -m luber_hardware probe --python "$TRAINING_PYTHON_EXECUTABLE"
uv run python -m luber_hardware readiness --python "$TRAINING_PYTHON_EXECUTABLE"
```

## 6. Starting the services

Four processes, from `docs/LOCAL_DEVELOPMENT.md`:

```bash
uv run uvicorn luber_api.main:app --port 8000
uv run arq luber_generation_worker.worker.WorkerSettings
uv run arq luber_audio_worker.worker.WorkerSettings
pnpm --filter web start          # only where a web front end is served
```

The generation worker runs `max_jobs = 1`, so scaling generation means
running several worker processes — which is exactly why Phase 31 put
circuit state in the database rather than in a process.

### Supervision

**This repository has no service supervisor**, and Phase 32 does not add
one. On macOS the mechanism is `launchd`: a `~/Library/LaunchAgents/*.plist`
per service with `RunAtLoad` and `KeepAlive`. Writing those is a
deliberate step for a later phase — an auto-start change that has not
been tested on the real machine is a change that fails at 3am — so the
processes above are started by hand or by whatever the operator already
trusts.

## 7. Keeping it a control plane

If the Mac mini ever trains locally while serving:

- **One local training job at a time.** `LOCAL_TRAINING_CONCURRENCY` is
  1. Two runs on shared unified memory is how the API stops answering.
- **Memory is never planned to 100%.** A machine flagged as running the
  control plane reserves 40% (floor 4 GB) before any training budget is
  computed.
- **Heavy training is refused locally** unless somebody explicitly
  passes `allow_local_fallback`. Not because a Mac cannot, but because a
  GPU job that quietly became a Mac job is indistinguishable afterwards.

There is no hard OS-level isolation here. macOS does not offer cgroups,
and pretending otherwise would be worse than saying so: these are policy
limits in the scheduler, not kernel enforcement.

## 8. Storage layout

| Location | Holds | Why |
|---|---|---|
| Internal SSD | the repository, virtualenvs, Postgres, Redis | fast, and small enough to fit |
| External SSD (optional) | datasets, audio cache, collected checkpoints | large, and none of it belongs in a backup of the boot volume |
| Remote GPU local disk | staged training artifacts | temporary; the run's own scratch |
| Durable artifact store | the authoritative results | the only copy that matters |

Transfer volume to and from a remote worker is recorded by Phase 27's
staging manifests — bytes moved, per run. There is no cost model here
and no provider pricing: what a byte costs depends on a contract this
repository knows nothing about.

No volume path is hard-coded anywhere. The directories come from
settings (`audio_storage_dir` and the training roots), so an external
disk is a configuration change rather than a code change.

## 9. Local training, honestly

Verified on an Apple M4 Pro with 24 GB, through the trainer's own
interpreter: MPS is built and available, fp32/fp16/bf16 all compute, a
tiny training loop runs, and checkpoints move between MPS and CPU with
optimizer state intact.

**Since Phase 33 and 34, more is known — about a different machine.**
On the development **M4 Pro / 24 GB**, a bounded canary loaded the real
ACE-Step DiT on MPS in bf16, injected a real LoRA, took an optimizer
step, wrote a checkpoint and resumed from it; and a bounded memory
profile at **production sequence length** (6000 latent frames = 240 s of
audio) peaked at **9 855 MiB of unified memory** with a LoRA rank of 32,
which the capacity policy qualifies on a 24 GB machine.

**That does not qualify this Mac mini.** An M4 Pro and a base M4 share a
memory capacity and not a GPU configuration. Memory-capacity evidence
may be informative across the two; it is not qualification.

**Still not verified anywhere:** that ACE-Step's LoRA training
*converges* on MPS. A memory profile measures memory.

The trainer accepts `--device mps` (read from its own parser at the
pinned commit) and defaults MPS to fp16 — **and fp16 cannot train**:
Fabric's GradScaler refuses to unscale fp16 gradients and the run dies
at the first clip, after the model has loaded. Request `bf16` or `fp32`
explicitly. `auto` resolves to fp16 here and hits the same wall.

### On-device qualification procedure

Run in this order on the Mac mini itself. Nothing may be skipped by
citing a measurement from another machine.

```bash
# 1. bootstrap: sections 2-6 of this runbook
# 2. hardware probe — what this machine can actually reach
uv run python -m luber_hardware probe --python ~/ace-step-1.5/.venv/bin/python
uv run python -m luber_hardware readiness --python ~/ace-step-1.5/.venv/bin/python

# 3. preflight — can this machine execute the plan at all
uv run python -m luber_training preflight --run-id <run-id> --device MPS \
    --intent CANARY --trainer-root ~/ace-step-1.5 \
    --python ~/ace-step-1.5/.venv/bin/python --model-dir ~/ace-step-1.5/checkpoints

# 4. bounded canary — does the trainer actually run here
uv run python -m luber_training canary ace-step --run-id <run-id> --device MPS \
    --trainer-root ~/ace-step-1.5 --python ~/ace-step-1.5/.venv/bin/python \
    --model-dir ~/ace-step-1.5/checkpoints --samples 2 --resume

# 5. memory profile — what the intended configuration costs HERE
uv run python -m luber_training profile-memory --run-id <run-id> --device MPS \
    --trainer-root ~/ace-step-1.5 --python ~/ace-step-1.5/.venv/bin/python \
    --model-dir ~/ace-step-1.5/checkpoints \
    --latent-length 6000 --encoder-length 256 --samples 2 --measure-resume

# 6. capacity — what the local evidence permits
uv run python -m luber_training capacity --run-id <run-id> --device MPS \
    --latent-length 6000 --encoder-length 256

# 7. only then: a full-training preflight
uv run python -m luber_training preflight --run-id <run-id> --device MPS \
    --intent FULL_TRAINING ...
```

Until step 6 returns `QUALIFIED` **on this machine**, this Mac mini's
status is `REQUIRES ON-DEVICE QUALIFICATION`. A profile from the
development M4 Pro is a different identity's evidence and the qualifier
will say so.

Note `--runs-control-plane` defaults on, which adds a 4 GiB reserve. On
this machine that default is correct: it is the API, Postgres, Redis and
the orchestrator before it is a trainer.

Details: `docs/TRAINING_MEMORY_CAPACITY.md`.

## 10. Restart and recovery

A control-plane restart must not lose work, and by existing design it
does not:

- **Training runs** are files in the registry, not process state.
- **Remote runs** keep their state on the worker; Phase 27's collection
  step re-reads it.
- **Provider circuits** are database rows with a compare-and-set, so a
  deploy during an outage does not reset every circuit to closed.
- **Generation jobs** are in Redis, and ARQ re-queues what was in flight.

Order on restart: Postgres, Redis, API, workers. Then:

```bash
uv run python -m luber_training run list        # what was in flight
uv run python -m luber_training worker list     # who can take it
uv run python -m luber_provider_resilience status
uv run python -m luber_hardware readiness
```

## 11. If the Mac mini is offline

**Nothing new is orchestrated.** Remote GPU training does not make the
system independent of the control plane: the plan is compiled here, the
staging is driven from here, and the registry lives here. A run already
executing on a remote worker keeps running and can be collected
afterwards; nothing new starts.

That is a single point of failure and it is a deliberate one at this
scale. It is worth knowing rather than discovering.

## 12. Access

- **SSH**: enable Remote Login in System Settings → General → Sharing.
  Key-based only; passwords off.
- **Headless**: enable Screen Sharing for the occasional GUI need. A
  Mac mini with no display attached runs fine.
- **The operator console is not for production.** It is refused when
  `ENVIRONMENT=production` and is not mounted unless switched on. If
  this machine ever serves real users, the console is off and the CLIs
  are the operator interface.

## 13. Backups

- **Postgres**: `pg_dump` on a schedule, to a volume that is not the
  boot disk. This is the one irreplaceable thing on the machine.
- **The registry** (training runs, evaluations): file trees, worth the
  same treatment.
- **Datasets and checkpoints**: large, and reproducible from locks and
  manifests. Back up the manifests; the audio is recoverable.
- **`.env`**: back it up somewhere a person controls, never into the
  repository.
