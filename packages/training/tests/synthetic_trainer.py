"""A fake trainer. TEST ONLY — this trains nothing and never will.

Stands in for `train.py` so the remote execution lifecycle can be
exercised end to end without a GPU, without ACE-Step, and without an
hour of wall clock. It writes metrics, writes logs, optionally writes a
checkpoint fixture, optionally fails, and handles SIGTERM the way a real
trainer must.

Three properties it deliberately has:

**Every checkpoint it writes is a fixture.** The adapter files contain
the string `SYNTHETIC_TEST_FIXTURE`, and the caller registers them in
Phase 25 as kind MOCK. A MOCK checkpoint can never become an evaluation
candidate — Phase 25 refuses at that boundary — so nothing this file
produces can be mistaken for a trained model.

**Every metric it emits is stamped SIMULATED.** A synthetic loss value
sitting in the same column as a real one would eventually be plotted
next to it. There is no `train_loss` here at all.

**It handles SIGTERM.** Cancellation is a real feature being tested, and
a fake trainer that ignored the signal would make the cancellation path
look like it worked when it had only waited for a process to finish.

It is never installed, never packaged, and lives under tests/ so that
nothing outside a test can reach it.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import FrameType

MARKER = "SYNTHETIC_TEST_FIXTURE"

#: Exactly what PyTorch prints, because the failure classifier looks for
#: this string and a paraphrase would test nothing.
CUDA_OOM_MESSAGE = (
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB "
    "(GPU 0; 23.65 GiB total capacity; 21.02 GiB already allocated)"
)
DISK_FULL_MESSAGE = (
    "OSError: [Errno 28] No space left on device: 'checkpoint-000/adapter_model.safetensors'"
)

_cancelled = False


def _on_term(signum: int, frame: FrameType | None) -> None:
    """Stop at the next step boundary, as a real trainer would.

    Not an immediate exit: the point of a grace period is that a trainer
    finishes writing whatever it had started, and a fixture that died
    instantly would never exercise that.
    """
    global _cancelled
    _cancelled = True
    print(f"[{MARKER}] SIGTERM received; stopping after this step", flush=True)


def write_checkpoint(directory: Path, *, step: int, valid: bool = True) -> Path:
    """A directory shaped like a PEFT adapter checkpoint.

    Written to a `.tmp` name and renamed, so a test that inspects the
    directory mid-write sees the same in-progress state a real trainer
    would produce.
    """
    target = directory / f"checkpoint-step{step:06d}"
    staging = target.with_name(target.name + ".tmp")
    staging.mkdir(parents=True, exist_ok=True)

    (staging / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "_luber_note": f"{MARKER}: contains no trained weights",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if valid:
        # Not a real safetensors file. It is bytes with a marker, which
        # is all the checkpoint contract can check without loading it —
        # and loading it is exactly what must never succeed.
        (staging / "adapter_model.safetensors").write_bytes(
            f"{MARKER}\nstep={step}\n".encode() + bytes(1024)
        )
    (staging / "README.txt").write_text(
        f"{MARKER}\nThis directory is a test fixture. It contains no model weights.\n",
        encoding="utf-8",
    )
    staging.rename(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="synthetic_trainer",
        description=f"{MARKER}: a fake trainer for remote execution tests. Trains nothing.",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--step-seconds", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="0 writes none")
    parser.add_argument("--invalid-checkpoint", action="store_true")
    parser.add_argument("--fail-at-step", type=int, help="exit non-zero at this step")
    parser.add_argument("--exit-code", type=int, default=1)
    parser.add_argument(
        "--simulate", choices=["oom", "disk-full", "crash"], help="what the failure looks like"
    )
    parser.add_argument("--ignore-sigterm", action="store_true", help="for the SIGKILL path")
    args = parser.parse_args(argv)

    if not args.ignore_sigterm:
        signal.signal(signal.SIGTERM, _on_term)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics = Path(args.metrics_file)
    metrics.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[{MARKER}] starting run {args.run_id} for {args.steps} step(s), pid {os.getpid()}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        if _cancelled:
            print(f"[{MARKER}] stopped at step {step} by request", flush=True)
            return 0

        time.sleep(args.step_seconds)

        if args.fail_at_step and step >= args.fail_at_step:
            if args.simulate == "oom":
                print(CUDA_OOM_MESSAGE, file=sys.stderr, flush=True)
            elif args.simulate == "disk-full":
                print(DISK_FULL_MESSAGE, file=sys.stderr, flush=True)
            else:
                print(
                    f"[{MARKER}] RuntimeError: the synthetic trainer was told to fail",
                    file=sys.stderr,
                    flush=True,
                )
            return args.exit_code

        with metrics.open("a", encoding="utf-8") as handle:
            for name, value, unit in (
                ("step_time_seconds", args.step_seconds, "seconds"),
                ("samples_per_second", 0.0, "samples/s"),
            ):
                handle.write(
                    json.dumps(
                        {
                            "run_id": args.run_id,
                            "metric_name": name,
                            "value": value,
                            # Never TRAINER. Nothing here measured
                            # anything about a model.
                            "source": "SIMULATED",
                            "step": step,
                            "epoch": 1,
                            "unit": unit,
                            "timestamp": f"1970-01-01T00:00:{step:02d}+00:00",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

        print(f"[{MARKER}] step {step}/{args.steps}", flush=True)

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            written = write_checkpoint(checkpoint_dir, step=step, valid=not args.invalid_checkpoint)
            print(f"[{MARKER}] wrote {written.name}", flush=True)

    print(f"[{MARKER}] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
