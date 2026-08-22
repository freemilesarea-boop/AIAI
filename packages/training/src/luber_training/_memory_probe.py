"""Measure the real trainer's memory, from inside the real trainer. No LUBER imports.

The same shape as ``_facts.py``, ``_smoke.py`` and ``_trainer_probe.py``,
and here the shape is the whole point. Phase 33's canary measured the
trainer from outside, as a subprocess, and the only number available
from there — a resident-set figure for a child process — turned out not
to be defensible. Everything that matters is inside: `torch.mps` will
only talk to the process that allocated, and a peak that happens between
a forward and a backward is invisible to anything watching from the
outside.

So this file **is** the trainer process. It imports ACE-Step, wraps a
handful of its callables with pass-through recorders, starts a sampler,
and then runs `train.py` in-process through `runpy` with the argv LUBER
compiled. The trainer does exactly what it would have done; nothing here
changes a config, a device, a precision or a shape.

Three things it is careful about.

**Wrappers do not change behaviour.** Each one calls the original,
records two numbers around it, and returns what the original returned.
Generators are wrapped as generators so `yield from` still works.

**The sampler always stops.** It is a daemon thread driven by an Event,
stopped in a `finally`, and joined with a timeout. A profiler that left
a thread sampling a machine after it exited would be worse than no
profiler.

**A safety trip is checked at real seams.** The sampler can observe that
memory has passed a configured boundary, but it does not try to predict
an OOM and it does not kill anything from a background thread. It sets a
flag, and the next wrapper to run in the main thread raises. That is a
real abort at a real point, not a guess.
"""

from __future__ import annotations

import json
import os
import platform
import runpy
import sys
import threading
import time
import traceback
from typing import Any

#: Bump when the shape of the emitted document changes.
MEMORY_PROBE_PROTOCOL_VERSION = "luber-memory-probe/1"

#: How many snapshots may accumulate before sampling stops.
#:
#: A bounded profile at one sample a second cannot reach this; a bug
#: that sampled in a loop would. The cap exists so a runaway produces a
#: truncated profile with a note rather than a gigabyte of JSON.
MAX_SNAPSHOTS = 20_000

#: Default seconds between samples. Conservative: sampling costs a
#: `torch.mps` call and a psutil read, and a tighter interval buys
#: precision on a number that is a lower bound either way.
DEFAULT_SAMPLE_INTERVAL = 0.25


class MemorySafetyAbort(RuntimeError):
    """Raised in the main thread when a configured boundary was crossed."""


# ── reading the machine ──────────────────────────────────────────────


def _psutil() -> Any:
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil
    except Exception:
        return None


class Reader:
    """Everything readable about memory, in one place, cheaply.

    Handles to the runtimes are resolved once. A sampler that re-imported
    torch every quarter second would be measuring its own import machinery
    as much as the trainer.
    """

    def __init__(self) -> None:
        self.psutil = _psutil()
        self.process = None
        if self.psutil is not None:
            try:
                self.process = self.psutil.Process(os.getpid())
            except Exception:
                self.process = None
        self.torch: Any = None
        self.mps: Any = None
        self.cuda_available = False
        self.system_total: int | None = self._system_total()

    def attach_torch(self) -> None:
        """Pick up torch once it is importable. Never imports it early."""
        if self.torch is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            return
        self.torch = torch
        candidate = getattr(torch, "mps", None)
        # Only kept if the backend is genuinely there. A handle to a
        # module whose every call raises is worse than no handle.
        try:
            if candidate is not None and torch.backends.mps.is_available():
                self.mps = candidate
        except Exception:
            self.mps = None
        try:
            self.cuda_available = bool(torch.cuda.is_available())
        except Exception:
            self.cuda_available = False

    def _system_total(self) -> int | None:
        if self.psutil is not None:
            try:
                return int(self.psutil.virtual_memory().total)
            except Exception:
                pass
        if sys.platform == "darwin":
            import subprocess

            try:
                out = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                return int(out.stdout.strip()) if out.returncode == 0 else None
            except Exception:
                return None
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            return None

    def snapshot(self, stage: str, *, started: float, note: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {
            "stage": stage,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "note": note,
            "host_rss_bytes": None,
            "host_available_bytes": None,
            "system_total_bytes": self.system_total,
            "mps_current_allocated_bytes": None,
            "mps_driver_allocated_bytes": None,
            "mps_recommended_max_bytes": None,
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "cuda_total_bytes": None,
            "cuda_free_bytes": None,
        }

        if self.process is not None:
            try:
                out["host_rss_bytes"] = int(self.process.memory_info().rss)
            except Exception:
                pass
        if self.psutil is not None:
            try:
                out["host_available_bytes"] = int(self.psutil.virtual_memory().available)
            except Exception:
                pass

        if self.mps is not None:
            for key, name in (
                ("mps_current_allocated_bytes", "current_allocated_memory"),
                ("mps_driver_allocated_bytes", "driver_allocated_memory"),
                ("mps_recommended_max_bytes", "recommended_max_memory"),
            ):
                function = getattr(self.mps, name, None)
                if function is None:
                    continue
                try:
                    out[key] = int(function())
                except Exception:
                    pass

        if self.cuda_available and self.torch is not None:
            cuda = self.torch.cuda
            for key, name in (
                ("cuda_allocated_bytes", "memory_allocated"),
                ("cuda_reserved_bytes", "memory_reserved"),
                ("cuda_peak_allocated_bytes", "max_memory_allocated"),
                ("cuda_peak_reserved_bytes", "max_memory_reserved"),
            ):
                function = getattr(cuda, name, None)
                if function is None:
                    continue
                try:
                    out[key] = int(function())
                except Exception:
                    pass
            try:
                free, total = cuda.mem_get_info()
                out["cuda_free_bytes"] = int(free)
                out["cuda_total_bytes"] = int(total)
            except Exception:
                pass
        return out

    def runtime_peaks(self) -> dict[str, int]:
        """Peaks the runtime kept for itself, where it keeps any.

        CUDA does. MPS does not: the pinned torch has no
        `torch.mps.max_memory_allocated`, which is why an Apple peak in
        this project is always sampled.
        """
        peaks: dict[str, int] = {}
        if self.cuda_available and self.torch is not None:
            try:
                peaks["CUDA_DEVICE"] = int(self.torch.cuda.max_memory_reserved())
            except Exception:
                pass
        return peaks

    def reset_runtime_peaks(self) -> None:
        if self.cuda_available and self.torch is not None:
            try:
                self.torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


# ── recording ────────────────────────────────────────────────────────


class Recorder:
    """Snapshots, stage markers and the safety trip, behind one lock."""

    def __init__(self, reader: Reader, *, limits: dict[str, Any] | None = None) -> None:
        self.reader = reader
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.snapshots: list[dict[str, Any]] = []
        self.truncated = False
        self.limits = limits or {}
        self.safety_tripped: str | None = None
        self.observed_stages: set[str] = set()
        self.optimizer_steps = 0
        self.resumed = False

    def record(self, stage: str, note: str = "") -> dict[str, Any] | None:
        snapshot = self.reader.snapshot(stage, started=self.started, note=note)
        with self.lock:
            if len(self.snapshots) >= MAX_SNAPSHOTS:
                self.truncated = True
                return None
            self.snapshots.append(snapshot)
            self.observed_stages.add(stage)
        self.check_safety(snapshot)
        return snapshot

    def check_safety(self, snapshot: dict[str, Any]) -> None:
        """Observe a boundary crossing. Never predicts one.

        Two boundaries, both from readings that already exist: Apple's
        own recommended working-set maximum, and the system's available
        memory. Crossing either is a fact, not a forecast.
        """
        floor = self.limits.get("host_available_floor_bytes")
        available = snapshot.get("host_available_bytes")
        if floor and available is not None and available < int(floor):
            self.safety_tripped = (
                f"system available memory fell to {available} bytes, below the configured "
                f"floor of {int(floor)} bytes"
            )
            return
        fraction = self.limits.get("mps_recommended_max_fraction")
        driver = snapshot.get("mps_driver_allocated_bytes")
        recommended = snapshot.get("mps_recommended_max_bytes")
        if fraction and driver is not None and recommended:
            ceiling = float(recommended) * float(fraction)
            if driver > ceiling:
                self.safety_tripped = (
                    f"MPS driver allocation reached {driver} bytes, past {fraction:.0%} of "
                    f"the runtime's own recommended maximum of {recommended} bytes"
                )

    def raise_if_tripped(self) -> None:
        """Called from the main thread, at real seams only."""
        if self.safety_tripped is not None:
            raise MemorySafetyAbort(self.safety_tripped)


class Sampler:
    """A background thread that snapshots on an interval, and stops.

    Daemon, so a wedged interpreter cannot be kept alive by it; driven by
    an Event, so stopping is immediate rather than eventual; joined with
    a timeout, so a stuck read cannot hang the profile.
    """

    def __init__(self, recorder: Recorder, interval: float) -> None:
        self.recorder = recorder
        self.interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="luber-memory-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.recorder.record("SAMPLE")
            except Exception:
                # A sampler that raised would take its thread down
                # silently and leave the profile looking complete.
                # Losing one sample is the smaller loss.
                continue
            if self.recorder.truncated:
                break

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ── instrumentation ──────────────────────────────────────────────────


def install(recorder: Recorder) -> dict[str, str]:
    """Wrap the trainer's own callables. Returns what could not be wrapped.

    Everything here is a pass-through: call the original, record around
    it, return what it returned. A seam that is not present in the
    installed trainer is reported in the result rather than faked, which
    is why this returns a mapping of stage to reason.
    """
    missing: dict[str, str] = {}

    def around(stage: str, before: str | None = None) -> Any:
        def decorate(original: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                recorder.raise_if_tripped()
                if before:
                    recorder.record(before)
                result = original(*args, **kwargs)
                recorder.record(stage)
                return result

            return wrapper

        return decorate

    # MODEL_LOADED — `train_fixed` holds the name it calls.
    try:
        from acestep.training_v2.cli import (  # type: ignore[import-not-found]
            train_fixed as cli_train_fixed,
        )

        cli_train_fixed.load_decoder_for_training = around("MODEL_LOADED")(
            cli_train_fixed.load_decoder_for_training
        )
    except Exception as exc:
        missing["MODEL_LOADED"] = f"{type(exc).__name__}: {exc}"

    # LORA_ATTACHED — injected from inside FixedLoRAModule.__init__.
    try:
        from acestep.training_v2 import (  # type: ignore[import-not-found]
            fixed_lora_module,
        )

        fixed_lora_module.inject_lora_into_dit = around("LORA_ATTACHED")(
            fixed_lora_module.inject_lora_into_dit
        )
    except Exception as exc:
        missing["LORA_ATTACHED"] = f"{type(exc).__name__}: {exc}"

    # BATCH_READY / FORWARD_COMPLETE — one call, two markers. The batch
    # is on the device by the time the step is entered, and the forward
    # is what the step returns.
    try:
        from acestep.training_v2.fixed_lora_module import (  # type: ignore[import-not-found]
            FixedLoRAModule,
        )

        FixedLoRAModule.training_step = around("FORWARD_COMPLETE", before="BATCH_READY")(
            FixedLoRAModule.training_step
        )
    except Exception as exc:
        missing["FORWARD_COMPLETE"] = f"{type(exc).__name__}: {exc}"

    # OPTIMIZER_CREATED and CHECKPOINT_* — the names `trainer_fixed` uses.
    try:
        from acestep.training_v2 import trainer_fixed

        trainer_fixed.build_optimizer = around("OPTIMIZER_CREATED")(trainer_fixed.build_optimizer)
        trainer_fixed.save_checkpoint = around("CHECKPOINT_COMPLETE", before="CHECKPOINT_BEGIN")(
            trainer_fixed.save_checkpoint
        )
        trainer_fixed.resume_checkpoint = _wrap_generator(
            trainer_fixed.resume_checkpoint, recorder, "RESUME_LOADED"
        )
        trainer_fixed.FixedLoRATrainer.train = _wrap_train(
            trainer_fixed.FixedLoRATrainer.train, recorder
        )
    except Exception as exc:
        for stage in ("OPTIMIZER_CREATED", "CHECKPOINT_COMPLETE", "RESUME_LOADED"):
            missing[stage] = f"{type(exc).__name__}: {exc}"

    # BACKWARD_COMPLETE — Fabric owns the backward in this loop.
    try:
        from lightning.fabric import Fabric  # type: ignore[import-not-found]

        Fabric.backward = around("BACKWARD_COMPLETE")(Fabric.backward)
    except Exception as exc:
        missing["BACKWARD_COMPLETE"] = f"{type(exc).__name__}: {exc}"

    return missing


def _wrap_generator(original: Any, recorder: Recorder, stage: str) -> Any:
    """Wrap a generator function without turning it into a function.

    `resume_checkpoint` is consumed with `yield from` and returns a
    value. A plain wrapper would break both, so this stays a generator
    and forwards the return.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = yield from original(*args, **kwargs)
        recorder.record(stage)
        recorder.resumed = True
        return result

    return wrapper


def _wrap_train(original: Any, recorder: Recorder) -> Any:
    """Observe the trainer's own progress updates as they are yielded.

    The loop reports a step by yielding, so this is where an optimizer
    step becomes observable without reaching into the loop. Each update
    is passed straight through.
    """

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        for update in original(self, *args, **kwargs):
            kind = getattr(update, "kind", "")
            if kind == "step":
                recorder.optimizer_steps += 1
                stage = "RESUME_STEP_COMPLETE" if recorder.resumed else "OPTIMIZER_STEP_COMPLETE"
                recorder.record(stage, note=f"step {getattr(update, 'step', '?')}")
            yield update

    return wrapper


# ── running it ───────────────────────────────────────────────────────


def _runtime_identity(reader: Reader, request: dict[str, Any]) -> dict[str, Any]:
    """Reproducibility facts only. No hostname, no user, no path."""
    return {
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": str(getattr(reader.torch, "__version__", "")) or None,
        "ace_step_commit": request.get("ace_step_commit"),
        "luber_commit": request.get("luber_commit"),
        "platform_class": f"{platform.system()} {platform.machine()}",
        "device_class": request.get("device"),
        "probe_protocol_version": MEMORY_PROBE_PROTOCOL_VERSION,
    }


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Profile one bounded trainer invocation, in this process."""
    argv = [str(item) for item in (request.get("argv") or [])]
    if not argv:
        return {"outcome": "BLOCKED", "failure_reason": "no trainer argv was supplied"}

    reader = Reader()
    recorder = Recorder(reader, limits=request.get("limits") or {})
    recorder.record("BASELINE")

    reader.attach_torch()
    reader.reset_runtime_peaks()
    recorder.record("RUNTIME_INITIALIZED")

    missing = install(recorder)
    sampler = Sampler(recorder, float(request.get("sample_interval") or DEFAULT_SAMPLE_INTERVAL))

    outcome = "COMPLETED"
    failure_reason = ""
    started_wall = time.time()
    started = time.perf_counter()
    sampler.start()
    try:
        saved_argv = list(sys.argv)
        sys.argv = argv
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exit_signal:
            code = exit_signal.code
            if code not in (0, None):
                outcome = "FAILED"
                failure_reason = f"the trainer exited {code}"
        finally:
            sys.argv = saved_argv
    except MemorySafetyAbort as abort:
        outcome = "FAILED"
        failure_reason = f"memory safety abort: {abort}"
    except BaseException as exc:
        # Broad on purpose: whatever the trainer did, the profile has to
        # come back with what was measured up to that point and a reason,
        # rather than the profiler dying alongside it.
        outcome = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"
        request.setdefault("_traceback", traceback.format_exc()[-4000:])
    finally:
        # Always. A sampler still running after the profile has finished
        # would keep measuring a machine nobody is profiling.
        sampler.stop()
        recorder.record("FINAL")

    if recorder.safety_tripped and outcome == "COMPLETED":
        outcome = "FAILED"
        failure_reason = f"memory safety boundary crossed: {recorder.safety_tripped}"

    return {
        "protocol_version": MEMORY_PROBE_PROTOCOL_VERSION,
        "outcome": outcome,
        "failure_reason": failure_reason,
        "traceback_tail": request.get("_traceback", ""),
        "snapshots": recorder.snapshots,
        "runtime_peaks": reader.runtime_peaks(),
        "not_observed": missing,
        "observed_stages": sorted(recorder.observed_stages),
        "optimizer_steps": recorder.optimizer_steps,
        "resumed": recorder.resumed,
        "truncated": recorder.truncated,
        "safety_tripped": recorder.safety_tripped,
        "sampler_running_after_stop": sampler.running,
        "sample_interval_seconds": sampler.interval,
        "started_at_epoch": started_wall,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "runtime_identity": _runtime_identity(reader, request),
    }


def main() -> int:
    try:
        raw = sys.stdin.read()
        request = json.loads(raw) if raw.strip() else {}
    except ValueError:
        request = {}
    if not isinstance(request, dict):
        request = {}

    result = run(request)
    destination = request.get("result_path")
    if destination:
        # Written to a file rather than trusted to stdout: the trainer
        # prints a great deal, and a profile that had to be recovered
        # from a log would be a profile that could be corrupted by one.
        try:
            with open(str(destination), "w", encoding="utf-8") as handle:
                json.dump(result, handle, sort_keys=True)
        except OSError as exc:
            print(json.dumps({"error": f"could not write the profile: {exc}"}))
            return 1
    print(json.dumps({"outcome": result.get("outcome"), "written": bool(destination)}))
    return 0 if result.get("outcome") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
