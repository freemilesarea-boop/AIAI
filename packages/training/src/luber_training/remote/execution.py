"""Running a trainer on a worker, and being able to say what happened.

A training process outlives the thing that launched it. The worker CLI
is invoked over SSH, does its work and exits; the trainer keeps running
for hours. Everything here follows from that: state lives in files under
the run root, not in memory, and any later invocation reconstructs what
it needs by reading them.

Four decisions worth stating.

**The trainer gets its own process group.** Cancellation signals the
group, so dataloader workers and any other children stop too. Signalling
a bare PID leaves orphans holding GPU memory, and the next run fails to
allocate for reasons nobody can see.

**Logs go to files, never through a pipe.** A parent that reads a pipe
must stay alive to drain it, and this parent exits immediately. stdout
and stderr are separate files opened in append mode, so a multi-hour log
never occupies memory and a restarted worker keeps what came before.

**Cancellation is graceful first.** SIGTERM to the group, a configurable
grace period, then SIGKILL only if the process is still there. A trainer
killed outright mid-checkpoint leaves a half-written directory, which is
precisely the artifact the whole checkpoint contract exists to prevent.

**Exit codes are classified conservatively.** A non-zero exit is a
crash. It becomes OOM only on a CUDA out-of-memory message in the log —
not on a SIGKILL, which the OOM killer and an operator's `kill -9`
produce identically.
"""

from __future__ import annotations

import errno
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from luber_training.entities import FailureCode
from luber_training.remote.paths import RunLayout
from luber_training.remote.protocol import WorkerState, now

#: How long a cancelled trainer is given to stop on its own. Long
#: enough to finish writing a checkpoint it had already started.
DEFAULT_GRACE_SECONDS = 60.0

#: How long to watch a freshly launched process before reporting that
#: it started. Half a second: a trainer that dies on an import error or
#: a missing CUDA library is gone well inside that, and a trainer that
#: is genuinely working has not finished.
#:
#: Not a training timeout. There is deliberately no timeout on training
#: itself — a multi-hour job killed by a global deadline is the most
#: expensive possible bug in this package.
DEFAULT_START_CONFIRM_SECONDS = 0.5

#: Evidence strong enough to call a failure OOM. These are the exact
#: strings PyTorch and the CUDA runtime emit; a SIGKILL is not on the
#: list, because the kernel OOM killer and `kill -9` are indistinguishable
#: from the outside.
CUDA_OOM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"cudaErrorMemoryAllocation", re.IGNORECASE),
)

#: Evidence of a full disk. ENOSPC by name or by number, and the
#: messages a Python traceback produces when a write fails.
DISK_FULL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"No space left on device", re.IGNORECASE),
    re.compile(r"\bENOSPC\b"),
    re.compile(r"errno\s*28\b", re.IGNORECASE),
    re.compile(r"Disk quota exceeded", re.IGNORECASE),
)

#: How much of a log tail is scanned for failure evidence. The relevant
#: traceback is always at the end, and reading a multi-gigabyte log to
#: classify an exit would defeat the purpose of streaming it.
CLASSIFY_TAIL_BYTES = 256 * 1024


class ExecutionError(RuntimeError):
    """Raised when a trainer cannot be launched or managed."""


@dataclass
class ProcessRecord:
    """What was launched, enough to find it again after a restart.

    ``started_at_monotonic`` is deliberately absent — a monotonic clock
    does not survive a reboot. ``pid`` plus ``process_group`` plus the
    recorded start time is what a later invocation uses to decide
    whether the thing at that PID is still the thing it launched.
    """

    pid: int
    process_group: int | None
    argv: list[str]
    working_directory: str
    started_at: str = field(default_factory=now)
    exit_code: int | None = None
    finished_at: str | None = None
    cancel_requested_at: str | None = None
    cancel_signal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProcessRecord:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)


def process_alive(pid: int) -> bool:
    """Whether a PID is a live process this user may signal.

    Signal 0 is the standard probe: it performs the permission and
    existence checks and delivers nothing. EPERM means the process
    exists and belongs to someone else, which is still "alive".
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


@dataclass
class ExecutionState:
    """The worker's durable record of one run's execution.

    Written to `status.json` after every transition. It is the only
    thing a restarted worker or a reconnecting control plane has to go
    on, so it holds everything needed to answer "what happened", not
    merely "what is the state now".
    """

    run_id: str
    state: str = WorkerState.IDLE.value
    training_plan_sha256: str = ""
    manifest_sha256: str = ""
    lease_id: str | None = None
    process: ProcessRecord | None = None
    exit_code: int | None = None
    failure_code: str | None = None
    detail: str = ""
    updated_at: str = field(default_factory=now)
    started_at: str | None = None
    completed_at: str | None = None
    cancelled_at: str | None = None
    failed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["process"] = self.process.to_dict() if self.process else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionState:
        known = {
            key: value
            for key, value in payload.items()
            if key in cls.__annotations__ and key != "process"
        }
        state = cls(**known)
        process = payload.get("process")
        state.process = ProcessRecord.from_dict(process) if process else None
        return state

    def write(self, layout: RunLayout) -> None:
        """Persist atomically. A torn status file is unrecoverable."""
        layout.root.mkdir(parents=True, exist_ok=True)
        self.updated_at = now()
        temporary = layout.status_json.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, layout.status_json)

    @classmethod
    def read(cls, layout: RunLayout) -> ExecutionState | None:
        if not layout.status_json.is_file():
            return None
        try:
            return cls.from_dict(json.loads(layout.status_json.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None


def _tail(path: Path, limit: int = CLASSIFY_TAIL_BYTES) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(size - limit)
        return handle.read().decode("utf-8", errors="replace")


def classify_failure(
    exit_code: int | None, *, stdout: Path | None = None, stderr: Path | None = None
) -> tuple[str, str]:
    """Why a trainer exited, on evidence, with UNKNOWN as a real answer.

    OOM is only claimed on an explicit CUDA out-of-memory message.
    Signal-based exits are especially tempting to classify — a SIGKILL
    is often the kernel OOM killer — but it is also what `kill -9` does,
    and a run mislabelled OOM sends the next experiment chasing a memory
    problem that never existed.
    """
    if exit_code == 0:
        return ("", "the trainer exited cleanly")

    evidence = "\n".join(
        part for part in (_tail(stderr) if stderr else "", _tail(stdout) if stdout else "") if part
    )

    for pattern in CUDA_OOM_PATTERNS:
        if pattern.search(evidence):
            return (
                FailureCode.OOM.value,
                f"the trainer reported CUDA out of memory (exit {exit_code})",
            )
    for pattern in DISK_FULL_PATTERNS:
        if pattern.search(evidence):
            return (
                FailureCode.CHECKPOINT_WRITE_FAILED.value,
                f"the trainer ran out of disk space (exit {exit_code})",
            )

    if exit_code is not None and exit_code < 0:
        name = (
            signal.Signals(-exit_code).name
            if -exit_code in signal.Signals.__members__.values()
            else str(-exit_code)
        )
        return (
            FailureCode.TRAINER_CRASH.value,
            (
                f"the trainer was terminated by {name}. This is not classified as OOM: a "
                "kernel OOM kill and an operator kill are indistinguishable from here, and "
                "no CUDA out-of-memory message was found in the logs"
            ),
        )
    return (
        FailureCode.TRAINER_CRASH.value,
        f"the trainer exited with code {exit_code} and no recognised failure signature",
    )


class TrainerProcess:
    """Launch, observe, and stop one trainer. State lives on disk."""

    def __init__(self, layout: RunLayout) -> None:
        self.layout = layout

    # ── launch ───────────────────────────────────────────────────────
    def launch(
        self,
        argv: list[str],
        *,
        working_directory: Path,
        environment: dict[str, str] | None = None,
        confirm_seconds: float = DEFAULT_START_CONFIRM_SECONDS,
    ) -> ProcessRecord:
        """Start the trainer detached, in its own process group.

        argv, never a shell string. Every element came from Phase 25's
        command compiler, and the whole point of that compiler producing
        a list is that nothing here reassembles it into something a
        shell would parse.
        """
        if not argv:
            raise ExecutionError("cannot launch an empty command")
        for element in argv:
            if not isinstance(element, str):
                raise ExecutionError(f"argv element {element!r} is not a string")
            if "\x00" in element:
                raise ExecutionError("argv element contains a null byte")

        self.layout.ensure()
        working_directory = Path(working_directory)
        if not working_directory.is_dir():
            raise ExecutionError(f"working directory {working_directory} does not exist")

        # Append, so a relaunch after a worker restart does not destroy
        # the previous attempt's output.
        stdout = self.layout.stdout_log.open("ab")
        stderr = self.layout.stderr_log.open("ab")

        merged = dict(os.environ)
        if environment:
            merged.update(environment)

        try:
            process = subprocess.Popen(
                argv,
                cwd=str(working_directory),
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                env=merged,
                # A new session detaches the trainer from the SSH channel
                # that launched it: closing the connection must not kill
                # a run that has been going for three hours.
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            stdout.close()
            stderr.close()
            raise ExecutionError(f"could not launch the trainer: {exc}") from exc
        finally:
            stdout.close()
            stderr.close()

        # `start_new_session=True` makes the child a session leader, so
        # its group id becomes its own pid — but setsid runs in the
        # child after the fork, so reading the group immediately here
        # can still see the *launcher's* group. Recording that would be
        # dangerous rather than merely wrong: the launcher exits
        # moments later, and signalling its group id afterwards could
        # reach whatever process reused it.
        #
        # So the group is only recorded once it is observably the
        # child's own. If that never becomes true, it stays None and
        # cancellation falls back to signalling the single pid.
        group: int | None = None
        for _ in range(50):
            try:
                observed = os.getpgid(process.pid)
            except OSError:
                break
            if observed == process.pid:
                group = observed
                break
            time.sleep(0.01)

        record = ProcessRecord(
            pid=process.pid,
            process_group=group,
            argv=list(argv),
            working_directory=str(working_directory),
        )

        # A process that exits instantly did not really start. Waiting a
        # moment turns "launched, then failed" into a launch failure the
        # operator sees now rather than a RUNNING state that resolves to
        # FAILED on the next poll.
        deadline = time.monotonic() + confirm_seconds
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                record.exit_code = code
                record.finished_at = now()
                break
            time.sleep(0.05)
        return record

    # ── observe ──────────────────────────────────────────────────────
    def poll(self, record: ProcessRecord) -> ProcessRecord:
        """Refresh a record from the operating system.

        A worker that launched the trainer in an earlier invocation has
        no `Popen` to wait on, so liveness is a signal-0 probe. That
        cannot recover an exit code — the process was never this
        process's child — which is why the exit code is written by the
        invocation that owns the run and, failing that, reported unknown
        rather than assumed.
        """
        if record.exit_code is not None:
            return record
        if not process_alive(record.pid):
            record.finished_at = record.finished_at or now()
        return record

    def running(self, record: ProcessRecord | None) -> bool:
        return record is not None and record.exit_code is None and process_alive(record.pid)

    # ── stop ─────────────────────────────────────────────────────────
    def cancel(
        self, record: ProcessRecord, *, grace_seconds: float = DEFAULT_GRACE_SECONDS
    ) -> ProcessRecord:
        """Stop the trainer, gracefully, then forcibly if it will not.

        The signal goes to the process group where one is known, so
        dataloader workers stop with the trainer. Where the group is
        unknown only the PID is signalled — narrower than intended, but
        signalling a group we are not certain of could hit unrelated
        processes on a shared machine.
        """
        record.cancel_requested_at = record.cancel_requested_at or now()

        if not process_alive(record.pid):
            record.finished_at = record.finished_at or now()
            record.cancel_signal = record.cancel_signal or "NONE"
            return record

        self._signal(record, signal.SIGTERM)
        record.cancel_signal = "SIGTERM"

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not process_alive(record.pid):
                record.finished_at = now()
                return record
            time.sleep(0.1)

        # Only now, and only because it did not stop. A checkpoint being
        # written at this moment is lost, which is why the grace period
        # is generous and configurable.
        self._signal(record, signal.SIGKILL)
        record.cancel_signal = "SIGKILL"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not process_alive(record.pid):
                break
            time.sleep(0.1)
        record.finished_at = record.finished_at or now()
        return record

    def _signal(self, record: ProcessRecord, number: int) -> None:
        """Signal the trainer's group, having re-proved it is its group.

        The recorded group is checked against the live process before it
        is used. A pid can be reused, and a group id belonging to
        something else entirely would turn a cancellation into an attack
        on an unrelated process. Where the two no longer agree, only the
        pid is signalled — narrower than intended, and the only safe
        narrowing available.
        """
        target_group: int | None = None
        if record.process_group is not None and record.process_group > 0:
            try:
                if os.getpgid(record.pid) == record.process_group:
                    target_group = record.process_group
            except OSError:
                target_group = None

        try:
            if target_group is not None:
                os.killpg(target_group, number)
            else:
                os.kill(record.pid, number)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return
            if exc.errno == errno.EPERM and target_group is not None:
                # The group moved out from under us between the check
                # and the call. Fall back to the pid rather than
                # insisting on a group we can no longer vouch for.
                try:
                    os.kill(record.pid, number)
                    return
                except OSError as inner:
                    if inner.errno == errno.ESRCH:
                        return
                    raise ExecutionError(f"could not signal the trainer: {inner}") from inner
            raise ExecutionError(f"could not signal the trainer: {exc}") from exc

    def reap(self, record: ProcessRecord) -> int | None:
        """Collect an exit code where this process is the parent.

        Returns None where it is not — an honest "unknown" rather than a
        fabricated zero. A worker that restarted between launch and exit
        genuinely cannot know how the trainer finished, and reporting
        COMPLETED on that basis would be the worst available answer.
        """
        if record.exit_code is not None:
            return record.exit_code
        try:
            pid, status = os.waitpid(record.pid, os.WNOHANG)
        except ChildProcessError:
            return None
        except OSError:
            return None
        if pid == 0:
            return None
        if os.WIFEXITED(status):
            record.exit_code = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            record.exit_code = -os.WTERMSIG(status)
        record.finished_at = record.finished_at or now()
        return record.exit_code


__all__ = [
    "CLASSIFY_TAIL_BYTES",
    "CUDA_OOM_PATTERNS",
    "DEFAULT_GRACE_SECONDS",
    "DISK_FULL_PATTERNS",
    "ExecutionError",
    "ExecutionState",
    "ProcessRecord",
    "TrainerProcess",
    "classify_failure",
    "process_alive",
]
