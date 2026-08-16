"""One generation worker per machine, enforced by the operating system.

Two workers on one queue will not process the same job twice — ARQ claims
each job in Redis before starting it — but they will run two inferences
at once against hardware sized for one. On this Mac that means two model
processes competing for the same memory.

The guard is an advisory lock (``flock``) held on a file for as long as
the process lives. Three properties make it the right size for the job:

* The kernel releases it when the process dies, however it dies. A PID
  file has to be validated and can go stale; a lock cannot.
* It is **per host**, which is exactly the scope of the problem. A
  future deployment running one worker on each of five machines is a
  legitimate topology, and a lock in Redis or the database would forbid
  it. This one cannot: each host has its own file.
* It needs no daemon, no dependency, and no cleanup path.

What it deliberately does not do is give a worker an identity across
hosts. If two machines are ever meant to share one queue, the thing that
keeps them honest is ARQ's per-job claim, not this.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType

#: Exit status when another worker already holds the lock. Distinct from
#: 1 so a supervisor's logs separate "already running" — which is
#: correct and needs no action — from a real startup failure.
EXIT_ALREADY_RUNNING = 3


def runtime_dir() -> Path:
    """Where LUBER keeps per-user runtime state (locks, pid files, logs).

    Under ``$HOME`` rather than ``/var/run``: no elevation needed, and
    nothing here is machine-wide. ``XDG_RUNTIME_DIR`` is honoured when
    set, which keeps the Linux deployment path identical.
    """
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(base) if base else Path.home() / ".luber"
    return root / "run"


def log_dir() -> Path:
    return Path(os.environ.get("LUBER_LOG_DIR") or (Path.home() / ".luber" / "log"))


class WorkerAlreadyRunningError(RuntimeError):
    """Raised when another generation worker holds the lock on this host."""

    def __init__(self, lock_path: Path, holder_pid: int | None) -> None:
        self.lock_path = lock_path
        self.holder_pid = holder_pid
        who = f"pid {holder_pid}" if holder_pid else "another process"
        super().__init__(
            f"a generation worker is already running on this machine ({who}). "
            f"Stop it before starting another, or use the supervised service."
        )


class SingleWorkerLock:
    """Hold the host's generation-worker lock for the life of a process.

    Usable as a context manager or acquired directly. The PID is written
    into the locked file purely so an operator can see *who* holds it;
    the lock itself, not the contents, is what enforces exclusion.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        # Coerced rather than assumed: a path arriving from a config
        # value or a script argument is a string, and failing on it deep
        # inside acquire() would be a confusing way to find that out.
        self.path = Path(path) if path is not None else runtime_dir() / "generation-worker.lock"
        self._handle: object | None = None

    def read_holder_pid(self) -> int | None:
        """The PID recorded in the lock file, if it is readable."""
        try:
            text = self.path.read_text().strip()
        except OSError:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Opened without truncating: the previous holder's pid stays
        # readable until this process wins the lock and rewrites it.
        handle = self.path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = self.read_holder_pid()
            handle.close()
            raise WorkerAlreadyRunningError(self.path, holder) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        except OSError:
            pass
        finally:
            handle.close()  # type: ignore[attr-defined]
        # The file is left behind on purpose: it records the last holder,
        # and an empty lock file is not a stale lock.

    def __enter__(self) -> SingleWorkerLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
