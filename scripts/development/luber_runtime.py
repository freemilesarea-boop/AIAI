#!/usr/bin/env python
"""Start, stop and inspect the LUBER services this repository owns.

Deliberately small. It manages **the generation worker** and nothing
else, because the worker is the only service whose unattended death
silently stops the product: PostgreSQL is already a Homebrew LaunchAgent,
Redis is a system package, and the API, the web dev server and ACE-Step
are things a developer starts on purpose and wants to watch.

    scripts/development/luber_runtime.py status
    scripts/development/luber_runtime.py start
    scripts/development/luber_runtime.py stop
    scripts/development/luber_runtime.py restart
    scripts/development/luber_runtime.py plist            # print the agent
    scripts/development/luber_runtime.py plist --install  # persistent; asks first

Processes are found through the worker's own lock file, never by pattern
matching. ``pkill -f python`` is one typo away from killing an unrelated
interpreter, and this tool must not be capable of that: before signalling
anything it confirms the PID's own command line is the LUBER worker.

Nothing here installs persistent state unless ``plist --install`` is run
explicitly, and that command confirms before writing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "services" / "generation-worker" / "src"))

from luber_generation_worker.singleton import (  # noqa: E402
    SingleWorkerLock,
    log_dir,
)

LABEL = "com.luber.generation-worker"
TEMPLATE = REPO_ROOT / "deploy" / "launchd" / f"{LABEL}.plist.template"
WORKER_MODULE = "luber_generation_worker.worker.WorkerSettings"

#: How long a graceful stop is given before it is reported as failed.
#: A worker mid-generation needs a moment to record the interruption and
#: let ARQ re-queue the job; killing it sooner throws that away.
STOP_TIMEOUT_SECONDS = 30


def python_path() -> Path:
    """The venv interpreter.

    Everything runs the worker as ``python -m arq`` rather than through
    the venv's ``arq`` console script. From a shell the two are
    equivalent; under launchd they are not, and the console script fails
    before Python starts (see the launchd template for the detail).
    Using one invocation everywhere keeps the supervised and manual
    workers identical.
    """
    return REPO_ROOT / ".venv" / "bin" / "python3"


def _command_of(pid: int) -> str:
    """The full command line of a PID, or empty if it is gone."""
    try:
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def running_worker_pid() -> int | None:
    """The PID of this host's worker, verified — not merely recorded.

    The lock file names a PID; that PID is only trusted after its command
    line is confirmed to be the worker. A recycled PID belonging to some
    unrelated program must never be returned from here, because callers
    signal what this returns.
    """
    pid = SingleWorkerLock().read_holder_pid()
    if pid is None:
        return None
    command = _command_of(pid)
    if not command:
        return None
    if WORKER_MODULE in command and "arq" in command:
        return pid
    return None


def _launchd_state() -> str:
    """What launchd knows about the agent, if anything."""
    try:
        result = subprocess.run(
            ["launchctl", "list", LABEL], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "not loaded"
    return "loaded"


def cmd_status(_: argparse.Namespace) -> int:
    pid = running_worker_pid()
    lock = SingleWorkerLock()
    print(f"generation worker : {'running (pid ' + str(pid) + ')' if pid else 'not running'}")
    print(f"  lock            : {lock.path}")
    print(f"  logs            : {log_dir()}")
    print(f"  launchd agent   : {LABEL} — {_launchd_state()}")
    recorded = lock.read_holder_pid()
    if recorded is not None and pid is None:
        # Not a stale lock — the lock itself was released by the kernel.
        # Only the recorded pid is old, which is worth saying plainly so
        # nobody goes looking for a process that ended.
        print(
            f"  note            : lock file records pid {recorded}, which is no longer the worker"
        )
    print()
    print("Other services are not managed here:")
    print("  postgresql  Homebrew LaunchAgent (brew services list)")
    print("  redis       system package, started manually — see OPERATIONS_RECOVERY.md")
    print("  ace-step    manual; heavy, started deliberately")
    print("  api / web   manual during development")
    return 0 if pid else 1


def cmd_start(args: argparse.Namespace) -> int:
    existing = running_worker_pid()
    if existing:
        print(f"already running (pid {existing}) — nothing to do")
        return 0
    python = python_path()
    if not python.exists():
        print(f"interpreter not found at {python}; is the venv installed?", file=sys.stderr)
        return 1
    logs = log_dir()
    logs.mkdir(parents=True, exist_ok=True)
    out = (logs / "generation-worker.out.log").open("a")
    err = (logs / "generation-worker.err.log").open("a")
    process = subprocess.Popen(
        [str(python), "-m", "arq", WORKER_MODULE],
        cwd=REPO_ROOT,
        stdout=out,
        stderr=err,
        start_new_session=True,
    )
    # The lock is taken inside the worker's own startup, so "started"
    # means the process is up *and* won the lock — not merely spawned.
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if running_worker_pid() == process.pid:
            print(f"started (pid {process.pid}); logs in {logs}")
            return 0
        if process.poll() is not None:
            print(
                f"worker exited immediately with status {process.returncode}; "
                f"see {logs / 'generation-worker.err.log'}",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.3)
    print("worker did not report ready in time", file=sys.stderr)
    return 1


def cmd_stop(args: argparse.Namespace) -> int:
    pid = running_worker_pid()
    if pid is None:
        print("not running — nothing to do")
        return 0
    # SIGTERM, never SIGKILL: ARQ needs the signal to cancel the running
    # job cleanly so it is re-queued rather than abandoned.
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if not _command_of(pid):
            print(f"stopped (pid {pid})")
            return 0
        time.sleep(0.3)
    print(
        f"pid {pid} did not exit within {args.timeout}s. It may be finishing a "
        f"generation; wait, or force it with: kill -9 {pid}",
        file=sys.stderr,
    )
    return 1


def cmd_restart(args: argparse.Namespace) -> int:
    if cmd_stop(args) != 0:
        return 1
    return cmd_start(args)


def render_plist(*, python: Path | None = None, repo: Path | None = None) -> str:
    """Fill the launchd template with this machine's paths."""
    template = TEMPLATE.read_text()
    resolved_repo = repo or REPO_ROOT
    resolved_python = python or python_path()
    # PATH needs the venv and the usual prefixes: launchd starts agents
    # with a minimal environment, and ffmpeg lives in Homebrew's.
    path = os.pathsep.join(
        [
            str(resolved_repo / ".venv" / "bin"),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
    )
    return (
        template.replace("@LABEL@", LABEL)
        .replace("@PYTHON@", str(resolved_python))
        .replace("@REPO@", str(resolved_repo))
        .replace("@LOG_DIR@", str(log_dir()))
        .replace("@PATH@", path)
    )


def installed_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def cmd_plist(args: argparse.Namespace) -> int:
    content = render_plist()
    target = installed_plist_path()

    if not args.install and not args.uninstall:
        print(content, end="")
        print(f"\n<!-- would install to: {target} -->", file=sys.stderr)
        return 0

    if args.uninstall:
        if not target.exists():
            print(f"{target} is not installed — nothing to do")
            return 0
        subprocess.run(["launchctl", "unload", str(target)], check=False, timeout=30)
        target.unlink()
        # Only the agent file is removed. Logs, the lock file, Redis data
        # and the database are untouched: uninstalling supervision is not
        # a request to delete anything.
        print(f"unloaded and removed {target}")
        print("logs, lock file and all data were left in place")
        return 0

    print("This installs a persistent user LaunchAgent:")
    print(f"  {target}")
    print("It will start the generation worker at login and restart it if it dies.")
    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("aborted; nothing was written")
            return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    log_dir().mkdir(parents=True, exist_ok=True)
    # Idempotent: writing over an existing agent then reloading it is the
    # supported way to apply a change, so running install twice is safe.
    if target.exists():
        subprocess.run(["launchctl", "unload", str(target)], check=False, timeout=30)
    target.write_text(content)
    result = subprocess.run(
        ["launchctl", "load", "-w", str(target)], capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"launchctl load failed: {result.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"installed and loaded {target}")
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    """Check the rendered agent parses, without loading it."""
    content = render_plist()
    plutil = shutil.which("plutil")
    if plutil is None:
        print("plutil not available; skipped syntax check", file=sys.stderr)
        return 1
    result = subprocess.run(
        [plutil, "-lint", "-"], input=content, capture_output=True, text=True, timeout=30
    )
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("status", cmd_status, "show worker state and who owns everything else"),
        ("start", cmd_start, "start the generation worker if it is not running"),
        ("stop", cmd_stop, "stop the generation worker gracefully (SIGTERM)"),
        ("restart", cmd_restart, "stop then start"),
        ("validate", cmd_validate, "check the launchd agent parses, without installing"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(handler=handler)
        if name in {"start", "stop", "restart"}:
            p.add_argument("--timeout", type=float, default=STOP_TIMEOUT_SECONDS)

    p = sub.add_parser("plist", help="print, install or uninstall the launchd agent")
    p.set_defaults(handler=cmd_plist)
    p.add_argument("--install", action="store_true", help="write and load it (persistent)")
    p.add_argument("--uninstall", action="store_true", help="unload and remove it")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
