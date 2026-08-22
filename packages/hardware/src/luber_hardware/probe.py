"""Asking a machine what it can do — including a machine's *other* Python.

The interesting case is not "probe this process". It is: the control
plane's environment has no `torch`, and the ACE-Step trainer's
environment has torch 2.10 with a working MPS backend. Both are on the
same Mac. A probe that could only ask its own interpreter would report
"no accelerator" on the machine that can train, which is the wrong
answer to the only question worth asking.

So `probe_machine` takes an optional interpreter path and runs
`_facts.py` — the same file, not a copy — under it. One implementation,
two callers, no drift.

Nothing here reads an environment variable, a hostname or a user
directory. The subprocess is invoked with a fixed argument list and its
stdout is parsed as one JSON document; anything else is a failed probe
rather than a partial one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from luber_hardware import _facts
from luber_hardware.capability import MachineCapability, capability_from_facts

#: How long another interpreter has to answer. Importing torch is not
#: instant — it is seconds, not milliseconds — but it is not a minute
#: either, and a hung probe must not hang an operator's terminal.
PROBE_TIMEOUT_SECONDS = 120.0


class ProbeError(RuntimeError):
    """The interpreter could not be asked, or did not answer usefully."""


def collect_facts(python_executable: str | Path | None = None) -> dict[str, Any]:
    """Raw facts from this interpreter, or from a named one.

    ``None`` means this process. Anything else is executed as a
    subprocess — including `sys.executable`, if a caller passes it
    explicitly, because a caller that names an interpreter is asking for
    the out-of-process path and should get it.
    """
    if python_executable is None:
        return _facts.collect()

    script = Path(_facts.__file__).resolve()
    if not script.is_file():
        # Installed from a zip or otherwise not on disk. Better to say
        # so than to fall back to this process and answer about the
        # wrong machine's Python.
        raise ProbeError(
            f"the fact collector is not available as a file at {script}, so another "
            "interpreter cannot be asked"
        )

    try:
        completed = subprocess.run(
            [str(python_executable), str(script)],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProbeError(f"no interpreter at {python_executable}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(
            f"{python_executable} did not answer within {PROBE_TIMEOUT_SECONDS:.0f}s"
        ) from exc
    except OSError as exc:
        raise ProbeError(f"could not run {python_executable}: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise ProbeError(
            f"{python_executable} exited {completed.returncode} while probing"
            + (f": {detail[-1]}" if detail else "")
        )
    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise ProbeError(f"{python_executable} did not print a JSON fact document") from exc
    if not isinstance(payload, dict):
        raise ProbeError(f"{python_executable} printed {type(payload).__name__}, not an object")
    return payload


def probe_machine(
    python_executable: str | Path | None = None,
    *,
    label: str | None = None,
    location: str = "LOCAL",
) -> MachineCapability:
    """This machine's capability, as some interpreter can see it."""
    return capability_from_facts(collect_facts(python_executable), label=label, location=location)


def probe_this_process() -> MachineCapability:
    """Capability as the *running* interpreter sees it.

    Named separately from `probe_machine()` because the difference
    matters when reading a report: this one describes what the control
    plane itself can do, which on a machine with a separate trainer
    environment is usually "CPU only, and that is fine".
    """
    return probe_machine(None)


def default_trainer_interpreter() -> str:
    """The interpreter to probe when nobody names one.

    `sys.executable` — this process's own Python. Deliberately not a
    guess at where a trainer virtualenv might live: searching the
    filesystem for a `.venv` would sometimes find the wrong one, and a
    capability report that silently described a different environment
    is worse than one that describes this one honestly.
    """
    return sys.executable


__all__ = [
    "PROBE_TIMEOUT_SECONDS",
    "ProbeError",
    "collect_facts",
    "default_trainer_interpreter",
    "probe_machine",
    "probe_this_process",
]
