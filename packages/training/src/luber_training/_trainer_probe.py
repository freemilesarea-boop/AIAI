"""Ask the trainer's own interpreter what it can do. No LUBER imports.

Same shape and the same reason as ``luber_hardware/_facts.py``: LUBER's
environment has no torch and no ACE-Step, and the trainer's environment
has both. A check that could only run in this process would answer about
the wrong Python.

So this file is executed *by the trainer's interpreter*, from the
trainer's own directory, reads one JSON request on stdin and prints one
JSON document on stdout. It never trains, never loads a model, never
downloads anything and never touches a dataset.

The interesting question it answers is whether the compiled command
would actually start. Offering the argv to the installed parser catches
two failures nothing else does:

* a flag this LUBER emits that the installed trainer has never heard of
* an argv that parses but leaves ``--yes`` unset, which means the
  trainer stops at an interactive confirmation prompt — and with stdin
  closed, as a detached launch has it, ``confirm_start`` returns False
  and the process exits **zero** having trained nothing

The second one looks exactly like a successful run from the outside.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any

#: Bump when the shape of the emitted document changes.
TRAINER_PROBE_VERSION = "luber-trainer-probe/1"


def _torch() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        out["torch_importable"] = False
        out["torch_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["torch_importable"] = True
    out["torch_version"] = str(torch.__version__)
    return out


def _packages(names: list[str]) -> dict[str, Any]:
    """Really import each name, because that is what the trainer does.

    ``find_spec`` would be cheaper and would answer a different
    question: ACE-Step's optimizer factory catches ``ImportError`` from
    a real import and substitutes AdamW, so a package that is present
    but unimportable has exactly the same effect as an absent one.
    """
    import importlib

    results: dict[str, Any] = {}
    for name in names:
        try:
            importlib.import_module(name)
            results[name] = {"importable": True}
        except Exception as exc:
            results[name] = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


def _command(argv: list[str]) -> dict[str, Any]:
    """Offer an argv to the installed parser without running anything."""
    out: dict[str, Any] = {}
    try:
        from acestep.training_v2.cli.common import (  # type: ignore[import-not-found]
            build_root_parser,
        )
    except Exception as exc:
        out["command_accepted"] = None
        out["command_detail"] = f"the trainer's argument parser could not be imported: {exc}"
        return out

    parser = build_root_parser()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            namespace = parser.parse_args(argv)
    except SystemExit:
        message = stderr.getvalue().strip().splitlines()
        out["command_accepted"] = False
        out["command_detail"] = (
            message[-1] if message else "the installed trainer rejected the compiled command"
        )
        return out
    except Exception as exc:
        out["command_accepted"] = False
        out["command_detail"] = f"{type(exc).__name__}: {exc}"
        return out

    assume_yes = bool(getattr(namespace, "yes", False))
    out["command_accepted"] = assume_yes
    out["parsed"] = {
        "subcommand": getattr(namespace, "subcommand", None),
        "device": getattr(namespace, "device", None),
        "precision": getattr(namespace, "precision", None),
        "optimizer_type": getattr(namespace, "optimizer_type", None),
        "epochs": getattr(namespace, "epochs", None),
        "dataset_dir": getattr(namespace, "dataset_dir", None),
        "output_dir": getattr(namespace, "output_dir", None),
        "checkpoint_dir": getattr(namespace, "checkpoint_dir", None),
        "resume_from": getattr(namespace, "resume_from", None),
        "yes": assume_yes,
    }
    out["command_detail"] = (
        "the installed trainer accepts every flag in the compiled command"
        if assume_yes
        else (
            "the command parses but does not set --yes, so the trainer stops at an "
            "interactive confirmation. Launched with stdin closed it exits 0 without "
            "training, which is indistinguishable from success"
        )
    )
    return out


def run(request: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"probe_version": TRAINER_PROBE_VERSION}
    out.update(_torch())
    out["packages"] = _packages([str(name) for name in (request.get("packages") or [])])
    argv = [str(item) for item in (request.get("argv") or [])]
    if argv:
        out.update(_command(argv))
    else:
        out["command_accepted"] = None
        out["command_detail"] = "no command was offered for checking"
    return out


def main() -> int:
    try:
        raw = sys.stdin.read()
        request = json.loads(raw) if raw.strip() else {}
    except ValueError:
        request = {}
    if not isinstance(request, dict):
        request = {}
    print(json.dumps(run(request), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
