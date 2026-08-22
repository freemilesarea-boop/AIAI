"""Operator CLI: what is here, what can run where, and what nobody knows.

Six verbs, all read-only. Nothing in this file trains anything, moves a
workload, downloads a weight or changes a configuration — it probes,
resolves and prints.

The flag worth knowing is `--python`. LUBER's own environment has no
torch; the ACE-Step trainer's does. Without naming an interpreter, every
verb answers about the control plane, which on this topology means "CPU
only, and that is correct". Pointing it at the trainer's Python is how
an operator learns what the machine can actually do:

    python -m luber_hardware probe --python ~/ace-step-1.5/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from luber_hardware import _smoke
from luber_hardware.capability import MachineCapability
from luber_hardware.devices import ComputePreference, Precision
from luber_hardware.placement import ExecutionTarget, PlacementRequest, place
from luber_hardware.probe import ProbeError, probe_machine
from luber_hardware.profiles import planned_cuda_worker, planned_mac_mini_24gb
from luber_hardware.readiness import readiness
from luber_hardware.report import build_report
from luber_hardware.workloads import WorkloadClass

#: How long the out-of-process smoke may take. It is eight steps on a
#: toy network; a minute is generous and a hang should not be silent.
SMOKE_TIMEOUT_SECONDS = 600.0


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _capability(args: argparse.Namespace) -> MachineCapability:
    return probe_machine(args.python, label=args.label)


def _targets(args: argparse.Namespace, capability: MachineCapability) -> list[ExecutionTarget]:
    """This machine, plus the planned hardware if it was asked for.

    Planned targets are opt-in. A readiness view that silently included
    a machine nobody owns would be planning material pretending to be a
    status report.
    """
    targets = [
        ExecutionTarget("this-machine", capability, runs_control_plane=True),
    ]
    if getattr(args, "include_planned", False):
        targets.append(ExecutionTarget("planned-mac-mini", planned_mac_mini_24gb()))
        targets.append(
            ExecutionTarget(
                "planned-cuda-worker",
                planned_cuda_worker(),
                location="REMOTE",
                worker_id="PLANNED",
            )
        )
    return targets


# ── verbs ────────────────────────────────────────────────────────────


def cmd_probe(args: argparse.Namespace) -> int:
    """What this machine is, as some interpreter can see it."""
    capability = _capability(args)
    if args.json:
        _print(capability.to_dict())
    else:
        print(capability.render())
    return 0


def cmd_compatibility(args: argparse.Namespace) -> int:
    """Every device, every precision, and what has not been measured."""
    capability = _capability(args)
    report = build_report(capability, targets=_targets(args, capability), smoke=_maybe_smoke(args))
    if args.markdown:
        print(report.to_markdown())
    elif args.json:
        print(report.to_json())
    else:
        for finding in report.findings:
            print(f"{finding.evidence:10} {finding.subject:34} {finding.verdict}")
            if finding.detail:
                print(f"           {finding.detail}")
    return 0


def cmd_placement(args: argparse.Namespace) -> int:
    """Where one workload would run, and why."""
    capability = _capability(args)
    decision = place(
        PlacementRequest(
            workload=args.workload,
            policy=args.policy,
            preference=args.device,
            precision=args.precision,
            required_operations=tuple(args.requires or ()),
            allow_local_fallback=args.allow_local_fallback,
        ),
        _targets(args, capability),
    )
    if args.json:
        _print(decision.to_dict())
    else:
        print(decision.render())
        if decision.placed:
            print(f"  reason: {decision.reason}")
        for item in decision.unknowns:
            print(f"  unknown: {item}")
    return 0 if decision.placed else 1


def cmd_readiness(args: argparse.Namespace) -> int:
    """What each compute target can be asked to do right now."""
    capability = _capability(args)
    report = readiness(_targets(args, capability))
    if args.json:
        _print(report.to_dict())
    else:
        print(report.render())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Actually run a tiny training step on each available device.

    The only verb that computes anything. A toy network on synthetic
    noise: no model weights, no music, nothing downloaded. It answers
    whether the mechanism works, and nothing about the model.
    """
    result = _run_smoke(args)
    if args.json:
        _print(result)
        return 0 if result.get("ok") else 1

    if not result.get("torch_installed"):
        print("torch is not installed in that interpreter, so nothing could be run.")
        print("Point --python at the environment that runs training.")
        return 1

    for device, entry in sorted((result.get("results") or {}).items()):
        status = "PASS" if entry.get("ok") else "FAIL"
        print(f"{device}: {status}")
        if entry.get("error"):
            print(f"    {entry['error']}")
            continue
        training = entry.get("training", {})
        print(
            f"    training: {training.get('steps')} steps, "
            f"loss {training.get('first_loss', 0):.4f} -> {training.get('last_loss', 0):.4f}"
        )
        for target, answer in sorted((entry.get("checkpoint", {}).get("loads") or {}).items()):
            mark = "ok" if answer.get("ok") else f"FAILED ({answer.get('error')})"
            print(f"    checkpoint -> {target}: {mark}")
        benchmark = entry.get("benchmark", {})
        if benchmark:
            print(
                f"    benchmark: matmul {benchmark.get('matmul_ms', 0):.3f} ms, "
                f"fwd+bwd {benchmark.get('forward_backward_ms', 0):.3f} ms "
                "(this machine only)"
            )
    return 0 if result.get("ok") else 1


def cmd_report(args: argparse.Namespace) -> int:
    """Write the compatibility report to disk, in both forms."""
    capability = _capability(args)
    report = build_report(capability, targets=_targets(args, capability), smoke=_maybe_smoke(args))

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "hardware_compatibility_report.json"
    markdown_path = destination / "hardware_compatibility_report.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


# ── the smoke, out of process ────────────────────────────────────────


def _maybe_smoke(args: argparse.Namespace) -> dict[str, Any] | None:
    return _run_smoke(args) if getattr(args, "run_smoke", False) else None


def _run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run `_smoke.py` under the named interpreter and read its JSON.

    Out of process for the same reason the probe is: the interpreter
    that can train is not the one running this CLI.
    """
    interpreter = args.python or sys.executable
    script = Path(_smoke.__file__).resolve()
    devices = list(getattr(args, "devices", None) or [])
    try:
        completed = subprocess.run(
            [str(interpreter), str(script), *devices],
            capture_output=True,
            text=True,
            timeout=SMOKE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"torch_installed": False, "ok": False, "error": str(exc)}
    if completed.returncode != 0:
        return {
            "torch_installed": False,
            "ok": False,
            "error": (completed.stderr or "").strip()[-500:],
        }
    try:
        return dict(json.loads(completed.stdout))
    except ValueError:
        return {"torch_installed": False, "ok": False, "error": "no JSON on stdout"}


# ── parsing ──────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "interpreter to probe. LUBER's own environment has no torch; point this at "
            "the one that runs training"
        ),
    )
    parser.add_argument("--label", default=None, help="a name for this target in the output")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def _add_planned(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-planned",
        action="store_true",
        help=(
            "include the planned 24 GB Mac mini and a planned CUDA worker. Off by "
            "default: hardware nobody owns does not belong in a status view"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="luber-hardware",
        description=(
            "Hardware capability, device resolution and execution placement. Probes and "
            "reports; trains nothing and moves nothing."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="what this machine is")
    _add_common(probe)
    probe.set_defaults(handler=cmd_probe)

    compatibility = subparsers.add_parser(
        "compatibility", help="every device and precision, with provenance"
    )
    _add_common(compatibility)
    _add_planned(compatibility)
    compatibility.add_argument("--markdown", action="store_true")
    compatibility.add_argument(
        "--run-smoke",
        action="store_true",
        help="also run the tiny training smoke and include its result",
    )
    compatibility.set_defaults(handler=cmd_compatibility, devices=None)

    placement = subparsers.add_parser("placement", help="where one workload would run")
    _add_common(placement)
    _add_planned(placement)
    placement.add_argument(
        "--workload",
        default=WorkloadClass.HEAVY_TRAINING.value,
        choices=sorted(item.value for item in WorkloadClass),
    )
    placement.add_argument("--policy", default=None)
    placement.add_argument(
        "--device",
        default=ComputePreference.AUTO.value,
        choices=sorted(item.value for item in ComputePreference),
    )
    placement.add_argument(
        "--precision",
        default=Precision.AUTO.value,
        choices=sorted(item.value for item in Precision),
    )
    placement.add_argument(
        "--requires",
        action="append",
        help="an operation this run needs, e.g. adamw8bit. May be repeated",
    )
    placement.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help=(
            "permit a remote-CUDA-preferring workload to run locally. Off by default: "
            "this is the flag that stops a GPU job silently becoming a Mac job"
        ),
    )
    placement.set_defaults(handler=cmd_placement)

    ready = subparsers.add_parser("readiness", help="what each compute target can take")
    _add_common(ready)
    _add_planned(ready)
    ready.set_defaults(handler=cmd_readiness)

    verify = subparsers.add_parser(
        "verify", help="run a tiny training step on each available device"
    )
    _add_common(verify)
    verify.add_argument(
        "devices",
        nargs="*",
        help="devices to try (cpu, mps, cuda). Default: every available device",
    )
    verify.set_defaults(handler=cmd_verify)

    report = subparsers.add_parser("report", help="write the compatibility report")
    _add_common(report)
    _add_planned(report)
    report.add_argument("--out", default="artifacts/hardware", help="directory to write into")
    report.add_argument("--run-smoke", action="store_true")
    report.set_defaults(handler=cmd_report, devices=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ProbeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
