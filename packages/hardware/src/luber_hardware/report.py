"""The compatibility report, in two forms and with four kinds of claim.

A hardware report is read by somebody deciding what to buy and what to
rent. That makes the *provenance* of each line more important than the
line itself, so every claim carries how it was established:

- **VERIFIED** — measured on this machine, now, by running it.
- **SYNTHETIC** — decision logic exercised against a fixture. Real code,
  hypothetical hardware.
- **NOT_RUN** — needs hardware nobody has. Not a failure and not a pass.
- **PLANNED** — a profile for a machine that does not exist yet.

The distinction is the whole document. A report where "MPS training
works" and "CUDA training works" look alike would be worse than no
report, because only one of them was tried.

Nothing host-specific goes in. No username, no home directory, no
hostname, no serial. The capability model has no field for any of them,
and the markdown is rendered from that model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_hardware.capability import UNKNOWN, MachineCapability
from luber_hardware.memory import LOCAL_TRAINING_CONCURRENCY, budget_for
from luber_hardware.placement import ExecutionTarget, PlacementRequest, place
from luber_hardware.precision import supported_precisions
from luber_hardware.readiness import TrainingExecutionReadiness, readiness
from luber_hardware.versions import version_block
from luber_hardware.workloads import WorkloadClass


class Evidence(StrEnum):
    """How a claim in this report was established."""

    VERIFIED = "VERIFIED"
    SYNTHETIC = "SYNTHETIC"
    NOT_RUN = "NOT_RUN"
    PLANNED = "PLANNED"


@dataclass(frozen=True)
class Finding:
    """One line of the report, with its provenance attached."""

    subject: str
    verdict: str
    evidence: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "detail": self.detail,
        }


@dataclass
class CompatibilityReport:
    """Everything known about where this deployment can run things."""

    at: datetime
    capability: MachineCapability
    readiness: TrainingExecutionReadiness
    findings: list[Finding] = field(default_factory=list)
    smoke: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "at": self.at.isoformat(),
            "machine": self.capability.to_dict(),
            "readiness": self.readiness.to_dict(),
            "findings": [item.to_dict() for item in self.findings],
            "smoke": self.smoke,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        return _markdown(self)


def build_report(
    capability: MachineCapability,
    *,
    targets: list[ExecutionTarget] | None = None,
    smoke: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> CompatibilityReport:
    """Assemble a report from a probe, and from a smoke run if one exists."""
    moment = now or datetime.now(UTC)
    all_targets = targets or [ExecutionTarget("this-machine", capability, runs_control_plane=True)]
    report = CompatibilityReport(
        at=moment,
        capability=capability,
        readiness=readiness(all_targets, now=moment),
        smoke=smoke,
    )
    report.findings = _findings(capability, all_targets, smoke)
    return report


def _findings(
    capability: MachineCapability,
    targets: list[ExecutionTarget],
    smoke: dict[str, Any] | None,
) -> list[Finding]:
    findings: list[Finding] = []

    # ── the probe itself ─────────────────────────────────────────────
    if not capability.torch_installed:
        findings.append(
            Finding(
                subject="torch",
                verdict="NOT INSTALLED in the interpreter that answered",
                evidence=Evidence.VERIFIED.value,
                detail=(
                    "This is normal for the control plane, which never imports torch. "
                    "Probe the interpreter that runs training to learn what the machine "
                    "can do."
                ),
            )
        )
    else:
        findings.append(
            Finding(
                subject="torch",
                verdict=capability.torch_version or UNKNOWN,
                evidence=Evidence.VERIFIED.value,
            )
        )

    findings.append(
        Finding(
            subject="Apple MPS",
            verdict=_availability(capability.mps_available, capability.mps_built),
            evidence=(
                Evidence.VERIFIED.value if capability.torch_installed else Evidence.NOT_RUN.value
            ),
            detail=(
                "torch.backends.mps.is_built() and .is_available(), not inferred from sys.platform"
            ),
        )
    )
    findings.append(
        Finding(
            subject="NVIDIA CUDA",
            verdict=(
                f"{capability.cuda_device_name} x {capability.cuda_device_count}"
                if capability.cuda_available
                else "NOT AVAILABLE on this machine"
            ),
            evidence=(
                Evidence.VERIFIED.value if capability.torch_installed else Evidence.NOT_RUN.value
            ),
            detail=(
                ""
                if capability.cuda_available
                else "No NVIDIA hardware is attached to this project. Every CUDA claim in "
                "this report is synthetic or not run."
            ),
        )
    )

    # ── precision ────────────────────────────────────────────────────
    for device in capability.devices():
        measured = supported_precisions(capability, device)
        findings.append(
            Finding(
                subject=f"precision on {device}",
                verdict=", ".join(measured) if measured else UNKNOWN,
                evidence=(Evidence.VERIFIED.value if measured else Evidence.NOT_RUN.value),
                detail=(
                    "each dtype allocated on the device and added to itself"
                    if measured
                    else "no probe has measured dtypes on this device"
                ),
            )
        )

    # ── the smoke, if it ran ─────────────────────────────────────────
    if smoke and smoke.get("torch_installed"):
        for device, result in sorted((smoke.get("results") or {}).items()):
            training = result.get("training") or {}
            findings.append(
                Finding(
                    subject=f"tiny training on {device}",
                    verdict="PASS" if result.get("ok") else "FAIL",
                    evidence=Evidence.VERIFIED.value,
                    detail=(
                        f"{training.get('steps', '?')} steps of forward, backward, AdamW, "
                        f"scheduler and gradient clip on a toy network — not ACE-Step, "
                        f"no music, no model weights"
                    ),
                )
            )
            loads = (result.get("checkpoint") or {}).get("loads") or {}
            if loads:
                targets_ok = [name for name, item in loads.items() if item.get("ok")]
                findings.append(
                    Finding(
                        subject=f"checkpoint written on {device}",
                        verdict=f"loads on {', '.join(sorted(targets_ok))}"
                        if targets_ok
                        else "FAILED to load",
                        evidence=Evidence.VERIFIED.value,
                        detail="model, optimizer state and a LoRA-shaped tensor pair",
                    )
                )
            benchmark = result.get("benchmark") or {}
            if benchmark:
                findings.append(
                    Finding(
                        subject=f"tiny benchmark on {device}",
                        verdict=(
                            f"matmul({benchmark.get('matmul_size')}) "
                            f"{benchmark.get('matmul_ms', 0):.3f} ms, "
                            f"fwd+bwd {benchmark.get('forward_backward_ms', 0):.3f} ms"
                        ),
                        evidence=Evidence.VERIFIED.value,
                        detail=(
                            "this machine only. Not comparable to other hardware and not "
                            "extrapolable to a different Mac or to any NVIDIA GPU."
                        ),
                    )
                )
    else:
        findings.append(
            Finding(
                subject="tiny training smoke",
                verdict="NOT RUN",
                evidence=Evidence.NOT_RUN.value,
                detail="no interpreter with torch was available to run it",
            )
        )

    # ── placement ────────────────────────────────────────────────────
    for workload in WorkloadClass:
        decision = place(PlacementRequest(workload=workload.value), targets)
        findings.append(
            Finding(
                subject=f"placement: {workload.value}",
                verdict=(
                    f"{decision.execution_location} + {decision.compute_device}"
                    if decision.placed
                    else decision.outcome
                ),
                # VERIFIED covers a refusal too. The claim being made
                # is "this workload cannot be placed", and that was
                # computed from a real probe by the same policy the
                # scheduler uses. NOT_RUN would read as "we did not
                # check", which is a different and false statement.
                evidence=(
                    Evidence.PLANNED.value if decision.planned_target else Evidence.VERIFIED.value
                ),
                detail=decision.reason,
            )
        )

    # ── memory ───────────────────────────────────────────────────────
    budget = budget_for(capability.memory_total_mb, shared_with_control_plane=True)
    findings.append(
        Finding(
            subject="memory budget",
            verdict=(
                f"{budget.usable_mb()} MB usable of {budget.total_mb} MB"
                if budget.usable_mb() is not None
                else UNKNOWN
            ),
            evidence=Evidence.VERIFIED.value,
            detail=(
                f"{budget.reserved_mb()} MB held back for the operating system and the "
                "control plane. No figure has been measured for what LUBER training "
                "actually needs, so whether a real run fits is UNKNOWN."
            ),
        )
    )
    return findings


def _availability(available: bool | None, built: bool | None) -> str:
    if available:
        return "AVAILABLE"
    if built is False:
        return "NOT BUILT into this torch"
    if available is False:
        return "NOT AVAILABLE"
    return UNKNOWN


def _markdown(report: CompatibilityReport) -> str:
    capability = report.capability
    lines: list[str] = [
        "# Hardware compatibility report",
        "",
        f"Generated {report.at.isoformat()}.",
        "",
        "Every line carries how it was established. **VERIFIED** was measured here and "
        "now; **SYNTHETIC** exercised real decision logic against a fixture; "
        "**NOT_RUN** needs hardware this project does not have; **PLANNED** describes a "
        "machine that does not exist yet.",
        "",
        "## Machine",
        "",
        "| Fact | Value |",
        "|---|---|",
        f"| Label | {capability.label} |",
        f"| Platform | {capability.system or UNKNOWN} ({capability.architecture or UNKNOWN}) |",
        f"| CPU | {capability.cpu_model or UNKNOWN} |",
        f"| Cores | {capability.cpu_count or UNKNOWN} |",
        f"| Memory | {_gib(capability.memory_total_mb)} |",
        f"| Python | {capability.python_version or UNKNOWN} |",
        f"| torch | {capability.torch_version or 'NOT INSTALLED'} |",
        f"| MPS built / available | {_tri(capability.mps_built)} / "
        f"{_tri(capability.mps_available)} |",
        f"| CUDA available | {_tri(capability.cuda_available)} |",
        f"| Capability digest | `{capability.digest()[:16]}` |",
        "",
    ]
    if capability.planned:
        lines.append("> **PLANNED PROFILE** — no machine matching this has been probed.")
        lines.append("")

    lines += ["## Findings", "", "| Subject | Verdict | Evidence | Detail |", "|---|---|---|---|"]
    for finding in report.findings:
        detail = finding.detail.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {finding.subject} | {finding.verdict} | {finding.evidence} | {detail} |")

    lines += ["", "## Execution readiness", "", "```", report.readiness.render(), "```", ""]
    lines += [
        "## Local training policy",
        "",
        f"- Concurrent local training jobs: **{LOCAL_TRAINING_CONCURRENCY}**. The machine "
        "that runs the control plane stays a control plane.",
        "- Memory is never planned to 100%. A headroom fraction and a floor are held back, "
        "and both are configurable.",
        "- Nothing has measured what LUBER training needs, so memory feasibility is "
        "`UNKNOWN` rather than a number.",
        "",
        "## What this report does not say",
        "",
        "- It makes no performance claim about any machine other than the one probed.",
        "- It does not compare Apple silicon with NVIDIA. No NVIDIA hardware was measured.",
        "- It says nothing about whether ACE-Step LoRA training *converges* on any device. "
        "The smoke test trains a toy network, not a music model.",
        "- Generation still runs through the local ARQ worker. Remote CUDA training exists; "
        "remote generation does not.",
    ]
    return "\n".join(lines) + "\n"


def _gib(value: int | None) -> str:
    return UNKNOWN if value is None else f"{value / 1024:.1f} GiB ({value} MB)"


def _tri(value: bool | None) -> str:
    return UNKNOWN if value is None else ("yes" if value else "no")


__all__ = [
    "CompatibilityReport",
    "Evidence",
    "Finding",
    "build_report",
]
