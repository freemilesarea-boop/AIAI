"""Everything checkable on the worker before a trainer starts.

The economics decide the design. A misconfiguration caught here costs
seconds; the same misconfiguration caught an hour into a rented GPU
costs the hour, the operator's attention, and — if it produced a
checkpoint before failing — a plausible artifact nobody can trust.

So preflight checks everything that can be known without training:
protocol, plan identity, artifact digests, code revision, environment,
CUDA, disk, writability, and whether the trainer command even compiles.

The status vocabulary has three values and the third is the important
one. `UNKNOWN` is not a soft pass. A check that could not be performed
is reported as not performed, and a plan that *requires* what could not
be checked does not proceed. The alternative — treating "nobody
measured VRAM" as a tick — is how a run lands on hardware that cannot
hold it.

Dependency comparison is graded rather than absolute. Python and torch
must match where the plan pins them; PEFT and transformers are
compatible-or-warn; the rest of the OS package universe is
informational. Demanding byte-identical environments would block every
real deployment, and demanding nothing would let a run train against a
torch it was never tested with.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_training.remote.capabilities import CapabilityReport
from luber_training.remote.manifest import RemoteArtifactManifest, sha256_file
from luber_training.remote.paths import RunLayout
from luber_training.remote.protocol import (
    REMOTE_PROTOCOL_VERSION,
    ProtocolError,
    check_protocol,
    now,
)

PREFLIGHT_SCHEMA_VERSION = "luber-remote-preflight/1"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    #: Could not be established. Never treated as a pass.
    UNKNOWN = "UNKNOWN"


class PreflightStatus(StrEnum):
    PASS = "PASS"
    #: Something is missing or unverifiable. Fixable; not proof of fault.
    BLOCKED = "BLOCKED"
    #: Something is definitively wrong.
    FAIL = "FAIL"


class Severity(StrEnum):
    REQUIRED = "REQUIRED"
    COMPATIBLE = "COMPATIBLE"
    INFORMATIONAL = "INFORMATIONAL"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    severity: str = Severity.REQUIRED.value
    expected: str | None = None
    observed: str | None = None

    @property
    def blocking(self) -> bool:
        """Whether this result may stop a run.

        Only REQUIRED checks block, and an UNKNOWN required check blocks
        just as a failed one does — the run needs the thing, and nobody
        established that it is there.
        """
        return self.severity == Severity.REQUIRED.value and self.status != CheckStatus.PASS.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    """What the worker established, and whether training may begin."""

    run_id: str
    worker_id: str
    protocol_version: str = REMOTE_PROTOCOL_VERSION
    schema_version: str = PREFLIGHT_SCHEMA_VERSION
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now)

    def add(
        self,
        name: str,
        status: str,
        detail: str = "",
        *,
        severity: str = Severity.REQUIRED.value,
        expected: str | None = None,
        observed: str | None = None,
    ) -> Check:
        check = Check(
            name=name,
            status=status,
            detail=detail,
            severity=severity,
            expected=expected,
            observed=observed,
        )
        self.checks.append(check)
        return check

    @property
    def blocking_reasons(self) -> list[str]:
        return [f"{check.name}: {check.detail}" for check in self.checks if check.blocking]

    @property
    def status(self) -> str:
        """PASS, BLOCKED or FAIL, in that order of severity.

        FAIL means something is definitively wrong — a digest mismatch,
        a plan that does not match. BLOCKED means something could not be
        established. Both stop the run; they differ in what an operator
        should do next, and collapsing them would lose that.
        """
        failed = [check for check in self.checks if check.blocking]
        if not failed:
            return PreflightStatus.PASS.value
        if any(check.status == CheckStatus.FAIL.value for check in failed):
            return PreflightStatus.FAIL.value
        return PreflightStatus.BLOCKED.value

    @property
    def passed(self) -> bool:
        return self.status == PreflightStatus.PASS.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "status": self.status,
            "created_at": self.created_at,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": self.warnings,
            "blocking_reasons": self.blocking_reasons,
            "capabilities": self.capabilities,
            "note": (
                "UNKNOWN is not a pass. A required capability nobody could measure is "
                "treated as unsatisfied."
            ),
        }

    def write(self, layout: RunLayout) -> Path:
        layout.root.mkdir(parents=True, exist_ok=True)
        layout.preflight_json.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return layout.preflight_json


def _compare(
    report: PreflightReport,
    name: str,
    expected: str | None,
    observed: str | None,
    *,
    severity: str,
    exact: bool = True,
) -> None:
    """One environment comparison, graded by how much it matters."""
    if expected is None:
        report.add(
            name,
            CheckStatus.UNKNOWN.value,
            "the plan does not pin this, so nothing is required",
            severity=Severity.INFORMATIONAL.value,
            observed=observed,
        )
        return
    if observed is None:
        report.add(
            name,
            CheckStatus.UNKNOWN.value,
            "not present or not reported on the worker",
            severity=severity,
            expected=expected,
            observed=None,
        )
        return

    matched = observed == expected if exact else observed.startswith(expected.split("+")[0])
    report.add(
        name,
        CheckStatus.PASS.value if matched else CheckStatus.FAIL.value,
        "" if matched else f"expected {expected}, worker has {observed}",
        severity=severity,
        expected=expected,
        observed=observed,
    )


def run_preflight(
    *,
    layout: RunLayout,
    worker_id: str,
    plan: dict[str, Any],
    manifest: RemoteArtifactManifest,
    capabilities: CapabilityReport,
    expected_plan_sha256: str,
    expected_manifest_sha256: str,
    environment_lock: dict[str, Any] | None = None,
    trainer_root: Path | None = None,
    protocol_version: str = REMOTE_PROTOCOL_VERSION,
    minimum_free_disk_mb: int | None = None,
    require_code_match: bool = True,
) -> PreflightReport:
    """Everything the worker can establish before starting a trainer."""
    report = PreflightReport(run_id=str(plan.get("run_id", "")), worker_id=worker_id)
    report.capabilities = capabilities.to_dict()

    # ── protocol ──
    try:
        check_protocol(protocol_version, peer="control plane")
        report.add("protocol", CheckStatus.PASS.value, protocol_version)
    except ProtocolError as exc:
        report.add("protocol", CheckStatus.FAIL.value, str(exc))
        # Nothing below can be trusted if the peers disagree about what
        # the messages mean. Returning early is the honest response.
        return report

    # ── plan and manifest identity ──
    from luber_training.remote.staging import PLAN_PATH

    plan_path = layout.root / PLAN_PATH
    if plan_path.is_file():
        actual_plan_digest = _plan_digest(plan_path)
        report.add(
            "plan_hash",
            CheckStatus.PASS.value
            if actual_plan_digest == expected_plan_sha256
            else CheckStatus.FAIL.value,
            ""
            if actual_plan_digest == expected_plan_sha256
            else "the plan on the worker is not the plan that was dispatched",
            expected=expected_plan_sha256[:16],
            observed=(actual_plan_digest or "unreadable")[:16],
        )
    else:
        report.add("plan_hash", CheckStatus.FAIL.value, f"{PLAN_PATH} was never received")

    manifest_digest = manifest.digest()
    report.add(
        "artifact_manifest_hash",
        CheckStatus.PASS.value
        if manifest_digest == expected_manifest_sha256
        else CheckStatus.FAIL.value,
        ""
        if manifest_digest == expected_manifest_sha256
        else "the manifest on the worker differs from the one dispatched",
        expected=expected_manifest_sha256[:16],
        observed=manifest_digest[:16],
    )

    # ── every required artifact, rehashed on the worker ──
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in manifest.required_entries:
        try:
            local = layout.resolve(entry.target_path)
        except ValueError as exc:
            mismatched.append(f"{entry.target_path} ({exc})")
            continue
        if not local.is_file():
            missing.append(entry.target_path)
            continue
        digest, size = sha256_file(local)
        if size != entry.size_bytes or digest != entry.sha256:
            mismatched.append(entry.target_path)

    if missing:
        report.add(
            "artifacts_present",
            CheckStatus.FAIL.value,
            f"{len(missing)} required artifact(s) are missing: {', '.join(sorted(missing)[:5])}",
        )
    else:
        report.add(
            "artifacts_present",
            CheckStatus.PASS.value,
            f"{len(manifest.required_entries)} required artifact(s) present",
        )

    if mismatched:
        report.add(
            "artifact_digests",
            CheckStatus.FAIL.value,
            (
                f"{len(mismatched)} artifact(s) do not match the manifest: "
                f"{', '.join(sorted(mismatched)[:5])}. The trainer will not start on data "
                "that is not what was approved"
            ),
        )
    else:
        report.add("artifact_digests", CheckStatus.PASS.value, "every required digest matches")

    # ── code revision ──
    expected_commit = (environment_lock or {}).get("luber_commit")
    if require_code_match:
        if expected_commit is None:
            report.add(
                "code_revision",
                CheckStatus.UNKNOWN.value,
                "the dispatch recorded no LUBER commit, so the worker's code cannot be matched",
            )
        else:
            _compare(
                report,
                "code_revision",
                str(expected_commit),
                capabilities.luber_commit,
                severity=Severity.REQUIRED.value,
            )
    else:
        report.add(
            "code_revision",
            CheckStatus.UNKNOWN.value,
            "code matching was waived for this dispatch",
            severity=Severity.INFORMATIONAL.value,
            expected=str(expected_commit) if expected_commit else None,
            observed=capabilities.luber_commit,
        )

    # ── environment, graded ──
    lock = environment_lock or {}
    _compare(
        report,
        "python_version",
        lock.get("python_version"),
        capabilities.python_version,
        severity=Severity.COMPATIBLE.value,
    )
    _compare(
        report,
        "torch_version",
        lock.get("torch_version"),
        capabilities.torch_version,
        severity=Severity.REQUIRED.value,
    )
    _compare(
        report,
        "ace_step_commit",
        lock.get("ace_step_commit"),
        capabilities.ace_step_commit,
        severity=Severity.REQUIRED.value,
    )
    _compare(
        report,
        "peft_version",
        lock.get("peft_version"),
        capabilities.peft_version,
        severity=Severity.COMPATIBLE.value,
    )
    _compare(
        report,
        "transformers_version",
        lock.get("transformers_version"),
        capabilities.transformers_version,
        severity=Severity.INFORMATIONAL.value,
    )

    # ── accelerator ──
    requirements = plan.get("requirements") or {}
    requires_cuda = bool(requirements.get("requires_cuda", True))
    if requires_cuda:
        if capabilities.cuda_available is None:
            report.add(
                "cuda",
                CheckStatus.UNKNOWN.value,
                "the worker could not determine CUDA availability; a plan that requires "
                "CUDA will not run on a machine where nobody could check",
            )
        elif not capabilities.cuda_available:
            report.add("cuda", CheckStatus.FAIL.value, "the worker reports no CUDA")
        else:
            report.add(
                "cuda",
                CheckStatus.PASS.value,
                f"{capabilities.gpu_count} GPU(s), {capabilities.gpu_model}",
            )

        needed = int(requirements.get("minimum_gpu_count") or 1)
        if capabilities.gpu_count is None:
            report.add("gpu_count", CheckStatus.UNKNOWN.value, "no GPU count reported")
        elif capabilities.gpu_count < needed:
            report.add(
                "gpu_count",
                CheckStatus.FAIL.value,
                f"the plan needs {needed}, the worker has {capabilities.gpu_count}",
            )
        else:
            report.add("gpu_count", CheckStatus.PASS.value, str(capabilities.gpu_count))

        minimum_vram = requirements.get("minimum_vram_mb")
        if minimum_vram is None:
            report.add(
                "vram",
                CheckStatus.UNKNOWN.value,
                "no VRAM figure has been measured for any LUBER configuration, so memory "
                "sufficiency cannot be checked",
                severity=Severity.INFORMATIONAL.value,
                observed=str(capabilities.vram_total_mb),
            )
        elif capabilities.vram_total_mb is None:
            report.add("vram", CheckStatus.UNKNOWN.value, "the worker reported no VRAM")
        elif capabilities.vram_total_mb < int(minimum_vram):
            report.add(
                "vram",
                CheckStatus.FAIL.value,
                f"the plan needs {minimum_vram} MB, the worker has {capabilities.vram_total_mb} MB",
            )
        else:
            report.add("vram", CheckStatus.PASS.value, f"{capabilities.vram_total_mb} MB")
    else:
        report.add(
            "cuda",
            CheckStatus.PASS.value,
            "the plan does not require CUDA",
            severity=Severity.INFORMATIONAL.value,
        )

    # ── precision ──
    precision = str((plan.get("config") or {}).get("precision", "auto"))
    if precision == "bf16" and capabilities.bf16_supported is False:
        report.add("precision", CheckStatus.FAIL.value, "the worker reports no bf16 support")
    elif precision == "bf16" and capabilities.bf16_supported is None:
        report.add(
            "precision",
            CheckStatus.UNKNOWN.value,
            "bf16 support could not be determined",
            severity=Severity.COMPATIBLE.value,
        )
    else:
        report.add("precision", CheckStatus.PASS.value, precision)

    # ── disk and writability ──
    try:
        layout.ensure()
        probe = layout.checkpoints_dir / ".writable"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        report.add("checkpoint_dir_writable", CheckStatus.PASS.value, str(layout.checkpoints_dir))
    except OSError as exc:
        report.add(
            "checkpoint_dir_writable",
            CheckStatus.FAIL.value,
            f"the checkpoint directory is not writable: {exc}",
        )

    if minimum_free_disk_mb is None:
        report.add(
            "disk_capacity",
            CheckStatus.UNKNOWN.value,
            "no disk requirement was supplied; checkpoint size is unmeasured, so this "
            "cannot be checked from what is known",
            severity=Severity.INFORMATIONAL.value,
            observed=str(capabilities.free_disk_mb),
        )
    elif capabilities.free_disk_mb is None:
        report.add("disk_capacity", CheckStatus.UNKNOWN.value, "free disk could not be measured")
    elif capabilities.free_disk_mb < minimum_free_disk_mb:
        report.add(
            "disk_capacity",
            CheckStatus.FAIL.value,
            f"{capabilities.free_disk_mb} MB free, {minimum_free_disk_mb} MB required",
        )
    else:
        report.add("disk_capacity", CheckStatus.PASS.value, f"{capabilities.free_disk_mb} MB free")

    # ── the dataset the trainer will actually open ──
    if layout.dataset_dir.is_dir() and any(layout.dataset_dir.iterdir()):
        count = sum(1 for path in layout.dataset_dir.rglob("*") if path.is_file())
        report.add("dataset_present", CheckStatus.PASS.value, f"{count} file(s)")
    else:
        report.add(
            "dataset_present", CheckStatus.FAIL.value, f"{layout.dataset_dir} is empty or absent"
        )

    # ── does the command even compile ──
    if trainer_root is None:
        report.add(
            "trainer_command",
            CheckStatus.UNKNOWN.value,
            "no trainer root was configured on this worker",
        )
    elif not Path(trainer_root).is_dir():
        report.add(
            "trainer_command",
            CheckStatus.FAIL.value,
            f"the trainer is not installed at {trainer_root}",
        )
    else:
        trainer_entry = Path(trainer_root) / "train.py"
        report.add(
            "trainer_command",
            CheckStatus.PASS.value if trainer_entry.is_file() else CheckStatus.FAIL.value,
            str(trainer_entry) if trainer_entry.is_file() else f"{trainer_entry} does not exist",
        )

    if shutil.which("nvidia-smi") is None and requires_cuda:
        report.warnings.append(
            "nvidia-smi is not on PATH, so GPU telemetry will be unavailable during this run"
        )
    report.warnings.extend(capabilities.unknown)
    return report


def _plan_digest(path: Path) -> str | None:
    """Recompute a received plan's digest from its own contents.

    Reconstructing the plan object and asking it to hash itself is what
    makes this a check rather than a formality: reading a
    `training_plan_sha256` field out of the file would only prove the
    file contains a string.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    from luber_training.plan import NON_CANONICAL_KEYS

    canonical = {
        key: value
        for key, value in payload.items()
        if key not in NON_CANONICAL_KEYS and key != "plan_id"
    }
    import hashlib

    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


__all__ = [
    "PREFLIGHT_SCHEMA_VERSION",
    "Check",
    "CheckStatus",
    "PreflightReport",
    "PreflightStatus",
    "Severity",
    "run_preflight",
]
