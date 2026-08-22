"""How much memory a workload may use, and what nobody has measured.

Two things are modelled here, and only one of them is a number.

**Headroom is arithmetic.** A machine with 24 GB does not have 24 GB
available to PyTorch. On the planned topology the same Mac is also the
API, Postgres, Redis, the orchestrator and whatever the operator has
open, and Apple's unified memory is shared with the GPU rather than
separate from it. So a budget reserves a fraction and a floor, and both
are configurable, because the right reservation on a dedicated trainer
is not the right reservation on a control plane.

**Feasibility is mostly UNKNOWN, and says so.** Nothing in this project
has measured what the 2B DiT plus optimizer state plus activations
actually needs. Writing "this model needs 18.4 GB" would be inventing
the single number every scheduling decision depends on. So the verdict
is `UNKNOWN` unless an estimate was *supplied* by something that
measured it, and `UNKNOWN` never reads as "fine".

Upstream's VRAM presets (`vram_8gb.json` … `vram_24gb_plus.json`) are
not a source for this. They are indexed by dedicated NVIDIA VRAM, and
Apple unified memory is a different resource shared with the operating
system. Reading one as the other is exactly the mistake this module
exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MemoryVerdict(StrEnum):
    """Whether a workload fits."""

    #: An estimate exists and fits inside the usable budget.
    KNOWN_SAFE = "KNOWN_SAFE"
    #: An estimate exists and exceeds it.
    LIKELY_TOO_LARGE = "LIKELY_TOO_LARGE"
    #: Nobody has measured what this needs. Not a pass.
    UNKNOWN = "UNKNOWN"


#: Fraction of total memory held back from ML work by default.
#:
#: Thirty percent, which on a 24 GB machine is a little over 7 GB. Not
#: derived from a measurement — no measurement exists — but chosen so
#: that a machine which is *also* the control plane keeps enough to
#: stay a control plane. An operator running a dedicated trainer should
#: lower it deliberately rather than inherit a number meant for a
#: shared machine.
DEFAULT_HEADROOM_FRACTION = 0.30

#: Never reserve less than this, whatever the fraction works out to.
#: A percentage of a small machine reserves too little to run an
#: operating system.
DEFAULT_RESERVED_FLOOR_MB = 4096

#: How many local training jobs may run at once.
#:
#: One. The planned Mac mini is the 24/7 control plane before it is a
#: trainer, and two concurrent local runs on shared unified memory is
#: how the API stops answering. Bounded here rather than left to
#: whoever launches the second one.
LOCAL_TRAINING_CONCURRENCY = 1


@dataclass(frozen=True)
class MemoryBudget:
    """What a target may spend, given what it has to keep."""

    total_mb: int | None
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION
    reserved_floor_mb: int = DEFAULT_RESERVED_FLOOR_MB
    #: Why this budget is shaped the way it is, for the report.
    note: str = ""

    def reserved_mb(self) -> int | None:
        if self.total_mb is None:
            return None
        return max(int(self.total_mb * self.headroom_fraction), self.reserved_floor_mb)

    def usable_mb(self) -> int | None:
        """Memory a workload may plan to use. Never the whole machine."""
        reserved = self.reserved_mb()
        if self.total_mb is None or reserved is None:
            return None
        return max(0, self.total_mb - reserved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_mb": self.total_mb,
            "headroom_fraction": self.headroom_fraction,
            "reserved_floor_mb": self.reserved_floor_mb,
            "reserved_mb": self.reserved_mb(),
            "usable_mb": self.usable_mb(),
            "note": self.note,
        }


@dataclass(frozen=True)
class MemoryAssessment:
    """Whether a workload fits, and how confident that is."""

    verdict: str
    budget: MemoryBudget
    estimated_mb: int | None = None
    reason: str = ""

    @property
    def blocks(self) -> bool:
        """UNKNOWN does not block. It is recorded and carried.

        Refusing every placement for want of a measurement nobody has
        taken would stop the project from ever taking one. What must not
        happen is UNKNOWN being *rendered* as a pass, which is why the
        verdict travels on the decision rather than being collapsed into
        a boolean here.
        """
        return self.verdict == MemoryVerdict.LIKELY_TOO_LARGE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "estimated_mb": self.estimated_mb,
            "reason": self.reason,
            "budget": self.budget.to_dict(),
        }


def budget_for(
    total_mb: int | None,
    *,
    shared_with_control_plane: bool = False,
    headroom_fraction: float | None = None,
    reserved_floor_mb: int | None = None,
) -> MemoryBudget:
    """A budget for a target with this much memory.

    ``shared_with_control_plane`` is the Mac mini case: the same machine
    is serving the API and holding Postgres and Redis while it trains.
    It raises the reservation rather than trusting the operator to
    remember, because the failure mode is the control plane becoming
    unresponsive during the exact run somebody is watching.
    """
    fraction = headroom_fraction
    if fraction is None:
        fraction = 0.40 if shared_with_control_plane else DEFAULT_HEADROOM_FRACTION
    floor = reserved_floor_mb if reserved_floor_mb is not None else DEFAULT_RESERVED_FLOOR_MB
    note = (
        "reservation raised: this machine also runs the control plane"
        if shared_with_control_plane
        else ""
    )
    return MemoryBudget(
        total_mb=total_mb,
        headroom_fraction=fraction,
        reserved_floor_mb=floor,
        note=note,
    )


def assess(budget: MemoryBudget, estimated_mb: int | None = None) -> MemoryAssessment:
    """Whether an estimate fits — or that there is no estimate.

    The `None` branch is the normal one today and it is the honest one.
    """
    usable = budget.usable_mb()

    if estimated_mb is None:
        return MemoryAssessment(
            verdict=MemoryVerdict.UNKNOWN.value,
            budget=budget,
            reason=(
                "no memory requirement has been measured for this workload, so whether it "
                "fits is UNKNOWN. This is not a pass: it means the first run on new "
                "hardware is the measurement"
            ),
        )
    if usable is None:
        return MemoryAssessment(
            verdict=MemoryVerdict.UNKNOWN.value,
            budget=budget,
            estimated_mb=estimated_mb,
            reason="the target has not reported how much memory it has",
        )
    if estimated_mb <= usable:
        return MemoryAssessment(
            verdict=MemoryVerdict.KNOWN_SAFE.value,
            budget=budget,
            estimated_mb=estimated_mb,
            reason=(
                f"{estimated_mb} MB fits inside the {usable} MB budget "
                f"({budget.reserved_mb()} MB held back)"
            ),
        )
    return MemoryAssessment(
        verdict=MemoryVerdict.LIKELY_TOO_LARGE.value,
        budget=budget,
        estimated_mb=estimated_mb,
        reason=(
            f"{estimated_mb} MB exceeds the {usable} MB budget "
            f"({budget.total_mb} MB total, {budget.reserved_mb()} MB held back for the "
            "system)"
        ),
    )


__all__ = [
    "DEFAULT_HEADROOM_FRACTION",
    "DEFAULT_RESERVED_FLOOR_MB",
    "LOCAL_TRAINING_CONCURRENCY",
    "MemoryAssessment",
    "MemoryBudget",
    "MemoryVerdict",
    "assess",
    "budget_for",
]
