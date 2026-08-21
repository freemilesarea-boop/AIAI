"""What QC is allowed to conclude, and how badly.

The vocabulary is closed and every member is implemented. A finding that
exists in the enum and is never produced would read, to anyone auditing
a trace, as a check that ran and passed — which is the same fabrication
this phase exists to avoid, one level up.

Three distinctions carry the design.

**Severity is not the same as retryability.** A CRITICAL finding means
the candidate cannot be delivered; whether *another* attempt would help
is a separate question. Severe clipping is critical and retryable. A
provider misconfiguration is critical and not.

**A soft finding is not a small hard finding.** Harshness, sibilance and
narrow stereo are recorded because Phase 22 may act on them and because
they are worth ranking by — never because they are defects. A dark mix
is a production choice. The console's own rule applies here: the
console does not reward brightness, and neither does this.

**NOT_MEASURABLE is a real outcome.** Where this repository has no
validated detector — vocal presence, lyric completeness, structure — the
finding says so and carries the reason. It never becomes a pass and it
never becomes a rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """How much a finding matters to delivery."""

    #: Worth recording. Never affects eligibility or ranking.
    INFO = "INFO"
    #: A small penalty when ranking eligible candidates.
    MINOR = "MINOR"
    #: A large penalty. Blocking only where the check says so.
    MAJOR = "MAJOR"
    #: The candidate cannot be delivered. Not a low score — excluded.
    CRITICAL = "CRITICAL"


class Finding(StrEnum):
    """Every conclusion this engine can reach about one candidate."""

    # ── structural ───────────────────────────────────────────────────
    #: The file is missing, empty, unparseable, or decodes to nothing.
    INVALID_AUDIO = "INVALID_AUDIO"
    #: Samples that are not numbers. A file that will make every
    #: downstream measurement meaningless.
    NON_FINITE_SAMPLES = "NON_FINITE_SAMPLES"

    # ── duration ─────────────────────────────────────────────────────
    DURATION_SHORT = "DURATION_SHORT"
    DURATION_LONG = "DURATION_LONG"

    # ── level ────────────────────────────────────────────────────────
    SILENT_OUTPUT = "SILENT_OUTPUT"
    #: The whole file is too quiet to be anything but noise.
    NEAR_SILENT = "NEAR_SILENT"
    #: A lot of the file is silent, without the positional evidence that
    #: would make it a collapse. Kept apart from NEAR_SILENT because a
    #: track with long structured gaps and a track mastered 40 dB down
    #: are different problems, and one code covering both would make a
    #: trace unreadable at exactly the moment somebody needs it.
    EXCESSIVE_SILENCE = "EXCESSIVE_SILENCE"
    #: Content stops long before the file does.
    EARLY_COLLAPSE = "EARLY_COLLAPSE"
    #: Clipping severe enough that it is in the source, not the ceiling.
    SEVERE_CLIPPING = "SEVERE_CLIPPING"
    #: A peak issue the finishing limiter handles routinely.
    PEAK_OVERSHOOT = "PEAK_OVERSHOOT"
    DC_OFFSET = "DC_OFFSET"

    # ── stereo ───────────────────────────────────────────────────────
    #: Broadband anti-phase: this will partially disappear in mono.
    PHASE_UNSAFE = "PHASE_UNSAFE"
    #: Low-frequency energy that cancels in mono. Phase 22 repairs it.
    LOW_END_PHASE_RISK = "LOW_END_PHASE_RISK"
    NARROW_STEREO = "NARROW_STEREO"
    CHANNEL_IMBALANCE = "CHANNEL_IMBALANCE"

    # ── spectral ─────────────────────────────────────────────────────
    #: Effectively no content above an implausibly low bandwidth. A
    #: technical failure, not a dark mix.
    SPECTRAL_COLLAPSE = "SPECTRAL_COLLAPSE"
    HIGH_HARSHNESS_PROXY = "HIGH_HARSHNESS_PROXY"
    HIGH_SIBILANCE_PROXY = "HIGH_SIBILANCE_PROXY"

    # ── control adherence ────────────────────────────────────────────
    CONTROL_BPM_MISMATCH = "CONTROL_BPM_MISMATCH"
    CONTROL_KEY_MISMATCH = "CONTROL_KEY_MISMATCH"
    #: Only ever produced by a detector that was supplied and confident.
    CONTROL_VOCAL_MISMATCH = "CONTROL_VOCAL_MISMATCH"
    #: The default for vocal presence in this build: no validated
    #: detector exists, and a guess would be indistinguishable from a
    #: measurement.
    CONTROL_VOCAL_UNKNOWN = "CONTROL_VOCAL_UNKNOWN"
    #: A control was requested and could not be measured with enough
    #: confidence to say anything.
    CONTROL_NOT_MEASURABLE = "CONTROL_NOT_MEASURABLE"

    # ── provider ─────────────────────────────────────────────────────
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    #: Configuration, credentials, or an unsupported capability. Another
    #: attempt produces the same failure.
    PROVIDER_MISCONFIGURED = "PROVIDER_MISCONFIGURED"

    # ── the absence of anything wrong ────────────────────────────────
    NO_CRITICAL_FINDINGS = "NO_CRITICAL_FINDINGS"


#: Findings that mean another attempt could plausibly do better. A
#: retryable finding is not automatically retried — the policy decides
#: that — but a non-retryable one never is, however much budget remains.
RETRYABLE: frozenset[str] = frozenset(
    {
        Finding.INVALID_AUDIO.value,
        Finding.NON_FINITE_SAMPLES.value,
        Finding.DURATION_SHORT.value,
        Finding.DURATION_LONG.value,
        Finding.SILENT_OUTPUT.value,
        Finding.NEAR_SILENT.value,
        Finding.EARLY_COLLAPSE.value,
        Finding.SEVERE_CLIPPING.value,
        Finding.SPECTRAL_COLLAPSE.value,
        Finding.PHASE_UNSAFE.value,
        Finding.CONTROL_BPM_MISMATCH.value,
        Finding.CONTROL_VOCAL_MISMATCH.value,
        Finding.PROVIDER_TIMEOUT.value,
        Finding.PROVIDER_ERROR.value,
    }
)

#: Findings no further attempt can fix. Retrying a misconfiguration
#: spends inference to reproduce the same error.
NON_RETRYABLE: frozenset[str] = frozenset({Finding.PROVIDER_MISCONFIGURED.value})


@dataclass(frozen=True)
class QCFinding:
    """One conclusion, with the number behind it.

    ``measured`` and ``threshold`` are carried so a trace can be argued
    with. A finding that said only "duration is wrong" would be a
    verdict nobody could check; one that says 41.2 s against 240 s with
    a 15% tolerance can be checked by anyone, including against a
    threshold somebody later decides was wrong.
    """

    code: str
    severity: str
    detail: str
    metric: str = ""
    measured: float | None = None
    threshold: float | None = None
    #: True when the check ran and could not establish an answer, as
    #: distinct from running and finding nothing wrong.
    not_measurable: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        """Whether this alone makes the candidate undeliverable."""
        return self.severity == Severity.CRITICAL.value

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
            "metric": self.metric,
            "measured": self.measured,
            "threshold": self.threshold,
            "not_measurable": self.not_measurable,
            "evidence": self.evidence,
        }


def critical(findings: list[QCFinding]) -> list[QCFinding]:
    return [item for item in findings if item.severity == Severity.CRITICAL.value]


def by_severity(findings: list[QCFinding], severity: Severity) -> list[QCFinding]:
    return [item for item in findings if item.severity == severity.value]


__all__ = [
    "NON_RETRYABLE",
    "RETRYABLE",
    "Finding",
    "QCFinding",
    "Severity",
    "by_severity",
    "critical",
]
