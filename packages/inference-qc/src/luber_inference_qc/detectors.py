"""Vocal presence: the check that exists so it can honestly say UNKNOWN.

`packages/dataset/.../classification.py` refused to build a vocal
classifier, and the reason applies with more force here:

> A spectral heuristic could be written in an afternoon and would be
> wrong often enough to matter — and its errors would be invisible,
> because a wrongly-labelled instrumental looks exactly like a
> correctly-labelled one.

There, a wrong label put a track in the wrong bucket of a manifest.
Here, a wrong label throws away a user's song and spends another
inference to replace it. The cost of guessing went up, so the answer
stays no.

What this module provides instead:

**A protocol.** `VocalPresenceDetector` is the seam a validated detector
plugs into. The mismatch path is written and tested against a stub, so
the day one exists it starts working rather than needing to be built
then.

**A null default that measures the evidence anyway.**
`centre_dominance_db` — mid versus side energy across 200-4000 Hz — is
*consistent with* a lead vocal, because lead vocals sit in the centre of
a stereo image. It is recorded on every candidate as evidence and it
never sets a verdict. A track can be centre-dominant because it is a
mono-ish mix, and a vocal can be wide.

**A confidence threshold that the default can never reach.** The null
detector reports 0.0. Nothing downstream needs to special-case it: the
same rule that discards a weak real detector discards this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from luber_audio_finishing import LoadedAudio

#: Below this, a detector's answer is not used. Chosen high because the
#: action it would justify is discarding a finished generation.
MINIMUM_VOCAL_CONFIDENCE = 0.85

#: The band a lead vocal occupies. Same range Phase 23 measures.
VOCAL_LOW_HZ = 200.0
VOCAL_HIGH_HZ = 4_000.0

_EPS = 1e-12


class VocalPresence(StrEnum):
    VOCAL = "VOCAL"
    INSTRUMENTAL = "INSTRUMENTAL"
    #: Nobody could establish it. Not a third kind of audio.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VocalAssessment:
    """A verdict, its confidence, and why it is what it is."""

    presence: str
    confidence: float
    reason: str
    #: Mid-minus-side energy in the vocal band, dB. Evidence, recorded
    #: whether or not any verdict was reached.
    centre_dominance_db: float | None = None
    detector: str = "null"

    @property
    def usable(self) -> bool:
        """Whether this answer may drive a decision."""
        return (
            self.presence != VocalPresence.UNKNOWN.value
            and self.confidence >= MINIMUM_VOCAL_CONFIDENCE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "presence": self.presence,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "centre_dominance_db": (
                round(self.centre_dominance_db, 2) if self.centre_dominance_db is not None else None
            ),
            "detector": self.detector,
        }


class VocalPresenceDetector(Protocol):
    """The seam a validated detector plugs into."""

    name: str

    def assess(self, audio: LoadedAudio) -> VocalAssessment:
        """Whether this audio carries a lead vocal, and how sure."""
        ...


def centre_dominance_db(audio: LoadedAudio) -> float | None:
    """Mid energy minus side energy across the vocal band, in dB.

    ``None`` for mono, where the question has no meaning: a mono file has
    no side channel, which says nothing about whether anyone is singing.

    A single FFT over the whole file rather than a windowed analysis —
    this is evidence, not a measurement anything depends on, and the
    cheap version is the honest amount of effort to spend on it.
    """
    if not audio.is_stereo:
        return None
    left = audio.samples[:, 0].astype(np.float64)
    right = audio.samples[:, 1].astype(np.float64)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    size = min(mid.size, 1 << 18)
    if size < 1024:
        return None
    window = np.hanning(size)
    freqs = np.fft.rfftfreq(size, 1.0 / audio.sample_rate)
    band = (freqs >= VOCAL_LOW_HZ) & (freqs <= VOCAL_HIGH_HZ)
    if not band.any():
        return None

    mid_power = float(np.sum(np.abs(np.fft.rfft(mid[:size] * window)[band]) ** 2))
    side_power = float(np.sum(np.abs(np.fft.rfft(side[:size] * window)[band]) ** 2))
    return float(10.0 * np.log10((mid_power + _EPS) / (side_power + _EPS)))


class NullVocalDetector:
    """The default: measures the evidence, reaches no verdict.

    This is not a placeholder waiting to be filled in with a heuristic.
    It is the correct implementation for a repository with no labelled
    data to validate one against, and it is what keeps a vocal
    "mismatch" from ever being asserted on a guess.
    """

    name = "null"

    def assess(self, audio: LoadedAudio) -> VocalAssessment:
        return VocalAssessment(
            presence=VocalPresence.UNKNOWN.value,
            confidence=0.0,
            reason=(
                "no validated vocal/instrumental detector exists in this repository and no "
                "labelled data to validate one against; a spectral heuristic would produce "
                "errors indistinguishable from measurements, and here the cost of one is a "
                "discarded generation"
            ),
            centre_dominance_db=centre_dominance_db(audio),
            detector="null",
        )


__all__ = [
    "MINIMUM_VOCAL_CONFIDENCE",
    "NullVocalDetector",
    "VocalAssessment",
    "VocalPresence",
    "VocalPresenceDetector",
    "centre_dominance_db",
]
