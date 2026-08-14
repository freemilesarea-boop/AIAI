"""Perceptual frequency bands, and what to do when they exceed Nyquist.

A 22.05 kHz-Nyquist file has no 16-20 kHz band. Reporting 0 energy there
would be a lie that reads as "no air" and would provoke a correction the
file cannot benefit from, so bands above Nyquist are reported as absent
and every consumer must handle ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SUB = "sub"
BASS = "bass"
LOW_MID = "low_mid"
MID = "mid"
PRESENCE = "presence"
BRILLIANCE = "brilliance"
AIR = "air"
ULTRA_HIGH = "ultra_high"

#: Edges in Hz. Contiguous by construction: each band starts where the
#: previous one ends, so band energies sum to the analysed spectrum
#: between 20 Hz and 20 kHz with nothing double-counted.
BAND_EDGES: Final[tuple[tuple[str, float, float], ...]] = (
    (SUB, 20.0, 60.0),
    (BASS, 60.0, 150.0),
    (LOW_MID, 150.0, 400.0),
    (MID, 400.0, 2_000.0),
    (PRESENCE, 2_000.0, 5_000.0),
    (BRILLIANCE, 5_000.0, 10_000.0),
    (AIR, 10_000.0, 16_000.0),
    (ULTRA_HIGH, 16_000.0, 20_000.0),
)

BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _, _ in BAND_EDGES)


@dataclass(frozen=True)
class BandCoverage:
    """How much of a band a given sample rate can actually represent."""

    name: str
    low_hz: float
    high_hz: float
    #: The part of the band below Nyquist. ``None`` when nothing is.
    usable_high_hz: float | None

    @property
    def is_absent(self) -> bool:
        """The sample rate cannot represent this band at all."""
        return self.usable_high_hz is None

    @property
    def is_partial(self) -> bool:
        """Present, but truncated by Nyquist."""
        return self.usable_high_hz is not None and self.usable_high_hz < self.high_hz


def band_coverage(sample_rate: int) -> tuple[BandCoverage, ...]:
    """Resolve every band against a sample rate's Nyquist frequency.

    A band is truncated rather than dropped when Nyquist falls inside it:
    a 44.1 kHz file really does have 16-20 kHz content up to 22.05 kHz,
    and pretending otherwise would discard measurable air.
    """
    nyquist = sample_rate / 2.0
    resolved: list[BandCoverage] = []
    for name, low, high in BAND_EDGES:
        if low >= nyquist:
            usable: float | None = None
        else:
            usable = min(high, nyquist)
        resolved.append(BandCoverage(name=name, low_hz=low, high_hz=high, usable_high_hz=usable))
    return tuple(resolved)
