"""Where every number in this engine comes from.

Not one of these was chosen to make a test pass. Each is either taken
from an existing part of this project that already decided the question,
or derived from the corpus, and the provenance is recorded beside it —
because a threshold whose origin nobody remembers is a threshold nobody
can argue with when it starts rejecting good songs.

Four are copied rather than imported, and that is worth explaining.
`luber_evaluation` (Phase 26) and the Phase 5 benchmark scripts both
hold numbers this engine needs, but importing them would drag the
training registry and a benchmark harness into the runtime generation
path. The values are restated here with their source named, and the test
suite asserts they still agree with the originals — so a drift is a test
failure rather than a silent divergence.
"""

from __future__ import annotations

# ── duration ─────────────────────────────────────────────────────────
#
# Two tolerances, both already decided elsewhere, doing different jobs.
#
# 5% is Phase 26's `DEFAULT_DURATION_TOLERANCE_RELATIVE`, taken from the
# P20 objective baseline where generated durations landed within a small
# fraction of the request. Past it, the duration is *wrong* — worth
# recording and worth ranking on.
#
# 20% is the Phase 5 benchmark's `DURATION_TOLERANCE_RATIO`, the point
# at which a generated duration stopped being a near miss. Past it, the
# provider did not answer the question that was asked: a 40-second reply
# to a 240-second request is not a short song, it is a failure.

#: Beyond this the duration is recorded and penalised. Not a rejection.
DURATION_SOFT_TOLERANCE_RATIO = 0.05

#: Beyond this the candidate is rejected and, where policy allows,
#: retried.
DURATION_HARD_TOLERANCE_RATIO = 0.20


# ── silence ──────────────────────────────────────────────────────────
#
# Both ratios are the Phase 5 benchmark's, which measured them against
# real generated output rather than picking round numbers.

#: Peak below this share of full scale is silence. -60 dBFS.
SILENCE_PEAK_RATIO = 0.001

#: Peak below this is audible only as noise. -34 dBFS.
NEAR_SILENCE_PEAK_RATIO = 0.02

#: Share of the track that may be silent before it is suspicious. Phase
#: 5's `EXCESSIVE_SILENCE_RATIO`. On its own this is a finding, not a
#: rejection: a track with long structured gaps is unusual, not broken.
#: Rejection needs the *positional* evidence that `collapse` provides.
EXCESSIVE_SILENCE_RATIO = 0.35


# ── clipping ─────────────────────────────────────────────────────────
#
# The distinction the brief asks for: a peak the limiter handles versus
# distortion baked into the samples.
#
# Phase 22 flags any clipped sample at all, which is right for deciding
# whether to engage a limiter and far too sensitive for deciding whether
# to throw a song away — a handful of samples at full scale is a normal
# master. Phase 5's `CLIPPING_SAMPLE_RATIO` is the share at which
# clipping stops being incidental.

#: Share of samples at or beyond full scale that means the distortion is
#: in the source. Phase 5's `CLIPPING_SAMPLE_RATIO`.
SEVERE_CLIPPING_SAMPLE_RATIO = 0.001

#: Peak above this is worth recording as a finding the limiter will
#: handle. Phase 22 targets -1.0 dBFS true peak, so anything at or above
#: -0.1 dBFS is going to be touched.
PEAK_OVERSHOOT_DBFS = -0.1


# ── DC offset ────────────────────────────────────────────────────────
#: Phase 22's `DC_OFFSET_LIMIT`, restated. Above it the engine acts; a
#: candidate is not rejected for it, because the engine's action is the
#: repair.
DC_OFFSET_LIMIT = 0.002

#: Ten times the repairable limit. At this level the offset is eating
#: headroom rather than sitting under the music.
DC_OFFSET_SEVERE = 0.02


# ── stereo and phase ─────────────────────────────────────────────────
#
# Phase 22's `BROADBAND_PHASE_CORRELATION = 0.20` is where it starts
# correcting. Rejection is reserved for genuine anti-phase, where the
# mono sum cancels rather than merely narrows.

#: Below this the material is broadly out of phase and will partially
#: disappear in mono. Phase 22 cannot repair a signal that is inverted
#: against itself.
PHASE_UNSAFE_CORRELATION = -0.2

#: Below this Phase 22 records a broadband phase risk and acts. A
#: finding here, never a rejection.
PHASE_RISK_CORRELATION = 0.20

#: Phase 22's `STEREO_NARROW_WIDTH`. A finding only — narrow is a
#: production choice and the engine widens it later if it is safe to.
NARROW_STEREO_WIDTH = 0.11

#: Phase 22's `STEREO_IMBALANCE_DB`, for the finding.
CHANNEL_IMBALANCE_DB = 0.8

#: Imbalance this large is a broken channel, not a mix decision.
CHANNEL_IMBALANCE_SEVERE_DB = 9.0


# ── spectral ─────────────────────────────────────────────────────────
#
# The one spectral rejection, and it is measured as *concentration*
# rather than darkness — because darkness turned out not to separate the
# two things at all.
#
# The first version of this rule used spectral rolloff: below 2 kHz, the
# top of the spectrum is empty. Measuring the corpus killed it. Real
# ACE-Step output reaches a rolloff of 352 Hz at the bottom of the
# distribution, with the 5th percentile at 668 Hz — there are genuine,
# deliverable, bass-heavy songs living exactly where a "collapsed" file
# would. A rolloff threshold low enough to spare them would be too low to
# catch anything.
#
# What does separate them is how much of the energy sits in *one* band.
# Real music spreads across several even when it is dark: across the
# corpus the largest single band holds 0.279 to 0.805 of the energy,
# with a median of 0.383 and a 95th percentile of 0.642. A degenerate
# tone puts 0.967 in one band. The gap is real and it is what this
# threshold sits in.
#
# Measured over all 97 tracks of the raw corpus; see
# docs/INFERENCE_QUALITY_CONTROL.md for the distribution.

#: Share of total energy in a single analysis band. Above this the file
#: is one tone, not a mix. Set clear of the corpus maximum of 0.805 —
#: not midway, because the cost of being wrong is a discarded song.
SPECTRAL_CONCENTRATION_SHARE = 0.90


# ── harshness and sibilance ──────────────────────────────────────────
#
# Phase 22's own thresholds, used here only to rank. Neither rejects:
# the finishing engine exists partly to correct them, and rejecting a
# candidate for a defect the next stage repairs would waste an inference
# to avoid a problem that was about to be solved.
HARSHNESS_PEAK_EXCESS_DB = 14.0
SIBILANCE_PEAK_EXCESS_DB = 17.0


# ── control adherence ────────────────────────────────────────────────

#: Phase 26's `DEFAULT_BPM_CONFIDENCE_FLOOR`. Below it the estimate is
#: not consulted, because Phase 23's estimator reports a BPM for
#: material with no pulse at all and an ungated comparison would be two
#: guesses disagreeing.
BPM_CONFIDENCE_FLOOR = 0.55

#: How far a tempo may drift before it is recorded. Half a BPM either
#: side of 120 is inaudible; five is a different song's tempo.
BPM_SOFT_TOLERANCE_RATIO = 0.04

#: Past this the request was not honoured. Double- and half-time
#: estimates land far outside it, which is intended: a track at 60 when
#: 120 was asked for is a real mismatch even when the estimator is
#: seeing the same pulse.
BPM_HARD_TOLERANCE_RATIO = 0.15

#: Phase 23's `MIN_KEY_CONFIDENCE` is 0.10, which is a floor for
#: *recording* an estimate. Acting on one needs more, and even then key
#: adherence stays advisory: a song in the relative minor of the
#: requested key is not a failure.
KEY_CONFIDENCE_FLOOR = 0.35


__all__ = [
    "BPM_CONFIDENCE_FLOOR",
    "BPM_HARD_TOLERANCE_RATIO",
    "BPM_SOFT_TOLERANCE_RATIO",
    "CHANNEL_IMBALANCE_DB",
    "CHANNEL_IMBALANCE_SEVERE_DB",
    "DC_OFFSET_LIMIT",
    "DC_OFFSET_SEVERE",
    "DURATION_HARD_TOLERANCE_RATIO",
    "DURATION_SOFT_TOLERANCE_RATIO",
    "EXCESSIVE_SILENCE_RATIO",
    "HARSHNESS_PEAK_EXCESS_DB",
    "KEY_CONFIDENCE_FLOOR",
    "NARROW_STEREO_WIDTH",
    "NEAR_SILENCE_PEAK_RATIO",
    "PEAK_OVERSHOOT_DBFS",
    "PHASE_RISK_CORRELATION",
    "PHASE_UNSAFE_CORRELATION",
    "SEVERE_CLIPPING_SAMPLE_RATIO",
    "SIBILANCE_PEAK_EXCESS_DB",
    "SILENCE_PEAK_RATIO",
    "SPECTRAL_CONCENTRATION_SHARE",
]
