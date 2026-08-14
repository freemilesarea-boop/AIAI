"""The committed baseline must keep saying what the documents claim.

Every threshold in the engine was justified by a number in
``baseline_summary.json``, and every "N of 40 tracks" in the audit was
counted from ``baseline_results.jsonl``. If those files change and the
prose does not, the reasoning becomes decoration. These tests fail when
that happens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK.parents[1] / "packages" / "audio-finishing" / "src"))

from luber_audio_finishing.risks import (  # noqa: E402
    AIR_DEFICIT_DB,
    HIGH_FREQUENCY_DEFICIT_AIR_DB,
    LOW_END_EXCESS_SHARE,
    LOW_MID_MUD_DB,
    STEREO_NARROW_WIDTH,
    STEREO_WIDE_WIDTH,
    TRANSIENT_FLAT_CREST_DB,
)

EXPECTED_TRACKS = 40


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((BENCHMARK / "baseline_summary.json").read_text())


@pytest.fixture(scope="module")
def records() -> list[dict]:
    lines = (BENCHMARK / "baseline_results.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def band_share(record: dict, name: str) -> float | None:
    for band in record["frequency"]["bands"]:
        if band["name"] == name:
            return band["share"]
    return None


class TestCorpus:
    def test_the_corpus_is_the_size_the_documents_claim(self, records, summary):
        assert len(records) == EXPECTED_TRACKS
        assert summary["track_count"] == EXPECTED_TRACKS

    def test_no_record_carries_a_filesystem_path(self, records):
        """Committed benchmark data must not leak machine paths."""
        for record in records:
            assert "/" not in record["path"]
            assert not record["label"].startswith("/")

    def test_the_corpus_spans_short_and_long_form(self, records):
        durations = {round(record["technical"]["duration_seconds"]) for record in records}
        assert min(durations) <= 30
        assert max(durations) >= 180


class TestTheHumanReport:
    """Does the listening report survive contact with the measurements?"""

    def test_generated_masters_all_arrive_with_no_headroom(self, summary):
        """Every correction has to be paid for out of 1 dB.

        This is why the level stage exists and why a boost can leave the
        finished file slightly quieter than the raw one.
        """
        peak = summary["metrics"]["peak_dbfs"]
        assert peak["min"] == pytest.approx(-1.0, abs=0.01)
        assert peak["max"] == pytest.approx(-1.0, abs=0.01)
        assert peak["stdev"] == pytest.approx(0.0, abs=0.01)

    def test_tonal_balance_is_measurably_inconsistent(self, summary):
        """The half of the report the numbers support without argument.

        A 25 dB spread in high-frequency content across one model's own
        output is the case for an adaptive engine over a fixed curve.
        """
        air = summary["metrics"]["air_ratio_db"]
        assert air["max"] - air["min"] > 20.0
        assert air["stdev"] > 4.0

    def test_a_high_frequency_deficit_is_real_but_not_universal(self, records):
        """The half that needs qualifying.

        "Upper frequencies often feel rolled off" is true of a substantial
        minority, not of the catalogue. Reporting it as universal would
        have justified a fixed shelf on every track.
        """
        dark = [
            record
            for record in records
            if record["frequency"]["air_ratio_db"]["p50"] < HIGH_FREQUENCY_DEFICIT_AIR_DB
            and record["frequency"]["spectral_slope_db_per_octave"] < -6.5
        ]
        assert 0 < len(dark) < len(records) / 2

    def test_the_opposite_failure_mode_is_present_too(self, records):
        """Which is why no fixed high-shelf boost could be correct."""
        spiky = [
            record
            for record in records
            if record["sibilance"]["sibilance_peak_excess_db"] > 17.0
            or record["sibilance"]["harshness_peak_excess_db"] > 14.0
        ]
        assert spiky

    def test_some_tracks_are_both_dark_and_spiky(self, records):
        """The contradiction the decision engine has to resolve.

        Not a hypothetical: these tracks are why the shelf ceiling drops
        when sibilance is present instead of the rules being independent.
        """
        both = [
            record
            for record in records
            if record["frequency"]["air_ratio_db"]["p50"] < AIR_DEFICIT_DB
            and (
                record["sibilance"]["sibilance_peak_excess_db"] > 17.0
                or record["sibilance"]["harshness_peak_excess_db"] > 14.0
            )
        ]
        assert both


class TestThresholdsAgainstTheCorpus:
    def test_the_low_end_threshold_sits_above_ordinary_music(self, records):
        """Music concentrates energy low; only the tail is excess."""
        shares = sorted(
            (band_share(r, "sub") or 0.0) + (band_share(r, "bass") or 0.0) for r in records
        )
        median = shares[len(shares) // 2]
        assert median < LOW_END_EXCESS_SHARE
        assert sum(1 for share in shares if share > LOW_END_EXCESS_SHARE) < len(shares) / 4

    def test_the_mud_threshold_sits_above_the_corpus_median(self, summary):
        assert summary["metrics"]["low_mid_ratio_db"]["p50"] < LOW_MID_MUD_DB

    def test_the_width_thresholds_bracket_the_corpus(self, summary):
        """Narrow catches a tail; wide catches nothing and is a ceiling.

        The widest track in the corpus is a legitimately wide mix that
        stays mono-compatible, so narrowing it would be taste, not repair.
        """
        width = summary["metrics"]["stereo_width"]
        assert width["min"] < STEREO_NARROW_WIDTH < width["p50"]
        assert width["max"] < STEREO_WIDE_WIDTH

    def test_the_transient_threshold_fires_on_nothing_here(self, summary):
        """It exists to catch a future regression, and says so."""
        assert summary["metrics"]["short_window_crest_db"]["min"] > TRANSIENT_FLAT_CREST_DB

    def test_no_threshold_flags_the_whole_corpus(self, records):
        """A flag that fires on everything describes the model, not a defect."""
        checks = {
            "air": lambda r: r["frequency"]["air_ratio_db"]["p50"] < AIR_DEFICIT_DB,
            "mud": lambda r: r["frequency"]["low_mid_ratio_db"]["p50"] > LOW_MID_MUD_DB,
            "narrow": lambda r: r["stereo"]["width"] < STEREO_NARROW_WIDTH,
        }
        for name, check in checks.items():
            hits = sum(1 for record in records if check(record))
            assert 0 < hits < len(records) * 0.6, name


class TestMeasurementIntegrity:
    def test_the_body_reference_band_overlaps_nothing_it_measures(self):
        """A ratio whose numerator sits in its denominator saturates.

        The first version used 300 Hz-3 kHz, which shared 300-400 Hz with
        the low-mid band and 2.5-3 kHz with the harshness band. Thick
        300-400 Hz content raised both sides of its own ratio and read as
        balanced.
        """
        from luber_audio_finishing.analysis import (
            AIR_RATIO_LOW_HZ,
            BODY_HIGH_HZ,
            BODY_LOW_HZ,
            HARSHNESS_LOW_HZ,
            LOW_MID_RATIO_HIGH_HZ,
            SIBILANCE_LOW_HZ,
        )

        assert LOW_MID_RATIO_HIGH_HZ <= BODY_LOW_HZ
        for band_low in (HARSHNESS_LOW_HZ, SIBILANCE_LOW_HZ, AIR_RATIO_LOW_HZ):
            assert band_low >= BODY_HIGH_HZ

    def test_every_record_has_the_fields_the_thresholds_read(self, records):
        for record in records:
            assert record["frequency"]["air_ratio_db"]["p50"] is not None
            assert record["frequency"]["spectral_slope_db_per_octave"] is not None
            assert record["sibilance"]["sibilance_peak_excess_db"] is not None
            assert record["stereo"]["width"] is not None

    def test_spatial_proxies_are_marked_as_driving_nothing(self, records):
        """They are recorded for a later phase to test, not to act on."""
        for record in records:
            assert record["spatial"]["drives_no_processing"] is True
