"""Tests for the Phase 13E calibration tooling.

The measurements decided a product classification, so the code that
produced them has to be shown to work on signals whose answer is known in
advance. A descriptor that silently returned a constant would have made
the whole experiment look like a null result.

Real audio is not needed: synthesised tones have known spectra, so the
expected ordering is arithmetic rather than a matter of opinion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from reference_analyse import compare, cosine, descriptors, mfcc  # noqa: E402

RATE = 48_000


def tone(freq: float, seconds: float = 2.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return 0.3 * np.sin(2 * np.pi * freq * t)


def noise(seconds: float = 2.0, rate: int = RATE) -> np.ndarray:
    rng = np.random.default_rng(0)
    return 0.3 * rng.standard_normal(int(seconds * rate))


class TestDescriptors:
    def test_centroid_tracks_pitch(self):
        """A brighter signal must measure brighter, or nothing else holds."""
        low = descriptors(tone(220.0), RATE)["spectral_centroid_hz"]
        high = descriptors(tone(3520.0), RATE)["spectral_centroid_hz"]
        assert high > low * 3

    def test_centroid_is_near_the_actual_tone(self):
        measured = descriptors(tone(1000.0), RATE)["spectral_centroid_hz"]
        assert 700 < measured < 1400

    def test_rolloff_tracks_pitch(self):
        low = descriptors(tone(220.0), RATE)["spectral_rolloff85_hz"]
        high = descriptors(tone(3520.0), RATE)["spectral_rolloff85_hz"]
        assert high > low

    def test_flatness_separates_noise_from_a_tone(self):
        """Flatness is the descriptor that should notice noisiness."""
        assert (
            descriptors(noise(), RATE)["spectral_flatness"]
            > descriptors(tone(440.0), RATE)["spectral_flatness"] * 5
        )

    def test_duration_is_reported_from_the_samples(self):
        assert descriptors(tone(440.0, seconds=3.0), RATE)["duration_seconds"] == pytest.approx(
            3.0, abs=0.01
        )


class TestMfcc:
    def test_returns_the_expected_shape(self):
        assert mfcc(tone(440.0), RATE).shape == (12,)

    def test_drops_the_loudness_coefficient(self):
        """Coefficient 0 is level; keeping it would compare volume."""
        quiet, loud = tone(440.0), tone(440.0) * 4
        assert cosine(mfcc(quiet, RATE), mfcc(loud, RATE)) > 0.99

    def test_is_finite_on_silence(self):
        assert np.all(np.isfinite(mfcc(np.zeros(RATE), RATE)))


class TestCompare:
    def test_identical_audio_is_maximally_similar(self):
        x = tone(440.0)
        result = compare(x, x, RATE)
        assert result["waveform_correlation"] == pytest.approx(1.0, abs=1e-6)
        assert result["centroid_delta_hz"] == pytest.approx(0.0, abs=1e-6)
        assert result["mfcc_cosine"] == pytest.approx(1.0, abs=1e-6)

    def test_centroid_delta_has_the_reported_direction(self):
        """Positive means the second argument is brighter.

        The whole reference finding is read off the sign of this number,
        so the convention is pinned rather than assumed.
        """
        assert compare(tone(220.0), tone(3520.0), RATE)["centroid_delta_hz"] > 0
        assert compare(tone(3520.0), tone(220.0), RATE)["centroid_delta_hz"] < 0

    def test_unrelated_audio_has_no_signal_relationship(self):
        result = compare(tone(440.0), noise(), RATE)
        assert abs(result["waveform_correlation"]) < 0.1
        assert result["si_sdr_db"] < 0

    def test_every_reported_field_is_present(self):
        result = compare(tone(440.0), tone(880.0), RATE)
        for field in (
            "waveform_correlation",
            "si_sdr_db",
            "mfcc_cosine",
            "centroid_delta_hz",
            "rolloff_delta_hz",
            "chroma_sequence_similarity",
            "onset_correlation",
        ):
            assert field in result, field


class TestRecordedResults:
    """The committed numbers must still say what the report says they do."""

    def test_the_analysis_file_covers_every_run(self):
        import json

        analysis = json.loads((SCRIPTS.parent / "analysis.json").read_text())
        for run in (
            "00_PROMPT_ONLY",
            "01_REFERENCE_A",
            "02_REFERENCE_B",
            "03_REFERENCE_A_CONTRADICTORY_PROMPT",
            "04_CONTRADICTORY_PROMPT_ONLY",
            "05_REFERENCE_A_DIFFERENT_SEED",
        ):
            assert run in analysis["descriptors"], run

    def test_the_reference_moved_brightness_in_both_directions(self):
        """The central finding, asserted against the stored measurements."""
        import json

        d = json.loads((SCRIPTS.parent / "analysis.json").read_text())["descriptors"]
        control = d["00_PROMPT_ONLY"]["spectral_centroid_hz"]
        with_bright = d["01_REFERENCE_A"]["spectral_centroid_hz"]
        with_dark = d["02_REFERENCE_B"]["spectral_centroid_hz"]
        assert with_bright > control, "the bright reference should brighten the output"
        assert with_dark < control, "the dark reference should darken the output"

    def test_the_reference_effect_exceeds_the_seed_noise_floor(self):
        import json

        d = json.loads((SCRIPTS.parent / "analysis.json").read_text())["descriptors"]
        control = d["00_PROMPT_ONLY"]["spectral_centroid_hz"]
        run_01 = d["01_REFERENCE_A"]["spectral_centroid_hz"]
        run_05 = d["05_REFERENCE_A_DIFFERENT_SEED"]["spectral_centroid_hz"]
        reference_effect = abs(run_01 - control)
        seed_noise = abs(run_05 - run_01)
        assert reference_effect > seed_noise * 5

    def test_no_output_shares_audio_with_a_reference(self):
        """The safety property: influence, never copying."""
        import json

        comparisons = json.loads((SCRIPTS.parent / "analysis.json").read_text())["comparisons"]
        for label, values in comparisons.items():
            if "->" in label:
                assert values["si_sdr_db"] < -20, label
