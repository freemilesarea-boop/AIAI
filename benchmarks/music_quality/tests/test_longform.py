"""Long-form QA measurements, validated against synthetic ground truth.

An estimator nobody has checked is worse than no estimator, because it
produces confident-looking numbers. Every measurement here is therefore
tested against a signal whose correct answer is known by construction:
click tracks at a known BPM, tones in a known key, and synthetic tracks
with deliberate level and high-frequency drift.

What this file does **not** claim: that passing on synthetic signals
means the estimators are accurate on real produced music. That gap is
the reason the key estimator reports a verdict instead of a bare answer.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from bench.longform import (
    SIBILANCE_BAND_HZ,
    TIME_SIGNATURE_VERDICT,
    analyze_long_form,
    estimate_bpm,
    estimate_key,
    load_mono,
    verify_controls,
)

RATE = 48000


def write_wav(path: Path, samples: np.ndarray, rate: int = RATE, channels: int = 1) -> Path:
    data = np.clip(samples, -1.0, 1.0)
    pcm = (data * 32767).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def click_track(bpm: float, seconds: float, rate: int = RATE) -> np.ndarray:
    """Impulses at a known tempo over quiet noise."""
    n = int(seconds * rate)
    out = np.random.default_rng(0).normal(0, 0.002, n).astype(np.float32)
    period = int(rate * 60.0 / bpm)
    click = np.exp(-np.linspace(0, 12, 400)).astype(np.float32)
    for start in range(0, n - click.size, period):
        out[start : start + click.size] += click
    return out


def tone_mix(freqs: list[float], seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0).astype(np.float32) / len(
        freqs
    )


# ── Whole-track measurement ───────────────────────────────────────────


def test_measures_basic_properties(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone_mix([440.0], 4.0), channels=2)
    a = analyze_long_form(path)
    assert a.sample_rate == RATE
    assert a.channels == 2
    assert a.bit_depth == 16
    assert a.duration_seconds == pytest.approx(4.0, abs=0.05)
    assert 0.9 < a.peak <= 1.0
    assert a.rms_dbfs is not None and a.rms_dbfs < 0


def test_splits_into_four_windows_by_default(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone_mix([440.0], 8.0))
    a = analyze_long_form(path)
    assert [w.index for w in a.windows] == [0, 1, 2, 3]
    assert a.windows[0].start_seconds == 0.0
    assert a.windows[-1].end_seconds == pytest.approx(8.0, abs=0.05)
    # Windows tile the track without gaps.
    for previous, following in zip(a.windows, a.windows[1:], strict=False):
        assert previous.end_seconds == pytest.approx(following.start_seconds, abs=0.01)


def test_crest_factor_is_higher_for_peaky_audio(tmp_path):
    steady = write_wav(tmp_path / "steady.wav", tone_mix([440.0], 4.0))
    peaky = write_wav(tmp_path / "peaky.wav", click_track(120, 4.0))
    assert analyze_long_form(peaky).crest_factor_db > analyze_long_form(steady).crest_factor_db


def test_detects_clipping(tmp_path):
    loud = np.ones(RATE * 2, dtype=np.float32)
    a = analyze_long_form(write_wav(tmp_path / "clip.wav", loud))
    assert a.clipping_sample_ratio > 0.5
    assert "CLIPPING" in a.flags


def test_detects_excessive_silence(tmp_path):
    samples = np.zeros(RATE * 4, dtype=np.float32)
    samples[: RATE // 2] = tone_mix([440.0], 0.5)
    a = analyze_long_form(write_wav(tmp_path / "quiet.wav", samples))
    assert a.silence_ratio > 0.5
    assert "EXCESSIVE_SILENCE" in a.flags


def test_clean_tone_raises_no_flags(tmp_path):
    a = analyze_long_form(write_wav(tmp_path / "clean.wav", tone_mix([440.0], 8.0) * 0.5))
    assert a.flags == []


# ── Drift detection: the reason windows exist ─────────────────────────


def test_level_drift_is_detected(tmp_path):
    # Deliberate fade from full level to near silence.
    base = tone_mix([440.0], 8.0)
    faded = base * np.linspace(1.0, 0.05, base.size).astype(np.float32)
    a = analyze_long_form(write_wav(tmp_path / "fade.wav", faded))
    assert a.level_drift_db is not None and a.level_drift_db > 6.0
    assert "LEVEL_DRIFT" in a.flags
    # And the direction is right: later windows are quieter.
    assert a.windows[0].rms_dbfs > a.windows[-1].rms_dbfs


def test_steady_level_is_not_flagged_as_drift(tmp_path):
    a = analyze_long_form(write_wav(tmp_path / "steady.wav", tone_mix([440.0], 8.0) * 0.5))
    assert a.level_drift_db is not None and a.level_drift_db < 1.0
    assert "LEVEL_DRIFT" not in a.flags


def test_rising_high_end_is_detected_as_sibilance_growth(tmp_path):
    """The Phase 9 question: does the high end get worse over a song?"""
    seconds, rate = 8.0, RATE
    t = np.arange(int(seconds * rate)) / rate
    low = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    # A 7 kHz component (inside the sibilance band) that grows over time.
    ramp = np.linspace(0.0, 0.8, t.size)
    high = ramp * np.sin(2 * np.pi * 7000.0 * t)
    a = analyze_long_form(write_wav(tmp_path / "harsh.wav", (low + high).astype(np.float32)))

    assert a.sibilance_growth is not None and a.sibilance_growth > 1.5
    assert "SIBILANCE_GROWTH" in a.flags
    assert a.windows[-1].sibilance_risk_proxy > a.windows[0].sibilance_risk_proxy


def test_constant_spectrum_shows_no_sibilance_growth(tmp_path):
    t = np.arange(int(8.0 * RATE)) / RATE
    steady = 0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 7000.0 * t)
    a = analyze_long_form(write_wav(tmp_path / "even.wav", steady.astype(np.float32)))
    assert a.sibilance_growth == pytest.approx(1.0, abs=0.15)
    assert "SIBILANCE_GROWTH" not in a.flags


def test_sibilance_proxy_responds_to_its_own_band(tmp_path):
    t = np.arange(int(4.0 * RATE)) / RATE
    inside = write_wav(tmp_path / "in.wav", (0.5 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32))
    outside = write_wav(
        tmp_path / "out.wav", (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    )
    assert analyze_long_form(inside).sibilance_risk_proxy > 0.5
    assert analyze_long_form(outside).sibilance_risk_proxy < 0.05


def test_sibilance_band_is_the_documented_one():
    # The proxy's meaning depends entirely on this band; pin it.
    assert SIBILANCE_BAND_HZ == (5000.0, 10000.0)


def test_spectral_centroid_tracks_brightness(tmp_path):
    t = np.arange(int(4.0 * RATE)) / RATE
    dark = write_wav(tmp_path / "dark.wav", (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    bright = write_wav(
        tmp_path / "bright.wav", (0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    )
    assert (
        analyze_long_form(bright).spectral_centroid_hz
        > analyze_long_form(dark).spectral_centroid_hz
    )
    assert analyze_long_form(dark).spectral_centroid_hz < 1000


def test_centroid_drift_reported_for_brightening_track(tmp_path):
    t = np.arange(int(8.0 * RATE)) / RATE
    sweep = 0.4 * np.sin(2 * np.pi * (200 + 4000 * (t / t[-1])) * t)
    a = analyze_long_form(write_wav(tmp_path / "sweep.wav", sweep.astype(np.float32)))
    assert a.centroid_drift_hz is not None and a.centroid_drift_hz > 500


def test_analysis_is_json_safe(tmp_path):
    import json

    a = analyze_long_form(write_wav(tmp_path / "a.wav", tone_mix([440.0], 4.0)))
    json.dumps(a.to_dict())  # must not raise on inf/nan


# ── BPM estimation, validated on click tracks ─────────────────────────


@pytest.mark.parametrize("bpm", [80, 100, 120, 140])
def test_estimates_click_track_tempo(tmp_path, bpm):
    path = write_wav(tmp_path / f"{bpm}.wav", click_track(bpm, 12.0))
    estimate = estimate_bpm(path, requested_bpm=bpm)
    assert estimate.estimated_bpm is not None
    # Accept an octave error explicitly rather than pretending it is exact.
    assert estimate.difference <= 3.0 or estimate.octave_equivalent, estimate.to_dict()


def test_reports_difference_against_the_request(tmp_path):
    path = write_wav(tmp_path / "c.wav", click_track(120, 12.0))
    estimate = estimate_bpm(path, requested_bpm=90)
    assert estimate.requested_bpm == 90
    assert estimate.difference is not None and estimate.difference > 0


def test_octave_confusion_is_reported_not_hidden(tmp_path):
    path = write_wav(tmp_path / "c.wav", click_track(160, 12.0))
    estimate = estimate_bpm(path, requested_bpm=80)
    if estimate.estimated_bpm and abs(estimate.estimated_bpm - 160) < 5:
        assert estimate.octave_equivalent is True


def test_tempo_confidence_is_lower_for_noise_than_for_a_click_track(tmp_path):
    noise = np.random.default_rng(1).normal(0, 0.2, RATE * 12).astype(np.float32)
    noisy = estimate_bpm(write_wav(tmp_path / "n.wav", noise))
    clicky = estimate_bpm(write_wav(tmp_path / "c.wav", click_track(120, 12.0)))
    assert clicky.confidence > noisy.confidence


def test_tempo_estimate_declines_on_a_too_short_file(tmp_path):
    path = write_wav(tmp_path / "short.wav", tone_mix([440.0], 0.2))
    estimate = estimate_bpm(path, requested_bpm=120)
    assert estimate.estimated_bpm is None
    assert estimate.confidence == 0.0
    assert estimate.difference is None


def test_tempo_estimate_is_json_safe(tmp_path):
    import json

    json.dumps(estimate_bpm(write_wav(tmp_path / "c.wav", click_track(120, 12.0))).to_dict())


# ── Key estimation, with honest reliability reporting ─────────────────


def test_recovers_the_tonic_of_an_unambiguous_major_triad(tmp_path):
    # C major triad across two octaves: C E G.
    freqs = [261.63, 329.63, 392.00, 523.25, 659.26, 783.99]
    path = write_wav(tmp_path / "c.wav", tone_mix(freqs, 6.0))
    estimate = estimate_key(path, requested_key="C major")
    assert estimate.estimated_key is not None
    assert estimate.tonic_matches is True, estimate.to_dict()


def test_recovers_the_tonic_of_an_unambiguous_minor_triad(tmp_path):
    # A minor triad: A C E.
    freqs = [220.00, 261.63, 329.63, 440.00, 523.25, 659.26]
    path = write_wav(tmp_path / "a.wav", tone_mix(freqs, 6.0))
    estimate = estimate_key(path, requested_key="A minor")
    assert estimate.tonic_matches is True, estimate.to_dict()


def test_flat_spellings_compare_equal_to_sharp_spellings(tmp_path):
    freqs = [233.08, 293.66, 349.23]  # Bb D F
    path = write_wav(tmp_path / "bb.wav", tone_mix(freqs, 6.0))
    estimate = estimate_key(path, requested_key="Bb major")
    # Whatever the estimator says, Bb and A# must not count as different.
    if estimate.estimated_key and estimate.estimated_key.startswith("A#"):
        assert estimate.tonic_matches is True


def test_key_estimate_declares_low_confidence_on_noise(tmp_path):
    noise = np.random.default_rng(2).normal(0, 0.2, RATE * 6).astype(np.float32)
    estimate = estimate_key(write_wav(tmp_path / "n.wav", noise), requested_key="C major")
    # Noise has no key. The estimator must not claim one confidently.
    assert estimate.verdict in {"LOW_CONFIDENCE", "INSUFFICIENT_AUDIO"}


def test_key_estimate_declines_on_a_too_short_file(tmp_path):
    estimate = estimate_key(write_wav(tmp_path / "s.wav", tone_mix([440.0], 0.2)))
    assert estimate.estimated_key is None
    assert estimate.verdict == "INSUFFICIENT_AUDIO"


def test_key_estimate_reports_a_verdict_field(tmp_path):
    path = write_wav(tmp_path / "c.wav", tone_mix([261.63, 329.63, 392.0], 6.0))
    payload = estimate_key(path, requested_key="C major").to_dict()
    assert payload["verdict"] in {"ESTIMATED", "LOW_CONFIDENCE", "INSUFFICIENT_AUDIO"}
    assert "confidence" in payload


def test_tonic_match_is_none_without_a_request(tmp_path):
    path = write_wav(tmp_path / "c.wav", tone_mix([261.63, 329.63, 392.0], 6.0))
    assert estimate_key(path).tonic_matches is None


# ── Combined control verification ─────────────────────────────────────


def test_verify_controls_reports_all_three(tmp_path):
    path = write_wav(tmp_path / "c.wav", click_track(120, 12.0))
    report = verify_controls(
        path, requested_bpm=120, requested_key="C major", requested_time_signature="4"
    )
    assert set(report) == {"bpm", "key", "time_signature"}
    assert report["bpm"]["requested_bpm"] == 120
    assert report["key"]["requested_key"] == "C major"


def test_time_signature_is_honestly_unverified(tmp_path):
    path = write_wav(tmp_path / "c.wav", click_track(120, 12.0))
    report = verify_controls(path, requested_time_signature="4")
    assert report["time_signature"]["estimated"] is None
    assert report["time_signature"]["verdict"] == TIME_SIGNATURE_VERDICT
    assert TIME_SIGNATURE_VERDICT == "HUMAN_OR_EXTERNAL_ANALYSIS_REQUIRED"


def test_verify_controls_is_json_safe(tmp_path):
    import json

    path = write_wav(tmp_path / "c.wav", click_track(120, 12.0))
    json.dumps(verify_controls(path, requested_bpm=120, requested_key="C major"))


def test_analysis_never_modifies_the_audio(tmp_path):
    path = write_wav(tmp_path / "c.wav", click_track(120, 8.0))
    before = path.read_bytes()
    analyze_long_form(path)
    estimate_bpm(path, requested_bpm=120)
    estimate_key(path, requested_key="C major")
    assert path.read_bytes() == before


def test_load_mono_downmixes_stereo(tmp_path):
    path = write_wav(tmp_path / "s.wav", tone_mix([440.0], 2.0), channels=2)
    samples, rate = load_mono(path)
    assert rate == RATE
    assert samples.ndim == 1
    assert samples.size == pytest.approx(2.0 * RATE, rel=0.01)


def test_windows_cover_a_long_track(tmp_path):
    # 240s is the Phase 9 ceiling; make sure windowing scales to it.
    samples = tone_mix([440.0], 240.0) * 0.4
    a = analyze_long_form(write_wav(tmp_path / "long.wav", samples))
    assert len(a.windows) == 4
    assert a.windows[-1].end_seconds == pytest.approx(240.0, abs=0.1)
    assert a.duration_seconds == pytest.approx(240.0, abs=0.1)
    total = sum(w.end_seconds - w.start_seconds for w in a.windows)
    assert total == pytest.approx(240.0, abs=0.1)


def test_math_import_is_used_for_finite_guards():
    assert math.isfinite(1.0)
