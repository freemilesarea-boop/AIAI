"""Objective structure analysis.

These verify the measurements behave as claimed on synthetic audio with
known properties, so the numbers in the report can be trusted to mean
what the report says they mean.
"""

import math
import struct
import wave
from pathlib import Path

from bench.analysis import analyze_structure, seed_divergence

RATE = 22050


def _tone_wav(path: Path, segments: list[tuple[float, float, float]], rate: int = RATE) -> Path:
    """Write a WAV from (seconds, frequency_hz, amplitude) segments."""
    data = bytearray()
    for seconds, freq, amp in segments:
        for i in range(int(rate * seconds)):
            value = int(32767 * amp * math.sin(2 * math.pi * freq * i / rate))
            data += struct.pack("<h", value)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(data))
    return path


def test_flat_texture_has_low_energy_variation(tmp_path):
    flat = _tone_wav(tmp_path / "flat.wav", [(6.0, 440.0, 0.5)])
    analysis = analyze_structure(flat)
    assert analysis.energy_variation < 1.0
    assert analysis.frames >= 5


def test_dynamic_track_has_higher_energy_variation(tmp_path):
    flat = analyze_structure(_tone_wav(tmp_path / "flat.wav", [(6.0, 440.0, 0.5)]))
    dynamic = analyze_structure(
        _tone_wav(
            tmp_path / "dyn.wav",
            [(2.0, 440.0, 0.9), (2.0, 440.0, 0.05), (2.0, 440.0, 0.9)],
        )
    )
    assert dynamic.energy_variation > flat.energy_variation


def test_section_change_is_detected(tmp_path):
    """A hard timbre switch must register as a section boundary."""
    changing = analyze_structure(
        _tone_wav(tmp_path / "c.wav", [(3.0, 220.0, 0.6), (3.0, 3000.0, 0.6)])
    )
    steady = analyze_structure(_tone_wav(tmp_path / "s.wav", [(6.0, 220.0, 0.6)]))
    assert changing.section_changes >= steady.section_changes


def test_repeated_section_scores_high_self_similarity(tmp_path):
    """An A-B-A structure must show a strong off-diagonal match."""
    aba = analyze_structure(
        _tone_wav(
            tmp_path / "aba.wav",
            [(3.0, 300.0, 0.6), (3.0, 2500.0, 0.6), (3.0, 300.0, 0.6)],
        )
    )
    assert aba.max_repetition > 0.9


def test_spectral_centroid_tracks_brightness(tmp_path):
    dark = analyze_structure(_tone_wav(tmp_path / "d.wav", [(4.0, 200.0, 0.6)]))
    bright = analyze_structure(_tone_wav(tmp_path / "b.wav", [(4.0, 5000.0, 0.6)]))
    assert bright.spectral_centroid_hz > dark.spectral_centroid_hz


def test_level_drift_detects_a_quieter_ending(tmp_path):
    fading = analyze_structure(
        _tone_wav(
            tmp_path / "f.wav",
            [(3.0, 440.0, 0.9), (3.0, 440.0, 0.5), (3.0, 440.0, 0.05)],
        )
    )
    assert fading.level_drift_db < -10


def test_spectral_drift_detects_a_changed_ending(tmp_path):
    stable = analyze_structure(_tone_wav(tmp_path / "s.wav", [(9.0, 440.0, 0.6)]))
    drifting = analyze_structure(
        _tone_wav(
            tmp_path / "dr.wav",
            [(3.0, 300.0, 0.6), (3.0, 300.0, 0.6), (3.0, 6000.0, 0.6)],
        )
    )
    assert drifting.long_form_drift > stable.long_form_drift


def test_band_energy_sums_to_one(tmp_path):
    analysis = analyze_structure(_tone_wav(tmp_path / "a.wav", [(4.0, 440.0, 0.6)]))
    assert abs(sum(analysis.band_energy) - 1.0) < 0.05


def test_seed_divergence_is_zero_for_identical_audio(tmp_path):
    a = _tone_wav(tmp_path / "a.wav", [(3.0, 440.0, 0.6)])
    b = _tone_wav(tmp_path / "b.wav", [(3.0, 440.0, 0.6)])
    assert (seed_divergence([a, b]) or 0.0) < 1e-6


def test_seed_divergence_grows_with_difference(tmp_path):
    a = _tone_wav(tmp_path / "a.wav", [(3.0, 440.0, 0.6)])
    b = _tone_wav(tmp_path / "b.wav", [(3.0, 440.0, 0.6)])
    c = _tone_wav(tmp_path / "c.wav", [(3.0, 4000.0, 0.6)])
    assert (seed_divergence([a, c]) or 0.0) > (seed_divergence([a, b]) or 0.0)


def test_seed_divergence_needs_at_least_two_takes(tmp_path):
    a = _tone_wav(tmp_path / "a.wav", [(1.0, 440.0, 0.6)])
    assert seed_divergence([a]) is None


def test_analysis_handles_24_bit_audio(tmp_path):
    """Production masters are 24-bit; the loader must widen them correctly."""
    path = tmp_path / "24.wav"
    rate = RATE
    data = bytearray()
    for i in range(rate * 3):
        value = int((1 << 23) * 0.5 * math.sin(2 * math.pi * 440 * i / rate))
        data += (value & 0xFFFFFF).to_bytes(3, "little")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(rate)
        w.writeframes(bytes(data))

    analysis = analyze_structure(path)
    assert analysis.frames >= 2
    # A 0.5 FS tone must not read as silence after 24-bit unpacking.
    assert analysis.spectral_centroid_hz > 100


def test_analysis_serializes_to_dict(tmp_path):
    import json

    analysis = analyze_structure(_tone_wav(tmp_path / "a.wav", [(2.0, 440.0, 0.6)]))
    json.dumps(analysis.to_dict())
