"""The one distinction a whole-file silence ratio cannot make.

A song that fades out over its last six seconds and a song that stops
dead after twelve of forty both contain a lot of silence. Averaged over
the file they can look identical. The difference is *where* the silence
is and whether anything follows it, which is why this detector is
positional rather than a ratio.

Getting this wrong in either direction is expensive: a false positive
throws away a finished song for having an outro, and a false negative
delivers twenty-eight seconds of nothing.
"""

from __future__ import annotations

import qc_fixtures as fx

from luber_audio_finishing import load_audio
from luber_inference_qc import measure_collapse
from luber_inference_qc.collapse import MINIMUM_TRAILING_SILENCE_SECONDS


def _measure(path):
    return measure_collapse(load_audio(path))


def test_content_that_stops_dead_is_found(audio_dir):
    result = _measure(fx.early_collapse(audio_dir / "stop.wav", seconds=40.0, content_seconds=12.0))
    assert result.content_end_seconds < 13.0
    assert result.trailing_silence_seconds > MINIMUM_TRAILING_SILENCE_SECONDS


def test_a_fade_to_nothing_is_not_a_collapse(audio_dir):
    """The audio gets quieter and quieter. It never stops."""
    result = _measure(fx.quiet_outro(audio_dir / "fade.wav", seconds=20.0))
    assert result.trailing_silence_seconds < MINIMUM_TRAILING_SILENCE_SECONDS


def test_a_healthy_file_has_no_trailing_silence_worth_naming(audio_dir):
    result = _measure(fx.healthy(audio_dir / "ok.wav", seconds=12.0))
    assert result.trailing_silence_seconds < 1.0
    assert result.content_ratio > 0.9
    assert result.entirely_silent is False


def test_digital_silence_is_reported_as_such_rather_than_as_a_collapse(audio_dir):
    """Nothing at all is a different failure, and it says so.

    A file with no content has no content *end*, so treating it as a
    collapse would report a position that does not exist.
    """
    result = _measure(fx.silent(audio_dir / "nothing.wav"))
    assert result.entirely_silent is True


def test_the_measurement_is_relative_so_a_quiet_master_is_not_a_collapse(audio_dir):
    """A mix mastered 20 dB down is quiet, not truncated.

    The floor is set relative to the file's own loud passages, so the
    same music at a lower level measures the same. An absolute floor
    would call every quiet master a collapse from the first bar.
    """
    loud = _measure(fx.healthy(audio_dir / "loud.wav", seconds=12.0))
    quiet = fx.write_wav(
        audio_dir / "quiet.wav",
        [(left * 0.1, right * 0.1) for left, right in fx._tone_frames(12.0)],
    )
    assert _measure(quiet).content_end_seconds == loud.content_end_seconds
