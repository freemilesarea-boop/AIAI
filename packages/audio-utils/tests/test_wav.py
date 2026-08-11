from pathlib import Path

import pytest

from luber_audio_utils import WavValidationError, inspect_wav

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


def test_inspect_fixture_wav():
    info = inspect_wav(FIXTURE)
    assert info.sample_rate == 48000
    assert info.channels == 2
    assert info.bit_depth == 16
    assert info.duration_seconds == pytest.approx(2.0)
    assert info.file_size > 0
    assert len(info.sha256) == 64


def test_sha256_is_deterministic():
    assert inspect_wav(FIXTURE).sha256 == inspect_wav(FIXTURE).sha256


def test_missing_file_raises(tmp_path):
    with pytest.raises(WavValidationError, match="does not exist"):
        inspect_wav(tmp_path / "nope.wav")


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(WavValidationError, match="empty"):
        inspect_wav(empty)


def test_non_wav_content_raises(tmp_path):
    fake = tmp_path / "fake.wav"
    fake.write_text('{"not": "audio"}')
    with pytest.raises(WavValidationError, match="not a valid WAV"):
        inspect_wav(fake)
