from luber_audio_utils import (
    MASTER_BIT_DEPTH,
    MASTER_CHANNELS,
    MASTER_SAMPLE_RATE,
)


def test_master_format_contract():
    assert MASTER_SAMPLE_RATE == 48_000
    assert MASTER_BIT_DEPTH == 24
    assert MASTER_CHANNELS == 2
