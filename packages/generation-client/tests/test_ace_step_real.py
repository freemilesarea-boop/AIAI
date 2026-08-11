"""REAL ACE-Step model integration test — requires a live server.

Deliberately excluded from normal CI: it needs a running ACE-Step 1.5
API server with downloaded model weights. This is the one sanctioned
conditional skip (external model runtime requirement).

Run with:

    RUN_ACE_STEP_INTEGRATION=1 ACE_STEP_BASE_URL=http://127.0.0.1:8001 \
        uv run pytest packages/generation-client/tests/test_ace_step_real.py -v
"""

import os
import wave
from pathlib import Path

import pytest

from luber_audio_utils import inspect_wav
from luber_generation_client import GenerationRequest
from luber_generation_client.ace_step import AceStepProvider, AceStepProviderConfig
from luber_schemas import VocalGender

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACE_STEP_INTEGRATION") != "1",
    reason="real ACE-Step server required (set RUN_ACE_STEP_INTEGRATION=1)",
)


async def test_real_ace_step_generation(tmp_path: Path):
    config = AceStepProviderConfig(
        base_url=os.environ.get("ACE_STEP_BASE_URL", "http://127.0.0.1:8001"),
        api_key=os.environ.get("ACE_STEP_API_KEY") or None,
        model=os.environ.get("ACE_STEP_MODEL", "acestep-v15-turbo"),
        output_dir=tmp_path,
        generation_timeout=float(os.environ.get("ACE_STEP_GENERATION_TIMEOUT", "900")),
    )
    provider = AceStepProvider(config)
    try:
        result = await provider.generate(
            GenerationRequest(
                title="PHASE 2 REAL TEST",
                prompt=(
                    "Dreamy Korean indie pop with warm electric piano, "
                    "soft drums and emotional female lead vocal"
                ),
                lyrics=(
                    "[Verse]\n오늘 밤 너를 생각해\n조용한 창가에 앉아\n"
                    "[Chorus]\n내 곁에 조금만 더 있어줘\n이 밤이 지나가기 전에"
                ),
                vocal_gender=VocalGender.FEMALE,
                duration_seconds=30,
                language="ko",
            )
        )
    finally:
        await provider.close()

    # This must be a NEW real model output — validate thoroughly.
    info = inspect_wav(result.audio_path)
    assert info.duration_seconds > 0
    assert info.sample_rate > 0
    assert info.channels > 0
    assert result.provider == "ace_step"
    assert result.model_name.startswith("acestep-v15")

    # Not silent: peak amplitude must be clearly non-zero.
    with wave.open(str(result.audio_path), "rb") as w:
        frames = w.readframes(min(w.getnframes(), w.getframerate() * 10))
    peak = max(
        abs(int.from_bytes(frames[i : i + 2], "little", signed=True))
        for i in range(0, len(frames), 2)
    )
    assert peak > 100, "generated audio appears to be silence"
    print(
        f"REAL GENERATION OK: {result.audio_path} sha256={info.sha256} "
        f"duration={info.duration_seconds:.2f}s rate={info.sample_rate} "
        f"channels={info.channels} peak={peak}"
    )
