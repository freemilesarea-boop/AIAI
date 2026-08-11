"""AceStepProvider plugged into the real GenerationService pipeline.

Uses the fake ACE-Step server (documented protocol) — proves the
provider satisfies the same contract MockGenerationProvider does and
flows through repository → storage → COMPLETED without the service
knowing anything ACE-Step-specific.
"""

from pathlib import Path

from ace_step_fake_server import FakeAceStepServer

from luber_audio_utils import LocalAudioStorage
from luber_generation_client import GenerationService, MusicGenerationProvider
from luber_generation_client.ace_step import AceStepClient, AceStepProvider, AceStepProviderConfig
from luber_generation_client.mock import MockGenerationProvider
from luber_schemas import AssetType, GenerationStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


def test_both_providers_satisfy_the_contract():
    assert issubclass(AceStepProvider, MusicGenerationProvider)
    assert issubclass(MockGenerationProvider, MusicGenerationProvider)
    assert AceStepProvider.name == "ace_step"
    assert MockGenerationProvider.name == "mock"


async def test_generation_service_completes_with_ace_step_provider(repository, tmp_path):
    gen = await repository.create_generation(
        title="PHASE 2 REAL TEST",
        prompt="Dreamy Korean indie pop with warm electric piano",
        lyrics="[Verse]\n오늘 밤 너를 생각해",
        vocal_gender="female",
        duration_requested=30,
        status=GenerationStatus.QUEUED.value,
        language="ko",
    )
    server = FakeAceStepServer(FIXTURE, polls_before_success=1)
    config = AceStepProviderConfig(
        base_url="http://acestep.test",
        output_dir=tmp_path / "raw",
        poll_interval=0.01,
        generation_timeout=5.0,
    )
    provider = AceStepProvider(
        config, client=AceStepClient(config.base_url, transport=server.transport())
    )
    service = GenerationService(repository, provider, LocalAudioStorage(tmp_path / "store"))

    final = await service.execute(gen.id, worker_id="phase2-test")

    assert final is GenerationStatus.COMPLETED
    fetched = await repository.get_generation(gen.id)
    assert fetched.provider == "ace_step"
    assert fetched.model_name == "acestep-v15-turbo"
    assert fetched.duration_actual and fetched.duration_actual > 0
    assets = await repository.get_audio_assets(gen.id)
    assert len(assets) == 1
    assert assets[0].asset_type == AssetType.MASTER.value
    stored = tmp_path / "store" / assets[0].storage_key
    assert stored.is_file()
    # No mock values anywhere.
    assert fetched.provider != "mock"
    assert fetched.model_name != "mock-generation-provider"
