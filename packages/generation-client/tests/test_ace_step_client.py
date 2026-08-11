from pathlib import Path

import pytest
from ace_step_fake_server import FakeAceStepServer

from luber_generation_client.ace_step import AceStepApiError, AceStepClient
from luber_generation_client.ace_step.types import AceStepTaskStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


def _client(server: FakeAceStepServer, **kwargs) -> AceStepClient:
    return AceStepClient("http://acestep.test", transport=server.transport(), **kwargs)


async def test_health_unwraps_envelope():
    server = FakeAceStepServer(FIXTURE)
    async with _client(server) as client:
        health = await client.health()
    assert health.status == "ok"
    assert health.service == "ACE-Step API"
    assert health.models_initialized is True
    assert health.loaded_model == "acestep-v15-turbo"


async def test_list_models():
    server = FakeAceStepServer(FIXTURE)
    async with _client(server) as client:
        models = await client.list_models()
    assert models.models == ["acestep-v15-turbo"]
    assert models.default_model == "acestep-v15-turbo"


async def test_submit_and_query_generation_parses_json_string_result():
    server = FakeAceStepServer(FIXTURE, polls_before_success=0)
    async with _client(server) as client:
        handle = await client.submit_generation({"prompt": "test", "batch_size": 1})
        assert handle.task_id
        result = await client.query_generation(handle.task_id)
    assert result.status is AceStepTaskStatus.SUCCEEDED
    assert len(result.tracks) == 1
    track = result.tracks[0]
    assert track.file_url.startswith("/v1/audio?path=")
    assert track.first_seed() == 12345  # "12345,67890" → first int
    assert track.dit_model == "acestep-v15-turbo"


async def test_download_audio_writes_real_bytes(tmp_path):
    server = FakeAceStepServer(FIXTURE, polls_before_success=0)
    async with _client(server) as client:
        handle = await client.submit_generation({"prompt": "x"})
        result = await client.query_generation(handle.task_id)
        dest = tmp_path / "out.wav"
        await client.download_audio(result.tracks[0].file_url, dest)
    assert dest.read_bytes() == FIXTURE.read_bytes()


async def test_api_key_sent_as_bearer_header():
    server = FakeAceStepServer(FIXTURE)
    async with _client(server, api_key="sk-test-123") as client:
        await client.health()
    assert server.auth_headers[-1] == "Bearer sk-test-123"


async def test_envelope_error_raises():
    import httpx

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": None, "code": 500, "error": "boom", "timestamp": 0, "extra": None},
        )

    async with AceStepClient(
        "http://acestep.test", transport=httpx.MockTransport(broken)
    ) as client:
        with pytest.raises(AceStepApiError, match="boom"):
            await client.health()
