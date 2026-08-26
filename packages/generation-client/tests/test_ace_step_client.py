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


# ── /v1/models response shapes ────────────────────────────────────────
#
# Two shapes are in the wild. The documented envelope is what the pinned
# upstream docs describe; the installed 1.5 server answers in the OpenAI
# listing style with no `code` field. Measured against a live server, the
# envelope validator rejected the installed shape outright — a healthy
# engine holding a loaded model reported an error. Both are pinned here.


def _models_client(payload, status_code: int = 200) -> AceStepClient:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(status_code, json=payload)
        return httpx.Response(404)

    return AceStepClient("http://acestep.test", transport=httpx.MockTransport(handler))


async def test_list_models_reads_the_installed_openai_style_listing():
    """`{object: "list", data: [{id, name}]}` — no envelope, no `code`."""
    async with _models_client(
        {
            "object": "list",
            "data": [
                {"id": "acestep/acestep-v15-turbo", "name": "ACE-Step acestep-v15-turbo"},
                {"id": "acestep/acestep-v15-base", "name": "ACE-Step acestep-v15-base"},
            ],
        }
    ) as client:
        models = await client.list_models()
    # The id is what the protocol addresses; `name` is a display label.
    assert models.models == ["acestep/acestep-v15-turbo", "acestep/acestep-v15-base"]


async def test_list_models_still_reads_the_documented_envelope():
    async with _models_client(
        {
            "code": 200,
            "error": None,
            "data": {
                "models": [{"name": "acestep-v15-turbo"}],
                "default_model": "acestep-v15-turbo",
            },
        }
    ) as client:
        models = await client.list_models()
    assert models.models == ["acestep-v15-turbo"]
    assert models.default_model == "acestep-v15-turbo"


async def test_list_models_falls_back_to_name_when_an_entry_has_no_id():
    async with _models_client({"object": "list", "data": [{"name": "solo-name"}]}) as client:
        models = await client.list_models()
    assert models.models == ["solo-name"]


async def test_list_models_skips_entries_that_name_nothing():
    """A nameless entry is dropped rather than given an invented name."""
    async with _models_client(
        {"object": "list", "data": [{"id": "real"}, {"context_length": 4096}, "junk"]}
    ) as client:
        models = await client.list_models()
    assert models.models == ["real"]


async def test_an_empty_listing_is_reported_as_empty_not_as_an_error():
    """Lazy-load leaves a healthy server with no models. That is a fact."""
    async with _models_client({"object": "list", "data": []}) as client:
        models = await client.list_models()
    assert models.models == []


async def test_a_malformed_body_raises_rather_than_reporting_no_models():
    """The failure this fix exists to prevent: silence that looks empty."""
    for payload in ({"object": "list"}, {"data": "not-a-list"}, {"data": {"models": 5}}):
        async with _models_client(payload) as client:
            with pytest.raises(AceStepApiError):
                await client.list_models()


async def test_an_envelope_error_is_still_surfaced():
    async with _models_client({"code": 500, "error": "engine down", "data": {}}) as client:
        with pytest.raises(AceStepApiError, match="engine down"):
            await client.list_models()


async def test_a_non_200_is_an_error():
    async with _models_client({"object": "list", "data": []}, 503) as client:
        with pytest.raises(AceStepApiError):
            await client.list_models()
