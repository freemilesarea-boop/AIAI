"""Phase 8 advanced controls on the ACE-Step provider.

The most important test in this file is
``test_legacy_request_payload_is_byte_for_byte_phase7``: a Phase 7
request — one with none of the Phase 8 fields set — must still produce
exactly the payload the working generation path produced before Phase 8
existed. The expected payload is a frozen golden captured from the
committed Phase 7 provider at c4ea7ea, not a restatement of what the
current code happens to do.

Field names and value shapes for the three controls are taken from the
pinned engine (``acestep/api/http/release_task_models.py`` @ 6d467e4b:
``bpm: Optional[int]``, ``key_scale: str``, ``time_signature: str``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ace_step_fake_server import FakeAceStepServer
from pydantic import ValidationError

from luber_generation_client import GenerationRequest, MockGenerationProvider
from luber_generation_client.ace_step import (
    ACE_STEP_VERSION,
    AceStepClient,
    AceStepProvider,
    AceStepProviderConfig,
)
from luber_schemas import BPM_MAX, BPM_MIN, VocalGender

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

#: Distinctive values so a leak into the trace is unmistakable.
SECRET_API_KEY = "sk-live-DO-NOT-LEAK-2f4a9c"
SECRET_BASE_URL = "http://internal-gpu-07.acme.invalid:8001"


def _provider(tmp_path: Path, **overrides) -> AceStepProvider:
    config_kwargs: dict = {
        "base_url": "http://acestep.test",
        "output_dir": tmp_path / "raw",
        "poll_interval": 0.01,
        "generation_timeout": 5.0,
    }
    config_kwargs.update(overrides)
    config = AceStepProviderConfig(**config_kwargs)
    server = FakeAceStepServer(FIXTURE)
    client = AceStepClient(config.base_url, api_key=config.api_key, transport=server.transport())
    provider = AceStepProvider(config, client=client)
    provider.test_server = server  # type: ignore[attr-defined]
    return provider


def _legacy_request(**overrides) -> GenerationRequest:
    """A Phase 7-era request: nothing from Phase 8 is set."""
    defaults = dict(
        title="PHASE 7 LEGACY",
        prompt="Dreamy Korean indie pop with warm electric piano",
        lyrics="[Verse]\n오늘 밤 너를 생각해",
        vocal_gender=VocalGender.FEMALE,
        duration_seconds=30,
        language="ko",
        seed=777,
    )
    defaults.update(overrides)
    return GenerationRequest(**defaults)


# ── Backward compatibility: the Phase 7 path must not move ────────────

#: Golden payload for ``_legacy_request()``, captured from the committed
#: Phase 7 provider (c4ea7ea, before Phase 8 touched ``_build_payload``).
#: Any change to this dict is a change to the working generation path and
#: must be justified, not absorbed.
PHASE7_GOLDEN_PAYLOAD: dict[str, object] = {
    "prompt": (
        "Dreamy Korean indie pop with warm electric piano, "
        "female lead vocal, natural female singing voice"
    ),
    "lyrics": "[Verse]\n오늘 밤 너를 생각해",
    "vocal_language": "ko",
    "audio_duration": 30.0,
    "audio_format": "wav",
    "model": "acestep-v15-turbo",
    "inference_steps": 8,
    "thinking": False,
    "batch_size": 1,
    "use_cot_caption": False,
    "use_cot_language": False,
    "use_random_seed": False,
    "seed": 777,
}


def test_legacy_request_payload_is_byte_for_byte_phase7(tmp_path):
    provider = _provider(tmp_path)
    assert provider._build_payload(_legacy_request()) == PHASE7_GOLDEN_PAYLOAD


async def test_legacy_request_over_the_wire_is_byte_for_byte_phase7(tmp_path):
    # Same assertion, but on what the fake server actually received —
    # proving nothing between the provider and the socket adds a field.
    provider = _provider(tmp_path)
    await provider.generate(_legacy_request())
    assert provider.test_server.release_payloads[0] == PHASE7_GOLDEN_PAYLOAD


def test_legacy_payload_carries_no_advanced_control_keys(tmp_path):
    provider = _provider(tmp_path)
    payload = provider._build_payload(_legacy_request())
    for absent in ("bpm", "key_scale", "time_signature"):
        assert absent not in payload


def test_unset_controls_are_omitted_not_sent_as_empty(tmp_path):
    # Upstream reads "" as "not specified", but an empty string in the
    # trace is indistinguishable from a deliberate choice. Omit instead.
    provider = _provider(tmp_path)
    payload = provider._build_payload(
        _legacy_request(bpm=None, key_scale=None, time_signature=None)
    )
    assert payload == PHASE7_GOLDEN_PAYLOAD


async def test_legacy_generation_still_produces_real_audio(tmp_path):
    provider = _provider(tmp_path)
    result = await provider.generate(_legacy_request())
    assert result.audio_path.read_bytes() == FIXTURE.read_bytes()
    assert result.provider == "ace_step"
    assert result.model_version == ACE_STEP_VERSION


# ── Control propagation ───────────────────────────────────────────────


def test_bpm_propagates_as_an_integer(tmp_path):
    provider = _provider(tmp_path)
    payload = provider._build_payload(_legacy_request(bpm=128))
    assert payload["bpm"] == 128
    assert isinstance(payload["bpm"], int)


def test_key_scale_propagates_verbatim(tmp_path):
    provider = _provider(tmp_path)
    payload = provider._build_payload(_legacy_request(key_scale="F# minor"))
    assert payload["key_scale"] == "F# minor"


def test_time_signature_propagates_as_bare_numerator(tmp_path):
    provider = _provider(tmp_path)
    payload = provider._build_payload(_legacy_request(time_signature="3"))
    # Engine vocabulary is [2, 3, 4, 6] — never "3/4".
    assert payload["time_signature"] == "3"


def test_all_three_controls_together(tmp_path):
    provider = _provider(tmp_path)
    payload = provider._build_payload(
        _legacy_request(bpm=92, key_scale="Bb major", time_signature="6")
    )
    assert payload["bpm"] == 92
    assert payload["key_scale"] == "Bb major"
    assert payload["time_signature"] == "6"
    # Everything else is untouched.
    advanced = {"bpm", "key_scale", "time_signature"}
    assert {k: v for k, v in payload.items() if k not in advanced} == PHASE7_GOLDEN_PAYLOAD


async def test_controls_reach_the_server_over_the_wire(tmp_path):
    provider = _provider(tmp_path)
    await provider.generate(_legacy_request(bpm=140, key_scale="A minor", time_signature="4"))
    sent = provider.test_server.release_payloads[0]
    assert sent["bpm"] == 140
    assert sent["key_scale"] == "A minor"
    assert sent["time_signature"] == "4"


def test_bpm_zero_is_out_of_range_not_silently_dropped():
    with pytest.raises(ValidationError):
        _legacy_request(bpm=0)


# ── Request-level validation ──────────────────────────────────────────


@pytest.mark.parametrize("bpm", [BPM_MIN, 120, BPM_MAX])
def test_bpm_within_engine_bounds_is_accepted(bpm):
    assert _legacy_request(bpm=bpm).bpm == bpm


@pytest.mark.parametrize("bpm", [BPM_MIN - 1, BPM_MAX + 1, -10, 10_000])
def test_bpm_outside_engine_bounds_is_rejected(bpm):
    with pytest.raises(ValidationError):
        _legacy_request(bpm=bpm)


@pytest.mark.parametrize("key_scale", ["C major", "A minor", "F# minor", "Bb major", "G# major"])
def test_engine_supported_key_scales_are_accepted(key_scale):
    assert _legacy_request(key_scale=key_scale).key_scale == key_scale


@pytest.mark.parametrize("key_scale", ["H minor", "C dorian", "c major", "C", "Cmajor", "C♯ major"])
def test_unsupported_key_scales_are_rejected(key_scale):
    with pytest.raises(ValidationError):
        _legacy_request(key_scale=key_scale)


@pytest.mark.parametrize("value", ["2", "3", "4", "6"])
def test_engine_supported_time_signatures_are_accepted(value):
    assert _legacy_request(time_signature=value).time_signature == value


@pytest.mark.parametrize("value", ["4/4", "5", "7", "0", "four"])
def test_unsupported_time_signatures_are_rejected(value):
    with pytest.raises(ValidationError):
        _legacy_request(time_signature=value)


def test_empty_string_controls_normalize_to_unset():
    # An HTML select with no choice submits ""; that is "not specified",
    # not an invalid value.
    request = _legacy_request(key_scale="", time_signature="")
    assert request.key_scale is None
    assert request.time_signature is None


# ── Request trace ─────────────────────────────────────────────────────


def test_describe_request_reports_the_payload_that_would_be_sent(tmp_path):
    provider = _provider(tmp_path)
    request = _legacy_request(bpm=100, key_scale="C major", time_signature="4")
    trace = provider.describe_request(request)
    assert trace["payload"] == provider._build_payload(request)


def test_describe_request_records_engine_identity(tmp_path):
    provider = _provider(tmp_path)
    trace = provider.describe_request(_legacy_request())
    assert trace["provider"] == "ace_step"
    assert trace["model"] == "acestep-v15-turbo"
    assert trace["engine_version"] == ACE_STEP_VERSION
    assert trace["inference_steps"] == 8


def test_describe_request_shows_prompt_compilation_both_sides(tmp_path):
    provider = _provider(tmp_path)
    trace = provider.describe_request(_legacy_request())
    # The point of the trace: what the user wrote vs what was sent.
    assert trace["original_prompt"] == "Dreamy Korean indie pop with warm electric piano"
    assert trace["compiled_prompt"] != trace["original_prompt"]
    assert "female lead vocal" in str(trace["compiled_prompt"])
    assert trace["added_conditioning"]


def test_describe_request_never_leaks_credentials_or_host(tmp_path):
    provider = _provider(tmp_path, base_url=SECRET_BASE_URL, api_key=SECRET_API_KEY)
    serialized = json.dumps(provider.describe_request(_legacy_request()), ensure_ascii=False)
    assert SECRET_API_KEY not in serialized
    assert SECRET_BASE_URL not in serialized
    assert "internal-gpu-07" not in serialized
    assert "api_key" not in serialized
    assert "base_url" not in serialized
    assert "Authorization" not in serialized
    assert "Bearer" not in serialized


def test_describe_request_never_leaks_local_paths(tmp_path):
    secret_dir = tmp_path / "home" / "someone" / "private-output"
    provider = _provider(tmp_path, output_dir=secret_dir)
    serialized = json.dumps(provider.describe_request(_legacy_request()), ensure_ascii=False)
    assert "private-output" not in serialized
    assert str(secret_dir) not in serialized


def test_describe_request_is_json_serializable(tmp_path):
    provider = _provider(tmp_path)
    trace = provider.describe_request(
        _legacy_request(bpm=128, key_scale="C major", time_signature="4")
    )
    round_tripped = json.loads(json.dumps(trace, ensure_ascii=False))
    assert round_tripped["payload"]["bpm"] == 128


def test_describe_request_does_not_send_anything(tmp_path):
    provider = _provider(tmp_path)
    provider.describe_request(_legacy_request())
    assert provider.test_server.release_payloads == []


def test_provider_without_a_trace_implementation_returns_empty():
    # The base implementation is concrete, so Phase 7-era providers keep
    # working untouched — they simply record no trace.
    assert MockGenerationProvider(FIXTURE).describe_request(_legacy_request()) == {}
