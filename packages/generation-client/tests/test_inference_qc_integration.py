"""Phase 29 through the whole service: request in, delivered file out.

The unit tests in `packages/inference-qc` prove the loop decides
correctly. These prove the decisions reach the parts of the system a
user can see — the stored status, the error code returned, the trace
persisted to the row, and the fact that a rejected candidate never
becomes an asset in somebody's library.

Providers here are scripted rather than real. A retry policy cannot be
tested against a model that might succeed by luck on the second attempt:
the test would pass without proving the retry happened for the reason it
was supposed to.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from luber_audio_utils import LocalAudioStorage
from luber_generation_client import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    MockGenerationProvider,
    MusicGenerationProvider,
)
from luber_generation_client.errors import GenerationProviderError
from luber_schemas import AssetType, ErrorCode, GenerationStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"
SAMPLE_RATE = 44_100


def _write(path: Path, frames: list[tuple[float, float]]) -> Path:
    payload = bytearray()
    for left, right in frames:
        for value in (left, right):
            payload += struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(payload))
    return path


def silent_wav(path: Path, seconds: float) -> Path:
    return _write(path, [(0.0, 0.0)] * int(seconds * SAMPLE_RATE))


def healthy_wav(path: Path, seconds: float) -> Path:
    """Broadband and stereo, so nothing but the intended defect is found."""
    partials = ((110.0, 0.9), (330.0, 0.4), (880.0, 0.2), (2640.0, 0.08), (7040.0, 0.03))
    norm = sum(weight for _, weight in partials)
    state = 20260821
    frames: list[tuple[float, float]] = []
    for index in range(int(seconds * SAMPLE_RATE)):
        value = (
            sum(
                weight * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
                for frequency, weight in partials
            )
            / norm
        )
        channel: list[float] = []
        for _ in range(2):
            state = (1103515245 * state + 12345) % (2**31)
            channel.append(0.5 * value + 0.02 * ((state / (2**30)) - 1.0))
        frames.append((channel[0], channel[1]))
    return _write(path, frames)


class ScriptedProvider(MusicGenerationProvider):
    """A provider whose answers are written down, one per call."""

    name = "scripted"

    def __init__(self, directory: Path, *script) -> None:
        self.directory = directory
        self.script = list(script)
        self.calls: list[int | None] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        entry = self.script[min(len(self.calls), len(self.script) - 1)]
        self.calls.append(request.seed)
        if isinstance(entry, Exception):
            raise entry
        path = entry(
            self.directory / f"call-{len(self.calls):02d}.wav", float(request.duration_seconds)
        )
        with wave.open(str(path), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        return GenerationResult(
            audio_path=path,
            duration_seconds=duration,
            sample_rate=SAMPLE_RATE,
            seed_used=request.seed,
            provider=self.name,
            model_name="scripted",
            model_version="test",
        )


async def _create(repository, **overrides):
    payload = {
        "title": "TEST SONG",
        "prompt": "Dreamy Korean indie pop",
        "lyrics": "[Verse]\n테스트 가사",
        "vocal_gender": "female",
        "duration_requested": 20,
        "status": GenerationStatus.QUEUED.value,
        "language": "ko",
        **overrides,
    }
    generation = await repository.create_generation(**payload)
    await repository.create_job(
        generation.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    return generation


def _service(repository, provider, tmp_path, **kwargs):
    return GenerationService(
        repository,
        provider,
        LocalAudioStorage(tmp_path / "storage"),
        candidate_workspace_dir=str(tmp_path / "candidates"),
        **kwargs,
    )


# ── the healthy path ─────────────────────────────────────────────────


async def test_a_good_first_candidate_is_delivered_for_one_call(repository, tmp_path):
    provider = ScriptedProvider(tmp_path / "provider", healthy_wav)
    generation = await _create(repository)

    final = await _service(repository, provider, tmp_path).execute(
        generation.id, worker_id="test-worker"
    )

    assert final is GenerationStatus.COMPLETED
    assert len(provider.calls) == 1


async def test_the_trace_is_stored_on_the_row(repository, tmp_path):
    provider = ScriptedProvider(tmp_path / "provider", healthy_wav)
    generation = await _create(repository)
    await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    trace = json.loads((await repository.get_generation(generation.id)).inference_qc_trace)

    assert trace["outcome"] == "SELECTED"
    assert trace["policy"]["name"] == "STANDARD"
    assert len(trace["attempts"]) == 1
    assert trace["budget"]["provider_calls_used"] == 1


async def test_the_stored_trace_carries_no_prompt_and_no_lyrics(repository, tmp_path):
    """Phase 29's privacy rule, asserted where the data actually lands."""
    provider = ScriptedProvider(tmp_path / "provider", healthy_wav)
    generation = await _create(repository)
    await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    stored = (await repository.get_generation(generation.id)).inference_qc_trace

    assert "Dreamy Korean indie pop" not in stored
    assert "테스트 가사" not in stored
    assert str(tmp_path) not in stored


# ── retry that works ─────────────────────────────────────────────────


async def test_a_silent_candidate_is_retried_and_the_user_gets_a_song(repository, tmp_path):
    provider = ScriptedProvider(tmp_path / "provider", silent_wav, healthy_wav)
    generation = await _create(repository, seed=1234)

    final = await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    assert final is GenerationStatus.COMPLETED
    assert len(provider.calls) == 2
    assert provider.calls[0] == 1234
    assert provider.calls[1] not in (None, 1234)


async def test_the_retry_is_invisible_to_the_user_and_recorded_internally(repository, tmp_path):
    """A customer has no business knowing there were two attempts, and
    an operator has every business knowing it."""
    provider = ScriptedProvider(tmp_path / "provider", silent_wav, healthy_wav)
    generation = await _create(repository)
    await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    stored = await repository.get_generation(generation.id)
    trace = json.loads(stored.inference_qc_trace)

    assert stored.status == GenerationStatus.COMPLETED.value
    assert stored.error_code is None
    assert [attempt["status"] for attempt in trace["attempts"]] == ["REJECTED", "ELIGIBLE"]
    assert trace["attempts"][1]["attribution"] == "QUALITY_RETRY"


async def test_only_the_winner_becomes_an_asset(repository, tmp_path):
    """A rejected candidate cannot reach a library. It is never uploaded."""
    provider = ScriptedProvider(tmp_path / "provider", silent_wav, healthy_wav)
    generation = await _create(repository)
    await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    assets = await repository.get_audio_assets(generation.id)
    masters = [asset for asset in assets if asset.asset_type == AssetType.MASTER.value]
    assert len(masters) == 1


# ── failure, reported honestly ───────────────────────────────────────


async def test_every_candidate_rejected_fails_the_generation(repository, tmp_path):
    provider = ScriptedProvider(tmp_path / "provider", silent_wav)
    generation = await _create(repository)

    final = await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    stored = await repository.get_generation(generation.id)
    assert final is GenerationStatus.FAILED
    assert stored.error_code in {
        ErrorCode.QUALITY_CHECK_FAILED.value,
        ErrorCode.QUALITY_RETRY_EXHAUSTED.value,
    }
    assert not await repository.get_audio_assets(generation.id)


async def test_a_provider_that_never_answered_is_not_reported_as_a_quality_failure(
    repository, tmp_path
):
    """It failed to answer. Calling that a bad song sends an operator to
    the wrong system."""
    provider = ScriptedProvider(
        tmp_path / "provider",
        GenerationProviderError("no model on this host", error_code=ErrorCode.MODEL_LOAD_FAILED),
    )
    generation = await _create(repository)

    await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    stored = await repository.get_generation(generation.id)
    assert stored.error_code == ErrorCode.MODEL_LOAD_FAILED.value
    assert len(provider.calls) == 1


async def test_the_failure_message_is_the_thing_that_went_wrong(repository, tmp_path):
    provider = ScriptedProvider(tmp_path / "provider", silent_wav)
    generation = await _create(repository)

    await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    stored = await repository.get_generation(generation.id)
    assert "no audible content" in (stored.error_message or "").lower()


async def test_a_failed_generation_still_leaves_its_trace(repository, tmp_path):
    """Failures are not hidden; they are the ones worth reading."""
    provider = ScriptedProvider(tmp_path / "provider", silent_wav)
    generation = await _create(repository)
    await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")

    trace = json.loads((await repository.get_generation(generation.id)).inference_qc_trace)
    assert trace["outcome"] != "SELECTED"
    assert trace["selected_candidate_id"] is None
    assert all(attempt["status"] == "REJECTED" for attempt in trace["attempts"])


# ── the workspace ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("script", "expected"),
    [((healthy_wav,), GenerationStatus.COMPLETED), ((silent_wav,), GenerationStatus.FAILED)],
)
async def test_candidate_audio_is_cleaned_up_on_a_terminal_outcome(
    repository, tmp_path, script, expected
):
    provider = ScriptedProvider(tmp_path / "provider", *script)
    generation = await _create(repository)

    assert (
        await _service(repository, provider, tmp_path).execute(generation.id, worker_id="w")
    ) is expected

    assert not (tmp_path / "candidates" / str(generation.id)).exists()


# ── the switch ───────────────────────────────────────────────────────


async def test_disabling_qc_restores_the_pre_phase_29_behaviour(repository, tmp_path):
    """One call, no measurement, no retry — and the silent file ships.

    Asserted deliberately: the switch has to be a real bypass rather
    than a quieter version of the same loop, or an operator turning it
    off during an incident would not get what they expected.
    """
    provider = ScriptedProvider(tmp_path / "provider", silent_wav)
    generation = await _create(repository)

    final = await _service(repository, provider, tmp_path, qc_enabled=False).execute(
        generation.id, worker_id="w"
    )

    assert final is GenerationStatus.COMPLETED
    assert len(provider.calls) == 1
    assert (await repository.get_generation(generation.id)).inference_qc_trace is None


async def test_the_reproducible_policy_makes_exactly_one_call(repository, tmp_path):
    provider = ScriptedProvider(tmp_path / "provider", silent_wav, healthy_wav)
    generation = await _create(repository, seed=99)

    final = await _service(repository, provider, tmp_path, qc_policy="STRICT_REPRODUCIBLE").execute(
        generation.id, worker_id="w"
    )

    assert final is GenerationStatus.FAILED
    assert len(provider.calls) == 1


async def test_the_mock_provider_still_passes_quality_control(repository, tmp_path):
    """The double every other suite depends on has to be deliverable.

    If it were not, every test in the repository that generates
    something would be exercising the failure path.
    """
    generation = await _create(repository)
    final = await _service(repository, MockGenerationProvider(FIXTURE), tmp_path).execute(
        generation.id, worker_id="w"
    )
    assert final is GenerationStatus.COMPLETED
