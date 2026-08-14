"""Phase 13B — the engine contract for source-audio-conditioned editing.

These tests hold the line between *editing* and *generating*. The
distinction is not stylistic: an extension that quietly became a
text-to-music run would return a plausible song, pass every smoke test,
and be wrong in the only way that matters.

What is pinned here:

- the outgoing ACE-Step payload really says ``repaint``, over the range
  derived from the measured source;
- the source audio is uploaded as bytes on the multipart field upstream
  reads, not referenced by path;
- the file handle is released whatever happens;
- the service routes an edit to ``edit()`` and never to ``generate()``,
  and fails when it cannot;
- no path or engine vocabulary leaks into the trace.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import httpx
import pytest

from luber_generation_client.ace_step.client import AceStepApiError, AceStepClient
from luber_generation_client.ace_step.provider import (
    ACE_STEP_REPAINT_TASK,
    AceStepProvider,
    AceStepProviderConfig,
)
from luber_generation_client.audio_to_audio import AudioToAudioRequest
from luber_generation_client.editing import AudioEditKind, AudioEditRequest
from luber_generation_client.errors import GenerationProviderError
from luber_schemas import VocalGender


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 48000) -> Path:
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<h", 1000) * frames * 2)
    return path


@pytest.fixture
def source_wav(tmp_path: Path) -> Path:
    return _write_wav(tmp_path / "parent-master.wav", seconds=2.0)


@pytest.fixture
def source_wav_bytes(source_wav: Path) -> bytes:
    """Read synchronously: async tests must not touch the filesystem."""
    return source_wav.read_bytes()


def _edit(source: Path, **overrides) -> AudioEditRequest:
    defaults = dict(
        source_audio=source,
        # The fixture is 2.0s; the default range extends past it, which is
        # what Phase 13B's tests are about. Interior cases override both.
        source_duration_seconds=2.0,
        start_seconds=30.0,
        end_seconds=45.0,
        title="Midnight Window",
        prompt="Dreamy Korean indie pop",
        lyrics="[Verse]\n오늘 밤",
        vocal_gender=VocalGender.FEMALE,
        language="ko",
    )
    defaults.update(overrides)
    return AudioEditRequest(**defaults)


# ── the request contract ──────────────────────────────────────────────


class TestAudioEditRequest:
    def test_rejects_a_source_that_does_not_exist(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            _edit(tmp_path / "nope.wav")

    def test_rejects_an_empty_source(self, tmp_path: Path):
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            _edit(empty)

    def test_rejects_an_unsupported_source_format(self, tmp_path: Path):
        mp3 = tmp_path / "source.mp3"
        mp3.write_bytes(b"ID3not-really")
        with pytest.raises(ValueError, match="unsupported source audio format"):
            _edit(mp3)

    def test_rejects_an_inverted_range(self, source_wav: Path):
        with pytest.raises(ValueError, match="greater than start_seconds"):
            _edit(source_wav, start_seconds=30.0, end_seconds=30.0)

    def test_rejects_a_negative_start(self, source_wav: Path):
        with pytest.raises(ValueError):
            _edit(source_wav, start_seconds=-1.0, end_seconds=10.0)

    def test_rejects_a_non_finite_boundary(self, source_wav: Path):
        with pytest.raises(ValueError, match="finite"):
            _edit(source_wav, start_seconds=0.0, end_seconds=float("inf"))

    def test_total_is_the_end_of_the_range(self, source_wav: Path):
        assert _edit(source_wav, start_seconds=30.0, end_seconds=45.0).total_seconds == 45.0


# ── the ACE-Step payload ──────────────────────────────────────────────


class TestAceStepEditPayload:
    def _provider(self) -> AceStepProvider:
        return AceStepProvider(AceStepProviderConfig(base_url="http://engine.invalid"))

    def test_the_task_is_repaint(self, source_wav: Path):
        payload = self._provider()._build_edit_payload(_edit(source_wav))
        assert payload["task_type"] == ACE_STEP_REPAINT_TASK
        assert payload["task_type"] == "repaint"

    def test_the_range_becomes_the_repaint_boundaries(self, source_wav: Path):
        payload = self._provider()._build_edit_payload(
            _edit(source_wav, start_seconds=31.5, end_seconds=46.5)
        )
        assert payload["repainting_start"] == pytest.approx(31.5)
        assert payload["repainting_end"] == pytest.approx(46.5)

    def test_the_canvas_is_the_whole_extended_song(self, source_wav: Path):
        """Upstream sizes its output from this, not from the added part."""
        payload = self._provider()._build_edit_payload(
            _edit(source_wav, start_seconds=30.0, end_seconds=45.0)
        )
        assert payload["audio_duration"] == pytest.approx(45.0)

    def test_conditioning_is_carried_through(self, source_wav: Path):
        payload = self._provider()._build_edit_payload(
            _edit(source_wav, bpm=128, key_scale="F# minor", time_signature="3", seed=4242)
        )
        assert payload["bpm"] == 128
        assert payload["key_scale"] == "F# minor"
        assert payload["time_signature"] == "3"
        assert payload["seed"] == 4242
        assert payload["use_random_seed"] is False

    def test_no_cover_reference_or_retake_fields(self, source_wav: Path):
        """Phase 13B uses exactly one primitive; the rest are unproven."""
        payload = self._provider()._build_edit_payload(_edit(source_wav))
        for field in (
            "reference_audio_path",
            "audio_cover_strength",
            "cover_noise_strength",
            "retake_seed",
            "retake_variance",
            "flow_edit_morph",
            "audio_code_string",
        ):
            assert field not in payload

    def test_preservation_maps_to_repaint_strength_inversely(self, source_wav: Path):
        high = self._provider()._build_edit_payload(_edit(source_wav, preservation=0.9))
        low = self._provider()._build_edit_payload(_edit(source_wav, preservation=0.1))
        assert high["repaint_strength"] == pytest.approx(0.1)
        assert low["repaint_strength"] == pytest.approx(0.9)


class TestCapabilityProbe:
    def test_turbo_can_repaint(self):
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://x", model="acestep-v15-turbo")
        )
        assert provider.supports_edit(AudioEditKind.REGENERATE_RANGE) is True

    def test_an_unknown_model_is_not_claimed(self):
        """Silence beats a false yes: upstream would accept and misbehave."""
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://x", model="some-future-checkpoint")
        )
        assert provider.supports_edit(AudioEditKind.REGENERATE_RANGE) is False

    async def test_an_unsupported_model_refuses_to_edit(self, source_wav: Path):
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://x", model="some-future-checkpoint")
        )
        with pytest.raises(GenerationProviderError, match="cannot perform"):
            await provider.edit(_edit(source_wav))


class TestEditTrace:
    def test_carries_no_path_and_no_bytes(self, source_wav: Path):
        provider = AceStepProvider(AceStepProviderConfig(base_url="http://x"))
        trace = provider.describe_edit(_edit(source_wav))
        rendered = str(trace)
        assert str(source_wav) not in rendered
        assert source_wav.name not in rendered
        assert trace["source_audio_bytes"] == source_wav.stat().st_size
        assert trace["source_audio_transport"] == "multipart"

    def test_carries_no_credentials_or_host(self, source_wav: Path):
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://engine.internal:9999", api_key="sekrit")
        )
        rendered = str(provider.describe_edit(_edit(source_wav)))
        assert "sekrit" not in rendered
        assert "engine.internal" not in rendered


# ── the multipart transport ───────────────────────────────────────────


class TestMultipartUpload:
    async def test_uploads_the_bytes_on_the_field_upstream_reads(
        self, source_wav: Path, source_wav_bytes: bytes
    ):
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["content_type"] = request.headers.get("content-type", "")
            seen["body"] = request.content
            return httpx.Response(
                200, json={"data": {"task_id": "t-1"}, "code": 200, "error": None}
            )

        client = AceStepClient("http://engine.invalid", transport=httpx.MockTransport(handler))
        try:
            handle = await client.submit_generation_with_source_audio(
                {"task_type": "repaint", "repainting_start": 30.0}, source_wav
            )
        finally:
            await client.close()

        assert handle.task_id == "t-1"
        assert "multipart/form-data" in str(seen["content_type"])
        body = bytes(seen["body"])  # type: ignore[arg-type]
        # ``src_audio`` is the field name upstream's parser looks for.
        assert b'name="src_audio"' in body
        # The real audio, not a path to it.
        assert source_wav_bytes in body
        assert str(source_wav).encode() not in body
        assert b'name="task_type"' in body and b"repaint" in body

    async def test_scalars_are_sent_as_form_fields(self, source_wav: Path):
        captured: dict[str, bytes] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content
            return httpx.Response(
                200, json={"data": {"task_id": "t-2"}, "code": 200, "error": None}
            )

        client = AceStepClient("http://x", transport=httpx.MockTransport(handler))
        try:
            await client.submit_generation_with_source_audio(
                {"thinking": False, "batch_size": 1, "omitted": None}, source_wav
            )
        finally:
            await client.close()

        body = captured["body"]
        # Booleans must be engine-readable text, not Python's "False".
        assert b"false" in body and b"False" not in body
        # None means "not specified" and is left out entirely.
        assert b'name="omitted"' not in body

    async def test_the_file_handle_is_closed_on_success(self, source_wav: Path):
        opened: list[object] = []
        real_open = Path.open

        def tracking_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            handle = real_open(self, *args, **kwargs)
            opened.append(handle)
            return handle

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"data": {"task_id": "t-3"}, "code": 200, "error": None}
            )

        client = AceStepClient("http://x", transport=httpx.MockTransport(handler))
        Path.open = tracking_open  # type: ignore[method-assign]
        try:
            await client.submit_generation_with_source_audio({}, source_wav)
        finally:
            Path.open = real_open  # type: ignore[method-assign]
            await client.close()

        assert opened and all(getattr(h, "closed", True) for h in opened)

    async def test_the_file_handle_is_closed_when_the_request_fails(self, source_wav: Path):
        opened: list[object] = []
        real_open = Path.open

        def tracking_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            handle = real_open(self, *args, **kwargs)
            opened.append(handle)
            return handle

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("engine gone")

        client = AceStepClient("http://x", transport=httpx.MockTransport(handler))
        Path.open = tracking_open  # type: ignore[method-assign]
        try:
            with pytest.raises(httpx.ConnectTimeout):
                await client.submit_generation_with_source_audio({}, source_wav)
        finally:
            Path.open = real_open  # type: ignore[method-assign]
            await client.close()

        assert opened and all(getattr(h, "closed", True) for h in opened)

    async def test_a_missing_task_id_is_an_error(self, source_wav: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {}, "code": 200, "error": None})

        client = AceStepClient("http://x", transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(AceStepApiError, match="no task_id"):
                await client.submit_generation_with_source_audio({}, source_wav)
        finally:
            await client.close()


# ── end to end through the provider ───────────────────────────────────


class TestProviderEdit:
    async def test_submits_repaint_and_returns_the_engine_canvas(
        self, source_wav: Path, tmp_path: Path
    ):
        produced = _write_wav(tmp_path / "engine-output.wav", seconds=3.0)
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            path = request.url.path
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "data": {"status": "ok", "models_initialized": True},
                        "code": 200,
                        "error": None,
                    },
                )
            if path == "/release_task":
                return httpx.Response(
                    200, json={"data": {"task_id": "task-9"}, "code": 200, "error": None}
                )
            if path == "/query_result":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "task_id": "task-9",
                                "status": 1,
                                "result": (
                                    '[{"file": "/v1/audio?path=out.wav", "seed_value": "777",'
                                    ' "dit_model": "acestep-v15-turbo",'
                                    ' "metas": {"duration": 3.0}}]'
                                ),
                            }
                        ],
                        "code": 200,
                        "error": None,
                    },
                )
            return httpx.Response(200, content=produced.read_bytes())

        client = AceStepClient("http://x", transport=httpx.MockTransport(handler))
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://x", output_dir=tmp_path / "out"),
            client=client,
        )
        try:
            result = await provider.edit(_edit(source_wav, start_seconds=30.0, end_seconds=45.0))
        finally:
            await provider.close()

        submit = next(r for r in requests if r.url.path == "/release_task")
        assert b'name="src_audio"' in submit.content
        assert b"repaint" in submit.content
        assert result.seed_used == 777
        assert result.duration_seconds == pytest.approx(3.0, abs=0.05)
        assert result.provider == "ace_step"


def test_editing_contract_declares_only_the_proven_primitive():
    """One member. Adding another means proving another primitive."""
    assert [k.value for k in AudioEditKind] == ["REGENERATE_RANGE"]


def test_the_edit_kind_is_not_engine_vocabulary():
    """The domain must not speak ACE-Step."""
    for member in AudioEditKind:
        assert "repaint" not in member.value.lower()


# ── canvas length: Phase 13C ──────────────────────────────────────────


class TestCanvasLength:
    """What the engine is told to return, for each shape of edit."""

    def test_an_interior_range_keeps_the_source_length(self, source_wav: Path):
        edit = _edit(source_wav, source_duration_seconds=30.0, start_seconds=10.0, end_seconds=20.0)
        # Not 20.0: the song is 30s and stays 30s.
        assert edit.total_seconds == pytest.approx(30.0)
        assert edit.extends_source is False

    def test_a_range_past_the_end_lengthens_the_canvas(self, source_wav: Path):
        edit = _edit(source_wav, source_duration_seconds=30.0, start_seconds=30.0, end_seconds=45.0)
        assert edit.total_seconds == pytest.approx(45.0)
        assert edit.extends_source is True

    def test_the_payload_asks_for_the_full_source_on_a_replacement(self, source_wav: Path):
        provider = AceStepProvider(AceStepProviderConfig(base_url="http://x"))
        payload = provider._build_edit_payload(
            _edit(source_wav, source_duration_seconds=30.0, start_seconds=10.0, end_seconds=20.0)
        )
        assert payload["audio_duration"] == pytest.approx(30.0)
        assert payload["repainting_start"] == pytest.approx(10.0)
        assert payload["repainting_end"] == pytest.approx(20.0)
        assert payload["task_type"] == "repaint"

    def test_a_short_interior_edit_does_not_trip_the_generation_floor(self, source_wav: Path):
        """An edit's canvas may be shorter than a generation may request."""
        provider = AceStepProvider(AceStepProviderConfig(base_url="http://x"))
        payload = provider._build_edit_payload(
            _edit(source_wav, source_duration_seconds=2.0, start_seconds=0.5, end_seconds=1.5)
        )
        assert payload["audio_duration"] == pytest.approx(2.0)


# ── source-conditioned generation (Phase 13D-2) ───────────────────────
#
# Cover is a third operation, not a variant of repaint. These pin the
# parts that would silently change its meaning: the engine task, the
# strength direction, the absence of the MLX-inert control, and the fact
# that the whole canvas is regenerated rather than partly preserved.


def _cover(source: Path, **overrides) -> AudioToAudioRequest:
    defaults = dict(
        source_audio=source,
        source_duration_seconds=2.0,
        title="Midnight Window",
        prompt="modern synth pop with glossy production",
        lyrics="[Verse]\n오늘 밤",
        vocal_gender=VocalGender.FEMALE,
        language="ko",
        source_adherence=1.0,
    )
    defaults.update(overrides)
    return AudioToAudioRequest(**defaults)


class TestCoverPayload:
    def _provider(self) -> AceStepProvider:
        return AceStepProvider(AceStepProviderConfig(base_url="http://x"))

    def test_the_task_is_cover_not_repaint(self, source_wav: Path):
        payload = self._provider()._build_cover_payload(_cover(source_wav))
        assert payload["task_type"] == "cover"
        assert payload["task_type"] != ACE_STEP_REPAINT_TASK

    def test_adherence_maps_straight_onto_the_engine_dial(self, source_wav: Path):
        """Both mean "closer to the source", so this is a rename not a flip."""
        provider = self._provider()
        high = provider._build_cover_payload(_cover(source_wav, source_adherence=1.0))
        low = provider._build_cover_payload(_cover(source_wav, source_adherence=0.75))
        assert high["audio_cover_strength"] == pytest.approx(1.0)
        assert low["audio_cover_strength"] == pytest.approx(0.75)

    def test_the_mlx_inert_control_is_never_sent(self, source_wav: Path):
        """cover_noise_strength does nothing on this runtime."""
        payload = self._provider()._build_cover_payload(_cover(source_wav))
        assert "cover_noise_strength" not in payload

    def test_no_repaint_or_duration_fields(self, source_wav: Path):
        """A cover has no range, and the engine sizes it from the source."""
        payload = self._provider()._build_cover_payload(_cover(source_wav))
        for field in (
            "repainting_start",
            "repainting_end",
            "repaint_mode",
            "repaint_strength",
            "audio_duration",
            "reference_audio_path",
        ):
            assert field not in payload

    def test_the_target_style_is_carried(self, source_wav: Path):
        payload = self._provider()._build_cover_payload(
            _cover(source_wav, prompt="warm contemporary R&B")
        )
        assert "R&B" in str(payload["prompt"])


class TestCoverCapability:
    def test_turbo_can_cover(self):
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://x", model="acestep-v15-turbo")
        )
        assert provider.supports_audio_to_audio() is True

    def test_an_unknown_model_is_not_claimed(self):
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://x", model="some-future-checkpoint")
        )
        assert provider.supports_audio_to_audio() is False

    def test_the_validated_band_matches_the_calibration(self):
        provider = AceStepProvider(AceStepProviderConfig(base_url="http://x"))
        assert provider.validated_adherence_range() == (0.75, 1.0)

    async def test_an_uncalibrated_adherence_is_refused_not_clamped(self, source_wav: Path):
        """0.50 measured as indistinguishable from an unrelated song.

        Clamping would silently give the user a different setting from the
        one their label promised.
        """
        provider = AceStepProvider(AceStepProviderConfig(base_url="http://x"))
        with pytest.raises(GenerationProviderError, match="outside the validated range"):
            await provider.create_from_audio(_cover(source_wav, source_adherence=0.5))

    async def test_a_model_that_cannot_cover_refuses(self, source_wav: Path):
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://x", model="some-future-checkpoint")
        )
        with pytest.raises(GenerationProviderError, match="cannot generate from audio"):
            await provider.create_from_audio(_cover(source_wav))


class TestCoverTrace:
    def test_carries_no_path_bytes_or_credentials(self, source_wav: Path):
        provider = AceStepProvider(
            AceStepProviderConfig(base_url="http://engine.internal:9999", api_key="sekrit")
        )
        rendered = str(provider.describe_audio_to_audio(_cover(source_wav)))
        assert "sekrit" not in rendered
        assert "engine.internal" not in rendered
        assert str(source_wav) not in rendered
        assert source_wav.name not in rendered

    def test_records_the_operation_and_adherence(self, source_wav: Path):
        provider = AceStepProvider(AceStepProviderConfig(base_url="http://x"))
        trace = provider.describe_audio_to_audio(_cover(source_wav, source_adherence=0.75))
        assert trace["operation"] == "cover"
        assert trace["source_adherence"] == pytest.approx(0.75)
        assert trace["source_audio_transport"] == "multipart"


class TestCoverUpload:
    async def test_the_source_bytes_are_uploaded(self, source_wav: Path, source_wav_bytes: bytes):
        captured: dict[str, bytes] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "data": {"status": "ok", "models_initialized": True},
                        "code": 200,
                        "error": None,
                    },
                )
            if request.url.path == "/release_task":
                captured["body"] = request.content
                return httpx.Response(
                    200, json={"data": {"task_id": "c-1"}, "code": 200, "error": None}
                )
            if request.url.path == "/query_result":
                return httpx.Response(
                    200,
                    json={
                        "data": [{"task_id": "c-1", "status": 2, "result": "engine said no"}],
                        "code": 200,
                        "error": None,
                    },
                )
            return httpx.Response(404)

        client = AceStepClient("http://x", transport=httpx.MockTransport(handler))
        provider = AceStepProvider(AceStepProviderConfig(base_url="http://x"), client=client)
        try:
            # The task is failed deliberately: this test is about what was
            # uploaded, and stopping there keeps it fast.
            with pytest.raises(GenerationProviderError):
                await provider.create_from_audio(_cover(source_wav))
        finally:
            await provider.close()

        body = captured["body"]
        assert b'name="src_audio"' in body
        assert source_wav_bytes in body
        assert b"cover" in body
        assert b"cover_noise_strength" not in body
