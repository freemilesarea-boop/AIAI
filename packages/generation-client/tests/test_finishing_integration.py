"""Finishing at the real post-processing boundary.

Phase 14A proved the engine in isolation. What it could not prove is the
part that only exists once the engine is wired in: that the raw master
survives, that the preview follows the right master, that a finishing
failure costs the enhancement rather than the song, and that a retry
converges instead of accumulating.

These run the actual ``produce_delivery_assets`` and the actual
``GenerationService`` against real files and real storage. The audio is
the repository's own mock-generation fixture — which the engine does act
on — plus a synthesised healthy mix for the NO_ACTION path, because the
fixture cannot produce both outcomes at once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import uuid
import wave
from pathlib import Path

import numpy as np
import pytest

from luber_audio_finishing import FinishingError, read_finishing_stamp
from luber_audio_finishing.version import FINISHING_VERSION
from luber_audio_utils import (
    LocalAudioStorage,
    finished_master_storage_key,
    master_storage_key,
    preview_storage_key,
)
from luber_database import GenerationRepository
from luber_generation_client import MockGenerationProvider
from luber_generation_client.postprocess import produce_delivery_assets
from luber_generation_client.service import GenerationService
from luber_schemas import (
    AssetType,
    ErrorCode,
    FinishingOutcome,
    GenerationStatus,
    select_delivery_master,
    select_finished_master,
    select_raw_master,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
#: The same fixture the rest of the suite uses. The engine acts on it,
#: so it exercises the FINISHED path with production audio rather than
#: something built to make the test pass.
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

RATE = 48_000


def healthy_wav(path: Path, seconds: float = 4.0, seed: int = 3) -> Path:
    """A mix with nothing wrong with it, so the engine declines.

    Shaped in the frequency domain at -4 dB/octave: steeper than pink
    noise, and decorrelated only above 200 Hz so the bass stays coherent.

    The trim above 10 kHz is what makes it healthy rather than merely
    deficit-free. Broadband noise at this slope carries far more top end
    than music does — it measures an air ratio around -9 dB, brighter
    than any real master in the corpus — which was invisible while the
    engine could only detect darkness. Now that it detects brightness
    too, an untrimmed baseline provokes a high-shelf cut and this fixture
    stops testing what it claims to. Mirrors ``NEUTRAL_TOP_TRIM`` in the
    audio-finishing conftest.
    """
    length = int(seconds * RATE)
    freqs = np.fft.rfftfreq(length, 1.0 / RATE)
    shape = np.zeros_like(freqs)
    shape[freqs > 0] = freqs[freqs > 0] ** (-4.0 / 3.0103 / 2.0)
    shape[freqs >= 10_000.0] *= 10.0 ** (-8.0 / 20.0)

    def component(component_seed: int, highpass: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(component_seed)
        spectrum = np.fft.rfft(rng.standard_normal(length))
        full = np.fft.irfft(spectrum * shape, n=length)
        scale = float(np.sqrt(np.mean(full**2)))
        if highpass <= 0:
            return full / scale
        limited = shape.copy()
        limited[freqs < highpass] = 0.0
        return np.fft.irfft(spectrum * limited, n=length) / scale

    common = component(seed)
    left = common + component(seed + 101, highpass=200.0)
    right = common + component(seed + 202, highpass=200.0)
    pair = np.stack([left, right], axis=1)
    pair *= 0.5 / float(np.abs(pair).max())

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(3)
        handle.setframerate(RATE)
        as_int = (np.clip(pair, -1.0, 1.0) * 8388607.0).astype(np.int32)
        packed = as_int.astype("<i4").tobytes()
        handle.writeframes(b"".join(packed[i : i + 3] for i in range(0, len(packed), 4)))
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def storage(tmp_path: Path) -> LocalAudioStorage:
    return LocalAudioStorage(tmp_path / "store")


async def _queued(repository: GenerationRepository):
    return await repository.create_generation(
        title="Finishing integration",
        prompt="warm indie pop",
        lyrics="",
        vocal_gender="female",
        duration_requested=30,
        seed=None,
        language="en",
        instrumental=True,
        status=GenerationStatus.QUEUED.value,
        idempotency_key=None,
    )


# ── A, B, C: the raw master survives, separately ──────────────────────


class TestRawPreservation:
    async def test_a_raw_master_exists_after_a_successful_run(self, storage):
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        assert result.master.asset_type is AssetType.MASTER
        assert await storage.exists(master_storage_key(gid))

    async def test_the_finished_master_is_a_separate_object(self, storage):
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        assert result.finished is not None, "fixture is expected to provoke finishing"
        assert result.finished.storage_key == finished_master_storage_key(gid)
        assert result.finished.storage_key != result.master.storage_key
        assert result.finished.sha256 != result.master.sha256
        assert await storage.exists(result.finished.storage_key)

    async def test_the_provider_output_is_never_touched(self, storage):
        """The fixture is the model's output stand-in. It is read-only."""
        before = digest(FIXTURE)
        await produce_delivery_assets(uuid.uuid4(), FIXTURE, storage)
        assert digest(FIXTURE) == before

    async def test_the_stored_raw_master_is_the_transcode_not_the_finish(self, storage):
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        stored = hashlib.sha256(await storage.open(master_storage_key(gid))).hexdigest()
        assert stored == result.master.sha256
        assert result.finished is not None
        assert stored != result.finished.sha256

    async def test_finishing_cannot_be_pointed_at_the_raw_master_key(self, storage):
        """Distinct keys are what make an overwrite impossible."""
        gid = uuid.uuid4()
        assert finished_master_storage_key(gid) != master_storage_key(gid)


# ── D, E: the preview follows the delivery master ─────────────────────


class TestPreviewSource:
    async def test_preview_is_encoded_from_the_finished_master(self, storage, tmp_path):
        """What a listener streams must match what they download."""
        from luber_audio_utils import encode_preview_mp3_async

        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        assert result.finished is not None

        finished_bytes = await storage.open(result.finished.storage_key)
        finished_copy = tmp_path / "finished.wav"
        finished_copy.write_bytes(finished_bytes)
        expected = tmp_path / "expected.mp3"
        await encode_preview_mp3_async(finished_copy, expected)

        assert result.preview.sha256 == digest(expected)

    async def test_preview_is_encoded_from_the_raw_master_on_no_action(self, storage, tmp_path):
        from luber_audio_utils import encode_preview_mp3_async

        source = healthy_wav(tmp_path / "healthy.wav")
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, source, storage)
        assert result.finished is None, "healthy audio must produce NO_ACTION"

        master_copy = tmp_path / "master.wav"
        master_copy.write_bytes(await storage.open(master_storage_key(gid)))
        expected = tmp_path / "expected.mp3"
        await encode_preview_mp3_async(master_copy, expected)

        assert result.preview.sha256 == digest(expected)

    async def test_no_action_stores_no_finished_object(self, storage, tmp_path):
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, healthy_wav(tmp_path / "h.wav"), storage)
        assert result.finished is None
        assert not await storage.exists(finished_master_storage_key(gid))
        assert result.delivery_master is result.master
        assert result.finishing.outcome is FinishingOutcome.NO_ACTION


# ── F, G: the stamp survives, and stops a second pass ─────────────────


class TestIdempotency:
    async def test_the_engine_version_survives_storage(self, storage, tmp_path):
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        assert result.finished is not None

        roundtripped = tmp_path / "roundtrip.wav"
        roundtripped.write_bytes(await storage.open(result.finished.storage_key))
        assert read_finishing_stamp(roundtripped) == FINISHING_VERSION
        assert result.finishing.finishing_version == FINISHING_VERSION

    async def test_a_stored_finished_master_cannot_be_finished_again(self, storage, tmp_path):
        from luber_audio_finishing import AlreadyFinishedError, finish_audio

        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        assert result.finished is not None
        stored = tmp_path / "stored.wav"
        stored.write_bytes(await storage.open(result.finished.storage_key))

        with pytest.raises(AlreadyFinishedError):
            finish_audio(stored, tmp_path / "again.wav")

    async def test_the_raw_master_carries_no_stamp(self, storage, tmp_path):
        """So a retry finishing it again is a fresh decision, not a stack."""
        gid = uuid.uuid4()
        await produce_delivery_assets(gid, FIXTURE, storage)
        raw = tmp_path / "raw.wav"
        raw.write_bytes(await storage.open(master_storage_key(gid)))
        assert read_finishing_stamp(raw) is None

    async def test_rerunning_produces_identical_bytes(self, storage):
        gid = uuid.uuid4()
        first = await produce_delivery_assets(gid, FIXTURE, storage)
        second = await produce_delivery_assets(gid, FIXTURE, storage)
        assert first.master.sha256 == second.master.sha256
        assert first.preview.sha256 == second.preview.sha256
        assert first.finished is not None and second.finished is not None
        assert first.finished.sha256 == second.finished.sha256


# ── H, I: failure costs the enhancement, never the song ───────────────


class TestFailurePolicy:
    async def test_a_finishing_failure_still_delivers_the_raw_master(self, storage, monkeypatch):
        """The chosen policy, asserted rather than assumed.

        Finishing is an enhancement, not a delivery requirement, so its
        failure must not discard a successful inference.
        """
        import luber_generation_client.postprocess as postprocess

        def explode(*args: object, **kwargs: object) -> object:
            raise FinishingError("synthetic finishing failure")

        monkeypatch.setattr(postprocess, "finish_audio", explode)

        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)

        assert result.finished is None
        assert result.finishing.outcome is FinishingOutcome.FAILED
        assert result.finishing.error == "synthetic finishing failure"
        assert await storage.exists(master_storage_key(gid))
        assert await storage.exists(preview_storage_key(gid))
        assert not await storage.exists(finished_master_storage_key(gid))

    async def test_a_failed_finish_records_no_finished_asset(self, storage, monkeypatch):
        """No fake FINISHED_MASTER row, ever."""
        import luber_generation_client.postprocess as postprocess

        monkeypatch.setattr(
            postprocess,
            "finish_audio",
            lambda *a, **k: (_ for _ in ()).throw(FinishingError("boom")),
        )
        result = await produce_delivery_assets(uuid.uuid4(), FIXTURE, storage)
        assert all(a.asset_type is not AssetType.FINISHED_MASTER for a in result.assets)

    async def test_an_unexpected_engine_error_is_not_swallowed(self, storage, monkeypatch):
        """Only FinishingError means "the enhancement did not work out".

        A TypeError from a bad wiring change is a defect in this code and
        must surface as a failed generation, not be quietly downgraded to
        an unfinished master.
        """
        import luber_generation_client.postprocess as postprocess

        monkeypatch.setattr(
            postprocess,
            "finish_audio",
            lambda *a, **k: (_ for _ in ()).throw(TypeError("wiring defect")),
        )
        with pytest.raises(TypeError):
            await produce_delivery_assets(uuid.uuid4(), FIXTURE, storage)


# ── J, K, L: the service, retries, and what consumers resolve ─────────


class TestServiceIntegration:
    async def test_a_completed_generation_records_all_three_assets(self, repository, storage):
        gen = await _queued(repository)
        service = GenerationService(repository, MockGenerationProvider(FIXTURE), storage)
        assert await service.execute(gen.id, worker_id="w") is GenerationStatus.COMPLETED

        assets = await repository.get_audio_assets(gen.id)
        assert select_raw_master(assets) is not None
        assert select_finished_master(assets) is not None
        assert select_delivery_master(assets) is select_finished_master(assets)

    async def test_the_finishing_trace_is_recorded(self, repository, storage):
        import json

        gen = await _queued(repository)
        service = GenerationService(repository, MockGenerationProvider(FIXTURE), storage)
        await service.execute(gen.id, worker_id="w")

        fetched = await repository.get_generation(gen.id)
        assets = await repository.get_audio_assets(gen.id)
        trace = json.loads(fetched.finishing_trace)
        assert trace["outcome"] == FinishingOutcome.FINISHED.value
        assert trace["finishing_version"] == FINISHING_VERSION
        assert trace["source_sha256"] == select_raw_master(assets).sha256
        assert trace["plan"]["no_action"] is False
        assert trace["plan"]["actions"]

    async def test_no_action_is_distinguishable_from_never_having_run(
        self, repository, storage, tmp_path
    ):
        import json

        gen = await _queued(repository)
        service = GenerationService(
            repository, MockGenerationProvider(healthy_wav(tmp_path / "h.wav")), storage
        )
        await service.execute(gen.id, worker_id="w")

        fetched = await repository.get_generation(gen.id)
        assert fetched.finishing_trace is not None, "a run must leave a record"
        assert json.loads(fetched.finishing_trace)["outcome"] == FinishingOutcome.NO_ACTION.value
        assets = await repository.get_audio_assets(gen.id)
        assert select_finished_master(assets) is None
        assert select_delivery_master(assets) is select_raw_master(assets)

    async def test_a_retry_does_not_accumulate_assets(self, repository, storage):
        gen = await _queued(repository)
        service = GenerationService(repository, MockGenerationProvider(FIXTURE), storage)
        await service.execute(gen.id, worker_id="w1")
        first = {a.asset_type: a.id for a in await repository.get_audio_assets(gen.id)}
        await service.execute(gen.id, worker_id="w2")
        second = {a.asset_type: a.id for a in await repository.get_audio_assets(gen.id)}
        assert first == second

    async def test_a_retry_retracts_a_finished_master_it_no_longer_produces(
        self, repository, storage, monkeypatch
    ):
        """The stale-row case a deterministic engine cannot reach on its own.

        A future engine version could decline where an earlier one acted.
        Left in place, the old row would keep winning delivery selection
        while pointing at bytes the current decision never endorsed.

        Reached here the way production reaches it. A COMPLETED
        generation is never re-executed — Phase 18 made a duplicate
        invocation a no-op precisely so a retry cannot replace delivered
        audio — so the run that leaves a stale finished master behind is
        an *interrupted* one: assets written, then cancelled before
        mark_completed. That row is FAILED, and its retry is what can
        decline where the first attempt acted.
        """
        import luber_generation_client.postprocess as postprocess

        gen = await _queued(repository)
        service = GenerationService(repository, MockGenerationProvider(FIXTURE), storage)
        await service.execute(gen.id, worker_id="w1")
        assert select_finished_master(await repository.get_audio_assets(gen.id)) is not None

        # The interruption: assets exist, the run never reached COMPLETED.
        await repository.mark_failed(
            gen.id,
            status=GenerationStatus.FAILED.value,
            error_code=ErrorCode.GENERATION_INTERRUPTED.value,
            error_message="interrupted after assets were written",
        )

        monkeypatch.setattr(
            postprocess,
            "finish_audio",
            lambda *a, **k: (_ for _ in ()).throw(FinishingError("declined")),
        )
        await service.execute(gen.id, worker_id="w2")

        assets = await repository.get_audio_assets(gen.id)
        assert select_finished_master(assets) is None
        assert select_raw_master(assets) is not None
        assert select_delivery_master(assets) is select_raw_master(assets)
        assert not await storage.exists(finished_master_storage_key(gen.id))

    async def test_generation_lifecycle_is_otherwise_unchanged(self, repository, storage):
        """Finishing must not alter provenance or the reported duration."""
        gen = await _queued(repository)
        service = GenerationService(repository, MockGenerationProvider(FIXTURE), storage)
        await service.execute(gen.id, worker_id="w")

        fetched = await repository.get_generation(gen.id)
        assert fetched.status == GenerationStatus.COMPLETED.value
        assert fetched.provider == "mock"
        assert fetched.error_code is None
        assert fetched.started_at is not None and fetched.completed_at is not None
        # Duration comes from the delivered audio, and finishing preserves
        # duration, so it must still match the raw master.
        raw = select_raw_master(await repository.get_audio_assets(gen.id))
        assert fetched.duration_actual == pytest.approx(raw.duration, abs=0.01)


# ── Phase 22: a render the engine judged worse than the raw master ────


class TestRejectionPolicy:
    """Rejection is a fourth outcome, not a dressed-up failure.

    Four ways to end up delivering the raw master, and the pipeline has
    to keep them apart: the engine never ran, it found nothing to do, it
    could not run, or it ran and judged its own output worse. Only the
    third is a fault; only the fourth says the rules need work.
    """

    def _rejecting(self, monkeypatch):
        """Patch the engine to reject whatever it renders.

        The real filter chain does not currently produce a render worth
        rejecting, which is the point of it — so the verdict is forced
        here rather than waited for.
        """
        import luber_generation_client.postprocess as postprocess
        from luber_audio_finishing import finish_audio as real_finish
        from luber_audio_finishing.acceptance import (
            AcceptanceCheck,
            AcceptanceOutcome,
            AcceptanceVerdict,
            CheckKind,
        )

        refused = AcceptanceVerdict(
            outcome=AcceptanceOutcome.REJECTED,
            checks=(
                AcceptanceCheck(
                    kind=CheckKind.REGRESSION,
                    name="sibilance not worsened",
                    passed=False,
                    detail="6-9 kHz peak excess worsened by +2.10, past the 0.75 tolerance",
                ),
            ),
        )

        def rejecting(source, destination, **kwargs):
            result = real_finish(source, destination, **kwargs)
            destination.unlink(missing_ok=True)
            return dataclasses.replace(result, output_path=None, verdict=refused)

        monkeypatch.setattr(postprocess, "finish_audio", rejecting)

    async def test_a_rejected_render_delivers_the_raw_master(self, storage, monkeypatch):
        self._rejecting(monkeypatch)
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)

        assert result.finished is None
        assert result.finishing.outcome is FinishingOutcome.REJECTED
        assert await storage.exists(master_storage_key(gid))
        assert await storage.exists(preview_storage_key(gid))

    async def test_no_finished_object_survives_a_rejection(self, storage, monkeypatch):
        """The rejected audio must not be reachable by any key."""
        self._rejecting(monkeypatch)
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        assert not await storage.exists(finished_master_storage_key(gid))
        assert all(a.asset_type is not AssetType.FINISHED_MASTER for a in result.assets)

    async def test_the_preview_comes_from_the_raw_master(self, storage, monkeypatch):
        """A preview cut from audio the engine refused would ship it anyway."""
        self._rejecting(monkeypatch)
        gid = uuid.uuid4()
        result = await produce_delivery_assets(gid, FIXTURE, storage)
        assert result.delivery_master is result.master

    async def test_a_rejection_is_not_recorded_as_a_failure(self, storage, monkeypatch):
        self._rejecting(monkeypatch)
        result = await produce_delivery_assets(uuid.uuid4(), FIXTURE, storage)
        assert result.finishing.outcome is not FinishingOutcome.FAILED
        assert result.finishing.error is None

    async def test_the_reasons_are_kept(self, storage, monkeypatch):
        """A rejection nobody can explain cannot be acted on."""
        self._rejecting(monkeypatch)
        result = await produce_delivery_assets(uuid.uuid4(), FIXTURE, storage)
        verdict = result.finishing.verdict
        assert verdict is not None
        assert verdict["outcome"] == "REJECTED"
        assert verdict["failed_checks"] == ["sibilance not worsened"]

    async def test_the_verdict_reaches_the_durable_trace(
        self, repository, storage, tmp_path, monkeypatch
    ):
        """It has to survive the process, not just the function call."""
        import json

        self._rejecting(monkeypatch)
        gen = await _queued(repository)
        service = GenerationService(
            repository, MockGenerationProvider(healthy_wav(tmp_path / "h.wav")), storage
        )
        await service.execute(gen.id, worker_id="w")

        fetched = await repository.get_generation(gen.id)
        trace = json.loads(fetched.finishing_trace)
        assert trace["outcome"] == FinishingOutcome.REJECTED.value
        assert trace["verdict"]["failed_checks"] == ["sibilance not worsened"]
