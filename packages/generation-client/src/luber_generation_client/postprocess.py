"""Post-processing: raw model output → shippable audio assets.

Sits between inference and storage. Takes whatever the provider wrote,
produces the delivery assets, and hands them to the storage backend.
Nothing here knows which storage backend is configured.

The order is the safety property:

    provider output
      → transcode          format normalisation only, no sound change
      → RAW MASTER         written first, never written again
      → finishing          reads the raw, writes a *different* object
      → FINISHED MASTER    only when the engine decided to act
      → preview            encoded from whichever master will be served

The raw master is produced and stored before finishing is attempted, so
there is no ordering in which a finishing failure can leave a generation
without a master. Finishing writes to its own key; it is never handed
the raw's destination.

**Two failure policies, deliberately different.** Anything required for
delivery — the transcode, the preview, their uploads — still fails the
whole step, and the caller must not mark the generation COMPLETED.
Finishing is not required for delivery: the raw master is a complete,
shippable product and was the entire product before Phase 14B. A
finishing failure is therefore logged, recorded in the trace, and the
raw ships. Failing the generation instead would convert a working
feature into a new failure mode and discard a successful inference.

That fallback cannot publish a bad file, because the engine verifies its
own output (clipping, ceiling, duration, rate, channels, balance) and
deletes it on failure rather than returning it. The pipeline only
decides whether to *ship* the enhancement, never whether it is safe.

Intermediate files live in a temporary directory removed on success and
failure alike. Stored objects use deterministic keys, so a retry
overwrites its own previous attempt rather than accumulating garbage.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from luber_audio_finishing import (
    FINISHING_VERSION,
    FinishingError,
    finish_audio,
    plan_to_dict,
    verdict_to_dict,
)
from luber_audio_utils import (
    MASTER_FILE_EXTENSION,
    MASTER_FORMAT,
    MASTER_MIME_TYPE,
    PREVIEW_FILE_EXTENSION,
    PREVIEW_FORMAT,
    PREVIEW_MIME_TYPE,
    AudioProbe,
    AudioStorage,
    encode_preview_mp3_async,
    finished_master_storage_key,
    master_storage_key,
    preview_storage_key,
    probe_audio,
    sha256_file,
    transcode_master_wav_async,
)
from luber_schemas import AssetType, FinishingOutcome

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProducedAsset:
    """One delivery asset, ready to be recorded in the database."""

    asset_type: AssetType
    format: str
    mime_type: str
    file_extension: str
    sample_rate: int
    bit_depth: int | None
    bitrate: int | None
    channels: int
    duration: float
    storage_key: str
    sha256: str
    file_size: int


@dataclass(frozen=True)
class FinishingRecord:
    """What finishing did, in a form that survives in the database.

    Written for every generation that reaches post-processing, including
    the ones nothing happened to. An absent record means the generation
    predates Phase 14B; a present one with ``NO_ACTION`` means the engine
    looked and found nothing worth correcting. Those are different facts
    and a system that cannot tell them apart cannot be reasoned about
    later.
    """

    outcome: FinishingOutcome
    finishing_version: str
    #: Digest of the raw master this decision was made from, so a stored
    #: record can be tied back to the exact bytes that produced it.
    source_sha256: str
    #: Serialised plan: risks, actions, ceilings, deferrals. ``None``
    #: when the engine raised before producing one.
    plan: dict[str, object] | None = None
    #: How the render was judged against the raw master: every check, its
    #: numbers, and the verdict. ``None`` when nothing was rendered.
    #: Kept for REJECTED and FINISHED alike, because "why was this one
    #: accepted?" is as much a question as "why was that one refused?".
    verdict: dict[str, object] | None = None
    #: Present only for FAILED, and never shown to users.
    error: str | None = None

    @property
    def acted(self) -> bool:
        return self.outcome is FinishingOutcome.FINISHED


@dataclass(frozen=True)
class PostProcessingResult:
    #: The raw generation master. Always present, never overwritten.
    master: ProducedAsset
    preview: ProducedAsset
    #: The finishing result, present only when the engine acted.
    finished: ProducedAsset | None
    finishing: FinishingRecord

    @property
    def assets(self) -> tuple[ProducedAsset, ...]:
        """Every asset to record, raw first."""
        produced = [self.master]
        if self.finished is not None:
            produced.append(self.finished)
        produced.append(self.preview)
        return tuple(produced)

    @property
    def delivery_master(self) -> ProducedAsset:
        """The master a listener will be served."""
        return self.finished if self.finished is not None else self.master


def _describe(
    *,
    asset_type: AssetType,
    fmt: str,
    mime_type: str,
    extension: str,
    probe: AudioProbe,
    storage_key: str,
    path: Path,
) -> ProducedAsset:
    return ProducedAsset(
        asset_type=asset_type,
        format=fmt,
        mime_type=mime_type,
        file_extension=extension,
        sample_rate=probe.sample_rate,
        bit_depth=probe.bit_depth,
        bitrate=probe.bitrate_bps,
        channels=probe.channels,
        duration=probe.duration_seconds,
        storage_key=storage_key,
        # Hash the exact bytes that were uploaded, so the recorded digest
        # is the digest of what a client will download.
        sha256=sha256_file(path),
        file_size=path.stat().st_size,
    )


async def _finish_master(
    generation_id: UUID,
    master_path: Path,
    finished_path: Path,
    master_sha256: str,
) -> tuple[Path | None, FinishingRecord]:
    """Run the finishing engine over the raw master.

    Returns the finished file when the engine acted, and always a record
    of what it decided. Never raises: the caller has a shippable raw
    master either way, and the decision about whether to *use* a
    finishing result is not the same decision as whether the generation
    succeeded.
    """
    try:
        result = await asyncio.to_thread(finish_audio, master_path, finished_path)
    except FinishingError as exc:
        # Includes the engine's own verification failures, which delete
        # their output before raising — so there is nothing half-written
        # to clean up here.
        logger.warning(
            "finishing failed; delivering the raw master",
            extra={"generation_id": str(generation_id), "error": str(exc)},
            exc_info=True,
        )
        return None, FinishingRecord(
            outcome=FinishingOutcome.FAILED,
            finishing_version=FINISHING_VERSION,
            source_sha256=master_sha256,
            error=str(exc),
        )

    plan = plan_to_dict(result.plan)
    verdict = None if result.verdict is None else verdict_to_dict(result.verdict)

    if result.rejected:
        # The engine rendered, measured its own output and judged the raw
        # master better. Logged at info: this is the safeguard working,
        # not a fault, and warning-level noise would train people to
        # ignore it.
        logger.info(
            "finishing was rejected on review; delivering the raw master",
            extra={
                "generation_id": str(generation_id),
                "reasons": list(result.rejection_reasons),
            },
        )
        return None, FinishingRecord(
            outcome=FinishingOutcome.REJECTED,
            finishing_version=result.finishing_version,
            source_sha256=master_sha256,
            plan=plan,
            verdict=verdict,
        )

    if not result.changed or result.output_path is None:
        logger.info(
            "finishing took no action; delivering the raw master",
            extra={"generation_id": str(generation_id)},
        )
        return None, FinishingRecord(
            outcome=FinishingOutcome.NO_ACTION,
            finishing_version=result.finishing_version,
            source_sha256=master_sha256,
            plan=plan,
        )

    return result.output_path, FinishingRecord(
        outcome=FinishingOutcome.FINISHED,
        finishing_version=result.finishing_version,
        source_sha256=master_sha256,
        plan=plan,
        verdict=verdict,
    )


async def produce_delivery_assets(
    generation_id: UUID,
    raw_audio: Path,
    storage: AudioStorage,
) -> PostProcessingResult:
    """Normalize, finish, encode, verify, and store the delivery assets.

    Raises when a *required* asset cannot be produced or stored; callers
    treat that as a failed generation. A finishing failure is not one of
    those — see the module docstring.
    """
    with tempfile.TemporaryDirectory(prefix=f"luber-post-{generation_id}-") as tmp:
        workdir = Path(tmp)
        master_path = workdir / f"master.{MASTER_FILE_EXTENSION}"
        finished_path = workdir / f"finished.{MASTER_FILE_EXTENSION}"
        preview_path = workdir / f"preview.{PREVIEW_FILE_EXTENSION}"

        # Format normalization only — verified against the contract
        # inside the transcoder itself.
        master_probe = await transcode_master_wav_async(raw_audio, master_path)
        master_key = master_storage_key(generation_id, MASTER_FILE_EXTENSION)
        master = _describe(
            asset_type=AssetType.MASTER,
            fmt=MASTER_FORMAT,
            mime_type=MASTER_MIME_TYPE,
            extension=MASTER_FILE_EXTENSION,
            probe=master_probe,
            storage_key=master_key,
            path=master_path,
        )

        finished_file, finishing = await _finish_master(
            generation_id, master_path, finished_path, master.sha256
        )

        finished: ProducedAsset | None = None
        if finished_file is not None:
            finished = _describe(
                asset_type=AssetType.FINISHED_MASTER,
                fmt=MASTER_FORMAT,
                mime_type=MASTER_MIME_TYPE,
                extension=MASTER_FILE_EXTENSION,
                probe=await asyncio.to_thread(probe_audio, finished_file),
                storage_key=finished_master_storage_key(generation_id, MASTER_FILE_EXTENSION),
                path=finished_file,
            )

        # The preview follows whichever master will be served, so what a
        # listener streams and what they download are the same audio.
        delivery_path = finished_file if finished_file is not None else master_path
        preview_probe = await encode_preview_mp3_async(delivery_path, preview_path)
        preview_key = preview_storage_key(generation_id, PREVIEW_FILE_EXTENSION)
        preview = _describe(
            asset_type=AssetType.PREVIEW,
            fmt=PREVIEW_FORMAT,
            mime_type=PREVIEW_MIME_TYPE,
            extension=PREVIEW_FILE_EXTENSION,
            probe=preview_probe,
            storage_key=preview_key,
            path=preview_path,
        )

        # Upload last: a storage failure leaves no asset rows written by
        # the caller, and the temp dir is cleaned up regardless. The raw
        # master goes first so that a partial upload can only ever be
        # missing the enhancement, never the product.
        await storage.put(master_key, master_path)
        if finished is not None and finished_file is not None:
            await storage.put(finished.storage_key, finished_file)
        await storage.put(preview_key, preview_path)

        logger.info(
            "post-processing produced delivery assets",
            extra={
                "generation_id": str(generation_id),
                "master_sha256": master.sha256,
                "preview_sha256": preview.sha256,
                "finishing_outcome": finishing.outcome.value,
                "finishing_version": finishing.finishing_version,
            },
        )
        return PostProcessingResult(
            master=master, preview=preview, finished=finished, finishing=finishing
        )
