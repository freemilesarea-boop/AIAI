"""Orchestration: scan, analyse in parallel, decide in one place.

The shape is deliberate. Everything expensive and independent — decode,
technical analysis, fingerprint, tempo, key — happens per file in a
worker pool. Everything that needs to see the whole corpus — dedup,
grouping, splitting — happens afterwards in the parent, single-threaded
and in sorted order.

That division is what makes the output deterministic despite the
parallelism. Workers may finish in any order; nothing downstream depends
on the order they finished in, because results are collected into a dict
keyed by track id and every later stage iterates sorted keys.

**A worker failure costs one track.** Anything a worker raises is caught,
recorded against that file, and the run continues. A single unreadable
file in a library of forty thousand must not end a six-hour job — and
the record of what failed is more useful than the exception would have
been anyway.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_audio_finishing import AudioLoadError, load_audio
from luber_dataset.factory import audio_analysis, classification, dedup, metadata, musical
from luber_dataset.factory.cache import AnalysisCache
from luber_dataset.factory.config import (
    DECODE_VERSION,
    FACTORY_VERSION,
    FINGERPRINT_VERSION,
    MUSICAL_ANALYSIS_VERSION,
    TECHNICAL_ANALYSIS_VERSION,
    FactoryConfig,
)
from luber_dataset.factory.decoder import DecodeResult, DecodeStatus, decode_check
from luber_dataset.factory.provenance import resolve as resolve_provenance
from luber_dataset.factory.quality import QualityTier
from luber_dataset.factory.quality import evaluate as evaluate_quality
from luber_dataset.factory.scanner import ScannedFile, scan, verify_unchanged
from luber_dataset.factory.schemas import (
    REVIEW_LANGUAGE_UNCERTAIN,
    REVIEW_NEAR_DUPLICATE,
    REVIEW_QUALITY_BORDERLINE,
    REVIEW_RIGHTS_UNKNOWN,
    REVIEW_VOCAL_CLASS_UNCERTAIN,
    RejectionRecord,
    ReviewItem,
    TrackRecord,
)
from luber_dataset.factory.splitting import (
    Split,
    assess_eligibility,
    assign_splits,
    build_groups,
    group_key,
    verify_no_leakage,
)

logger = logging.getLogger(__name__)

#: Leave headroom: a machine pinned at 100% for six hours is unusable
#: for anything else, and the operator is normally sitting at it.
DEFAULT_WORKER_HEADROOM = 2


@dataclass
class FileAnalysis:
    """Everything a worker produces for one file. Picklable by design."""

    sha256: str
    source_path: str
    decode: dict[str, Any] = field(default_factory=dict)
    technical: dict[str, Any] = field(default_factory=dict)
    music: dict[str, Any] = field(default_factory=dict)
    fingerprint: str | None = None
    embedded_tags: dict[str, str] = field(default_factory=dict)
    sidecar_fields: dict[str, Any] | None = None
    sidecar_error: str | None = None
    lyrics_file: tuple[str, str] | None = None
    centre_dominance_db: float | None = None
    worker_error: str | None = None
    cache_hits: int = 0
    cache_misses: int = 0


def worker_count(requested: int) -> int:
    if requested > 0:
        return requested
    available = os.cpu_count() or 2
    return max(1, available - DEFAULT_WORKER_HEADROOM)


def _cached(cache_payload: dict[str, dict[str, Any]] | None, stage: str) -> dict[str, Any] | None:
    return (cache_payload or {}).get(stage)


def analyse_file(
    scanned: ScannedFile,
    *,
    cached: dict[str, dict[str, Any]] | None = None,
    measure_loudness: bool = True,
) -> FileAnalysis:
    """All per-file work. Runs in a worker process; never raises.

    Every stage is attempted independently, so a file whose tempo
    estimation fails still contributes its technical analysis. Partial
    information about a track is worth far more than none.
    """
    result = FileAnalysis(sha256=scanned.sha256, source_path=scanned.source_path)
    path = Path(scanned.source_path)

    try:
        # ── decode ───────────────────────────────────────────────────
        decode_payload = _cached(cached, "decode")
        if decode_payload is not None:
            result.cache_hits += 1
            decode = DecodeResult(
                status=DecodeStatus(decode_payload["status"]),
                decode_error=decode_payload.get("decode_error"),
                duration_seconds=decode_payload.get("duration_seconds"),
                sample_rate=decode_payload.get("sample_rate"),
                channels=decode_payload.get("channels"),
                bit_depth=decode_payload.get("bit_depth"),
                codec=decode_payload.get("codec"),
                container=decode_payload.get("container"),
            )
        else:
            result.cache_misses += 1
            decode = decode_check(path)
        result.decode = {
            "status": decode.status.value,
            "decode_error": decode.decode_error,
            "duration_seconds": decode.duration_seconds,
            "sample_rate": decode.sample_rate,
            "channels": decode.channels,
            "bit_depth": decode.bit_depth,
            "codec": decode.codec,
            "container": decode.container,
        }

        # ── metadata: cheap, and useful even for undecodable files ───
        try:
            sidecar = metadata.load_sidecar(path)
            result.sidecar_fields = sidecar.fields if sidecar else None
        except metadata.SidecarError as exc:
            # Recorded, not raised. The operator meant to say something
            # and it did not arrive; that belongs in the review queue.
            result.sidecar_error = str(exc)
        result.embedded_tags = metadata.read_embedded_tags(path)
        result.lyrics_file = metadata.find_lyrics_sidecar(path)

        if not decode.usable:
            return result

        # ── technical analysis ───────────────────────────────────────
        technical_payload = _cached(cached, "technical")
        if technical_payload is not None:
            result.cache_hits += 1
            result.technical = technical_payload
        else:
            result.cache_misses += 1
            analysis = audio_analysis.analyse(path, decode, measure_loudness=measure_loudness)
            result.technical = analysis.to_dict()

        # ── everything needing decoded samples ───────────────────────
        music_payload = _cached(cached, "music")
        fingerprint_payload = _cached(cached, "fingerprint")
        needs_samples = music_payload is None or fingerprint_payload is None

        if needs_samples:
            try:
                loaded = load_audio(path)
            except (AudioLoadError, OSError, ValueError) as exc:
                result.worker_error = f"sample load failed: {exc}"
                loaded = None
            if loaded is not None:
                mono = loaded.mono()
                if fingerprint_payload is None:
                    result.cache_misses += 1
                    result.fingerprint = dedup.compute_fingerprint(mono, loaded.sample_rate)
                if music_payload is None:
                    result.cache_misses += 1
                    result.music = musical.analyse(mono, loaded.sample_rate).to_dict()
                rolloff, cutoff = audio_analysis.spectral_shape(mono, loaded.sample_rate)
                if result.technical:
                    result.technical.setdefault("spectral_rolloff_hz", rolloff)
                    result.technical["spectral_rolloff_hz"] = rolloff
                    result.technical["high_frequency_cutoff_hz"] = cutoff

        if fingerprint_payload is not None:
            result.cache_hits += 1
            result.fingerprint = fingerprint_payload.get("fingerprint")
        if music_payload is not None:
            result.cache_hits += 1
            result.music = music_payload

        result.centre_dominance_db = classification.centre_dominance(
            result.technical.get("mid_energy_db"), result.technical.get("side_energy_db")
        )
    except Exception as exc:
        result.worker_error = f"{type(exc).__name__}: {exc}"
    return result


def _stage_versions() -> dict[str, str]:
    return {
        "decode": DECODE_VERSION,
        "technical": TECHNICAL_ANALYSIS_VERSION,
        "music": MUSICAL_ANALYSIS_VERSION,
        "fingerprint": FINGERPRINT_VERSION,
    }


@dataclass
class FactoryResult:
    records: list[TrackRecord] = field(default_factory=list)
    rejections: list[RejectionRecord] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    review_queue: list[ReviewItem] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    source_integrity_ok: bool = True
    changed_sources: list[str] = field(default_factory=list)
    leaked_groups: list[str] = field(default_factory=list)
    cache_stats: dict[str, Any] = field(default_factory=dict)
    worker_failures: list[tuple[str, str]] = field(default_factory=list)


def run(
    input_root: Path,
    output_root: Path,
    config: FactoryConfig,
    *,
    resume: bool = True,
    force_reanalyze: bool = False,
    max_files: int | None = None,
    measure_loudness: bool = True,
    progress: bool = False,
) -> FactoryResult:
    """The whole factory, end to end."""
    started = datetime.now(UTC)
    scanned = scan(input_root, max_files=max_files)
    result = FactoryResult()

    cache = AnalysisCache(output_root / "cache" / "analysis_cache.json")
    if force_reanalyze:
        cache = AnalysisCache(Path(os.devnull + "-missing"))
        cache.path = output_root / "cache" / "analysis_cache.json"

    versions = _stage_versions()
    configuration_key = config.configuration_hash()[:16]

    def cached_for(item: ScannedFile) -> dict[str, dict[str, Any]] | None:
        if not resume or force_reanalyze:
            return None
        found: dict[str, dict[str, Any]] = {}
        for stage, version in versions.items():
            key = AnalysisCache.key(item.sha256, stage, version, configuration_key)
            entry = cache.get(key)
            if entry is not None:
                found[stage] = entry
        return found or None

    analyses: dict[str, FileAnalysis] = {}
    workers = worker_count(config.workers)
    # One analysis per *identity*, not per file. Byte-identical copies
    # would otherwise be decoded and measured several times over for the
    # same answer, and the last one to finish would silently overwrite
    # the others.
    representatives: dict[str, ScannedFile] = {}
    for item in sorted(scanned.files, key=lambda f: (f.file_id, f.source_path)):
        representatives.setdefault(item.file_id, item)
    payloads = [(item, cached_for(item)) for item in representatives.values()]

    if workers == 1 or len(payloads) <= 1:
        for item, cached in payloads:
            analyses[item.file_id] = analyse_file(
                item, cached=cached, measure_loudness=measure_loudness
            )
    else:
        # Bounded pool, never one process per file.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    analyse_file, item, cached=cached, measure_loudness=measure_loudness
                ): item
                for item, cached in payloads
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    analyses[item.file_id] = future.result()
                except Exception as exc:
                    logger.warning("worker failed for %s: %s", item.source_filename, exc)
                    analyses[item.file_id] = FileAnalysis(
                        sha256=item.sha256,
                        source_path=item.source_path,
                        worker_error=f"{type(exc).__name__}: {exc}",
                    )

    _adopt_sibling_metadata(scanned.files, analyses)

    for item in representatives.values():
        analysis = analyses[item.file_id]
        if analysis.worker_error:
            result.worker_failures.append((item.source_path, analysis.worker_error))
        # Repopulate the cache under the current keys.
        if analysis.decode:
            cache.put(
                AnalysisCache.key(item.sha256, "decode", versions["decode"], configuration_key),
                analysis.decode,
            )
        if analysis.technical:
            cache.put(
                AnalysisCache.key(
                    item.sha256, "technical", versions["technical"], configuration_key
                ),
                analysis.technical,
            )
        if analysis.music:
            cache.put(
                AnalysisCache.key(item.sha256, "music", versions["music"], configuration_key),
                analysis.music,
            )
        if analysis.fingerprint is not None:
            cache.put(
                AnalysisCache.key(
                    item.sha256, "fingerprint", versions["fingerprint"], configuration_key
                ),
                {"fingerprint": analysis.fingerprint},
            )

    cache.prune({item.sha256 for item in scanned.files})
    cache.flush()
    result.cache_stats = cache.stats.to_dict()

    _assemble(scanned.files, analyses, config, result, started)

    changed = verify_unchanged(scanned.files)
    result.changed_sources = changed
    result.source_integrity_ok = not changed
    return result


def _adopt_sibling_metadata(files: list[ScannedFile], analyses: dict[str, FileAnalysis]) -> None:
    """Recover metadata attached to a copy rather than to the original.

    Analysis is a property of the audio; a sidecar is a property of a
    *path*. When the same bytes sit in two folders and only one of them
    carries `track.json`, analysing the identity once means whichever
    path was picked decides whether the operator's rights declaration is
    seen at all.

    That is not a cosmetic loss. In the first integration run it turned
    three tracks the operator had explicitly declared as their own into
    RIGHTS_UNKNOWN, because the copy sorted first. So every path sharing
    an identity is checked, and the first declaration found is adopted.
    """
    by_id: dict[str, list[ScannedFile]] = {}
    for item in sorted(files, key=lambda f: (f.file_id, f.source_path)):
        by_id.setdefault(item.file_id, []).append(item)

    for track_id, group in by_id.items():
        analysis = analyses.get(track_id)
        if analysis is None or len(group) == 1:
            continue
        for sibling in group:
            path = Path(sibling.source_path)
            if analysis.sidecar_fields is None and analysis.sidecar_error is None:
                try:
                    sidecar = metadata.load_sidecar(path)
                except metadata.SidecarError as exc:
                    analysis.sidecar_error = str(exc)
                else:
                    if sidecar is not None:
                        analysis.sidecar_fields = sidecar.fields
                        analysis.source_path = sibling.source_path
            if analysis.lyrics_file is None:
                analysis.lyrics_file = metadata.find_lyrics_sidecar(path)


def _assemble(
    files: list[ScannedFile],
    analyses: dict[str, FileAnalysis],
    config: FactoryConfig,
    result: FactoryResult,
    started: datetime,
) -> None:
    """Corpus-wide decisions, single-threaded and in sorted order."""
    # Identity is content, so byte-identical files are *one* track with
    # several source paths rather than several tracks. Collapsing here
    # rather than later is what keeps a track id unique in the manifest:
    # emitting one record per file would emit the same id twice.
    by_id: dict[str, list[ScannedFile]] = {}
    for item in sorted(files, key=lambda f: (f.file_id, f.source_path)):
        by_id.setdefault(item.file_id, []).append(item)

    candidates = [
        dedup._Candidate(
            track_id=track_id,
            sha256=group[0].sha256,
            source_path=group[0].source_path,
            fingerprint=analyses[track_id].fingerprint,
            duration_seconds=(analyses[track_id].decode or {}).get("duration_seconds"),
        )
        for track_id, group in sorted(by_id.items())
    ]
    dedup_records = dedup.analyse_duplicates(candidates, config.dedup)

    # The same loss again, one level up. When a WAV and its lossless
    # transcode are merged, the canonical is chosen by track id — which
    # has nothing to do with which of the two the operator put their
    # sidecar beside. A declaration must not depend on that coin flip.
    for track_id, merge in dedup_records.items():
        target = merge.canonical_track_id
        if target == track_id:
            continue
        canonical_analysis = analyses.get(target)
        merged_analysis = analyses.get(track_id)
        if canonical_analysis is None or merged_analysis is None:
            continue
        if canonical_analysis.sidecar_fields is None and merged_analysis.sidecar_fields:
            canonical_analysis.sidecar_fields = merged_analysis.sidecar_fields
        if canonical_analysis.lyrics_file is None and merged_analysis.lyrics_file:
            canonical_analysis.lyrics_file = merged_analysis.lyrics_file

    grouping: list[tuple[str, str]] = []
    staged: list[tuple[Any, ...]] = []

    for track_id, group in sorted(by_id.items()):
        item = group[0]
        analysis = analyses[track_id]
        path = Path(item.source_path)
        dedup_record = dedup_records[track_id]
        # Every path the same bytes were found at, so nothing is lost by
        # collapsing them and the operator can still find each copy.
        every_path = sorted({member.source_path for member in group})
        dedup_record.all_source_paths = sorted(set(dedup_record.all_source_paths) | set(every_path))
        if len(group) > 1:
            dedup_record.duplicate_type = dedup.DuplicateType.EXACT_FILE.value
            dedup_record.similarity_score = 1.0
            dedup_record.duplicate_group_id = (
                dedup_record.duplicate_group_id or f"dup_{item.sha256[:12]}"
            )

        sidecar = (
            metadata.Sidecar(path=str(metadata.sidecar_path(path)), fields=analysis.sidecar_fields)
            if analysis.sidecar_fields is not None
            else None
        )
        provenance = resolve_provenance(path, sidecar, embedded=analysis.embedded_tags)

        decode = DecodeResult(
            status=DecodeStatus(analysis.decode.get("status", DecodeStatus.INVALID.value))
            if analysis.decode
            else DecodeStatus.INVALID,
            decode_error=(analysis.decode or {}).get("decode_error"),
            duration_seconds=(analysis.decode or {}).get("duration_seconds"),
            sample_rate=(analysis.decode or {}).get("sample_rate"),
            channels=(analysis.decode or {}).get("channels"),
            bit_depth=(analysis.decode or {}).get("bit_depth"),
            codec=(analysis.decode or {}).get("codec"),
            container=(analysis.decode or {}).get("container"),
        )
        technical = audio_analysis.TechnicalAnalysis(**_technical_kwargs(analysis.technical))
        quality = evaluate_quality(
            decode,
            technical,
            config.quality,
            near_duplicate=dedup_record.dedup_decision == "REVIEW_REQUIRED",
        )
        eligibility = assess_eligibility(
            decoded=decode.usable,
            quality=quality,
            provenance=provenance,
            dedup=dedup_record,
            min_tier=config.min_training_tier,
            include_rights_unknown=config.include_rights_unknown,
        )

        vocals = classification.assess_vocals(
            sidecar, centre_dominance_db=analysis.centre_dominance_db
        )
        text = classification.assess_text(sidecar, analysis.lyrics_file, analysis.embedded_tags)
        language = classification.assess_language(
            sidecar, analysis.embedded_tags, lyrics=text.lyrics
        )

        artist = (sidecar.get("artist") if sidecar else None) or analysis.embedded_tags.get(
            "artist"
        )
        album = (sidecar.get("album") if sidecar else None) or analysis.embedded_tags.get("album")
        key = group_key(
            track_id=item.file_id,
            duplicate_group_id=dedup_record.duplicate_group_id,
            artist=artist,
            album=album,
            parent_directory=path.parent.name,
        )
        grouping.append((track_id, key))
        staged.append(
            (
                item,
                group,
                analysis,
                dedup_record,
                provenance,
                quality,
                eligibility,
                vocals,
                text,
                language,
            )
        )

    groups = build_groups(grouping)
    assignment = assign_splits(groups, config.split)
    result.leaked_groups = verify_no_leakage(groups, assignment)

    for (
        item,
        group,
        analysis,
        dedup_record,
        provenance,
        quality,
        eligibility,
        vocals,
        text,
        language,
    ) in staged:
        is_canonical = dedup_record.canonical_track_id == item.file_id
        split = (
            assignment.get(item.file_id, Split.EXCLUDED.value)
            if eligibility.training_eligible
            else Split.EXCLUDED.value
        )

        record = TrackRecord(
            track_id=item.file_id,
            source={
                "source_path": item.source_path,
                "source_filename": item.source_filename,
                "source_extension": item.source_extension,
                "source_size_bytes": item.source_size_bytes,
                "source_mtime": item.source_mtime,
                "sha256": item.sha256,
            },
            audio=analysis.decode,
            analysis=analysis.technical,
            music=analysis.music,
            vocals=vocals.to_dict(),
            text=text.to_dict(),
            quality=quality.to_dict(),
            provenance=provenance.to_dict(),
            dedup=dedup_record.to_dict(),
            eligibility=eligibility.to_dict(),
            metadata={
                "language": language.to_dict(),
                "embedded_tags": dict(sorted(analysis.embedded_tags.items())),
                "sidecar": analysis.sidecar_fields,
                "sidecar_error": analysis.sidecar_error,
                "worker_error": analysis.worker_error,
            },
            split=split,
        )
        # Extra copies of identical bytes are reported as duplicates
        # without being separate tracks.
        for extra in group[1:]:
            result.duplicates.append(
                {
                    "track_id": item.file_id,
                    "canonical_track_id": dedup_record.canonical_track_id,
                    "source_path": extra.source_path,
                    "duplicate_type": dedup.DuplicateType.EXACT_FILE.value,
                    "similarity_score": 1.0,
                    "dedup_decision": dedup.DedupDecision.MERGED.value,
                    "duplicate_group_id": dedup_record.duplicate_group_id,
                }
            )

        if is_canonical:
            result.records.append(record)
        else:
            result.duplicates.append(
                {
                    "track_id": item.file_id,
                    "canonical_track_id": dedup_record.canonical_track_id,
                    **dedup_record.to_dict(),
                }
            )

        if not eligibility.training_eligible:
            result.rejections.append(
                RejectionRecord(
                    track_id=item.file_id,
                    source_path=item.source_path,
                    reasons=eligibility.eligibility_reasons,
                    quality_flags=quality.quality_flags,
                    quality_tier=quality.quality_tier,
                    decode_status=analysis.decode.get("status", "INVALID")
                    if analysis.decode
                    else "INVALID",
                    detail="; ".join(quality.reasons),
                )
            )

        result.review_queue.extend(
            _review_items(item, analysis, dedup_record, provenance, quality, vocals, language)
        )

    result.summary = _summarize(result, started, config)


def _technical_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Only the fields TechnicalAnalysis declares, so a cache written by
    an older version cannot crash a newer one on an unexpected key."""
    allowed = set(audio_analysis.TechnicalAnalysis.__dataclass_fields__)
    return {key: value for key, value in (payload or {}).items() if key in allowed}


def _review_items(
    item: ScannedFile,
    analysis: FileAnalysis,
    dedup_record: Any,
    provenance: Any,
    quality: Any,
    vocals: Any,
    language: Any,
) -> Iterable[ReviewItem]:
    items: list[ReviewItem] = []

    if dedup_record.dedup_decision == "REVIEW_REQUIRED":
        items.append(
            ReviewItem(
                track_id=item.file_id,
                reason=REVIEW_NEAR_DUPLICATE,
                detail=(
                    f"fingerprint matches {dedup_record.duplicate_of} at "
                    f"{dedup_record.similarity_score}"
                ),
                source_path=item.source_path,
                recommended_action="compare both files by ear; merge or keep both deliberately",
                metrics={
                    "similarity_score": dedup_record.similarity_score,
                    "duplicate_of": dedup_record.duplicate_of,
                },
            )
        )

    if provenance.hard_blocks:
        pass  # Not a review item: hard blocks are not negotiable.
    elif provenance.commercial_training_allowed == "UNKNOWN":
        items.append(
            ReviewItem(
                track_id=item.file_id,
                reason=REVIEW_RIGHTS_UNKNOWN,
                detail=(
                    f"rights_status={provenance.rights_status}, "
                    f"training permission={provenance.commercial_training_allowed}"
                ),
                source_path=item.source_path,
                recommended_action=(
                    "add a sidecar declaring rights_status and "
                    "commercial_training_allowed, or leave excluded"
                ),
                metrics={"source_type": provenance.source_type},
            )
        )

    if vocals.vocal_class == "UNCERTAIN":
        items.append(
            ReviewItem(
                track_id=item.file_id,
                reason=REVIEW_VOCAL_CLASS_UNCERTAIN,
                detail=vocals.reason,
                source_path=item.source_path,
                recommended_action="declare vocal_type in a sidecar if it matters downstream",
                metrics={"centre_dominance_db": vocals.centre_dominance_db},
            )
        )

    if language.language == "unknown":
        items.append(
            ReviewItem(
                track_id=item.file_id,
                reason=REVIEW_LANGUAGE_UNCERTAIN,
                detail=language.reason,
                source_path=item.source_path,
                recommended_action="declare language in a sidecar, or supply lyrics",
                metrics={},
            )
        )

    if quality.quality_tier == QualityTier.C.value:
        items.append(
            ReviewItem(
                track_id=item.file_id,
                reason=REVIEW_QUALITY_BORDERLINE,
                detail="; ".join(quality.reasons) or "borderline score",
                source_path=item.source_path,
                recommended_action="listen before including; raise or lower the tier floor",
                metrics={
                    "quality_score": quality.quality_score,
                    "quality_flags": quality.quality_flags,
                },
            )
        )
    return items


def _summarize(result: FactoryResult, started: datetime, config: FactoryConfig) -> dict[str, Any]:
    records = result.records
    total_seconds = 0.0
    per_split: dict[str, float] = {
        Split.TRAIN.value: 0.0,
        Split.VALIDATION.value: 0.0,
        Split.TEST.value: 0.0,
    }
    counters: dict[str, dict[str, int]] = {
        "sample_rate_distribution": {},
        "channel_distribution": {},
        "duration_distribution": {},
        "quality_flag_distribution": {},
        "language_distribution": {},
    }

    def bump(name: str, key: Any) -> None:
        bucket = counters[name]
        text = str(key)
        bucket[text] = bucket.get(text, 0) + 1

    valid = invalid = 0
    tiers = {"A": 0, "B": 0, "C": 0, "REJECT": 0}
    vocal = instrumental = uncertain_vocals = 0
    rights_verified = rights_unknown = 0
    training_eligible = 0

    for record in records:
        duration = (record.analysis or {}).get("duration_seconds") or 0.0
        total_seconds += duration
        if record.split in per_split:
            per_split[record.split] += duration

        status = (record.audio or {}).get("status")
        if status in ("VALID", "PARTIAL"):
            valid += 1
        else:
            invalid += 1

        tier = (record.quality or {}).get("quality_tier", "REJECT")
        tiers[tier] = tiers.get(tier, 0) + 1
        for flag in (record.quality or {}).get("quality_flags", []):
            bump("quality_flag_distribution", flag)

        bump("sample_rate_distribution", (record.audio or {}).get("sample_rate"))
        bump("channel_distribution", (record.audio or {}).get("channels"))
        bump("duration_distribution", _duration_bucket(duration))
        bump("language_distribution", (record.metadata or {}).get("language", {}).get("language"))

        vocal_class = (record.vocals or {}).get("vocal_class")
        if vocal_class == "VOCAL":
            vocal += 1
        elif vocal_class == "INSTRUMENTAL":
            instrumental += 1
        else:
            uncertain_vocals += 1

        provenance = record.provenance or {}
        if provenance.get("rights_status") == "VERIFIED":
            rights_verified += 1
        if provenance.get("commercial_training_allowed") == "UNKNOWN":
            rights_unknown += 1
        if (record.eligibility or {}).get("training_eligible"):
            training_eligible += 1

    return {
        "factory_version": FACTORY_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration_hash": config.configuration_hash(),
        "total_files": len(records) + len(result.duplicates),
        "valid_audio": valid,
        "invalid_audio": invalid,
        "canonical_tracks": len(records),
        "exact_duplicates": sum(
            1 for d in result.duplicates if d.get("duplicate_type") in ("EXACT_FILE", "EXACT_AUDIO")
        ),
        "near_duplicates": sum(
            1 for r in records if (r.dedup or {}).get("duplicate_type") == "NEAR_AUDIO"
        ),
        "review_required": len(result.review_queue),
        "quality_A": tiers.get("A", 0),
        "quality_B": tiers.get("B", 0),
        "quality_C": tiers.get("C", 0),
        "rejected": len(result.rejections),
        "training_eligible": training_eligible,
        "rights_verified": rights_verified,
        "rights_unknown": rights_unknown,
        "vocal_tracks": vocal,
        "instrumental_tracks": instrumental,
        "vocal_class_uncertain": uncertain_vocals,
        "duration_total_hours": round(total_seconds / 3600.0, 4),
        "train_hours": round(per_split[Split.TRAIN.value] / 3600.0, 4),
        "validation_hours": round(per_split[Split.VALIDATION.value] / 3600.0, 4),
        "test_hours": round(per_split[Split.TEST.value] / 3600.0, 4),
        "split_counts": {
            name: sum(1 for r in records if r.split == name)
            for name in (
                Split.TRAIN.value,
                Split.VALIDATION.value,
                Split.TEST.value,
                Split.EXCLUDED.value,
            )
        },
        "worker_failures": len(result.worker_failures),
        "cache": result.cache_stats,
        "source_integrity_ok": result.source_integrity_ok,
        "split_leakage_groups": result.leaked_groups,
        **{name: dict(sorted(bucket.items())) for name, bucket in counters.items()},
    }


def _duration_bucket(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    for edge, label in (
        (30, "0-30s"),
        (60, "30-60s"),
        (120, "1-2m"),
        (300, "2-5m"),
        (600, "5-10m"),
    ):
        if seconds < edge:
            return label
    return "10m+"
