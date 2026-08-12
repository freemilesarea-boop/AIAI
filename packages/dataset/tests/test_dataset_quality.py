"""Audio and lyric quality gates, schema, and manifest assembly."""

import math
import struct
import wave
from pathlib import Path

import pytest

from luber_dataset import (
    Delivery,
    OriginType,
    PronunciationStyle,
    QualityGrade,
    RightsBasis,
    RightsRecord,
    TrainingRightsStatus,
    TrainingRunManifest,
    TrainingTrack,
    VibratoAmount,
    VibratoCharacter,
    VocalAnnotation,
    VocalStyle,
    build_manifest,
    inspect_lyrics,
    inspect_training_audio,
    summarize_quality_grade,
    validate_run_manifest,
)
from luber_dataset.quality import (
    CLIPPING,
    LOW_SAMPLE_RATE,
    LYRICS_DUPLICATE_LINES,
    LYRICS_EMPTY,
    LYRICS_ENCODING_CORRUPTION,
    LYRICS_LANGUAGE_MISMATCH,
    LYRICS_NO_SECTIONS,
    LYRICS_SECTION_MISMATCH,
    MONO_SOURCE,
    OVER_COMPRESSED,
    TOO_SHORT,
    UNREADABLE,
)


def _wav(
    path: Path,
    *,
    seconds=30.0,
    rate=48000,
    channels=2,
    amplitude=0.5,
    square=False,
) -> Path:
    frames = int(rate * seconds)
    peak = int(32767 * min(amplitude, 1.0))
    data = bytearray()
    period = max(1, int(rate * 0.5))
    for i in range(frames):
        if square:
            value = peak if (i // 40) % 2 else -peak
        else:
            # Dynamic envelope: a pure sine has a 3 dB crest factor and
            # would be (correctly) flagged as brickwalled. Real music has
            # loud passages against quieter ones.
            envelope = 1.0 if (i % period) < period * 0.2 else 0.15
            value = int(peak * envelope * math.sin(2 * math.pi * 220 * i / rate))
        data += struct.pack("<h", max(-32768, min(32767, value))) * channels
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(data))
    return path


def _rights(**overrides) -> RightsRecord:
    base = {
        "origin_type": OriginType.HUMAN_RECORDED,
        "training_rights_status": TrainingRightsStatus.CONFIRMED,
        "basis": RightsBasis.ORIGINAL_WORK,
        "source": "commissioned session",
        "rights_holder": "LUBER",
        "document_reference": "contract-1",
        "confirmed_on": "2026-08-12",
        "audio_use_confirmed": True,
        "lyrics_rights_confirmed": True,
        "performer_rights_confirmed": True,
        "commercial_training_allowed": True,
    }
    base.update(overrides)
    return RightsRecord(**base)


def _vocal(style=VocalStyle.CONTEMPORARY_KPOP) -> VocalAnnotation:
    return VocalAnnotation(
        vocal_style=style,
        delivery=Delivery.SMOOTH,
        vibrato_amount=VibratoAmount.SUBTLE,
        vibrato_character=VibratoCharacter.NATURAL,
        pronunciation_style=PronunciationStyle.MODERN_STANDARD,
        genre_vocal_identity="contemporary korean pop",
    )


def _track(track_id="t1", **overrides) -> TrainingTrack:
    base = dict(
        track_id=track_id,
        source="studio",
        rights=_rights(),
        audio_sha256="a" * 64,
        duration_seconds=180.0,
        sample_rate=48000,
        channels=2,
        language="ko",
        genre="KPOP",
        subgenre="dance pop",
        vocal_gender="female",
        lyrics_available=True,
        bpm=124,
        key_scale="C major",
        time_signature="4",
        vocal=_vocal(),
        quality_grade=QualityGrade.GOOD,
    )
    base.update(overrides)
    return TrainingTrack(**base)  # type: ignore[arg-type]


# ── audio gate ────────────────────────────────────────────────────────


def test_clean_audio_is_accepted(tmp_path):
    quality = inspect_training_audio(_wav(tmp_path / "ok.wav"))
    assert quality.readable
    assert quality.sample_rate == 48000
    assert quality.channels == 2
    assert quality.acceptable, quality.flags


def test_clipped_audio_is_flagged(tmp_path):
    quality = inspect_training_audio(_wav(tmp_path / "clip.wav", amplitude=1.0, square=True))
    assert CLIPPING in quality.flags
    assert not quality.acceptable


def test_over_compressed_audio_is_flagged(tmp_path):
    """A square wave has almost no crest factor — brickwalled mastering."""
    quality = inspect_training_audio(_wav(tmp_path / "sq.wav", amplitude=0.9, square=True))
    assert OVER_COMPRESSED in quality.flags


def test_low_sample_rate_is_flagged(tmp_path):
    quality = inspect_training_audio(_wav(tmp_path / "lo.wav", rate=22050))
    assert LOW_SAMPLE_RATE in quality.flags


def test_mono_is_flagged(tmp_path):
    quality = inspect_training_audio(_wav(tmp_path / "mono.wav", channels=1))
    assert MONO_SOURCE in quality.flags


def test_too_short_is_flagged(tmp_path):
    quality = inspect_training_audio(_wav(tmp_path / "short.wav", seconds=5.0))
    assert TOO_SHORT in quality.flags


def test_unreadable_audio_is_flagged(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio")
    quality = inspect_training_audio(junk)
    assert quality.readable is False
    assert UNREADABLE in quality.flags
    assert inspect_training_audio(tmp_path / "missing.wav").flags == [UNREADABLE]


def test_decoding_alone_does_not_qualify_a_track(tmp_path):
    """The stated rule: decoding is necessary, not sufficient."""
    quality = inspect_training_audio(_wav(tmp_path / "bad.wav", rate=22050, channels=1))
    assert quality.readable is True
    assert quality.acceptable is False


# ── lyric QA ──────────────────────────────────────────────────────────


def test_well_formed_korean_lyrics_pass():
    lyrics = "[Verse]\n오늘 밤 너를 생각해\n조용한 창가에 앉아\n\n[Chorus]\n조금만 더 있어줘"
    result = inspect_lyrics(lyrics, language="ko")
    assert result.acceptable, result.flags
    assert result.sections == ["Verse", "Chorus"]
    assert result.line_count == 3


def test_empty_lyrics_flagged():
    assert LYRICS_EMPTY in inspect_lyrics("   ", language="ko").flags


def test_missing_section_tags_flagged():
    assert LYRICS_NO_SECTIONS in inspect_lyrics("가사 한 줄", language="ko").flags


def test_duplicate_consecutive_lines_flagged():
    lyrics = "[Verse]\n같은 줄\n같은 줄"
    assert LYRICS_DUPLICATE_LINES in inspect_lyrics(lyrics, language="ko").flags


def test_encoding_corruption_flagged():
    assert LYRICS_ENCODING_CORRUPTION in inspect_lyrics("[Verse]\n���ë", language="ko").flags


def test_language_mismatch_flagged():
    """Korean-labelled lyrics with no Hangul are mislabelled or wrong."""
    assert (
        LYRICS_LANGUAGE_MISMATCH in inspect_lyrics("[Verse]\nall english here", language="ko").flags
    )


def test_section_count_mismatch_flagged():
    lyrics = "[Verse]\n한 줄"
    assert (
        LYRICS_SECTION_MISMATCH in inspect_lyrics(lyrics, language="ko", expected_sections=4).flags
    )


def test_line_breaks_are_preserved_not_normalized():
    lyrics = "[Verse]\n첫 줄\n둘째 줄\n\n[Chorus]\n셋째 줄"
    result = inspect_lyrics(lyrics, language="ko")
    assert result.line_count == 3
    assert result.section_count == 2


# ── manifest assembly ─────────────────────────────────────────────────


def test_manifest_accepts_cleared_quality_tracks():
    manifest = build_manifest("LUBER_TRAINSET_V1", [_track("a"), _track("b")])
    assert manifest.track_count == 2
    assert manifest.exclusions == []
    assert manifest.content_hash()


def test_manifest_excludes_unverified_rights_with_a_reason():
    manifest = build_manifest(
        "T",
        [
            _track("ok"),
            _track("bad", rights=_rights(training_rights_status=TrainingRightsStatus.UNVERIFIED)),
        ],
    )
    assert [t.track_id for t in manifest.tracks] == ["ok"]
    assert manifest.exclusions[0].track_id == "bad"
    assert manifest.exclusions[0].reason == "RIGHTS"


def test_manifest_excludes_flagged_audio_and_lyrics():
    manifest = build_manifest(
        "T",
        [
            _track("audio", audio_quality_flags=[CLIPPING]),
            _track("lyric", lyrics_qa_flags=[LYRICS_DUPLICATE_LINES]),
        ],
    )
    assert manifest.track_count == 0
    assert {e.reason for e in manifest.exclusions} == {"AUDIO_QUALITY", "LYRICS_QA"}


def test_manifest_excludes_rejected_grade():
    manifest = build_manifest("T", [_track("r", quality_grade=QualityGrade.REJECTED)])
    assert manifest.exclusions[0].reason == "QUALITY_GRADE"


def test_manifest_excludes_the_bias_it_exists_to_correct():
    """Trot-style vocals are the Phase 5 failure; do not train more of it."""
    manifest = build_manifest("T", [_track("trot", vocal=_vocal(VocalStyle.TRADITIONAL_TROT))])
    assert manifest.track_count == 0
    assert manifest.exclusions[0].reason == "DISCOURAGED_STYLE"
    # Overridable only with an explicit, named argument.
    allowed = build_manifest(
        "T",
        [_track("trot", vocal=_vocal(VocalStyle.TRADITIONAL_TROT))],
        allow_discouraged_styles=True,
    )
    assert allowed.track_count == 1


def test_rights_are_checked_before_quality():
    """Unlicensed material is excluded on rights, not on audio flags."""
    manifest = build_manifest(
        "T",
        [
            _track(
                "x",
                rights=_rights(training_rights_status=TrainingRightsStatus.UNVERIFIED),
                audio_quality_flags=[CLIPPING],
            )
        ],
    )
    assert manifest.exclusions[0].reason == "RIGHTS"


def test_content_hash_is_stable_and_order_independent():
    a = build_manifest("T", [_track("a"), _track("b")])
    b = build_manifest("T", [_track("b"), _track("a")])
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_when_audio_changes():
    a = build_manifest("T", [_track("a", audio_sha256="a" * 64)])
    b = build_manifest("T", [_track("a", audio_sha256="b" * 64)])
    assert a.content_hash() != b.content_hash()


def test_manifest_reports_distributions_and_serializes(tmp_path):
    manifest = build_manifest(
        "T",
        [
            _track("a"),
            _track("b", language="en", vocal=_vocal(VocalStyle.INDIE_BREATHY)),
            _track("c", vocal_gender="instrumental", vocal=None, lyrics_available=False),
        ],
    )
    assert manifest.language_distribution() == {"en": 1, "ko": 2}
    assert manifest.style_distribution()["instrumental"] == 1
    path = manifest.write(tmp_path / "m.json")
    assert path.is_file()
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["track_count"] == 3
    assert data["tracks"][0]["training_allowed"] is True


def test_training_allowed_is_derived_not_declared():
    """A manifest cannot assert rights the record does not support."""
    track = _track("x", rights=_rights(training_rights_status=TrainingRightsStatus.UNVERIFIED))
    assert track.training_allowed is False
    assert track.to_dict()["training_allowed"] is False


def test_quality_grade_is_derived_from_flags():
    assert summarize_quality_grade([], []) == QualityGrade.GOOD
    assert summarize_quality_grade([CLIPPING], []) == QualityGrade.REJECTED
    assert summarize_quality_grade([], [LYRICS_EMPTY]) == QualityGrade.REJECTED


# ── training run reproducibility ──────────────────────────────────────


def _run(**overrides) -> TrainingRunManifest:
    base = dict(
        run_id="LUBER_LORA_PILOT_V1",
        base_model="acestep-v15-turbo",
        ace_step_commit="6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        dataset_id="LUBER_TRAINSET_V1",
        dataset_content_hash="c" * 64,
        lora_rank=8,
        lora_alpha=16,
        learning_rate=1e-4,
        optimizer="AdamW8bit",
        batch_size=1,
        gradient_accumulation_steps=4,
        max_epochs=800,
        save_every_n_epochs=50,
        val_split=0.1,
        seed=42,
        gpu="L40S 48GB",
        cuda_version="12.8",
    )
    base.update(overrides)
    return TrainingRunManifest(**base)  # type: ignore[arg-type]


def test_complete_run_manifest_validates():
    validate_run_manifest(_run())


@pytest.mark.parametrize(
    "field",
    ["run_id", "base_model", "ace_step_commit", "dataset_id", "dataset_content_hash", "gpu"],
)
def test_run_manifest_requires_reproducibility_fields(field):
    with pytest.raises(ValueError, match=field):
        validate_run_manifest(_run(**{field: ""}))


def test_run_manifest_rejects_zero_val_split():
    """Without validation, overtraining cannot be detected (Step 16)."""
    with pytest.raises(ValueError, match="overtraining cannot be detected"):
        validate_run_manifest(_run(val_split=0.0))


def test_run_manifest_rejects_nonsense_hyperparameters():
    with pytest.raises(ValueError, match="rank and alpha"):
        validate_run_manifest(_run(lora_rank=0))
    with pytest.raises(ValueError, match="learning rate"):
        validate_run_manifest(_run(learning_rate=0))


def test_run_manifest_serializes(tmp_path):
    path = _run().write(tmp_path / "run.json")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["dataset_content_hash"] == "c" * 64
    assert data["val_split"] == 0.1


# ── Step 3 vocal vocabulary ───────────────────────────────────────────


def test_step3_descriptors_are_all_expressible():
    """Every descriptor the Phase 7 spec names must have a label."""
    from luber_dataset import Delivery, VibratoAmount, VocalStyle, VocalTimbre

    styles = {s.value for s in VocalStyle}
    deliveries = {d.value for d in Delivery}
    timbres = {t.value for t in VocalTimbre}
    vibratos = {v.value for v in VibratoAmount}

    assert {
        "contemporary_kpop",
        "contemporary_krnb",
        "trot_like".replace("trot_like", "traditional_trot"),
        "ballad_legacy",
    } <= styles
    assert {"breathy", "clean", "intimate", "powerful"} <= deliveries
    assert {"nasal", "airy"} <= timbres
    # restrained_vibrato / strong_vibrato map onto the amount axis.
    assert {"subtle", "heavy"} <= vibratos


def test_both_phase5_failure_styles_are_excluded_by_default():
    from luber_dataset import DISCOURAGED_STYLES, VocalStyle

    assert VocalStyle.TRADITIONAL_TROT in DISCOURAGED_STYLES
    assert VocalStyle.BALLAD_LEGACY in DISCOURAGED_STYLES


def test_ballad_legacy_track_is_excluded_from_a_manifest():
    from luber_dataset import VocalStyle

    manifest = build_manifest("T", [_track("legacy", vocal=_vocal(VocalStyle.BALLAD_LEGACY))])
    assert manifest.track_count == 0
    assert manifest.exclusions[0].reason == "DISCOURAGED_STYLE"


def test_timbre_defaults_to_neutral_and_serializes():
    from luber_dataset import VocalTimbre

    track = _track("t")
    assert track.vocal.timbre == VocalTimbre.NEUTRAL
    assert track.to_dict()["vocal"]["timbre"] == "neutral"
