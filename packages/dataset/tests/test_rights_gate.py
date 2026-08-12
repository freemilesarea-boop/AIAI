"""The rights gate.

The most important tests in the dataset package: if this leaks,
unlicensed audio reaches training.

The model here separates two independent facts — where audio came from
(`origin_type`) and whether we may train on it
(`training_rights_status`). AI origin alone is no longer disqualifying;
unconfirmed rights always are, and self-model output always is.
"""

import pytest

from luber_dataset import (
    OriginType,
    RightsBasis,
    RightsError,
    RightsRecord,
    SourceClass,
    TrainingRightsStatus,
    classify,
    is_trainable,
    validate_rights,
)


def _rights(**overrides) -> RightsRecord:
    base = {
        "origin_type": OriginType.HUMAN_RECORDED,
        "training_rights_status": TrainingRightsStatus.CONFIRMED,
        "basis": RightsBasis.ORIGINAL_WORK,
        "source": "commissioned session, LUBER studio",
        "rights_holder": "LUBER",
        "document_reference": "contract-2026-001",
        "confirmed_on": "2026-08-12",
        "audio_use_confirmed": True,
        "lyrics_rights_confirmed": True,
        "performer_rights_confirmed": True,
        "commercial_training_allowed": True,
        "notes": "",
    }
    base.update(overrides)
    return RightsRecord(**base)


# ── AI origin is no longer disqualifying ──────────────────────────────


def test_ai_generated_with_confirmed_rights_is_trainable():
    """The policy change: owned AI audio is legitimate training data."""
    record = _rights(
        origin_type=OriginType.AI_GENERATED,
        basis=RightsBasis.AI_SERVICE_OUTPUT_OWNED,
        source="operator's own generations, paid account",
    )
    validate_rights(record, has_lyrics=True, has_vocals=True)


def test_hybrid_origin_with_confirmed_rights_is_trainable():
    validate_rights(_rights(origin_type=OriginType.HYBRID), has_lyrics=True, has_vocals=True)


def test_ai_generated_without_confirmed_rights_is_not_trainable():
    record = _rights(
        origin_type=OriginType.AI_GENERATED,
        training_rights_status=TrainingRightsStatus.UNVERIFIED,
    )
    with pytest.raises(RightsError, match="requires CONFIRMED"):
        validate_rights(record, has_lyrics=True, has_vocals=True)


def test_human_recording_without_confirmed_rights_is_not_trainable():
    """Human origin is not a licence either — symmetry matters."""
    record = _rights(training_rights_status=TrainingRightsStatus.UNVERIFIED)
    assert is_trainable(record, has_lyrics=True, has_vocals=True) is False


@pytest.mark.parametrize("status", [TrainingRightsStatus.UNVERIFIED, TrainingRightsStatus.DENIED])
def test_only_confirmed_rights_permit_training(status):
    with pytest.raises(RightsError, match="requires CONFIRMED"):
        validate_rights(_rights(training_rights_status=status), has_lyrics=False, has_vocals=False)


# ── self-model output is absolutely refused ───────────────────────────


def test_self_model_origin_refused_even_with_confirmed_rights():
    """Training ACE-Step on ACE-Step output is never permitted."""
    record = _rights(origin_type=OriginType.SELF_MODEL_OUTPUT)
    with pytest.raises(RightsError, match="own model output"):
        validate_rights(record, has_lyrics=True, has_vocals=True)


@pytest.mark.parametrize(
    "source",
    ["acestep benchmark output", "ace-step render", "luber-generated take", "self-model demo"],
)
def test_self_model_source_markers_refused(source):
    with pytest.raises(RightsError, match="own model output"):
        validate_rights(_rights(source=source), has_lyrics=True, has_vocals=True)


def test_studio_is_not_mistaken_for_a_generator():
    """Word boundaries: "studio" contains "udio"."""
    validate_rights(
        _rights(source="commissioned session, LUBER studio"),
        has_lyrics=True,
        has_vocals=True,
    )


# ── unlawful acquisition is still absolute ────────────────────────────


@pytest.mark.parametrize(
    "source",
    [
        "scraped from a streaming site",
        "web crawler output",
        "crawled catalogue",
        "torrent",
        "yt-dlp rip",
        "leaked promo",
        "pirated copy",
    ],
)
def test_unlawful_acquisition_refused_regardless_of_claim(source):
    record = _rights(basis=RightsBasis.LICENSED_FOR_TRAINING, source=source)
    with pytest.raises(RightsError, match="unlawful acquisition"):
        validate_rights(record, has_lyrics=True, has_vocals=True)


def test_unlawful_markers_checked_in_notes_and_documents():
    with pytest.raises(RightsError, match="unlawful acquisition"):
        validate_rights(
            _rights(notes="originally scraped, later cleared"),
            has_lyrics=True,
            has_vocals=True,
        )


# ── required evidence ─────────────────────────────────────────────────


def test_confirmed_without_a_basis_is_refused():
    with pytest.raises(RightsError, match="no basis is recorded"):
        validate_rights(_rights(basis=RightsBasis.NONE), has_lyrics=False, has_vocals=False)


def test_audio_use_must_be_confirmed():
    with pytest.raises(RightsError, match="audio use"):
        validate_rights(_rights(audio_use_confirmed=False), has_lyrics=False, has_vocals=False)


def test_commercial_training_must_be_confirmed():
    with pytest.raises(RightsError, match="commercial ML training"):
        validate_rights(
            _rights(commercial_training_allowed=False), has_lyrics=False, has_vocals=False
        )


def test_lyrics_rights_required_only_when_there_are_lyrics():
    record = _rights(lyrics_rights_confirmed=False)
    with pytest.raises(RightsError, match="lyrics rights"):
        validate_rights(record, has_lyrics=True, has_vocals=False)
    validate_rights(record, has_lyrics=False, has_vocals=False)


def test_performer_rights_required_only_when_there_are_vocals():
    record = _rights(performer_rights_confirmed=False)
    with pytest.raises(RightsError, match="performer rights"):
        validate_rights(record, has_lyrics=False, has_vocals=True)
    validate_rights(record, has_lyrics=False, has_vocals=False)


@pytest.mark.parametrize("field", ["source", "rights_holder", "document_reference", "confirmed_on"])
def test_provenance_fields_cannot_be_blank(field):
    with pytest.raises(RightsError, match=f"missing {field}"):
        validate_rights(_rights(**{field: "   "}), has_lyrics=False, has_vocals=False)


# ── classification is reporting only ──────────────────────────────────


def test_classification_covers_every_required_class():
    assert (
        classify(
            origin_type=OriginType.AI_GENERATED,
            training_rights_status=TrainingRightsStatus.CONFIRMED,
        )
        is SourceClass.AI_GENERATED_RIGHTS_CLEARED
    )
    assert (
        classify(
            origin_type=OriginType.AI_GENERATED,
            training_rights_status=TrainingRightsStatus.UNVERIFIED,
        )
        is SourceClass.AI_GENERATED_RIGHTS_UNVERIFIED
    )
    assert (
        classify(
            origin_type=OriginType.SELF_MODEL_OUTPUT,
            training_rights_status=TrainingRightsStatus.CONFIRMED,
        )
        is SourceClass.SELF_MODEL_GENERATED
    )
    assert (
        classify(
            origin_type=OriginType.HUMAN_RECORDED,
            training_rights_status=TrainingRightsStatus.CONFIRMED,
        )
        is SourceClass.HUMAN_PRODUCED_RIGHTS_CLEARED
    )
    assert (
        classify(
            origin_type=OriginType.UNKNOWN,
            training_rights_status=TrainingRightsStatus.UNVERIFIED,
            commercial_reference=True,
        )
        is SourceClass.COMMERCIAL_REFERENCE_UNVERIFIED
    )
    assert (
        classify(
            origin_type=OriginType.UNKNOWN,
            training_rights_status=TrainingRightsStatus.UNVERIFIED,
        )
        is SourceClass.UNKNOWN
    )


def test_self_model_classification_wins_over_commercial_flag():
    assert (
        classify(
            origin_type=OriginType.SELF_MODEL_OUTPUT,
            training_rights_status=TrainingRightsStatus.UNVERIFIED,
            commercial_reference=True,
        )
        is SourceClass.SELF_MODEL_GENERATED
    )


def test_classification_confers_nothing():
    """A cleared *class* still goes through validate_rights to train."""
    record = _rights(
        origin_type=OriginType.AI_GENERATED,
        training_rights_status=TrainingRightsStatus.CONFIRMED,
        basis=RightsBasis.NONE,
    )
    assert (
        classify(
            origin_type=record.origin_type,
            training_rights_status=record.training_rights_status,
        )
        is SourceClass.AI_GENERATED_RIGHTS_CLEARED
    )
    assert is_trainable(record, has_lyrics=False, has_vocals=False) is False


# ── no bypass ─────────────────────────────────────────────────────────


def test_there_is_no_override_to_bypass_the_gate():
    import inspect

    from luber_dataset import rights as rights_module

    assert set(inspect.signature(rights_module.validate_rights).parameters) == {
        "record",
        "has_lyrics",
        "has_vocals",
    }
    for name, obj in vars(rights_module).items():
        if name.startswith("_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != rights_module.__name__:
            continue
        try:
            params = set(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            continue
        for escape_hatch in ("force", "skip_rights", "override", "ignore_rights"):
            assert escape_hatch not in params, f"{name} exposes {escape_hatch}"


def test_rights_module_cannot_acquire_audio():
    import inspect

    from luber_dataset import rights as rights_module

    source = inspect.getsource(rights_module)
    for forbidden in ("urllib", "requests", "httpx", "socket", "subprocess"):
        assert forbidden not in source
