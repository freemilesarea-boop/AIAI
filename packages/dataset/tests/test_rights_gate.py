"""The rights gate.

These are the tests that matter most in the whole dataset package: if
the gate leaks, unlicensed audio reaches training. Every path that
could admit an unconfirmed track is exercised.
"""

import pytest

from luber_dataset import (
    RightsError,
    RightsRecord,
    RightsStatus,
    is_trainable,
    validate_rights,
)


def _rights(**overrides) -> RightsRecord:
    base = {
        "status": RightsStatus.ORIGINAL_WORK,
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


# ── the gate admits properly cleared work ─────────────────────────────


@pytest.mark.parametrize(
    "status",
    [
        RightsStatus.ORIGINAL_WORK,
        RightsStatus.LICENSED_FOR_TRAINING,
        RightsStatus.PUBLIC_DOMAIN,
        RightsStatus.CC_TRAINING_PERMITTED,
        RightsStatus.RIGHTS_HOLDER_PERMISSION,
    ],
)
def test_acceptable_statuses_pass(status):
    validate_rights(_rights(status=status), has_lyrics=True, has_vocals=True)


# ── and refuses everything else ───────────────────────────────────────


@pytest.mark.parametrize("status", [RightsStatus.UNVERIFIED, RightsStatus.PROHIBITED])
def test_unverified_and_prohibited_are_excluded(status):
    with pytest.raises(RightsError, match="not acceptable"):
        validate_rights(_rights(status=status), has_lyrics=True, has_vocals=True)


def test_unverified_is_the_default_failure_mode():
    """A track nobody cleared must not be trainable."""
    assert (
        is_trainable(_rights(status=RightsStatus.UNVERIFIED), has_lyrics=True, has_vocals=True)
        is False
    )


@pytest.mark.parametrize(
    "source",
    [
        "downloaded from Suno",
        "udio export",
        "scraped from a streaming site",
        "web crawler output",
        "crawled catalogue",
        "torrent",
        "yt-dlp rip",
        "spotify-rip collection",
        "leaked promo",
    ],
)
def test_prohibited_provenance_is_refused_regardless_of_claimed_status(source):
    """Even a claim of full clearance cannot launder scraped audio."""
    record = _rights(status=RightsStatus.LICENSED_FOR_TRAINING, source=source)
    with pytest.raises(RightsError, match="prohibited provenance"):
        validate_rights(record, has_lyrics=True, has_vocals=True)


def test_prohibited_markers_are_checked_in_notes_and_documents_too():
    with pytest.raises(RightsError, match="prohibited provenance"):
        validate_rights(
            _rights(notes="originally scraped, later cleared"),
            has_lyrics=True,
            has_vocals=True,
        )
    with pytest.raises(RightsError, match="prohibited provenance"):
        validate_rights(
            _rights(document_reference="suno-export-2026.csv"),
            has_lyrics=True,
            has_vocals=True,
        )


# ── every required sub-right ──────────────────────────────────────────


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
    # An instrumental without lyrics is unaffected.
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


def test_there_is_no_override_to_bypass_the_gate():
    """The gate must not expose a force/skip parameter."""
    import inspect

    from luber_dataset import rights as rights_module

    signature = inspect.signature(rights_module.validate_rights)
    assert set(signature.parameters) == {"record", "has_lyrics", "has_vocals"}

    # No public callable in the module may accept a bypass parameter.
    for name, obj in vars(rights_module).items():
        if name.startswith("_") or not callable(obj) or not hasattr(obj, "__module__"):
            continue
        if obj.__module__ != rights_module.__name__:
            continue
        try:
            params = set(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            continue
        for escape_hatch in (
            "force",
            "skip_rights",
            "override",
            "ignore_rights",
            "allow_unverified",
        ):
            assert escape_hatch not in params, f"{name} exposes {escape_hatch}"


def test_rights_module_cannot_acquire_audio():
    """Data is supplied by the operator; nothing here fetches it."""
    import inspect

    from luber_dataset import rights as rights_module

    source = inspect.getsource(rights_module)
    for forbidden in ("urllib", "requests", "httpx", "socket", "subprocess"):
        assert forbidden not in source
