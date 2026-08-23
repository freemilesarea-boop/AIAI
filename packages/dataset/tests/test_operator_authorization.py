"""Operator-authorised scope: what it grants, and what it must not.

Phase 35B let an operator authorise a whole directory for training
instead of annotating every track. That is a real weakening of the
evidence, so these tests exist to hold the line in two directions at
once: the authorisation has to actually be recorded before it grants
anything, and it must never quietly grow into the stronger claims —
ownership, licence, publisher clearance, performer agreement — that
nobody produced.
"""

import pytest

from luber_dataset import (
    OriginType,
    RightsBasis,
    RightsError,
    RightsRecord,
    TrainingRightsStatus,
    is_trainable,
    validate_rights,
)
from luber_dataset.factory.provenance import PERMISSIVE_STATUSES, RightsStatus


def _operator_record(**overrides) -> RightsRecord:
    base = {
        "origin_type": OriginType.UNKNOWN,
        "training_rights_status": TrainingRightsStatus.CONFIRMED,
        "basis": RightsBasis.OPERATOR_AUTHORIZED_SCOPE,
        "source": "operator-authorised directory, group 'POP'",
        "rights_holder": "the operator",
        "document_reference": "operator authorisation of ~/Desktop/LUBER_TRAINING_DATA/**",
        "confirmed_on": "2026-08-23",
        "audio_use_confirmed": True,
        "lyrics_rights_confirmed": False,
        "performer_rights_confirmed": False,
        "commercial_training_allowed": True,
        "notes": "no contract, licence or performer agreement was produced",
        "authorization_source": "OPERATOR_EXPLICIT_AUTHORIZATION",
        "authorization_scope": "~/Desktop/LUBER_TRAINING_DATA/**",
        "authorization_recorded_at": "2026-08-23",
    }
    base.update(overrides)
    return RightsRecord(**base)


class TestTheAuthorizationMustBeRecorded:
    def test_it_admits_a_fully_recorded_operator_authorization(self):
        validate_rights(_operator_record(), has_lyrics=False, has_vocals=True)

    @pytest.mark.parametrize(
        "missing",
        ["authorization_source", "authorization_scope", "authorization_recorded_at"],
    )
    def test_it_refuses_an_authorization_missing_who_what_or_when(self, missing):
        record = _operator_record(**{missing: ""})
        with pytest.raises(RightsError, match="OPERATOR_AUTHORIZED_SCOPE"):
            validate_rights(record, has_lyrics=False, has_vocals=False)

    def test_a_directory_name_alone_still_grants_nothing(self):
        """The rule the whole basis exists under.

        A record that names the basis but carries none of the
        authorisation fields is a folder name wearing a rights claim.
        """
        bare = _operator_record(
            authorization_source="",
            authorization_scope="",
            authorization_recorded_at="",
        )
        assert not is_trainable(bare, has_lyrics=False, has_vocals=False)


class TestItDoesNotBecomeAStrongerClaim:
    def test_a_vocal_track_needs_no_fabricated_performer_agreement(self):
        """Vocals are admitted without a performer-rights claim.

        The operator authorised the works as supplied. Requiring the
        flag here would only teach the ingestion path to set it, which
        would put a performer agreement nobody signed into the record.
        """
        record = _operator_record(performer_rights_confirmed=False)
        validate_rights(record, has_lyrics=False, has_vocals=True)
        assert record.performer_rights_confirmed is False

    def test_lyrics_are_admitted_without_a_publisher_clearance_claim(self):
        record = _operator_record(lyrics_rights_confirmed=False)
        validate_rights(record, has_lyrics=True, has_vocals=True)
        assert record.lyrics_rights_confirmed is False

    def test_every_other_basis_still_demands_those_confirmations(self):
        """The relaxation is scoped to this basis and nothing else."""
        record = _operator_record(
            basis=RightsBasis.ORIGINAL_WORK,
            performer_rights_confirmed=False,
        )
        with pytest.raises(RightsError, match="performer rights"):
            validate_rights(record, has_lyrics=False, has_vocals=True)

    def test_the_status_is_its_own_value_not_verified_or_user_owned(self):
        """A reader must be able to see which evidence they are holding."""
        assert RightsStatus.OPERATOR_AUTHORIZED.value == "OPERATOR_AUTHORIZED"
        assert RightsStatus.OPERATOR_AUTHORIZED.value in PERMISSIVE_STATUSES
        assert RightsStatus.OPERATOR_AUTHORIZED is not RightsStatus.VERIFIED
        assert RightsStatus.OPERATOR_AUTHORIZED is not RightsStatus.USER_OWNED


class TestTheOtherRefusalsStillApply:
    def test_self_model_output_is_refused_however_it_was_authorised(self):
        record = _operator_record(origin_type=OriginType.SELF_MODEL_OUTPUT)
        with pytest.raises(RightsError, match="model output"):
            validate_rights(record, has_lyrics=False, has_vocals=False)

    def test_unlawful_acquisition_is_refused_however_it_was_authorised(self):
        record = _operator_record(source="ripped from a streaming service")
        with pytest.raises(RightsError, match="unlawful"):
            validate_rights(record, has_lyrics=False, has_vocals=False)

    def test_unconfirmed_rights_are_still_unconfirmed(self):
        record = _operator_record(training_rights_status=TrainingRightsStatus.UNVERIFIED)
        with pytest.raises(RightsError, match="CONFIRMED"):
            validate_rights(record, has_lyrics=False, has_vocals=False)
