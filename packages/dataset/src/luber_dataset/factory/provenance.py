"""The rights gate. The one place where silence must not become consent.

Everything else in the factory can be re-run, re-tuned or argued with.
This cannot: audio trained on without the right to do so cannot be
untrained, and a dataset that quietly promoted a guess to a permission
is not fixable after the fact.

So one rule governs the whole module, and the tests assert it directly:

    **UNKNOWN never becomes TRUE.**

Not through a default, not through a folder name, not through an
embedded tag, not through the absence of an objection. The only thing
that can set ``commercial_training_allowed`` to TRUE is an operator
statement in a sidecar, and even then only when the surrounding rights
fields agree.

The factory is still free to *analyse* unknown-rights audio — measuring
a file is not using it — and it must be, because the operator needs the
inventory in order to decide. What UNKNOWN blocks is entry into a
training export. That default can be overridden, deliberately and
visibly, by ``include_rights_unknown``; nothing overrides it silently.

Path-based hypotheses come from :mod:`luber_dataset.discovery` and stay
what they were built to be: a prompt for a human, recorded as
``INFERRED``, incapable of granting anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_dataset.discovery import hypothesize_origin
from luber_dataset.factory.metadata import MetadataSource, Sidecar
from luber_dataset.rights import SELF_MODEL_MARKERS, UNLAWFUL_ACQUISITION_MARKERS, _matches


class RightsStatus(StrEnum):
    VERIFIED = "VERIFIED"
    USER_OWNED = "USER_OWNED"
    LICENSED = "LICENSED"
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    #: A named operator authorised a named scope for training, and
    #: that authorisation is the whole of the evidence. Weaker than
    #: every status above it: nobody produced a licence, a contract or
    #: an ownership document, and this value must never be read as
    #: though somebody had. It is separate from VERIFIED and USER_OWNED
    #: precisely so the difference stays visible in every export.
    OPERATOR_AUTHORIZED = "OPERATOR_AUTHORIZED"
    UNKNOWN = "UNKNOWN"
    RESTRICTED = "RESTRICTED"


class SourceType(StrEnum):
    USER_ORIGINAL = "USER_ORIGINAL"
    LICENSED_LIBRARY = "LICENSED_LIBRARY"
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    AI_GENERATED = "AI_GENERATED"
    #: This project's own model output. Never trainable, at any rights
    #: status: training ACE-Step on ACE-Step output teaches it its own
    #: artifacts back.
    SELF_MODEL_OUTPUT = "SELF_MODEL_OUTPUT"
    COMMERCIAL_REFERENCE = "COMMERCIAL_REFERENCE"
    UNKNOWN = "UNKNOWN"


class TrainingPermission(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


#: Statuses that can support training, *if* permission is also TRUE.
PERMISSIVE_STATUSES: frozenset[str] = frozenset(
    {
        RightsStatus.VERIFIED.value,
        RightsStatus.USER_OWNED.value,
        RightsStatus.LICENSED.value,
        RightsStatus.PUBLIC_DOMAIN.value,
        # Permissive because an operator's explicit authorisation is a
        # decision, not a guess — which is the line this module polices.
        # It is still the weakest entry here, and the status name is
        # what carries that downstream rather than a footnote.
        RightsStatus.OPERATOR_AUTHORIZED.value,
    }
)


@dataclass
class Provenance:
    source_type: str = SourceType.UNKNOWN.value
    source_reference: str = ""
    rights_status: str = RightsStatus.UNKNOWN.value
    license: str | None = None
    commercial_training_allowed: str = TrainingPermission.UNKNOWN.value
    provenance_notes: str = ""
    #: Where each of the above came from, so an inferred hypothesis can
    #: never be mistaken for an operator statement.
    field_sources: dict[str, str] = field(default_factory=dict)
    #: Reasons this track is barred from training regardless of status.
    hard_blocks: list[str] = field(default_factory=list)

    @property
    def training_permitted(self) -> bool:
        """The gate itself. Every condition must hold affirmatively."""
        if self.hard_blocks:
            return False
        return (
            self.commercial_training_allowed == TrainingPermission.TRUE.value
            and self.rights_status in PERMISSIVE_STATUSES
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "rights_status": self.rights_status,
            "license": self.license,
            "commercial_training_allowed": self.commercial_training_allowed,
            "provenance_notes": self.provenance_notes,
            "field_sources": dict(sorted(self.field_sources.items())),
            "hard_blocks": sorted(self.hard_blocks),
            "training_permitted": self.training_permitted,
        }


def _valid_enum(value: str | None, allowed: type[StrEnum]) -> str | None:
    if value is None:
        return None
    candidate = value.strip().upper().replace("-", "_").replace(" ", "_")
    return candidate if candidate in {member.value for member in allowed} else None


def resolve(
    audio: Path,
    sidecar: Sidecar | None,
    *,
    embedded: dict[str, str] | None = None,
) -> Provenance:
    """Establish provenance for one file.

    Order matters and is the point: hard blocks are found first and
    cannot be argued away, the path hypothesis is recorded but never
    promoted, and only an explicit sidecar can raise the status above
    UNKNOWN.
    """
    provenance = Provenance()
    hypothesis, commercial = hypothesize_origin(audio)

    # ── 1. hard blocks ───────────────────────────────────────────────
    # Checked against the operator's own words as well as the path, and
    # nothing later can clear them.
    haystack = " ".join(
        filter(
            None,
            [
                str(audio).lower(),
                (sidecar.get("source") or "").lower() if sidecar else "",
                (sidecar.get("notes") or "").lower() if sidecar else "",
                (sidecar.get("license") or "").lower() if sidecar else "",
            ],
        )
    )
    if (marker := _matches(haystack, UNLAWFUL_ACQUISITION_MARKERS)) is not None:
        provenance.hard_blocks.append(f"UNLAWFUL_ACQUISITION:{marker}")
    if hypothesis == "SELF_MODEL_OUTPUT" or _matches(haystack, SELF_MODEL_MARKERS) is not None:
        provenance.hard_blocks.append("SELF_MODEL_OUTPUT")
        provenance.source_type = SourceType.SELF_MODEL_OUTPUT.value
        provenance.field_sources["source_type"] = MetadataSource.INFERRED.value

    # ── 2. the path hypothesis: recorded, never promoted ─────────────
    if provenance.source_type == SourceType.UNKNOWN.value:
        if commercial:
            provenance.source_type = SourceType.COMMERCIAL_REFERENCE.value
        elif hypothesis == "AI_GENERATED":
            provenance.source_type = SourceType.AI_GENERATED.value
        provenance.field_sources["source_type"] = MetadataSource.INFERRED.value
    provenance.provenance_notes = f"path hypothesis: {hypothesis}"
    provenance.source_reference = audio.parent.name

    if sidecar is None:
        # No operator statement. Everything stays UNKNOWN — which is a
        # finding, not a failure, and the review queue will surface it.
        provenance.field_sources.setdefault("rights_status", MetadataSource.NONE.value)
        provenance.field_sources["commercial_training_allowed"] = MetadataSource.NONE.value
        return provenance

    # ── 3. operator statements ───────────────────────────────────────
    declared_type = _valid_enum(sidecar.get("source_type"), SourceType)
    if declared_type is not None and not provenance.hard_blocks:
        provenance.source_type = declared_type
        provenance.field_sources["source_type"] = MetadataSource.USER.value

    declared_status = _valid_enum(sidecar.get("rights_status"), RightsStatus)
    if declared_status is not None:
        provenance.rights_status = declared_status
        provenance.field_sources["rights_status"] = MetadataSource.USER.value
    else:
        provenance.field_sources["rights_status"] = MetadataSource.NONE.value

    if (license_text := sidecar.get("license")) is not None:
        provenance.license = license_text
        provenance.field_sources["license"] = MetadataSource.USER.value

    declared_permission = sidecar.get("commercial_training_allowed")
    if declared_permission in {member.value for member in TrainingPermission}:
        provenance.commercial_training_allowed = declared_permission
        provenance.field_sources["commercial_training_allowed"] = MetadataSource.USER.value
    else:
        provenance.field_sources["commercial_training_allowed"] = MetadataSource.NONE.value

    if (source_text := sidecar.get("source")) is not None:
        provenance.source_reference = source_text
        provenance.field_sources["source_reference"] = MetadataSource.USER.value
    if (notes := sidecar.get("notes")) is not None:
        provenance.provenance_notes = f"{provenance.provenance_notes}; operator: {notes}"

    # ── 4. the consistency check ─────────────────────────────────────
    # A sidecar claiming permission while leaving the status unknown is
    # incoherent, and resolving it in favour of permission is exactly the
    # silent promotion this module exists to prevent. Resolve it the
    # other way and say so.
    if (
        provenance.commercial_training_allowed == TrainingPermission.TRUE.value
        and provenance.rights_status not in PERMISSIVE_STATUSES
    ):
        provenance.commercial_training_allowed = TrainingPermission.UNKNOWN.value
        provenance.field_sources["commercial_training_allowed"] = MetadataSource.NONE.value
        provenance.provenance_notes += (
            "; permission claimed but rights_status is "
            f"{provenance.rights_status} — downgraded to UNKNOWN"
        )
    return provenance
