"""Typed schema for training-dataset manifests.

The whole point of this module is that a file cannot become training
data by accident. Three separate gates have to be passed deliberately:

* **Rights.** Every item carries an explicit :class:`DataRights` value.
  ``UNKNOWN`` is the default, and ``UNKNOWN`` is not trainable. A file
  does not become licensed because it sits in a folder called
  ``licensed``, and provenance is never inferred from a path.
* **Tier.** Quality is a separate axis from rights. Material can be
  perfectly licensed and still be too poor to train on.
* **Split.** Evaluation material is marked as such and is refused by the
  training manifest, because a benchmark a model trained on measures
  nothing.

Paths are dataset-root-relative by construction. An absolute path in a
committed manifest is both a portability bug and a disclosure of
somebody's home directory, so the schema rejects one.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DataRights(StrEnum):
    """Why this audio is, or is not, allowed to train a model.

    Ordered from most to least permissive. Only the members in
    :data:`TRAINABLE_RIGHTS` may enter a training manifest; everything
    else is either evaluation material or quarantined.
    """

    #: Produced by this project or its owner.
    OWNED = "OWNED"
    #: A licence that explicitly permits model training. The permission
    #: must be recorded in ``rights_note`` — "we assumed it was fine" is
    #: not a licence.
    LICENSED_FOR_TRAINING = "LICENSED_FOR_TRAINING"
    #: Public domain in the relevant jurisdiction.
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    #: Machine-generated audio whose terms allow training. Trainable, but
    #: tracked separately: see ``is_synthetic``.
    AI_GENERATED_ALLOWED = "AI_GENERATED_ALLOWED"
    #: Usable for measurement and comparison only. Never training.
    REFERENCE_ONLY = "REFERENCE_ONLY"
    #: Provenance not established. The default, and deliberately so.
    UNKNOWN = "UNKNOWN"
    #: Positively excluded — a rights holder said no, or the project did.
    DO_NOT_TRAIN = "DO_NOT_TRAIN"


#: The only values that may appear in a training manifest.
TRAINABLE_RIGHTS: frozenset[DataRights] = frozenset(
    {
        DataRights.OWNED,
        DataRights.LICENSED_FOR_TRAINING,
        DataRights.PUBLIC_DOMAIN,
        DataRights.AI_GENERATED_ALLOWED,
    }
)


class QualityTier(StrEnum):
    """How good the item is, independently of whether it is allowed."""

    #: Manually approved: clean audio, coherent arrangement, good mix,
    #: accurate lyrics and metadata, strong vocal where present.
    GOLD = "GOLD"
    #: Usable, with weaker or partial annotation.
    SILVER = "SILVER"
    #: Not for training. Low quality, corrupt, duplicate, artefact-heavy,
    #: or held back for evaluation.
    REJECT = "REJECT"


class DataSplit(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    #: Benchmark material. Never trained on, never validated on — held
    #: apart so a score against it stays meaningful.
    EVALUATION_ONLY = "EVALUATION_ONLY"


#: Splits a training run may consume.
TRAINABLE_SPLITS: frozenset[DataSplit] = frozenset({DataSplit.TRAIN})


class SourceType(StrEnum):
    LUBER_GENERATED = "LUBER_GENERATED"
    USER_PROVIDED = "USER_PROVIDED"
    COMMISSIONED = "COMMISSIONED"
    PUBLIC_DATASET = "PUBLIC_DATASET"
    OTHER = "OTHER"


class DatasetItem(BaseModel):
    """One audio file, with everything needed to decide its fate."""

    model_config = {"extra": "forbid"}

    item_id: str = Field(min_length=1, max_length=128)
    #: Relative to the dataset root. Never absolute, never containing
    #: ``..`` — a manifest is committed and must not carry a machine path.
    audio_path: str = Field(min_length=1)
    sha256: str
    #: Hash of the decoded PCM. Catches the same recording re-encoded to
    #: a different container, which sha256 alone cannot.
    pcm_sha256: str | None = None

    source_type: SourceType
    source_identifier: str = Field(default="", max_length=512)
    rights: DataRights = DataRights.UNKNOWN
    rights_note: str = Field(default="", max_length=1024)
    ingested_at: str

    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1, le=2)

    language: str | None = None
    instrumental: bool = False
    lyrics: str | None = None
    lyrics_source: str | None = None
    genre_tags: list[str] = Field(default_factory=list)
    bpm: float | None = None

    quality_tier: QualityTier = QualityTier.REJECT
    split: DataSplit = DataSplit.EVALUATION_ONLY
    notes: str = Field(default="", max_length=2048)

    @field_validator("sha256", "pcm_sha256")
    @classmethod
    def _hex_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.match(value):
            raise ValueError("expected a lowercase hex sha256 digest")
        return value

    @field_validator("audio_path")
    @classmethod
    def _relative_and_contained(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or value.startswith("~"):
            raise ValueError("audio_path must be relative to the dataset root")
        if ".." in path.parts:
            raise ValueError("audio_path must not escape the dataset root")
        return value

    @model_validator(mode="after")
    def _lyrics_match_the_vocal_claim(self) -> DatasetItem:
        if self.instrumental and self.lyrics:
            raise ValueError("an instrumental item cannot carry lyrics")
        return self

    @model_validator(mode="after")
    def _training_needs_provenance(self) -> DatasetItem:
        """The gate that makes the rest of the schema worth having."""
        if self.split in TRAINABLE_SPLITS:
            if self.rights not in TRAINABLE_RIGHTS:
                raise ValueError(
                    f"rights={self.rights.value} may not be used for training; "
                    "an item reaches TRAIN only with established provenance"
                )
            if self.quality_tier is QualityTier.REJECT:
                raise ValueError("a REJECT item may not be placed in TRAIN")
        return self

    @property
    def is_synthetic(self) -> bool:
        """Machine-generated, and therefore measured separately.

        Training a generative model on its own output narrows it. The
        flag exists so the proportion is always a known number rather
        than something discovered afterwards.
        """
        return (
            self.source_type is SourceType.LUBER_GENERATED
            or self.rights is DataRights.AI_GENERATED_ALLOWED
        )

    @property
    def is_trainable(self) -> bool:
        return (
            self.rights in TRAINABLE_RIGHTS
            and self.quality_tier is not QualityTier.REJECT
            and self.split in TRAINABLE_SPLITS
        )


class DatasetManifest(BaseModel):
    """A set of items, with the invariants a training run relies on."""

    model_config = {"extra": "forbid"}

    manifest_version: str = "p20"
    dataset_name: str = Field(min_length=1)
    created_at: str
    #: Recorded for provenance, never written into the manifest's items:
    #: the root is machine-specific and the paths must stay relative.
    dataset_root_note: str = Field(default="", max_length=512)
    items: list[DatasetItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identifiers_and_content_are_unique(self) -> DatasetManifest:
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("item_id values must be unique within a manifest")
        paths = [item.audio_path for item in self.items]
        if len(paths) != len(set(paths)):
            raise ValueError("audio_path values must be unique within a manifest")
        return self

    @model_validator(mode="after")
    def _no_content_crosses_a_split(self) -> DatasetManifest:
        """Leakage check, by content rather than by filename.

        The same recording under two names in TRAIN and TEST is the
        classic way a benchmark quietly stops measuring anything.
        """
        by_split: dict[str, set[str]] = {}
        for item in self.items:
            for digest in (item.sha256, item.pcm_sha256):
                if digest:
                    by_split.setdefault(digest, set()).add(item.split.value)
        crossing = sorted(d for d, splits in by_split.items() if len(splits) > 1)
        if crossing:
            raise ValueError(f"the same audio appears in more than one split: {crossing[:3]}")
        return self

    def trainable_items(self) -> list[DatasetItem]:
        """Exactly what a training run is allowed to read."""
        return [item for item in self.items if item.is_trainable]

    def quarantined_items(self) -> list[DatasetItem]:
        """Everything held back, for the reason it was held back."""
        return [item for item in self.items if not item.is_trainable]

    def synthetic_fraction(self) -> float:
        """Share of trainable audio that the machine produced itself."""
        trainable = self.trainable_items()
        if not trainable:
            return 0.0
        return sum(1 for item in trainable if item.is_synthetic) / len(trainable)
