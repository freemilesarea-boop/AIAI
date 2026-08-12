"""API request/response DTOs for the generation endpoints.

These are the public wire contract — separate from ORM models and from
the provider-level ``GenerationRequest`` so each layer can evolve
independently.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from luber_schemas import GenerationStatus, VocalGender


class GenerationCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=4000)
    lyrics: str = Field(default="", max_length=20000)
    vocal_gender: VocalGender
    duration: int = Field(default=180, ge=10, le=360)
    seed: int | None = None
    language: str | None = Field(default=None, max_length=16)
    instrumental: bool = False

    @model_validator(mode="after")
    def _sync_instrumental(self) -> GenerationCreateRequest:
        # vocal_gender=instrumental implies the instrumental flag.
        if self.vocal_gender is VocalGender.INSTRUMENTAL:
            self.instrumental = True
        return self


class GenerationCreateResponse(BaseModel):
    generation_id: uuid.UUID
    status: GenerationStatus


class AudioAssetResponse(BaseModel):
    id: uuid.UUID
    asset_type: str
    format: str
    mime_type: str
    file_extension: str
    sample_rate: int
    bit_depth: int | None
    bitrate: int | None
    channels: int
    duration: float
    #: Relative, UUID-scoped object key — never a filesystem path, bucket
    #: name, or URL. Clients address audio by generation id, not by key.
    storage_key: str
    sha256: str
    file_size: int
    created_at: datetime

    model_config = {"from_attributes": True}


class GenerationResponse(BaseModel):
    id: uuid.UUID
    title: str
    prompt: str
    lyrics: str
    vocal_gender: str
    duration_requested: int
    duration_actual: float | None
    seed: int | None
    language: str | None
    instrumental: bool
    status: str
    provider: str | None
    model_name: str | None
    model_version: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    error_message: str | None
    audio_assets: list[AudioAssetResponse]

    model_config = {"from_attributes": True}


class GenerationListResponse(BaseModel):
    items: list[GenerationResponse]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str
