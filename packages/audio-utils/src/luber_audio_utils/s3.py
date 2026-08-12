"""S3-compatible object storage adapter.

Targets the S3 API rather than any one vendor: AWS S3, Cloudflare R2,
MinIO, Backblaze B2, and Supabase Storage all work through the same
client by varying ``endpoint_url``, ``region``, and path-style
addressing.

Credentials are never read from source. They arrive through
:class:`S3StorageConfig`, which services populate from environment
variables, and they are never logged or returned to clients.

Downloads use short-lived presigned URLs so audio bytes do not transit
the API process in production. The URL carries the response content
type and download filename, so a client cannot coerce either.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from luber_audio_utils.storage import (
    AudioStorage,
    AudioStorageError,
    DownloadTarget,
    _validate_key,
    generation_prefix,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


@dataclass(frozen=True)
class S3StorageConfig:
    """Connection settings for an S3-compatible endpoint.

    ``endpoint_url`` is optional so plain AWS S3 works unset; any other
    provider supplies its own endpoint. ``force_path_style`` is required
    by MinIO and some self-hosted gateways.
    """

    bucket: str
    region: str | None = None
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    force_path_style: bool = False
    #: Optional key prefix, e.g. to share one bucket across environments.
    prefix: str = ""

    def scoped_key(self, storage_key: str) -> str:
        _validate_key(storage_key)
        if not self.prefix:
            return storage_key
        return f"{self.prefix.strip('/')}/{storage_key}"


class S3AudioStorage(AudioStorage):
    """:class:`AudioStorage` backed by any S3-compatible object store."""

    def __init__(self, config: S3StorageConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client or self._build_client(config)

    @staticmethod
    def _build_client(config: S3StorageConfig) -> Any:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise AudioStorageError(
                "boto3 is required for S3-compatible storage; install the 's3' extra"
            ) from exc

        boto_config = BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path" if config.force_path_style else "auto"},
        )
        return boto3.client(
            "s3",
            region_name=config.region,
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=boto_config,
        )

    def local_path(self, storage_key: str) -> Path | None:
        # Objects live remotely; there is no local file to stream.
        return None

    async def put(self, storage_key: str, source: Path) -> str:
        key = self._config.scoped_key(storage_key)
        if not await asyncio.to_thread(source.is_file):
            raise AudioStorageError(f"source audio missing: {source}")

        def _upload() -> None:
            self._client.upload_file(str(source), self._config.bucket, key)

        try:
            await asyncio.to_thread(_upload)
        except Exception as exc:  # boto3 raises many client-error subclasses
            raise AudioStorageError(f"failed to upload {storage_key}") from exc
        return storage_key

    async def open(self, storage_key: str) -> bytes:
        key = self._config.scoped_key(storage_key)

        def _get() -> bytes:
            response = self._client.get_object(Bucket=self._config.bucket, Key=key)
            body: bytes = response["Body"].read()
            return body

        try:
            return await asyncio.to_thread(_get)
        except Exception as exc:
            raise AudioStorageError(f"failed to read {storage_key}") from exc

    async def exists(self, storage_key: str) -> bool:
        key = self._config.scoped_key(storage_key)

        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self._config.bucket, Key=key)
            except Exception:
                return False
            return True

        return await asyncio.to_thread(_head)

    async def delete(self, storage_key: str) -> None:
        key = self._config.scoped_key(storage_key)

        def _delete() -> None:
            self._client.delete_object(Bucket=self._config.bucket, Key=key)

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            raise AudioStorageError(f"failed to delete {storage_key}") from exc

    async def delete_generation_audio(self, generation_id: UUID) -> None:
        prefix = self._config.scoped_key(f"{generation_prefix(generation_id)}placeholder").rsplit(
            "placeholder", 1
        )[0]

        def _delete_prefix() -> None:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
                contents = page.get("Contents") or []
                if not contents:
                    continue
                self._client.delete_objects(
                    Bucket=self._config.bucket,
                    Delete={"Objects": [{"Key": item["Key"]} for item in contents]},
                )

        try:
            await asyncio.to_thread(_delete_prefix)
        except Exception as exc:
            raise AudioStorageError(f"failed to delete audio for {generation_id}") from exc

    async def download_target(
        self, storage_key: str, *, filename: str, content_type: str, expires_in_seconds: int
    ) -> DownloadTarget:
        """Mint a short-lived presigned GET URL.

        The response content type and disposition are pinned into the
        signature, so the delivered object cannot be reinterpreted as a
        different media type or filename.
        """
        if expires_in_seconds <= 0:
            raise AudioStorageError("signed URL expiry must be positive")
        key = self._config.scoped_key(storage_key)
        # Quote-escape defensively; filenames are already ASCII-slugged.
        safe_filename = filename.replace('"', "")

        def _sign() -> str:
            url: str = self._client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self._config.bucket,
                    "Key": key,
                    "ResponseContentType": content_type,
                    "ResponseContentDisposition": f'attachment; filename="{safe_filename}"',
                },
                ExpiresIn=expires_in_seconds,
            )
            return url

        try:
            url = await asyncio.to_thread(_sign)
        except Exception as exc:
            raise AudioStorageError(f"failed to sign download for {storage_key}") from exc
        return DownloadTarget(url=url, expires_in_seconds=expires_in_seconds)
