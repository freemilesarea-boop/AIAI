"""Storage backend selection from configuration."""

from dataclasses import dataclass

import pytest

from luber_audio_utils import (
    AudioStorageError,
    LocalAudioStorage,
    S3AudioStorage,
    storage_from_settings,
)


@dataclass
class _Settings:
    storage_provider: str = "local"
    audio_storage_dir: str = "data"
    storage_bucket: str = ""
    storage_region: str | None = None
    storage_endpoint: str | None = None
    storage_access_key_id: str | None = None
    storage_secret_access_key: str | None = None
    storage_force_path_style: bool = False
    storage_key_prefix: str = ""


def test_defaults_to_the_local_adapter():
    assert isinstance(storage_from_settings(_Settings()), LocalAudioStorage)


def test_selects_the_s3_adapter(monkeypatch):
    built = {}

    class _FakeClient:
        pass

    monkeypatch.setattr(
        S3AudioStorage, "_build_client", staticmethod(lambda config: built.setdefault("c", config))
    )
    storage = storage_from_settings(
        _Settings(
            storage_provider="s3",
            storage_bucket="luber-prod",
            storage_region="auto",
            storage_endpoint="https://account.r2.cloudflarestorage.com",
            storage_access_key_id="from-environment",
            storage_secret_access_key="from-environment",
            storage_force_path_style=True,
            storage_key_prefix="prod",
        )
    )
    assert isinstance(storage, S3AudioStorage)
    # Vendor-neutral: endpoint and path style are configuration, not code.
    assert built["c"].bucket == "luber-prod"
    assert built["c"].endpoint_url == "https://account.r2.cloudflarestorage.com"
    assert built["c"].force_path_style is True
    assert built["c"].prefix == "prod"


def test_s3_without_a_bucket_is_a_configuration_error():
    with pytest.raises(AudioStorageError, match="STORAGE_BUCKET"):
        storage_from_settings(_Settings(storage_provider="s3"))


def test_unknown_provider_is_rejected():
    with pytest.raises(AudioStorageError, match="unknown storage provider"):
        storage_from_settings(_Settings(storage_provider="dropbox"))


def test_provider_name_is_case_and_space_insensitive():
    assert isinstance(
        storage_from_settings(_Settings(storage_provider=" Local ")), LocalAudioStorage
    )


def test_error_never_echoes_credentials():
    settings = _Settings(
        storage_provider="s3",
        storage_access_key_id="AKIAEXAMPLESECRET",
        storage_secret_access_key="super-secret-value",
    )
    with pytest.raises(AudioStorageError) as excinfo:
        storage_from_settings(settings)
    assert "AKIAEXAMPLESECRET" not in str(excinfo.value)
    assert "super-secret-value" not in str(excinfo.value)
