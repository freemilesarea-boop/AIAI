"""Storage abstraction: local adapter, S3 adapter, and key safety."""

import uuid
from pathlib import Path

import pytest

from luber_audio_utils import (
    AudioStorageError,
    LocalAudioStorage,
    S3AudioStorage,
    S3StorageConfig,
    master_storage_key,
    preview_storage_key,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

TRAVERSAL_KEYS = [
    "../../../../etc/passwd",
    "audio/../../../etc/passwd",
    "/etc/passwd",
    "audio/../../secrets.env",
    "..",
    "audio/./../../x.wav",
    "\\etc\\passwd",
    "audio/..\\..\\x.wav",
]


# ── deterministic keys ────────────────────────────────────────────────


def test_storage_keys_are_deterministic_and_scoped_by_generation():
    gid = uuid.uuid4()
    assert master_storage_key(gid) == f"audio/{gid}/master.wav"
    assert preview_storage_key(gid) == f"audio/{gid}/preview.mp3"
    # Stable across calls: a retry targets the same object.
    assert master_storage_key(gid) == master_storage_key(gid)


def test_different_generations_never_collide():
    keys = {master_storage_key(uuid.uuid4()) for _ in range(200)}
    assert len(keys) == 200


def test_master_and_preview_keys_differ_for_same_generation():
    gid = uuid.uuid4()
    assert master_storage_key(gid) != preview_storage_key(gid)


# ── local adapter ─────────────────────────────────────────────────────


async def test_local_put_get_exists_delete_roundtrip(tmp_path):
    storage = LocalAudioStorage(tmp_path / "store")
    gid = uuid.uuid4()
    key = master_storage_key(gid)

    assert await storage.exists(key) is False
    assert await storage.put(key, FIXTURE) == key
    assert await storage.exists(key) is True
    assert await storage.open(key) == FIXTURE.read_bytes()

    await storage.delete(key)
    assert await storage.exists(key) is False
    # Deleting twice is a no-op, not an error.
    await storage.delete(key)


async def test_local_put_is_idempotent_and_overwrites_in_place(tmp_path):
    """A retry must overwrite its own object, not create a second one."""
    storage = LocalAudioStorage(tmp_path / "store")
    gid = uuid.uuid4()
    key = master_storage_key(gid)

    await storage.put(key, FIXTURE)
    other = tmp_path / "other.wav"
    other.write_bytes(b"RIFF" + b"\x00" * 100)
    await storage.put(key, other)

    generation_dir = (tmp_path / "store" / "audio" / str(gid)).resolve()
    assert [p.name for p in generation_dir.iterdir()] == ["master.wav"]
    assert await storage.open(key) == other.read_bytes()


async def test_local_put_missing_source_raises(tmp_path):
    storage = LocalAudioStorage(tmp_path / "store")
    with pytest.raises(AudioStorageError, match="missing"):
        await storage.put(master_storage_key(uuid.uuid4()), tmp_path / "ghost.wav")


async def test_local_delete_generation_audio_removes_all_assets(tmp_path):
    storage = LocalAudioStorage(tmp_path / "store")
    gid = uuid.uuid4()
    await storage.put(master_storage_key(gid), FIXTURE)
    await storage.put(preview_storage_key(gid), FIXTURE)

    await storage.delete_generation_audio(gid)

    assert await storage.exists(master_storage_key(gid)) is False
    assert await storage.exists(preview_storage_key(gid)) is False
    await storage.delete_generation_audio(gid)  # idempotent


async def test_local_download_target_is_backend_mediated(tmp_path):
    """No signing locally: the API must stream the bytes itself."""
    storage = LocalAudioStorage(tmp_path / "store")
    gid = uuid.uuid4()
    key = master_storage_key(gid)
    await storage.put(key, FIXTURE)

    target = await storage.download_target(
        key, filename="x.wav", content_type="audio/wav", expires_in_seconds=300
    )
    assert target.is_signed_url is False
    assert target.url is None
    assert storage.local_path(key) is not None


# ── path traversal ────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_key", TRAVERSAL_KEYS)
def test_local_resolve_path_rejects_traversal(tmp_path, bad_key):
    storage = LocalAudioStorage(tmp_path / "store")
    with pytest.raises(AudioStorageError, match="escapes storage root"):
        storage.resolve_path(bad_key)


@pytest.mark.parametrize("bad_key", TRAVERSAL_KEYS)
async def test_local_operations_reject_traversal(tmp_path, bad_key):
    storage = LocalAudioStorage(tmp_path / "store")
    for operation in (storage.open, storage.exists, storage.delete):
        with pytest.raises(AudioStorageError):
            await operation(bad_key)
    with pytest.raises(AudioStorageError):
        await storage.put(bad_key, FIXTURE)


def test_local_rejects_empty_and_null_byte_keys(tmp_path):
    storage = LocalAudioStorage(tmp_path / "store")
    for bad in ("", "  ", "audio/\x00/master.wav"):
        with pytest.raises(AudioStorageError):
            storage.resolve_path(bad)


async def test_traversal_cannot_read_a_file_outside_the_root(tmp_path):
    """The decisive check: a secret next to the root stays unreachable."""
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    storage = LocalAudioStorage(tmp_path / "store")

    with pytest.raises(AudioStorageError):
        await storage.open("../secret.txt")


def test_local_accepts_normal_key(tmp_path):
    storage = LocalAudioStorage(tmp_path / "store")
    resolved = storage.resolve_path(master_storage_key(uuid.uuid4()))
    assert str(resolved).startswith(str((tmp_path / "store").resolve()))


# ── S3-compatible adapter (fake client; no network, no credentials) ───


class FakeS3Client:
    """Minimal in-memory stand-in for the boto3 S3 client surface."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.signed: list[dict] = []

    def upload_file(self, filename, bucket, key):
        self.objects[key] = Path(filename).read_bytes()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(self.objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                matching = [{"Key": k} for k in client.objects if k.startswith(Prefix)]
                yield {"Contents": matching}

        return _Paginator()

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.signed.append({"params": Params, "expires_in": ExpiresIn})
        return f"https://example-storage.invalid/{Params['Key']}?X-Expires={ExpiresIn}"


@pytest.fixture
def s3_storage():
    client = FakeS3Client()
    config = S3StorageConfig(bucket="luber-test", region="auto", endpoint_url="https://s3.invalid")
    return S3AudioStorage(config, client=client), client


async def test_s3_put_get_exists_delete_roundtrip(s3_storage):
    storage, client = s3_storage
    key = master_storage_key(uuid.uuid4())

    assert await storage.exists(key) is False
    await storage.put(key, FIXTURE)
    assert await storage.exists(key) is True
    assert await storage.open(key) == FIXTURE.read_bytes()
    assert client.objects[key] == FIXTURE.read_bytes()

    await storage.delete(key)
    assert await storage.exists(key) is False


async def test_s3_signed_url_has_expiry_and_pinned_content_type(s3_storage):
    storage, client = s3_storage
    key = master_storage_key(uuid.uuid4())
    await storage.put(key, FIXTURE)

    target = await storage.download_target(
        key, filename="song.wav", content_type="audio/wav", expires_in_seconds=300
    )

    assert target.is_signed_url is True
    assert target.expires_in_seconds == 300
    assert target.url and target.url.startswith("https://")
    signed = client.signed[-1]
    assert signed["expires_in"] == 300
    # Content type and filename are part of the signature, so a client
    # cannot reinterpret the object as another media type.
    assert signed["params"]["ResponseContentType"] == "audio/wav"
    assert 'filename="song.wav"' in signed["params"]["ResponseContentDisposition"]


async def test_s3_rejects_non_positive_expiry(s3_storage):
    storage, _ = s3_storage
    with pytest.raises(AudioStorageError, match="expiry must be positive"):
        await storage.download_target(
            master_storage_key(uuid.uuid4()),
            filename="a.wav",
            content_type="audio/wav",
            expires_in_seconds=0,
        )


@pytest.mark.parametrize("bad_key", TRAVERSAL_KEYS)
async def test_s3_rejects_traversal_keys(s3_storage, bad_key):
    storage, _ = s3_storage
    with pytest.raises(AudioStorageError):
        await storage.put(bad_key, FIXTURE)


async def test_s3_prefix_scopes_keys_without_escaping(s3_storage):
    client = FakeS3Client()
    storage = S3AudioStorage(S3StorageConfig(bucket="b", prefix="staging"), client=client)
    gid = uuid.uuid4()
    await storage.put(master_storage_key(gid), FIXTURE)
    assert f"staging/audio/{gid}/master.wav" in client.objects


async def test_s3_delete_generation_audio_removes_every_asset(s3_storage):
    storage, client = s3_storage
    gid = uuid.uuid4()
    await storage.put(master_storage_key(gid), FIXTURE)
    await storage.put(preview_storage_key(gid), FIXTURE)
    other = uuid.uuid4()
    await storage.put(master_storage_key(other), FIXTURE)

    await storage.delete_generation_audio(gid)

    assert not [k for k in client.objects if str(gid) in k]
    # Another generation's objects are untouched.
    assert [k for k in client.objects if str(other) in k]


def test_s3_has_no_local_path(s3_storage):
    storage, _ = s3_storage
    assert storage.local_path(master_storage_key(uuid.uuid4())) is None


def test_s3_config_carries_no_hardcoded_credentials():
    config = S3StorageConfig(bucket="b")
    assert config.access_key_id is None
    assert config.secret_access_key is None
    assert config.endpoint_url is None
