"""Reference-audio ingestion and its refusals.

This is the only route that takes a file from a browser, so most of what
matters is what it *rejects*. Each test below corresponds to a way a
hostile or broken upload could otherwise become a stored object, a
filesystem path, or somebody's downloadable master.
"""

from __future__ import annotations

import io
import shutil
import struct
import uuid
import wave

import pytest

from luber_schemas import MAX_REFERENCE_FILE_BYTES

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required to decode uploads",
)

UPLOAD_URL = "/v1/reference-audio"


def wav_bytes(seconds: float = 3.0, rate: int = 44_100, channels: int = 1) -> bytes:
    """A real, decodable WAV — deliberately not already canonical."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = int(seconds * rate)
        # A quiet tone rather than silence, so duration checks have signal.
        samples = b"".join(
            struct.pack("<h", int(3000 * ((i // 40) % 2 * 2 - 1))) * channels for i in range(frames)
        )
        handle.writeframes(samples)
    return buffer.getvalue()


async def upload(client, data: bytes, name: str = "ref.wav", content_type: str = "audio/wav"):
    return await client.post(UPLOAD_URL, files={"file": (name, data, content_type)})


class TestAcceptance:
    async def test_a_valid_upload_is_accepted_and_returns_a_stable_id(self, client):
        resp = await upload(client, wav_bytes())
        assert resp.status_code == 201, resp.text
        body = resp.json()
        # The id must parse as a UUID and be usable as a handle later.
        reference_id = uuid.UUID(body["reference_id"])
        assert body["duration_seconds"] == pytest.approx(3.0, abs=0.1)
        assert body["display_name"] == "ref.wav"
        assert str(reference_id) == body["reference_id"]

    async def test_the_upload_is_normalised_to_the_canonical_form(self, client):
        """Uploaded at 44.1 kHz mono; stored at 48 kHz stereo."""
        body = (await upload(client, wav_bytes(rate=44_100, channels=1))).json()
        assert body["sample_rate"] == 48_000
        assert body["channels"] == 2

    async def test_the_response_never_leaks_a_storage_key_or_path(self, client):
        body = (await upload(client, wav_bytes())).json()
        serialised = repr(body)
        assert "reference/" not in serialised
        assert "/" not in str(body.get("display_name") or "")
        assert "storage_key" not in body

    async def test_two_uploads_of_the_same_bytes_get_distinct_ids(self, client):
        """Lifecycles stay independent even when the audio is identical."""
        data = wav_bytes()
        first = (await upload(client, data)).json()["reference_id"]
        second = (await upload(client, data)).json()["reference_id"]
        assert first != second

    async def test_limits_are_published(self, client):
        body = (await client.get(f"{UPLOAD_URL}/limits")).json()
        assert body["max_file_bytes"] == MAX_REFERENCE_FILE_BYTES
        assert "wav" in body["supported_formats"]


class TestRejection:
    async def test_a_corrupt_file_is_rejected(self, client):
        resp = await upload(client, b"RIFF" + b"\x00" * 400, name="broken.wav")
        assert resp.status_code == 400
        assert "audio" in resp.json()["detail"].lower()

    async def test_a_renamed_non_audio_file_is_rejected(self, client):
        """The bytes decide, not the extension or the content type."""
        resp = await upload(client, b"<html><body>not audio</body></html>" * 40)
        assert resp.status_code == 400

    async def test_an_unsupported_format_is_rejected(self, client):
        resp = await upload(
            client, wav_bytes(), name="ref.exe", content_type="application/x-msdownload"
        )
        assert resp.status_code == 400
        assert "format" in resp.json()["detail"].lower()

    async def test_an_empty_file_is_rejected(self, client):
        resp = await upload(client, b"")
        assert resp.status_code == 400

    async def test_an_oversized_upload_is_rejected(self, client):
        """Refused while streaming, not after buffering it all."""
        resp = await upload(client, b"\x00" * (MAX_REFERENCE_FILE_BYTES + 1024))
        assert resp.status_code == 400
        assert "larger" in resp.json()["detail"].lower()

    async def test_a_too_short_clip_is_rejected(self, client):
        resp = await upload(client, wav_bytes(seconds=0.2))
        assert resp.status_code == 400

    async def test_no_reference_is_stored_when_the_upload_is_refused(self, client, app):
        """A refusal must not leave a row behind."""
        from sqlalchemy import func, select

        from luber_database.models.generation import ReferenceAudio

        async with app.state.session_factory() as session:
            before = await session.scalar(select(func.count()).select_from(ReferenceAudio))
        await upload(client, b"not audio at all")
        async with app.state.session_factory() as session:
            after = await session.scalar(select(func.count()).select_from(ReferenceAudio))
        assert after == before


class TestPathSafety:
    @pytest.mark.parametrize(
        "hostile",
        [
            "../../../../etc/passwd.wav",
            "..\\..\\windows\\system32\\evil.wav",
            "/absolute/path/ref.wav",
            "ref.wav\x00.png",
        ],
    )
    async def test_a_hostile_filename_never_becomes_a_path(self, client, app, hostile):
        """The key comes from a server UUID; the name is a label only."""

        from luber_database.models.generation import ReferenceAudio

        resp = await upload(client, wav_bytes(), name=hostile)
        assert resp.status_code == 201, resp.text
        reference_id = resp.json()["reference_id"]

        async with app.state.session_factory() as session:
            row = await session.get(ReferenceAudio, uuid.UUID(reference_id))
        assert row is not None
        assert row.storage_key == f"reference/{reference_id}/source.wav"
        assert ".." not in row.storage_key
        for label in (row.display_name or "", row.storage_key):
            assert "/etc/" not in label
            assert "\x00" not in label


class TestAssetRoleSeparation:
    async def test_a_reference_is_not_an_audio_asset(self, client, app):
        """It must never appear in the table the download route reads."""
        from sqlalchemy import select

        from luber_database.models.generation import AudioAsset

        resp = await upload(client, wav_bytes())
        reference_id = resp.json()["reference_id"]
        async with app.state.session_factory() as session:
            keys = (await session.scalars(select(AudioAsset.storage_key))).all()
        assert all(not key.startswith("reference/") for key in keys)
        assert all(reference_id not in key for key in keys)

    async def test_a_reference_cannot_be_downloaded_through_the_master_route(self, client):
        """The download route resolves generations, not references.

        A reference id is a different namespace entirely, so the strongest
        thing a caller can do with one here is get a 404.
        """
        reference_id = (await upload(client, wav_bytes())).json()["reference_id"]
        resp = await client.get(f"/v1/generations/{reference_id}/audio?asset=master")
        assert resp.status_code == 404
