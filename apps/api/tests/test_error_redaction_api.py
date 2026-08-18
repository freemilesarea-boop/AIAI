"""What a failing generation is allowed to tell the browser.

``ErrorCode``'s own docstring says raw exception strings are never sent
to clients, and the database deliberately keeps ``error_message`` so an
operator can read it. The response model used to carry that column
straight through, which meant a storage failure — whose message is
``source audio missing: /Users/…`` — would have handed the browser an
absolute path.

These tests assert the boundary, not the wording: the client gets a
stable code and nothing that describes the machine LUBER runs on.
"""

from __future__ import annotations

import uuid

import pytest

from luber_database import GenerationRepository
from luber_schemas import ErrorCode, GenerationStatus

LEAKY_MESSAGE = (
    "source audio missing: /Users/someone/Desktop/luber-music-ai/data/audio/x/master.wav"
)


@pytest.fixture
async def failed_generation(app, client):
    """A FAILED row whose stored message leaks a path, as a real one would.

    Owned by the authenticated client, because a row belonging to anyone
    else is now correctly invisible — which would test 404 handling
    rather than redaction.
    """
    async with app.state.session_factory() as session:
        repository = GenerationRepository(session, owner=uuid.UUID(client.user_id))
        generation = await repository.create_generation(
            title="Doomed",
            prompt="p",
            lyrics="",
            vocal_gender="instrumental",
            duration_requested=30,
            status=GenerationStatus.QUEUED.value,
            language="en",
        )
        await repository.mark_failed(
            generation.id,
            status=GenerationStatus.FAILED.value,
            error_code=ErrorCode.UPLOAD_FAILED.value,
            error_message=LEAKY_MESSAGE,
        )
        return generation.id


class TestFailedGenerationResponse:
    async def test_the_stable_code_is_returned(self, client, failed_generation):
        body = (await client.get(f"/v1/generations/{failed_generation}")).json()
        assert body["status"] == GenerationStatus.FAILED.value
        assert body["error_code"] == ErrorCode.UPLOAD_FAILED.value

    async def test_the_raw_message_is_not(self, client, failed_generation):
        body = (await client.get(f"/v1/generations/{failed_generation}")).json()
        assert "error_message" not in body

    async def test_no_filesystem_path_reaches_the_client(self, client, failed_generation):
        raw = (await client.get(f"/v1/generations/{failed_generation}")).text
        assert "/Users/" not in raw
        assert "luber-music-ai/data" not in raw
        assert LEAKY_MESSAGE not in raw

    async def test_the_list_endpoint_leaks_nothing_either(self, client, failed_generation):
        raw = (await client.get("/v1/generations")).text
        assert "/Users/" not in raw
        assert LEAKY_MESSAGE not in raw

    async def test_the_operator_can_still_read_it_in_the_database(self, app, failed_generation):
        """Redaction is at the boundary, not destruction of the record."""
        async with app.state.session_factory() as session:
            row = await GenerationRepository(session, owner=None).get_generation(failed_generation)
        assert row.error_message == LEAKY_MESSAGE


class TestUnknownGeneration:
    async def test_a_missing_generation_says_so_without_internals(self, client):
        response = await client.get(f"/v1/generations/{uuid.uuid4()}")
        assert response.status_code == 404
        assert "/Users/" not in response.text
        assert "Traceback" not in response.text
