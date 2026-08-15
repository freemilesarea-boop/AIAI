"""Lineage traversal and the delete rule that protects it.

Two things are under test. The classifier, which must read only durable
provenance and must never crash on data that predates it. And the
traversal, which walks user-controlled structure and therefore has to
survive a shape nobody intended — a row that is its own parent, a cycle,
a parent that was deleted years ago, a tree deeper or wider than anyone
planned for.
"""

from __future__ import annotations

import uuid

import pytest

from luber_database import GenerationHasDescendantsError, GenerationRepository
from luber_schemas import GenerationStatus, LineageOperation, classify_operation

PARENT = uuid.uuid4()


class TestClassification:
    """Durable fields only — never title, prompt, group or reference."""

    def test_no_parent_is_an_original(self):
        assert (
            classify_operation(parent_generation_id=None, edit_kind=None)
            is LineageOperation.ORIGINAL
        )

    def test_a_parent_without_an_edit_kind_is_generate_again(self):
        assert (
            classify_operation(parent_generation_id=PARENT, edit_kind=None)
            is LineageOperation.GENERATE_AGAIN
        )

    @pytest.mark.parametrize(
        ("edit_kind", "expected"),
        [
            ("EXTEND", LineageOperation.EXTEND),
            ("REPLACE_RANGE", LineageOperation.REPLACE_SECTION),
            ("COVER", LineageOperation.COVER),
        ],
    )
    def test_stored_edit_kinds_map_to_product_operations(self, edit_kind, expected):
        assert classify_operation(parent_generation_id=PARENT, edit_kind=edit_kind) is expected

    def test_the_engine_word_never_survives_classification(self):
        """``REPLACE_RANGE`` is stored; ``REPLACE_SECTION`` is spoken."""
        operation = classify_operation(parent_generation_id=PARENT, edit_kind="REPLACE_RANGE")
        assert operation.value == "REPLACE_SECTION"
        assert "RANGE" not in operation.value

    def test_an_unknown_edit_kind_degrades_instead_of_raising(self):
        """Legacy or corrupt metadata must not take down Song Detail.

        The one fact still known is that it was derived from that parent.
        """
        assert (
            classify_operation(parent_generation_id=PARENT, edit_kind="SOMETHING_OLD")
            is LineageOperation.GENERATE_AGAIN
        )

    def test_an_edit_kind_without_a_parent_is_still_an_original(self):
        """A row derived from nothing is an original, whatever it claims."""
        assert (
            classify_operation(parent_generation_id=None, edit_kind="EXTEND")
            is LineageOperation.ORIGINAL
        )


async def make(repository: GenerationRepository, *, parent=None, edit_kind=None, title="t"):
    return await repository.create_generation(
        title=title,
        prompt="p",
        lyrics="",
        vocal_gender="instrumental",
        duration_requested=30,
        seed=None,
        language="en",
        instrumental=True,
        status=GenerationStatus.COMPLETED.value,
        idempotency_key=None,
        parent_generation_id=parent,
        edit_kind=edit_kind,
    )


class TestTraversal:
    async def test_an_original_has_no_ancestry_and_no_descendants(self, repository):
        row = await make(repository)
        assert await repository.get_ancestry(row.id) == []
        assert await repository.get_descendants(row.id) == []

    async def test_ancestry_runs_child_to_root(self, repository):
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")
        c = await make(repository, parent=b.id, edit_kind="REPLACE_RANGE", title="C")

        assert [row.title for row in await repository.get_ancestry(c.id)] == ["B", "A"]
        assert (await repository.get_ancestry(c.id))[-1].id == a.id

    async def test_descendants_cover_every_branch(self, repository):
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")
        await make(repository, parent=a.id, edit_kind="COVER", title="C")
        await make(repository, parent=b.id, edit_kind="REPLACE_RANGE", title="D")

        assert {row.title for row in await repository.get_descendants(a.id)} == {"B", "C", "D"}

    async def test_a_self_parent_does_not_hang(self, repository):
        """Impossible through the product, but not impossible in the table."""
        row = await make(repository)
        row.parent_generation_id = row.id
        await repository._session.commit()

        assert await repository.get_ancestry(row.id) == []
        assert await repository.get_descendants(row.id) == []
        assert await repository.count_descendants(row.id) == 0

    async def test_a_cycle_terminates(self, repository):
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")
        a.parent_generation_id = b.id  # A <-> B
        await repository._session.commit()

        # Bounded rather than infinite; the exact contents matter less
        # than the fact that both calls return.
        assert len(await repository.get_ancestry(b.id)) <= 2
        assert len(await repository.get_descendants(a.id)) <= 2

    async def test_a_missing_parent_ends_the_walk(self, repository):
        """A legacy row pointing at something long gone."""
        orphan = await make(repository, parent=uuid.uuid4(), edit_kind="EXTEND")
        assert await repository.get_ancestry(orphan.id) == []

    async def test_a_large_sibling_set_is_capped(self, repository):
        root = await make(repository, title="root")
        for index in range(25):
            await make(repository, parent=root.id, edit_kind="COVER", title=f"c{index}")

        capped = await repository.get_descendants(root.id, max_nodes=10)
        assert len(capped) == 10
        assert len(await repository.get_descendants(root.id)) == 25

    async def test_depth_is_bounded(self, repository):
        current = await make(repository, title="d0")
        root_id = current.id
        for depth in range(1, 8):
            current = await make(
                repository, parent=current.id, edit_kind="EXTEND", title=f"d{depth}"
            )

        assert len(await repository.get_descendants(root_id, max_depth=3)) == 3
        assert len(await repository.get_descendants(root_id)) == 7


class TestDeletePolicy:
    async def test_a_leaf_can_be_deleted(self, repository):
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")

        assert await repository.delete_generation(b.id) is True
        assert await repository.get_generation(b.id) is None
        assert await repository.get_generation(a.id) is not None

    async def test_a_parent_with_a_child_is_refused(self, repository):
        a = await make(repository, title="A")
        await make(repository, parent=a.id, edit_kind="EXTEND", title="B")

        with pytest.raises(GenerationHasDescendantsError) as caught:
            await repository.delete_generation(a.id)
        assert caught.value.descendant_count == 1

    async def test_a_root_with_deep_descendants_is_refused(self, repository):
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")
        await make(repository, parent=b.id, edit_kind="REPLACE_RANGE", title="C")

        with pytest.raises(GenerationHasDescendantsError) as caught:
            await repository.delete_generation(a.id)
        # Both levels count, not just the direct child.
        assert caught.value.descendant_count == 2

    async def test_a_refused_delete_leaves_the_link_intact(self, repository):
        """The regression that motivated the whole rule.

        The old behaviour nulled this link, leaving a row that still said
        EXTEND while descending from nothing.
        """
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")

        with pytest.raises(GenerationHasDescendantsError):
            await repository.delete_generation(a.id)

        child = await repository.get_generation(b.id)
        assert child.parent_generation_id == a.id
        assert child.edit_kind == "EXTEND"
        assert (
            classify_operation(
                parent_generation_id=child.parent_generation_id, edit_kind=child.edit_kind
            )
            is LineageOperation.EXTEND
        )

    async def test_a_refused_delete_leaves_the_parent_intact(self, repository):
        a = await make(repository, title="A")
        await make(repository, parent=a.id, edit_kind="EXTEND", title="B")

        with pytest.raises(GenerationHasDescendantsError):
            await repository.delete_generation(a.id)

        assert await repository.get_generation(a.id) is not None

    async def test_a_refused_delete_removes_no_assets(self, repository):
        """Descendants are checked before a single asset row is touched."""
        a = await make(repository, title="A")
        await repository.create_audio_asset(
            a.id,
            asset_type="MASTER",
            format="wav",
            mime_type="audio/wav",
            file_extension="wav",
            sample_rate=48_000,
            bit_depth=24,
            bitrate=None,
            channels=2,
            duration=30.0,
            storage_key=f"audio/{a.id}/master.wav",
            sha256="a" * 64,
            file_size=1,
        )
        await make(repository, parent=a.id, edit_kind="EXTEND", title="B")

        with pytest.raises(GenerationHasDescendantsError):
            await repository.delete_generation(a.id)

        assert len(await repository.get_audio_assets(a.id)) == 1

    async def test_deleting_an_unrelated_original_is_unchanged(self, repository):
        lonely = await make(repository, title="unrelated")
        assert await repository.delete_generation(lonely.id) is True

    async def test_deleting_the_last_child_then_frees_the_parent(self, repository):
        """The refusal is recoverable, which is what makes it acceptable."""
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")

        await repository.delete_generation(b.id)
        assert await repository.delete_generation(a.id) is True

    async def test_a_missing_row_still_reports_false(self, repository):
        assert await repository.delete_generation(uuid.uuid4()) is False


class TestProjectIndependence:
    async def test_a_child_does_not_inherit_its_parents_project(self, repository):
        project = await repository.create_project(name="Album")
        a = await make(repository, title="A")
        await repository.set_generation_project(a.id, project.id)
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")

        assert (await repository.get_generation(b.id)).project_id is None

    async def test_filing_a_child_does_not_move_its_parent(self, repository):
        project = await repository.create_project(name="Album")
        a = await make(repository, title="A")
        b = await make(repository, parent=a.id, edit_kind="EXTEND", title="B")
        await repository.set_generation_project(b.id, project.id)

        assert (await repository.get_generation(a.id)).project_id is None
        assert (await repository.get_generation(b.id)).project_id == project.id
