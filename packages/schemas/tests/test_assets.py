"""Master selection: the one place the raw/finished choice is made.

Before Phase 14B a generation had one master and picking it was
unambiguous. It now has up to two for different purposes, so the failure
mode these guard against is silent: serving unfinished audio to
listeners, or feeding finished audio back into the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from luber_schemas import (
    AssetType,
    select_delivery_master,
    select_finished_master,
    select_raw_master,
)


@dataclass(frozen=True)
class Asset:
    asset_type: str


RAW = Asset(AssetType.MASTER.value)
FINISHED = Asset(AssetType.FINISHED_MASTER.value)
PREVIEW = Asset(AssetType.PREVIEW.value)
STEM = Asset(AssetType.STEM.value)


class TestDeliveryMaster:
    def test_prefers_the_finished_master(self):
        assert select_delivery_master([RAW, FINISHED, PREVIEW]) is FINISHED

    def test_preference_does_not_depend_on_list_order(self):
        """Rows come back in whatever order the database returns them."""
        assert select_delivery_master([FINISHED, RAW]) is FINISHED
        assert select_delivery_master([RAW, FINISHED]) is FINISHED

    def test_falls_back_to_the_raw_master(self):
        """NO_ACTION is normal: the raw master is then the product."""
        assert select_delivery_master([RAW, PREVIEW]) is RAW

    def test_never_returns_a_non_master(self):
        assert select_delivery_master([PREVIEW, STEM]) is None

    def test_returns_none_when_there_is_no_master_at_all(self):
        assert select_delivery_master([]) is None


class TestRawMaster:
    def test_returns_the_raw_master_even_when_a_finished_one_exists(self):
        """Edits are fed back into the model and must start from the raw.

        Otherwise finishing corrections compound across generations: a
        track extended five times would carry five high-shelf lifts.
        """
        assert select_raw_master([FINISHED, RAW]) is RAW

    def test_never_substitutes_the_finished_master(self):
        assert select_raw_master([FINISHED, PREVIEW]) is None


class TestFinishedMaster:
    def test_absence_is_representable(self):
        """ "No finished master" is a normal state, not an error."""
        assert select_finished_master([RAW, PREVIEW]) is None

    def test_found_when_present(self):
        assert select_finished_master([RAW, FINISHED]) is FINISHED
