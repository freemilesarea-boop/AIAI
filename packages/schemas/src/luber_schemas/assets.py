"""Choosing the right master, in one place.

Before Phase 14 a generation had exactly one master and "the master"
was unambiguous. It now has up to two, and the two are for different
purposes — so every caller that used to filter for ``MASTER`` by hand is
now a place where the wrong one can be picked silently. These selectors
exist so that decision is made once and can be changed once.

Two questions, two different answers:

*What should a listener hear?* The finished master when the engine
produced one, otherwise the raw. That is
:func:`select_delivery_master`, and it backs downloads, playback and the
preview encode.

*What should be fed back into the model?* Always the raw. That is
:func:`select_raw_master`, and it backs extend, replace-section and
cover. Feeding a finished master into a new generation and then
finishing the result would stack corrections across generations — a
track extended five times would carry five high-shelf lifts — and the
child gets its own finishing pass anyway, so nothing is lost by starting
from the raw.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from luber_schemas.enums import AssetType


class HasAssetType(Protocol):
    """Anything with an ``asset_type``: an ORM row or an API model."""

    @property
    def asset_type(self) -> str: ...


AssetT = TypeVar("AssetT", bound=HasAssetType)

#: Delivery preference, most preferred first.
DELIVERY_MASTER_PRIORITY: tuple[AssetType, ...] = (
    AssetType.FINISHED_MASTER,
    AssetType.MASTER,
)


def select_delivery_master(assets: list[AssetT]) -> AssetT | None:
    """The master a listener should get.

    Returns ``None`` only when the generation has no master at all, which
    for a completed generation means something is wrong upstream.
    """
    for wanted in DELIVERY_MASTER_PRIORITY:
        for asset in assets:
            if asset.asset_type == wanted.value:
                return asset
    return None


def select_raw_master(assets: list[AssetT]) -> AssetT | None:
    """The unprocessed master, for feeding a further generation."""
    for asset in assets:
        if asset.asset_type == AssetType.MASTER.value:
            return asset
    return None


def select_finished_master(assets: list[AssetT]) -> AssetT | None:
    """The finishing result, or ``None`` when the engine took no action."""
    for asset in assets:
        if asset.asset_type == AssetType.FINISHED_MASTER.value:
            return asset
    return None
