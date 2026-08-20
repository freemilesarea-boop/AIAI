"""Fixtures for model evaluation.

The helpers and fixtures live in ``evaluation_fixtures.py`` rather than here.
pytest imports every ``conftest.py`` as a module named ``conftest``, so
in a whole-repository run only the first one collected keeps that name
and every other package's ``from conftest import ...`` resolves to a
stranger's file. Giving each package's helpers a distinct module name
removes the collision; this file re-exports the fixtures so pytest still
discovers them by directory.
"""

from __future__ import annotations

from evaluation_fixtures import (  # noqa: F401
    orchestrator,
    registry_root,
    repository_root,
    seeded,
)
