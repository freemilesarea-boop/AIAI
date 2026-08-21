"""Fixtures for inference QC.

The helpers live in ``qc_fixtures.py`` rather than here, following the
convention this repository already uses: pytest imports every
``conftest.py`` as a module named ``conftest``, so in a whole-repository
run only the first one collected keeps that name and every other
package's ``from conftest import ...`` resolves to a stranger's file.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "audio"
    directory.mkdir()
    return directory
