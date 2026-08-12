"""Make the benchmark toolkit importable from the tests."""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS))
