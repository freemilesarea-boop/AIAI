#!/usr/bin/env python3
"""Fail when a Dockerfile's workspace list has drifted from the workspace.

This exists because the drift already happened twice, silently, and both
times the symptom appeared at container start rather than at build.

`uv sync` needs every workspace member's `pyproject.toml` present in the
build context to resolve the lockfile. When a member is missing, the sync
does not error usefully — it produces an environment without the
workspace packages installed, and the container then dies on
`ModuleNotFoundError: No module named 'luber_api'`. By the time anyone
sees that, a deploy is already in flight.

Phase 7 made the cost concrete: `apps/api` gained `luber-billing`, and an
API image without it has no PayApp endpoints at all — the checkout route
and both callback routes simply do not exist. A payment provider would
be posting notifications into a 404.

So this compares `[tool.uv.workspace] members` against the `COPY` lines
in each Dockerfile and reports the difference. It reads files and prints;
it changes nothing.

    uv run python scripts/deployment/check_docker_workspace.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILES = (
    REPO_ROOT / "infra" / "docker" / "api.Dockerfile",
    REPO_ROOT / "infra" / "docker" / "worker.Dockerfile",
)

#: `COPY <member>/pyproject.toml <member>/pyproject.toml`
_MANIFEST_COPY = re.compile(r"^COPY\s+(\S+)/pyproject\.toml\s", re.MULTILINE)


def workspace_members() -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return set(data["tool"]["uv"]["workspace"]["members"])


def copied_members(dockerfile: Path) -> set[str]:
    return set(_MANIFEST_COPY.findall(dockerfile.read_text()))


def main() -> int:
    expected = workspace_members()
    failures = 0

    for dockerfile in DOCKERFILES:
        if not dockerfile.exists():
            print(f"missing: {dockerfile.relative_to(REPO_ROOT)}", file=sys.stderr)
            failures += 1
            continue

        copied = copied_members(dockerfile)
        missing = sorted(expected - copied)
        # Stale entries are harmless to the build but mean the file is
        # describing a workspace that no longer exists, which is the same
        # kind of rot in the other direction.
        stale = sorted(copied - expected)
        name = dockerfile.relative_to(REPO_ROOT)

        if not missing and not stale:
            print(f"ok   {name}  ({len(copied)} workspace members)")
            continue

        failures += 1
        print(f"FAIL {name}", file=sys.stderr)
        for member in missing:
            print(f"       missing COPY: {member}/pyproject.toml", file=sys.stderr)
        for member in stale:
            print(f"       no longer a workspace member: {member}", file=sys.stderr)

    if failures:
        print(
            "\nAdd the missing COPY lines, or remove the stale ones. An image "
            "built from a drifted list installs no workspace packages and "
            "fails at import time inside the container.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
