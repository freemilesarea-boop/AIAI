from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field

from luber_shared import BaseServiceSettings


class ApiSettings(BaseServiceSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    # "queue" (production: dispatch to the ARQ worker) or "inline"
    # (tests/single-process dev only — never production).
    generation_execution_mode: Literal["queue", "inline"] = "queue"

    # ── operator training console (Phase 28) ──────────────────────────
    #
    # The console exposes training internals — datasets, checkpoints,
    # worker hosts, logs — to whoever can reach it. LUBER has no
    # operator role on `User` yet, so the boundary is the deployment
    # rather than the session: off by default, refused outright in
    # production, and a shared operator token required even when on.
    #
    # Inventing an `is_admin` column to hang this off would create a
    # privilege model the product has not designed, and a bug in it
    # would expose training control to ordinary accounts. A deployment
    # switch cannot be escalated into by a logged-in user.
    ops_console_enabled: bool = False
    #: Shared secret, compared in constant time. The console is refused
    #: when this is unset — an enabled console with no token is an open
    #: one, and failing closed is the only safe reading of a half
    #: configuration.
    ops_operator_token: str | None = None

    #: Where the Phase 25 training registry lives. Read-mostly: the
    #: console reads records and writes only through the orchestrator.
    ops_registry_root: str = "training-registry"
    #: Run artifact directories (plans, metrics, logs, environment
    #: locks). Defaults to the registry's own convention.
    ops_artifacts_root: str | None = None

    #: Roots the dataset and curation build catalogues are enumerated
    #: from. The console offers what it finds *inside* these, by id — an
    #: operator never types a path, so no request can name a directory
    #: the deployment did not sanction.
    ops_dataset_builds_root: str | None = None
    ops_curation_builds_root: str | None = None

    # ── remote worker transport ──────────────────────────────────────
    #
    # "none" (default): the console holds no way to reach a worker, so
    # cancel and reconcile record intent and say so plainly. "local": a
    # worker root on this machine, which is the real Phase 27 worker
    # driven through `LocalWorkerClient` — enough to exercise every
    # remote path without a GPU or an SSH key.
    #
    # There is deliberately no "ssh" option. SSH endpoints, key
    # references and known-hosts material are operator credentials; the
    # CLI holds them and a browser-reachable process does not.
    ops_worker_transport: Literal["none", "local"] = "none"
    ops_worker_root: str | None = None

    #: How many records a list endpoint will return at once, whatever
    #: the caller asks for. A registry of ten thousand runs must not
    #: become one response.
    ops_page_size_limit: int = 200


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()
