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

    # ── generation availability ──────────────────────────────────────
    #
    # A kill switch for the one endpoint that spends a user's allowance.
    #
    # Generation needs two things this deployment does not have: a GPU
    # serving ACE-Step, and object storage that survives a redeploy. With
    # neither, `POST /v1/generations` would still succeed — it would
    # reserve an allowance slot, enqueue a job, and leave the row QUEUED
    # for ever, because no worker exists to take it. An unsettled
    # reservation holds its slot by design, so the user would pay a song
    # out of their monthly allowance and receive nothing.
    #
    # Off means the route refuses *before* it reserves anything. That is
    # the whole mechanism: no partial state, no orphaned rows, nothing to
    # clean up when the GPU arrives and this goes back to true.
    generation_enabled: bool = True

    # ── PayApp recurring billing (Phase 7) ───────────────────────────
    #
    # Server-side only, all of it. None of these values is ever returned
    # in an API response, rendered into a page, or sent to the browser
    # in any form — the frontend asks for a plan by name and receives a
    # PayApp-hosted URL, which is the whole of what it needs to know.
    #
    # Billing is off unless configured. A half-configured integration is
    # more dangerous than none: it could register real recurring
    # contracts while being unable to validate the notifications that
    # confirm them, so `billing_available()` requires the whole set.
    payapp_userid: str | None = None
    #: Sent on outbound API calls. Also arrives on notifications, where
    #: it is one of the three values compared in constant time.
    payapp_linkkey: str | None = None
    #: PayApp's 연동 VALUE. Notification-only: this is never sent
    #: outbound, so a leak of it cannot be used to place charges — but it
    #: could be used to forge a notification, which is why it is held to
    #: the same standard as the key.
    payapp_linkval: str | None = None
    payapp_api_url: str = "https://api.payapp.kr/oapi/apiLoad.html"

    #: Public base URL PayApp calls back on. Must be reachable from the
    #: internet, which localhost is not — a developer testing callbacks
    #: needs a tunnel, and there is no way around that.
    payapp_public_base_url: str = "http://localhost:8000"
    #: Where the browser lands after PayApp. UX only: reaching it proves
    #: nothing and activates nothing.
    payapp_return_base_url: str = "http://localhost:3000"

    #: How long a notification-shaped request may be repeated from one
    #: address before it is throttled. The endpoint cannot require a
    #: session — PayApp has none — so this is what stops the public
    #: surface from being a free anomaly-row generator.
    payapp_feedback_rate_limit: int = 120
    payapp_feedback_rate_window_seconds: int = 60

    def billing_available(self) -> bool:
        """Whether checkout may be offered at all.

        All three secrets or nothing. Offering a Subscribe button that
        cannot validate the resulting payment would take a customer's
        money and fail to give them the plan.
        """
        return bool(self.payapp_userid and self.payapp_linkkey and self.payapp_linkval)


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()
