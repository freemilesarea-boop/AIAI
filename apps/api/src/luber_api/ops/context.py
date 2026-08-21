"""What the console is allowed to touch, resolved once.

The registries are directories of JSON on the operator's own machine.
That is a deliberate Phase 25 decision and this layer does not change
it — no database, no mirror, no cache that can disagree with the files.
What this module adds is the part a browser-reachable process needs and
a CLI does not: a bounded idea of *which* directories exist.

Two ideas do that work.

**Roots are configuration, ids are input.** Every path the console can
reach is derived from a configured root by joining an identifier that
has been checked against the entries actually present. A request never
carries a path, so no request can name one. That is what makes Step 50
— "do not permit manually typing arbitrary paths" — a property of the
API rather than a habit of the UI.

**The transport is opt-in and never SSH.** Reaching a rented GPU needs a
host, a user, a key reference and a known-hosts file: operator
credentials that the CLI holds. Putting them into a process a browser
can reach would move the Phase 27 security boundary for the sake of a
button. So the console reaches a worker only when one is configured
*on this machine*, and otherwise says plainly that it cannot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Request

from luber_api.settings import ApiSettings, get_settings
from luber_evaluation.registry import EvaluationRegistry
from luber_training.orchestrator import Orchestrator
from luber_training.registry import Registry
from luber_training.remote.identity import LivenessPolicy

#: Files that mark a directory as a usable build. A directory without
#: its lock is not offered: a build whose lock is missing cannot be
#: verified, and a run created against it would fail its first gate.
DATASET_LOCK_NAME = "dataset_lock.json"
CURATION_LOCK_NAME = "curation_lock.json"


class OpsConfigurationError(RuntimeError):
    """Raised when the console is asked for something it has no root for."""


def _safe_child(root: Path, name: str) -> Path:
    """Resolve *name* inside *root*, or refuse.

    Refuses rather than sanitises. A name that would escape is not a
    name with a typo in it, and quietly stripping the `..` would turn a
    hostile request into a successful one against a neighbouring
    directory.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise OpsConfigurationError(f"unsafe build identifier {name!r}")
    candidate = (root / name).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise OpsConfigurationError(f"build identifier {name!r} escapes its configured root")
    return candidate


@dataclass(frozen=True)
class BuildEntry:
    """One dataset or curation build the console may reference.

    ``build_id`` is the directory name and the only thing a request
    carries. The path is never returned to a browser — an operator
    picking a build does not need to know where the deployment keeps
    it, and a filesystem layout is not something to publish.
    """

    build_id: str
    lock: dict[str, Any]


@dataclass(frozen=True)
class BuildCatalogue:
    """The builds a configured root contains, enumerated on demand.

    Read each time rather than cached: an operator who has just finished
    a curation should see it without restarting the API, and the cost of
    listing a directory is not worth a staleness bug.
    """

    root: Path | None
    lock_name: str

    @property
    def configured(self) -> bool:
        return self.root is not None

    def entries(self) -> list[BuildEntry]:
        if self.root is None or not self.root.is_dir():
            return []
        found: list[BuildEntry] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            lock_path = child / self.lock_name
            if not lock_path.is_file():
                continue
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # A build whose lock cannot be read is not offered. It
                # would fail its first gate anyway, and offering it
                # would turn an unreadable file into a run that looks
                # like it was rejected on its merits.
                continue
            if isinstance(lock, dict):
                found.append(BuildEntry(build_id=child.name, lock=lock))
        return found

    def problems(self) -> list[str]:
        if self.root is None:
            return ["no build root is configured for this deployment"]
        if not self.root.is_dir():
            return ["the configured build root does not exist on this machine"]
        return []

    def directory_for(self, build_id: str) -> Path:
        if self.root is None:
            raise OpsConfigurationError("no build root is configured for this deployment")
        path = _safe_child(self.root, build_id)
        if not (path / self.lock_name).is_file():
            raise OpsConfigurationError(f"{build_id!r} is not a build this deployment offers")
        return path

    def lock_for(self, build_id: str) -> dict[str, Any]:
        path = self.directory_for(build_id) / self.lock_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise OpsConfigurationError(f"{build_id!r} has a lock that is not a JSON object")
        return payload


@dataclass
class OpsContext:
    """Everything an operator route reads or writes, in one place."""

    settings: ApiSettings
    registry: Registry
    evaluations: EvaluationRegistry
    orchestrator: Orchestrator
    datasets: BuildCatalogue
    curations: BuildCatalogue
    liveness: LivenessPolicy

    @property
    def registry_root(self) -> Path:
        return self.registry.root

    # ── remote transport ─────────────────────────────────────────────
    @property
    def worker_transport(self) -> str:
        return self.settings.ops_worker_transport

    def transport_unavailable_reason(self) -> str | None:
        """Why a worker cannot be reached, in words, or None if it can.

        Returned rather than raised because "we cannot see the worker"
        is a legitimate answer to a status question, not an error in
        having asked it.
        """
        if self.settings.ops_worker_transport == "none":
            return (
                "This console holds no transport to a worker. Remote credentials belong to "
                "the operator CLI, so reconciliation and cancellation delivery run there."
            )
        root = self.settings.ops_worker_root
        if not root:
            return "The local worker transport is selected but no worker root is configured."
        if not Path(root).expanduser().is_dir():
            return f"The configured worker root does not exist on this machine: {root}"
        return None

    def worker_root(self) -> Path:
        reason = self.transport_unavailable_reason()
        if reason is not None:
            raise OpsConfigurationError(reason)
        assert self.settings.ops_worker_root is not None  # narrowed by the check above
        return Path(self.settings.ops_worker_root).expanduser()


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


@lru_cache
def _build_context(
    registry_root: str,
    artifacts_root: str | None,
    dataset_root: str | None,
    curation_root: str | None,
) -> tuple[Registry, EvaluationRegistry, Orchestrator, BuildCatalogue, BuildCatalogue]:
    """Construct the registries once per configuration.

    Cached on the configuration rather than globally so a test that
    points at a temporary directory gets its own, and two tests with
    two registries never share one. `Registry.__init__` creates its
    collection directories, which is cheap but not free.
    """
    registry = Registry(Path(registry_root).expanduser())
    evaluations = EvaluationRegistry(registry)
    orchestrator = Orchestrator(
        registry,
        artifacts_root=_optional_path(artifacts_root),
        repository_root=Path.cwd(),
    )
    datasets = BuildCatalogue(root=_optional_path(dataset_root), lock_name=DATASET_LOCK_NAME)
    curations = BuildCatalogue(root=_optional_path(curation_root), lock_name=CURATION_LOCK_NAME)
    return registry, evaluations, orchestrator, datasets, curations


def context_from_settings(settings: ApiSettings) -> OpsContext:
    registry, evaluations, orchestrator, datasets, curations = _build_context(
        settings.ops_registry_root,
        settings.ops_artifacts_root,
        settings.ops_dataset_builds_root,
        settings.ops_curation_builds_root,
    )
    return OpsContext(
        settings=settings,
        registry=registry,
        evaluations=evaluations,
        orchestrator=orchestrator,
        datasets=datasets,
        curations=curations,
        liveness=LivenessPolicy(),
    )


def get_ops_context(request: Request) -> OpsContext:
    """The console's context for this request.

    Read from ``app.state`` when a test has put one there, so a test can
    hand the routes a temporary registry without touching global
    settings; otherwise built from configuration.
    """
    override = getattr(request.app.state, "ops_context", None)
    if isinstance(override, OpsContext):
        return override
    return context_from_settings(get_settings())
