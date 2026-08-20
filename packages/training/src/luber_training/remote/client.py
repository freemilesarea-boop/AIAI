"""Driving a worker from the control plane.

The client speaks the worker's CLI. Two implementations, and both build
an argv list and hand it to a subprocess without a shell anywhere in the
path.

`LocalWorkerClient` invokes the worker CLI as a subprocess against a
different root directory. It is not a mock: real processes, real files,
real exit codes, real detached trainers. It exists so the whole
lifecycle can be exercised on a laptop, and so the code paths the tests
cover are the code paths a rented GPU will run.

`SshWorkerClient` prefixes the same argv with `ssh`. That is the entire
difference, and it is deliberate — a bug that appears only over SSH
would otherwise be a bug in a code path nothing tested.

The safety property that matters: **no string this module builds is ever
interpreted by a shell.** `ssh` does concatenate its remote arguments
and hand them to a login shell on the far side, so every remote argument
is quoted with `shlex.quote` before it goes into that command, and every
identifier that reaches it has already been validated against a pattern.
A run id is not a place for shell syntax to hide.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.remote.protocol import (
    REMOTE_PROTOCOL_VERSION,
    Envelope,
    ProtocolError,
    RemoteCommand,
    check_protocol,
)
from luber_training.remote.secrets import NullSecretResolver, SecretResolver, redact

#: Identifiers that may cross to a remote command line. Deliberately
#: narrow: everything this project generates fits, and anything that
#: does not is refused rather than escaped and hoped for.
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: How long each verb may take. Split by kind, because they differ by
#: orders of magnitude and one global timeout would either kill a
#: transfer or let a hung status query block an operator for an hour.
DEFAULT_CONNECT_TIMEOUT = 15.0
DEFAULT_COMMAND_TIMEOUT = 120.0
DEFAULT_PREFLIGHT_TIMEOUT = 600.0
#: Cancellation waits out the trainer's grace period, then some.
DEFAULT_CANCEL_TIMEOUT = 300.0


class ClientError(RuntimeError):
    """Raised when a worker cannot be reached or refuses a request."""


class WorkerUnreachable(ClientError):
    """Raised when the worker did not answer at all.

    Distinct from a refusal on purpose. "The worker said no" and "the
    worker said nothing" call for different responses, and the second
    one must never be resolved by assuming the first.
    """


def safe_identifier(value: str, *, what: str = "identifier") -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER.match(value):
        raise ClientError(
            f"{value!r} is not a usable {what}: only letters, digits, dot, dash and "
            "underscore are permitted, because this value reaches a command line"
        )
    return value


@dataclass
class WorkerEndpoint:
    """How to reach one worker. References only, never secrets."""

    #: Where the worker's root lives on the far side.
    worker_root: str
    #: The interpreter that runs the worker CLI there.
    python_executable: str = "python"
    host: str | None = None
    user: str | None = None
    port: int | None = None
    #: Names, resolved out of band at the moment of use.
    ssh_key_ref: str | None = None
    known_hosts_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_root": self.worker_root,
            "python_executable": self.python_executable,
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "ssh_key_ref": self.ssh_key_ref,
            "known_hosts_ref": self.known_hosts_ref,
        }


class RemoteWorkerClient(ABC):
    """What the control plane can ask of a worker."""

    name: str = "abstract"

    @abstractmethod
    def invoke(
        self, command: str, arguments: list[str], *, timeout: float = DEFAULT_COMMAND_TIMEOUT
    ) -> Envelope:
        """Run one worker verb and return its envelope."""

    # ── the verbs, in terms of invoke ────────────────────────────────
    def probe_worker(self) -> dict[str, Any]:
        return self.invoke(RemoteCommand.PROBE.value, []).payload

    def identity(self) -> dict[str, Any]:
        return self.invoke(RemoteCommand.IDENTITY.value, []).payload

    def heartbeat(self) -> dict[str, Any]:
        return self.invoke(RemoteCommand.HEARTBEAT.value, []).payload

    def prepare_run(self, run_id: str, plan_sha256: str, manifest_sha256: str) -> dict[str, Any]:
        return self.invoke(
            RemoteCommand.PREPARE.value,
            [
                "--run-id",
                safe_identifier(run_id, what="run id"),
                "--plan-sha256",
                safe_identifier(plan_sha256, what="plan digest"),
                "--manifest-sha256",
                safe_identifier(manifest_sha256, what="manifest digest"),
            ],
        ).payload

    def receive_manifest(self, run_id: str, remote_manifest_path: str) -> dict[str, Any]:
        return self.invoke(
            RemoteCommand.RECEIVE.value,
            [
                "--run-id",
                safe_identifier(run_id, what="run id"),
                "--manifest",
                remote_manifest_path,
            ],
        ).payload

    def preflight(
        self,
        run_id: str,
        *,
        minimum_free_disk_mb: int | None = None,
        allow_code_mismatch: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        arguments = ["--run-id", safe_identifier(run_id, what="run id")]
        if minimum_free_disk_mb is not None:
            arguments += ["--minimum-free-disk-mb", str(int(minimum_free_disk_mb))]
        if allow_code_mismatch:
            arguments.append("--allow-code-mismatch")
        envelope = self.invoke(
            RemoteCommand.PREFLIGHT.value, arguments, timeout=DEFAULT_PREFLIGHT_TIMEOUT
        )
        return envelope.ok, envelope.payload

    def launch(self, run_id: str) -> dict[str, Any]:
        return self.invoke(
            RemoteCommand.START.value, ["--run-id", safe_identifier(run_id, what="run id")]
        ).payload

    def status(self, run_id: str) -> dict[str, Any]:
        return self.invoke(
            RemoteCommand.STATUS.value, ["--run-id", safe_identifier(run_id, what="run id")]
        ).payload

    def cancel(self, run_id: str, *, grace_seconds: float = 60.0) -> dict[str, Any]:
        return self.invoke(
            RemoteCommand.CANCEL.value,
            [
                "--run-id",
                safe_identifier(run_id, what="run id"),
                "--grace-seconds",
                str(float(grace_seconds)),
            ],
            timeout=DEFAULT_CANCEL_TIMEOUT,
        ).payload

    def logs(
        self, run_id: str, *, stream: str = "stdout", offset: int = 0, limit: int = 262_144
    ) -> dict[str, Any]:
        if stream not in ("stdout", "stderr"):
            raise ClientError(f"{stream!r} is not a log stream")
        return self.invoke(
            RemoteCommand.LOGS.value,
            [
                "--run-id",
                safe_identifier(run_id, what="run id"),
                "--stream",
                stream,
                "--offset",
                str(int(offset)),
                "--limit",
                str(int(limit)),
            ],
        ).payload

    def metrics(self, run_id: str, *, cursor: int = 0, limit: int = 2000) -> dict[str, Any]:
        return self.invoke(
            RemoteCommand.METRICS.value,
            [
                "--run-id",
                safe_identifier(run_id, what="run id"),
                "--cursor",
                str(int(cursor)),
                "--limit",
                str(int(limit)),
            ],
        ).payload

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        payload = self.invoke(
            RemoteCommand.CHECKPOINTS.value, ["--run-id", safe_identifier(run_id, what="run id")]
        ).payload
        checkpoints: list[dict[str, Any]] = payload.get("checkpoints", [])
        return checkpoints

    def collect_result(self, run_id: str) -> dict[str, Any]:
        return self.invoke(
            RemoteCommand.RESULT.value, ["--run-id", safe_identifier(run_id, what="run id")]
        ).payload

    def cleanup(self, run_id: str, *, remove_dataset: bool = False) -> dict[str, Any]:
        arguments = ["--run-id", safe_identifier(run_id, what="run id")]
        if remove_dataset:
            arguments.append("--remove-dataset")
        return self.invoke(RemoteCommand.CLEANUP.value, arguments).payload


def _parse(stdout: str, command: str) -> Envelope:
    """Turn a worker's reply into an envelope, or explain why not."""
    text = stdout.strip()
    if not text:
        raise WorkerUnreachable(f"the worker returned nothing for {command!r}")
    # The last line, because a poorly-behaved remote shell may print a
    # login banner before the reply.
    line = text.splitlines()[-1]
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ClientError(
            f"the worker's reply to {command!r} was not JSON: {redact(line[:200])}"
        ) from exc
    try:
        return Envelope.from_dict(payload, peer="worker")
    except ProtocolError:
        raise
    except Exception as exc:
        raise ClientError(f"the worker's reply to {command!r} was malformed: {exc}") from exc


class LocalWorkerClient(RemoteWorkerClient):
    """A worker on this machine, in a different directory.

    Runs the real CLI as a real subprocess. The only thing it does not
    simulate is the network, and the transport is where a network would
    be simulated — so between them the lifecycle tests exercise
    everything except the wire itself.
    """

    name = "local"

    def __init__(
        self,
        worker_root: Path,
        *,
        python_executable: str | None = None,
        protocol_version: str = REMOTE_PROTOCOL_VERSION,
    ) -> None:
        import sys

        self.worker_root = Path(worker_root)
        self.python_executable = python_executable or sys.executable
        self.protocol_version = protocol_version

    def _argv(self, command: str, arguments: list[str]) -> list[str]:
        return [
            self.python_executable,
            "-m",
            "luber_training.remote",
            "--root",
            str(self.worker_root),
            "--protocol-version",
            self.protocol_version,
            command,
            *arguments,
        ]

    def invoke(
        self, command: str, arguments: list[str], *, timeout: float = DEFAULT_COMMAND_TIMEOUT
    ) -> Envelope:
        argv = self._argv(command, arguments)
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerUnreachable(
                f"the worker did not answer {command!r} within {timeout}s"
            ) from exc
        except OSError as exc:
            raise WorkerUnreachable(f"could not run the worker: {exc}") from exc

        envelope = _parse(completed.stdout, command)
        if not envelope.ok and envelope.error:
            # A refusal is data, not an exception: preflight returning
            # BLOCKED is a normal outcome the caller has to inspect.
            return envelope
        return envelope


class SshWorkerClient(RemoteWorkerClient):
    """A worker reached over SSH. Provider-neutral by construction.

    Host key verification is **on**. There is no constructor argument
    that disables it, because the only reason to disable it is to make a
    warning go away, and the warning is the security property. First
    contact with a new host is an explicit enrolment step an operator
    performs once, documented in the runbook.

    No password is ever passed on a command line. Authentication is a
    key file, resolved by reference at the moment of use, and
    `BatchMode=yes` guarantees ssh fails rather than blocking on a
    prompt no one is there to answer.
    """

    name = "ssh"

    def __init__(
        self,
        endpoint: WorkerEndpoint,
        *,
        secrets: SecretResolver | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        protocol_version: str = REMOTE_PROTOCOL_VERSION,
        ssh_executable: str = "ssh",
    ) -> None:
        if not endpoint.host:
            raise ClientError("an SSH endpoint needs a host")
        self.endpoint = endpoint
        self.secrets = secrets or NullSecretResolver()
        self.connect_timeout = connect_timeout
        self.protocol_version = protocol_version
        self.ssh_executable = ssh_executable

    def ssh_options(self) -> list[str]:
        """The ssh flags, with verification never optional.

        `StrictHostKeyChecking=yes` refuses an unknown host outright.
        `accept-new` — which trusts on first use — is not used, because
        an operator dispatching a job is not in a position to notice
        that the host they enrolled last week has a different key today.
        """
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"ConnectTimeout={int(self.connect_timeout)}",
            # Password and keyboard-interactive are refused so a
            # misconfiguration cannot silently fall back to a prompt.
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
        ]
        if self.endpoint.known_hosts_ref:
            known_hosts = self.secrets.resolve_path(self.endpoint.known_hosts_ref)
            options += ["-o", f"UserKnownHostsFile={known_hosts}"]
        if self.endpoint.ssh_key_ref:
            # The key is used by path. Its bytes are never read here and
            # never copied into a staging directory.
            key = self.secrets.resolve_path(self.endpoint.ssh_key_ref)
            options += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
        if self.endpoint.port:
            options += ["-p", str(int(self.endpoint.port))]
        return options

    def _remote_command(self, command: str, arguments: list[str]) -> str:
        """The worker invocation, quoted for the far side's shell.

        ssh hands its remote arguments to a login shell, so this is the
        one place a string is built that something will parse. Every
        element goes through `shlex.quote`, and every identifier was
        validated before it got here — belt and braces, because the
        consequence of getting it wrong is arbitrary command execution
        on a machine holding the training data.
        """
        parts = [
            self.endpoint.python_executable,
            "-m",
            "luber_training.remote",
            "--root",
            self.endpoint.worker_root,
            "--protocol-version",
            self.protocol_version,
            command,
            *arguments,
        ]
        return " ".join(shlex.quote(part) for part in parts)

    def invoke(
        self, command: str, arguments: list[str], *, timeout: float = DEFAULT_COMMAND_TIMEOUT
    ) -> Envelope:
        target = (
            f"{self.endpoint.user}@{self.endpoint.host}"
            if self.endpoint.user
            else str(self.endpoint.host)
        )
        argv = [
            self.ssh_executable,
            *self.ssh_options(),
            target,
            self._remote_command(command, arguments),
        ]
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerUnreachable(
                f"{target} did not answer {command!r} within {timeout}s"
            ) from exc
        except OSError as exc:
            raise WorkerUnreachable(f"could not reach {target}: {exc}") from exc

        if completed.returncode == 255 and not completed.stdout.strip():
            # ssh's own failure code. Distinguishable from the worker
            # exiting non-zero, which still prints an envelope.
            raise WorkerUnreachable(
                f"ssh could not connect to {target}: {redact(completed.stderr.strip()[:300])}"
            )
        return _parse(completed.stdout, command)


@dataclass
class ClientRetryPolicy:
    """Which operations may be retried without thinking, and which may not.

    The distinction is the whole point. Reading a status is idempotent:
    ask again. Launching a trainer is not — a launch whose reply was
    lost may well have started a trainer, and retrying it would start a
    second one against the same checkpoint directory. That case is
    resolved by reconciliation, never by another launch.
    """

    attempts: int = 3
    backoff_seconds: float = 2.0

    #: Safe to repeat: they change nothing, or changing them twice is
    #: the same as changing them once.
    idempotent: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                RemoteCommand.IDENTITY.value,
                RemoteCommand.PROBE.value,
                RemoteCommand.HEARTBEAT.value,
                RemoteCommand.STATUS.value,
                RemoteCommand.LOGS.value,
                RemoteCommand.METRICS.value,
                RemoteCommand.CHECKPOINTS.value,
                RemoteCommand.RESULT.value,
                RemoteCommand.PREFLIGHT.value,
                # prepare is idempotent by construction: the lease check
                # makes a repeat return the existing state.
                RemoteCommand.PREPARE.value,
                RemoteCommand.RECEIVE.value,
            }
        )
    )

    def may_retry(self, command: str) -> bool:
        return command in self.idempotent

    def reason(self, command: str) -> str:
        if self.may_retry(command):
            return f"{command} changes nothing that repeating it would change twice"
        return (
            f"{command} may have taken effect before the failure. Reconcile with the "
            "worker to find out what actually happened; retrying blindly risks a second "
            "trainer on one run"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "backoff_seconds": self.backoff_seconds,
            "idempotent": sorted(self.idempotent),
        }


def call_with_retry(
    client: RemoteWorkerClient,
    command: str,
    arguments: list[str],
    *,
    policy: ClientRetryPolicy | None = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> Envelope:
    """Invoke, retrying only where repetition is provably harmless."""
    import time

    policy = policy or ClientRetryPolicy()
    if not policy.may_retry(command):
        return client.invoke(command, arguments, timeout=timeout)

    last: Exception | None = None
    for attempt in range(policy.attempts):
        try:
            return client.invoke(command, arguments, timeout=timeout)
        except WorkerUnreachable as exc:
            last = exc
            if attempt + 1 < policy.attempts:
                time.sleep(policy.backoff_seconds * (attempt + 1))
    raise WorkerUnreachable(str(last))


__all__ = [
    "DEFAULT_CANCEL_TIMEOUT",
    "DEFAULT_COMMAND_TIMEOUT",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_PREFLIGHT_TIMEOUT",
    "ClientError",
    "ClientRetryPolicy",
    "LocalWorkerClient",
    "RemoteWorkerClient",
    "SshWorkerClient",
    "WorkerEndpoint",
    "WorkerUnreachable",
    "call_with_retry",
    "check_protocol",
    "safe_identifier",
]
