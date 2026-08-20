"""Artifact transport over SSH, provider-neutral and injection-proof.

Uses the system `ssh` and `scp` rather than a Python SSH library. Two
reasons: they are present on every machine an operator would use, and
their host-key behaviour is the one operators already know how to
configure. Adding a dependency to reimplement that would mean a second
set of trust decisions to get right.

The hazard here is that ssh concatenates its remote arguments and hands
them to a login shell. Every remote path this module builds therefore
goes through `shlex.quote`, and every path is validated for traversal
before that. Both, because they fail differently: quoting stops a path
from becoming syntax, and validation stops a perfectly-quoted path from
being `../../../etc/passwd`.

Host key verification is on and cannot be turned off from here.
`StrictHostKeyChecking=yes`, no `accept-new`, no `-o
UserKnownHostsFile=/dev/null`. First contact is an enrolment step the
operator performs deliberately, once, and the runbook says how.

Resume is file-granular. Before sending anything the transport asks the
far side for digests and skips what matches. Partial files are *not*
resumed from an offset — `scp` cannot do it safely, and claiming
otherwise would invite someone to rely on it for a 40 GB dataset.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from luber_training.remote.client import WorkerEndpoint, WorkerUnreachable
from luber_training.remote.paths import validate_relative
from luber_training.remote.secrets import NullSecretResolver, SecretResolver, redact
from luber_training.remote.transport import (
    PARTIAL_SUFFIX,
    ArtifactTransport,
    IntegrityError,
    RemoteFile,
    TransportError,
)

DEFAULT_SSH_TIMEOUT = 60.0
#: Transfers get their own, much longer budget. A dataset is not a
#: status query, and one global timeout would either abort a real
#: transfer or leave a dead connection hanging for an hour.
DEFAULT_TRANSFER_TIMEOUT = 3600.0

#: How the far side is asked to hash a file. Coreutils and the BSD
#: variant differ; both are tried, in order, and the first that works
#: on a given host is what gets used.
DIGEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("sha256sum", "--"),
    ("shasum", "-a", "256", "--"),
)


class SshArtifactTransport(ArtifactTransport):
    """Files to and from one host, verified at both ends."""

    name = "ssh"
    #: Stated rather than implied. scp restarts a file from the
    #: beginning; nothing here resumes a byte offset.
    supports_byte_range_resume = False

    def __init__(
        self,
        endpoint: WorkerEndpoint,
        remote_root: str,
        *,
        secrets: SecretResolver | None = None,
        connect_timeout: float = 15.0,
        ssh_executable: str = "ssh",
        scp_executable: str = "scp",
    ) -> None:
        if not endpoint.host:
            raise TransportError("an SSH transport needs a host")
        self.endpoint = endpoint
        self.remote_root = remote_root.rstrip("/")
        self.secrets = secrets or NullSecretResolver()
        self.connect_timeout = connect_timeout
        self.ssh_executable = ssh_executable
        self.scp_executable = scp_executable
        self._digest_command: tuple[str, ...] | None = None

    # ── connection ───────────────────────────────────────────────────
    @property
    def target(self) -> str:
        return (
            f"{self.endpoint.user}@{self.endpoint.host}"
            if self.endpoint.user
            else str(self.endpoint.host)
        )

    def _options(self, *, for_scp: bool = False) -> list[str]:
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            f"ConnectTimeout={int(self.connect_timeout)}",
        ]
        if self.endpoint.known_hosts_ref:
            options += [
                "-o",
                f"UserKnownHostsFile={self.secrets.resolve_path(self.endpoint.known_hosts_ref)}",
            ]
        if self.endpoint.ssh_key_ref:
            options += [
                "-i",
                str(self.secrets.resolve_path(self.endpoint.ssh_key_ref)),
                "-o",
                "IdentitiesOnly=yes",
            ]
        if self.endpoint.port:
            # scp spells the port flag differently from ssh. Getting
            # this wrong is a silent connection to the default port.
            options += (
                ["-P", str(int(self.endpoint.port))]
                if for_scp
                else ["-p", str(int(self.endpoint.port))]
            )
        return options

    def _remote_path(self, relative: str) -> str:
        """A validated, quoted path under this transport's root."""
        return shlex.quote(f"{self.remote_root}/{validate_relative(relative)}")

    def _run_ssh(
        self, remote_command: str, *, timeout: float = DEFAULT_SSH_TIMEOUT
    ) -> tuple[int, str, str]:
        argv = [self.ssh_executable, *self._options(), self.target, remote_command]
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerUnreachable(f"{self.target} did not respond within {timeout}s") from exc
        except OSError as exc:
            raise WorkerUnreachable(f"could not reach {self.target}: {exc}") from exc
        if completed.returncode == 255 and not completed.stdout:
            raise WorkerUnreachable(
                f"ssh could not connect to {self.target}: {redact(completed.stderr.strip()[:300])}"
            )
        return completed.returncode, completed.stdout, completed.stderr

    # ── digesting on the far side ────────────────────────────────────
    def _digest_tool(self) -> tuple[str, ...]:
        """Find a working sha256 tool on the remote host, once.

        Cached because it does not change, and because asking on every
        file would double the round trips for a ten-thousand-file
        manifest.
        """
        if self._digest_command is not None:
            return self._digest_command
        for candidate in DIGEST_COMMANDS:
            code, _, _ = self._run_ssh(f"command -v {shlex.quote(candidate[0])}")
            if code == 0:
                self._digest_command = candidate
                return candidate
        raise TransportError(
            f"{self.target} has neither sha256sum nor shasum; artifacts cannot be verified "
            "there, and an unverified transfer is not one this project will perform"
        )

    def _remote_digest(self, relative: str) -> str | None:
        tool = self._digest_tool()
        command = " ".join(shlex.quote(part) for part in tool) + " " + self._remote_path(relative)
        code, stdout, _ = self._run_ssh(command)
        if code != 0 or not stdout.strip():
            return None
        return stdout.split()[0].strip() or None

    # ── the interface ────────────────────────────────────────────────
    def probe(self) -> dict[str, Any]:
        code, stdout, stderr = self._run_ssh(
            f"mkdir -p {shlex.quote(self.remote_root)} && "
            f"df -Pm {shlex.quote(self.remote_root)} | tail -1"
        )
        free_mb: int | None = None
        if code == 0 and stdout.strip():
            parts = stdout.split()
            if len(parts) >= 4:
                try:
                    free_mb = int(parts[3])
                except ValueError:
                    free_mb = None
        return {
            "transport": self.name,
            "target": self.target,
            "remote_root": self.remote_root,
            "reachable": code == 0,
            "free_disk_mb": free_mb,
            "detail": redact(stderr.strip()[:200]) if code != 0 else "",
        }

    def exists(self, relative: str) -> bool:
        code, _, _ = self._run_ssh(f"test -f {self._remote_path(relative)}")
        return code == 0

    def stat(self, relative: str) -> RemoteFile | None:
        path = self._remote_path(relative)
        code, stdout, _ = self._run_ssh(f"test -f {path} && wc -c < {path}")
        if code != 0 or not stdout.strip():
            return None
        try:
            size = int(stdout.strip().split()[0])
        except (ValueError, IndexError):
            return None
        return RemoteFile(path=relative, size_bytes=size, sha256=self._remote_digest(relative))

    def upload(self, local: Path, relative: str, *, expected_sha256: str) -> RemoteFile:
        """Send, verify on the far side, then rename into place.

        The verification happens remotely, on the bytes that landed. A
        local hash proves what was sent; only a remote hash proves what
        arrived, and the difference between those two is exactly the
        failure this guards against.
        """
        local = Path(local)
        if not local.is_file():
            raise TransportError(f"{local} does not exist")

        safe = validate_relative(relative)
        final = f"{self.remote_root}/{safe}"
        partial = final + PARTIAL_SUFFIX
        parent = str(Path(safe).parent)

        directory = f"{self.remote_root}/{parent}" if parent != "." else self.remote_root
        code, _, stderr = self._run_ssh(f"mkdir -p {shlex.quote(directory)}")
        if code != 0:
            raise TransportError(f"could not create the remote directory: {redact(stderr[:200])}")

        argv = [
            self.scp_executable,
            *self._options(for_scp=True),
            str(local),
            f"{self.target}:{partial}",
        ]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=DEFAULT_TRANSFER_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerUnreachable(f"the transfer of {relative} timed out") from exc
        except OSError as exc:
            raise TransportError(f"scp could not run: {exc}") from exc
        if completed.returncode != 0:
            raise TransportError(
                f"could not upload {relative}: {redact(completed.stderr.strip()[:300])}"
            )

        tool = " ".join(shlex.quote(part) for part in self._digest_tool())
        code, stdout, _ = self._run_ssh(f"{tool} {shlex.quote(partial)}")
        actual = stdout.split()[0].strip() if code == 0 and stdout.strip() else None
        if actual != expected_sha256:
            self._run_ssh(f"rm -f {shlex.quote(partial)}")
            raise IntegrityError(
                f"{relative} arrived on {self.target} with digest "
                f"{(actual or 'unreadable')[:12]}, expected {expected_sha256[:12]}"
            )

        code, _, stderr = self._run_ssh(f"mv -f {shlex.quote(partial)} {shlex.quote(final)}")
        if code != 0:
            raise TransportError(
                f"could not finalise {relative} on the worker: {redact(stderr[:200])}"
            )
        return RemoteFile(path=relative, size_bytes=local.stat().st_size, sha256=expected_sha256)

    def download(self, relative: str, local: Path, *, expected_sha256: str | None = None) -> Path:
        import os

        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + PARTIAL_SUFFIX)

        argv = [
            self.scp_executable,
            *self._options(for_scp=True),
            f"{self.target}:{self.remote_root}/{validate_relative(relative)}",
            str(partial),
        ]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=DEFAULT_TRANSFER_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            partial.unlink(missing_ok=True)
            raise WorkerUnreachable(f"the download of {relative} timed out") from exc
        except OSError as exc:
            raise TransportError(f"scp could not run: {exc}") from exc
        if completed.returncode != 0:
            partial.unlink(missing_ok=True)
            raise TransportError(
                f"could not download {relative}: {redact(completed.stderr.strip()[:300])}"
            )

        if expected_sha256 is not None:
            from luber_training.remote.manifest import sha256_file

            actual, _ = sha256_file(partial)
            if actual != expected_sha256:
                partial.unlink(missing_ok=True)
                raise IntegrityError(
                    f"{relative} downloaded with digest {actual[:12]}, expected "
                    f"{expected_sha256[:12]}"
                )

        os.replace(partial, local)
        return local

    def list_files(self, relative_dir: str) -> list[RemoteFile]:
        base = (
            f"{self.remote_root}/{validate_relative(relative_dir)}"
            if relative_dir
            else self.remote_root
        )
        code, stdout, _ = self._run_ssh(
            f"find {shlex.quote(base)} -type f -printf '%s %P\\n' 2>/dev/null || "
            f"find {shlex.quote(base)} -type f -exec stat -f '%z %N' {{}} +"
        )
        if code != 0:
            return []
        files: list[RemoteFile] = []
        prefix = f"{self.remote_root}/"
        for line in stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                size = int(parts[0])
            except ValueError:
                continue
            name = parts[1].strip()
            if name.endswith(PARTIAL_SUFFIX):
                continue
            relative = name[len(prefix) :] if name.startswith(prefix) else name
            if relative_dir and not relative.startswith(relative_dir):
                relative = f"{relative_dir}/{relative}"
            files.append(RemoteFile(path=relative, size_bytes=size))
        return files

    def remove_temp(self) -> list[str]:
        """Delete this transport's partial files. Nothing else, ever."""
        pattern = shlex.quote("*" + PARTIAL_SUFFIX)
        code, stdout, _ = self._run_ssh(
            f"find {shlex.quote(self.remote_root)} -type f -name {pattern} -print -delete"
        )
        if code != 0:
            return []
        prefix = f"{self.remote_root}/"
        return [
            line[len(prefix) :] if line.startswith(prefix) else line
            for line in stdout.splitlines()
            if line.strip()
        ]


__all__ = [
    "DEFAULT_SSH_TIMEOUT",
    "DEFAULT_TRANSFER_TIMEOUT",
    "DIGEST_COMMANDS",
    "SshArtifactTransport",
]
