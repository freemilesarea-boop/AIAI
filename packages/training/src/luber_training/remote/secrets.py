"""Resolving credentials by name, and keeping the values out of everything.

The rule inherited from Phase 25 is that no entity, plan, log or
registry record ever holds a secret value. This module is what makes
that rule liveable: the things that *do* need a key ask for it by
reference at the moment of use, and never write it down.

Two resolvers. `EnvironmentSecretResolver` reads process environment
variables, which is what an operator running a dispatch by hand will
have. `FileSecretResolver` reads a directory of files, which is what a
key on disk looks like — and it checks the file's permissions first,
because an SSH key readable by every account on the box is a key that
has to be assumed compromised.

`redact` is the other half. Errors and logs are written by code that
does not know whether the string it was handed contains a token. Rather
than auditing every call site, values that have been resolved during
this process are registered and scrubbed from any text on its way out.
"""

from __future__ import annotations

import os
import re
import stat
from abc import ABC, abstractmethod
from pathlib import Path

#: What replaces a secret in any text that leaves this process.
REDACTION = "«redacted»"

#: Permission bits that must not be set on a private key file. Group
#: and world access on a key is the condition ssh itself refuses to
#: proceed under, and matching that behaviour is more useful than
#: inventing a laxer rule.
UNSAFE_KEY_MODE = stat.S_IRWXG | stat.S_IRWXO

#: Environment variable names that look like secrets, for the sweep that
#: keeps them out of an environment lock. Matched case-insensitively
#: against the whole name.
SECRET_NAME_PATTERN = re.compile(
    r"(secret|token|password|passwd|api[_-]?key|private[_-]?key|credential|auth)", re.IGNORECASE
)


class SecretError(RuntimeError):
    """Raised when a named secret cannot be resolved safely."""


def valid_reference(name: str) -> str:
    """A secret reference is a name, and names have a shape.

    Constrained because a reference ends up in a filename and in an
    environment variable lookup. A reference containing a slash could
    read a file the operator did not intend to expose.
    """
    if not isinstance(name, str) or not name.strip():
        raise SecretError("a secret reference may not be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise SecretError(
            f"{name!r} is not a valid secret reference; use letters, digits, dot, "
            "dash and underscore only"
        )
    return name


class SecretResolver(ABC):
    """Turns a reference into a value, at the last possible moment."""

    name: str = "abstract"

    @abstractmethod
    def resolve(self, reference: str) -> str:
        """The secret named *reference*, or raise SecretError."""

    @abstractmethod
    def available(self, reference: str) -> bool:
        """Whether *reference* can be resolved, without resolving it.

        Separate from `resolve` so a preflight can report a missing
        credential without the value passing through a check that might
        log its argument.
        """

    def resolve_path(self, reference: str) -> Path:
        """A reference naming a file path rather than a value.

        SSH keys are used by path — ssh reads the file itself, and
        copying the bytes anywhere else, including into a staging
        directory, creates a second copy nobody is tracking.
        """
        raise SecretError(f"{self.name} cannot resolve {reference!r} to a path")


class EnvironmentSecretResolver(SecretResolver):
    """Reads `LUBER_SECRET_<REFERENCE>` from the environment.

    The prefix is deliberate. Without it, a reference could name any
    variable in the process, and `resolve("PATH")` would quietly
    succeed.
    """

    name = "environment"
    prefix = "LUBER_SECRET_"

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else dict(os.environ)

    def _variable(self, reference: str) -> str:
        return self.prefix + valid_reference(reference).upper().replace("-", "_").replace(".", "_")

    def available(self, reference: str) -> bool:
        return bool(self._environ.get(self._variable(reference)))

    def resolve(self, reference: str) -> str:
        variable = self._variable(reference)
        value = self._environ.get(variable)
        if not value:
            raise SecretError(
                f"secret {reference!r} is not set; export {variable} in the operator "
                "shell before dispatching"
            )
        register_secret(value)
        return value


class FileSecretResolver(SecretResolver):
    """Reads secrets from files in one directory, checking permissions.

    Intended for SSH keys, which are files by nature. The directory
    itself must not be inside the repository — a resolver pointed at a
    working tree would be one `git add -A` away from committing a
    private key — and the constructor refuses that outright.
    """

    name = "file"

    def __init__(self, directory: Path, *, repository_root: Path | None = None) -> None:
        self.directory = Path(directory).expanduser()
        if repository_root is not None:
            root = Path(repository_root).resolve()
            here = self.directory.absolute()
            if here == root or root in here.parents:
                raise SecretError(
                    f"{self.directory} is inside the repository at {root}; secrets kept "
                    "in a working tree get committed eventually"
                )

    def _path(self, reference: str) -> Path:
        return self.directory / valid_reference(reference)

    def available(self, reference: str) -> bool:
        return self._path(reference).is_file()

    def _check_permissions(self, path: Path) -> None:
        """Refuse a key any other account on the machine can read.

        Skipped where the OS has no meaningful mode bits. Reporting a
        pass on a filesystem that cannot express permissions would be
        asserting something nobody checked.
        """
        if os.name != "posix":
            return
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise SecretError(f"cannot read permissions of {path}: {exc}") from exc
        if mode & UNSAFE_KEY_MODE:
            raise SecretError(
                f"{path} is readable or writable beyond its owner (mode "
                f"{stat.filemode(mode)}); run chmod 600 on it before using it"
            )

    def resolve(self, reference: str) -> str:
        path = self._path(reference)
        if not path.is_file():
            raise SecretError(f"secret {reference!r} is not present at {path}")
        self._check_permissions(path)
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise SecretError(f"secret {reference!r} at {path} is empty")
        register_secret(value)
        return value

    def resolve_path(self, reference: str) -> Path:
        path = self._path(reference)
        if not path.is_file():
            raise SecretError(f"secret {reference!r} is not present at {path}")
        self._check_permissions(path)
        return path


class NullSecretResolver(SecretResolver):
    """Resolves nothing. The default where no credentials are needed.

    Used by the local worker, which reaches a "remote" machine that is a
    directory. It exists so that code paths requiring a resolver do not
    have to special-case its absence, and so a component that
    unexpectedly asks for a secret fails loudly rather than finding one.
    """

    name = "null"

    def available(self, reference: str) -> bool:
        return False

    def resolve(self, reference: str) -> str:
        raise SecretError(
            f"no secret resolver is configured, so {reference!r} cannot be resolved; "
            "this transport was not expected to need a credential"
        )


# ── redaction ────────────────────────────────────────────────────────

#: Values seen by a resolver in this process. Held so that text written
#: later — an error, a log line, an audit entry — can be scrubbed
#: without every writer having to know what it is handling.
_KNOWN_SECRETS: set[str] = set()

#: Below this length a "secret" is more likely to be a common substring,
#: and blanking it out of unrelated text would corrupt diagnostics.
MIN_REDACTABLE_LENGTH = 6


def register_secret(value: str) -> None:
    if isinstance(value, str) and len(value.strip()) >= MIN_REDACTABLE_LENGTH:
        _KNOWN_SECRETS.add(value.strip())


def forget_secrets() -> None:
    """Drop the registry. For tests, and for long-lived processes."""
    _KNOWN_SECRETS.clear()


def redact(text: str) -> str:
    """Remove any known secret value from *text*.

    Longest first, so a secret that contains another is replaced whole
    rather than leaving a fragment behind.
    """
    if not text:
        return text
    for value in sorted(_KNOWN_SECRETS, key=len, reverse=True):
        if value in text:
            text = text.replace(value, REDACTION)
    return text


def redact_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Redact a structure on its way into a log or a registry record.

    Applies to values by content *and* to keys by name: a field called
    ``api_token`` is replaced whatever it contains, because the field
    name is a better signal than any pattern match on the value.
    """
    cleaned: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(key, str) and SECRET_NAME_PATTERN.search(key) and not key.endswith("_ref"):
            cleaned[key] = REDACTION
        elif isinstance(value, str):
            cleaned[key] = redact(value)
        elif isinstance(value, dict):
            cleaned[key] = redact_mapping(value)
        elif isinstance(value, list):
            cleaned[key] = [redact(item) if isinstance(item, str) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


__all__ = [
    "MIN_REDACTABLE_LENGTH",
    "REDACTION",
    "SECRET_NAME_PATTERN",
    "UNSAFE_KEY_MODE",
    "EnvironmentSecretResolver",
    "FileSecretResolver",
    "NullSecretResolver",
    "SecretError",
    "SecretResolver",
    "forget_secrets",
    "redact",
    "redact_mapping",
    "register_secret",
    "valid_reference",
]
