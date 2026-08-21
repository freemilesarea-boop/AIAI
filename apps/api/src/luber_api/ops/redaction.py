"""Keeping credentials out of the browser, on the server side.

Step 34 is one sentence — *do not rely on UI redaction* — and it is the
whole design here. A React component that hides a token has already
received it: the value is in the JSON payload, in the browser's memory,
in the devtools network tab and in any error reporter the page loads. So
redaction happens before the response is built, and the UI is never
asked to be trusted with something it should not have.

Two layers, because they catch different things.

**Values this process knows are secret.** Phase 27's resolver registers
every secret it resolves, and :func:`luber_training.remote.secrets.redact`
removes those exact strings. In the API process that set is usually
empty — the console resolves no secrets — so this layer is defence
against a future in which it does, not the working one today.

**Shapes that are secret whatever their value.** A private key block, an
``Authorization`` header, a ``token=`` assignment: these are recognisable
without knowing the value, and they are what actually turns up in a
trainer log that echoed its environment. This is the layer that does the
work.

The patterns are deliberately conservative about *length*. Blanking every
six-character run of hex out of a stack trace would corrupt the
diagnostic an operator came to read, and a log nobody can read is its own
kind of failure.
"""

from __future__ import annotations

import re
from typing import Any

from luber_training.remote.secrets import REDACTION, SECRET_NAME_PATTERN, redact

#: A PEM block, from its header to its footer. Matched across lines
#: because that is the only way a key is ever written.
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

#: An OpenSSH public key line. Not secret in itself, but it identifies a
#: host's trust configuration and there is no reason for it to reach a
#: browser.
_SSH_KEY_LINE = re.compile(r"\b(ssh-(?:rsa|ed25519|dss)|ecdsa-sha2-\S+)\s+[A-Za-z0-9+/=]{20,}")

#: ``name=value`` or ``name: value`` where the name looks like a secret.
#: The value is taken up to whitespace, a quote or a comma, so a log line
#: that continues after the token keeps everything else.
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Za-z0-9_.\-]*"
    r"(?:secret|token|password|passwd|api[_-]?key|private[_-]?key|credential|auth)"
    r"[A-Za-z0-9_.\-]*)"
    r"(\s*[:=]\s*)"
    r"([\"']?)([^\s\"',;]{4,})\3"
)

#: An HTTP authorization header, scheme preserved so the diagnostic
#: still says *what kind* of credential was used.
_BEARER = re.compile(r"(?i)\b(bearer|basic|token)\s+([A-Za-z0-9._~+/=\-]{8,})")

#: A URL with credentials in it. Both parts go: a username in a
#: connection string is half of a credential.
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)([^\s/@:]+):([^\s/@]+)@")


#: Runs of the redaction marker, collapsed to one.
_REPEATED_REDACTION = re.compile(rf"(?:{re.escape(REDACTION)}\s+)+{re.escape(REDACTION)}")


def redact_text(text: str) -> str:
    """Strip credential-shaped content from free text.

    Applied to every log line and every error message that leaves this
    process. Order matters only in that the known-value pass runs first,
    so a registered secret is replaced whole rather than being partly
    caught by a shape rule.
    """
    if not text:
        return text
    cleaned = redact(text)
    cleaned = _PEM_BLOCK.sub(f"{REDACTION} (private key)", cleaned)
    cleaned = _SSH_KEY_LINE.sub(rf"\1 {REDACTION}", cleaned)
    cleaned = _URL_CREDENTIALS.sub(rf"\1{REDACTION}:{REDACTION}@", cleaned)
    cleaned = _BEARER.sub(rf"\1 {REDACTION}", cleaned)
    cleaned = _ASSIGNMENT.sub(rf"\1\2{REDACTION}", cleaned)
    # Two rules can both fire on one value — `Authorization: Bearer x`
    # is matched as a bearer token and as a secret-named assignment —
    # and "«redacted» «redacted»" reads as two secrets where there was
    # one.
    return _REPEATED_REDACTION.sub(REDACTION, cleaned)


#: Key names whose value is credential-adjacent even though the value
#: itself is only a *reference*. Phase 25 is right that a reference is a
#: name and never a value — but the name of an operator's SSH key is
#: still a fact about their infrastructure, and the console reports that
#: a credential is configured as a boolean rather than naming it.
#:
#: Broader than `SECRET_NAME_PATTERN`, which does not match `ssh_key_ref`
#: (it looks for `private_key` and `api_key`, not a bare `key`). That gap
#: is exactly how a key reference reached a browser in a reproducibility
#: bundle, so this pattern exists to close it rather than to duplicate
#: the other one.
_CREDENTIAL_KEY = re.compile(r"(?i)(ssh|known_hosts|credential|key_ref|keyfile|passphrase)")

#: Key names whose value is a filesystem path on the machine the console
#: runs on. An operator does not need the deployment's directory layout
#: in their browser, and a path pasted from a console into an issue is
#: how a home directory ends up in a bug tracker. The last component is
#: kept, because *which* checkpoint a reference points at is the part
#: that carries meaning.
_PATH_KEYS: frozenset[str] = frozenset(
    {
        "artifacts_root",
        "audio_root",
        "cache_root",
        "checkpoint_root",
        "code_root",
        "curation_dir",
        "data_root",
        "dataset_dir",
        "output_directory",
        "path",
        "reference",
        "repository_root",
        "run_root",
        "source_path",
        "staging_dir",
        "trainer_root",
        "worker_root",
    }
)


def _shorten_path(value: str) -> str:
    """Keep the identifying tail of a path and drop the machine's layout."""
    if not value:
        return value
    if "://" in value:
        scheme, _, remainder = value.partition("://")
        tail = remainder.rsplit("/", 1)[-1]
        return f"{scheme}://…/{tail}" if tail else value
    if "/" not in value:
        return value
    return "…/" + value.rstrip("/").rsplit("/", 1)[-1]


def redact_document(payload: Any) -> Any:
    """Redact a JSON document on its way to the browser.

    Recurses through lists and objects. A key whose *name* looks like a
    secret is blanked whatever it contains — the name is a better signal
    than any inspection of the value — and a key whose value is a path on
    this machine keeps only its last component.

    Applied to documents this layer does not define the shape of: an
    environment lock, an audit event's metadata, a reproducibility
    bundle. The typed response models need none of it, because none of
    them has a field a credential could occupy.
    """
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(key, str) and (
                SECRET_NAME_PATTERN.search(key) or _CREDENTIAL_KEY.search(key)
            ):
                cleaned[key] = REDACTION
            elif isinstance(key, str) and key in _PATH_KEYS and isinstance(value, str):
                cleaned[key] = _shorten_path(value)
            else:
                cleaned[key] = redact_document(value)
        return cleaned
    if isinstance(payload, list):
        return [redact_document(item) for item in payload]
    if isinstance(payload, str):
        return redact_text(payload)
    return payload


__all__ = ["REDACTION", "redact_document", "redact_text"]
