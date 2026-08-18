"""Password hashing and session tokens.

Two kinds of secret pass through here and they need opposite treatment,
which is the reason they live side by side rather than being handled ad
hoc at their call sites.

A **password** is low-entropy and chosen by a human, so the only defence
against an offline attack on a stolen database is making each guess
expensive. Argon2id, via ``argon2-cffi``, with the library's own
parameter defaults and its own salt handling. Nothing here invents
crypto, generates salts, or encodes parameters by hand.

A **session token** is 256 bits from the OS entropy source. Guessing it
is not a threat model, so it needs no slow hash — but a database leak
should not hand the attacker live sessions, so the raw token never
reaches storage. The server keeps SHA-256 of it and compares digests.
Fast is correct here for the same reason slow is correct above.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

#: Bytes of randomness in a session token. 32 bytes is 256 bits, which
#: is far beyond brute force and keeps the cookie a reasonable length.
SESSION_TOKEN_BYTES = 32

#: Passwords are not truncated. A user who types a passphrase gets the
#: whole passphrase hashed; silently keeping the first N characters
#: would make a long password weaker than the user believes it to be.
#: The ceiling exists only so a multi-megabyte body cannot be turned
#: into a denial of service against a deliberately slow hash.
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 1024

_hasher = PasswordHasher()


class PasswordPolicyError(ValueError):
    """A password the policy will not accept, with a usable reason."""


def validate_password(password: str) -> str:
    """Check a candidate password. Length only, deliberately.

    No composition rules. Requiring an uppercase letter and a symbol
    reliably produces ``Password1!`` and pushes users toward patterns an
    attacker already knows; length is the property that actually costs a
    guesser anything. Spaces and any Unicode are allowed so passphrases
    work.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
    return password


def hash_password(password: str) -> str:
    """Argon2id hash, salt and parameters encoded in the returned string."""
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Whether the password matches. Never raises on a bad password.

    A malformed stored hash returns False rather than propagating: a
    corrupt row must fail the login, not the request.
    """
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, InvalidHashError, ValueError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    """True when the hash predates the current parameters.

    Argon2 parameters are expected to increase over time. Because the
    plaintext is available at exactly one moment — a successful login —
    that is the only opportunity to upgrade a hash without asking the
    user for anything.
    """
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except (InvalidHashError, ValueError):
        return False


def generate_session_token() -> str:
    """A fresh opaque token. URL-safe, so it is cookie-safe."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_session_token(token: str) -> str:
    """What the server stores in place of the token itself.

    SHA-256 is the right primitive here precisely because it is fast:
    the input is 256 bits of OS randomness, so there is no dictionary to
    run and nothing for a slow hash to buy. What this does buy is that a
    dump of the sessions table contains no usable credentials.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(candidate_digest: str, stored_digest: str) -> bool:
    """Constant-time digest comparison."""
    return hmac.compare_digest(candidate_digest, stored_digest)


def normalise_email(email: str) -> str:
    """Trim and lower-case, and nothing more.

    Deliberately conservative. Provider-specific rewriting — stripping
    Gmail dots, cutting ``+`` tags — silently merges addresses that
    their owners consider distinct, and gets it wrong for every provider
    that treats them literally. Case folding the whole address is safe
    in practice: the local part is technically case-sensitive per RFC
    5321, but no mainstream provider treats it so, and the alternative
    is a user unable to log in because they capitalised their own name.
    """
    return email.strip().lower()
