"""Provider-side errors mapped to standard platform error codes."""

from __future__ import annotations

from luber_schemas import ErrorCode


class GenerationProviderError(Exception):
    """A provider failed to produce audio.

    Carries a standard :class:`ErrorCode` so the service layer can
    persist a machine-readable failure without leaking raw exception
    strings to clients.
    """

    def __init__(self, message: str, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.error_code = error_code
