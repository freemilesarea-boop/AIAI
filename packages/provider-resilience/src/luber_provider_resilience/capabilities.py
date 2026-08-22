"""What a request needs, what a provider offers, and whether they match.

This module exists to make one thing impossible: quietly delivering a
different song than the one somebody asked for.

Failover is attractive precisely when things are going badly, which is
exactly when "just send it to the other provider" is most tempting and
most wrong. If the fallback cannot carry the reference track, the
lyrics, the requested length or the task itself, then routing there does
not rescue the request — it answers a different question and reports
success. The user gets a song they did not ask for and no indication
that anything happened.

So the rule is: **failover is permitted only when every element of the
request survives it.** Where it cannot, the request fails, and the
failure says which capability was missing.

The requirements are read off the request rather than declared
separately, and the capabilities are read off the provider objects the
repository already defines — `supports_reference_audio`,
`supports_edit(kind)`, `supports_audio_to_audio()`. A second declaration
somebody has to keep in sync would drift, and it would drift silently in
the direction of claiming more than is true.

Note the shapes differ: the first is a **property** on the generation
client's ABC and the others are methods. `_asks` handles both, because
insisting on one of them is how a real provider comes back unable to
take a reference track it has always accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Capability(StrEnum):
    """What a provider can be asked to do.

    Named for request shapes rather than for implementation details,
    because the question being answered is "can this provider serve this
    request", not "what classes does it inherit".
    """

    TEXT_TO_MUSIC = "TEXT_TO_MUSIC"
    #: Text-to-music steered by a user-supplied reference track.
    REFERENCE_CONDITIONED = "REFERENCE_CONDITIONED"
    EXTEND = "EXTEND"
    REPLACE_RANGE = "REPLACE_RANGE"
    COVER = "COVER"
    #: Explicit musical controls: BPM, key, time signature.
    MUSICAL_CONTROLS = "MUSICAL_CONTROLS"
    #: Lyrics carried through to the model rather than dropped.
    LYRICS = "LYRICS"
    #: The instrumental / vocal distinction honoured.
    VOCAL_CONTROL = "VOCAL_CONTROL"


#: Task type → the capability that task fundamentally requires.
TASK_CAPABILITY: dict[str, str] = {
    "TEXT_TO_MUSIC": Capability.TEXT_TO_MUSIC.value,
    "REFERENCE_CONDITIONED": Capability.REFERENCE_CONDITIONED.value,
    "EXTEND": Capability.EXTEND.value,
    "REPLACE_RANGE": Capability.REPLACE_RANGE.value,
    "COVER": Capability.COVER.value,
}


@dataclass(frozen=True)
class ProviderProfile:
    """What one provider can do, and how it is identified.

    Built by `profile_provider` from the provider object itself rather
    than from configuration. A profile somebody wrote by hand would
    eventually claim a capability the provider had lost.
    """

    name: str
    capabilities: frozenset[str]
    revision: str = "UNKNOWN"
    #: Requested durations this provider will accept, in seconds.
    #: ``None`` on either side means "no bound this provider declares" —
    #: never "any duration is fine", which is why the check treats an
    #: unknown bound as unconstrained rather than as satisfied.
    minimum_duration_seconds: float | None = None
    maximum_duration_seconds: float | None = None

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "capabilities": sorted(self.capabilities),
            "minimum_duration_seconds": self.minimum_duration_seconds,
            "maximum_duration_seconds": self.maximum_duration_seconds,
        }


@dataclass(frozen=True)
class RequestNeeds:
    """Everything about a request that failover must preserve.

    Assembled once, at the point the request is understood, and carried
    unchanged through routing. A router that re-derived needs per
    candidate provider could derive them differently for the second one.
    """

    task_type: str
    duration_seconds: float | None = None
    has_reference: bool = False
    has_lyrics: bool = False
    instrumental: bool | None = None
    has_musical_controls: bool = False
    #: Set when the caller named a provider explicitly. Failover away
    #: from an explicit choice is refused unless separately permitted.
    requested_provider: str | None = None

    def required(self) -> frozenset[str]:
        """Capabilities a provider must have to serve this request."""
        needed: set[str] = set()
        task = TASK_CAPABILITY.get(self.task_type)
        if task is not None:
            needed.add(task)
        else:
            # A task type this build does not recognise is not silently
            # treated as text-to-music. Refusing is the conservative
            # reading; guessing would route a request nobody understood.
            needed.add(f"UNKNOWN_TASK:{self.task_type}")
        if self.has_reference:
            needed.add(Capability.REFERENCE_CONDITIONED.value)
        if self.has_lyrics:
            needed.add(Capability.LYRICS.value)
        if self.instrumental is not None:
            needed.add(Capability.VOCAL_CONTROL.value)
        if self.has_musical_controls:
            needed.add(Capability.MUSICAL_CONTROLS.value)
        return frozenset(needed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "duration_seconds": self.duration_seconds,
            "has_reference": self.has_reference,
            "has_lyrics": self.has_lyrics,
            "instrumental": self.instrumental,
            "has_musical_controls": self.has_musical_controls,
            "requested_provider": self.requested_provider,
        }


@dataclass(frozen=True)
class EquivalenceVerdict:
    """Whether a provider can serve a request without changing it."""

    equivalent: bool
    missing: tuple[str, ...] = ()
    reasons: tuple[str, ...] = field(default=())

    def explain(self) -> str:
        if self.equivalent:
            return "the provider can represent this request unchanged"
        return "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "missing": list(self.missing),
            "reasons": list(self.reasons),
            "explanation": self.explain(),
        }


def check_equivalence(profile: ProviderProfile, needs: RequestNeeds) -> EquivalenceVerdict:
    """Can *profile* serve *needs* without dropping anything?

    Every missing capability is collected rather than returning on the
    first: an operator reading why a failover was refused wants the
    whole answer, not the first thing checked.
    """
    missing = sorted(needs.required() - profile.capabilities)
    reasons: list[str] = []

    for capability in missing:
        if capability.startswith("UNKNOWN_TASK:"):
            reasons.append(
                f"{profile.name} was not asked to serve {capability.split(':', 1)[1]!r}: "
                "this build does not know that task type"
            )
        else:
            reasons.append(f"{profile.name} does not support {capability}")

    duration = needs.duration_seconds
    if duration is not None:
        low = profile.minimum_duration_seconds
        high = profile.maximum_duration_seconds
        if low is not None and duration < low:
            reasons.append(
                f"{profile.name} does not accept {duration:.0f}s: its minimum is {low:.0f}s"
            )
        if high is not None and duration > high:
            reasons.append(
                f"{profile.name} does not accept {duration:.0f}s: its maximum is {high:.0f}s"
            )

    return EquivalenceVerdict(
        equivalent=not reasons,
        missing=tuple(missing),
        reasons=tuple(reasons),
    )


def profile_provider(
    provider: Any,
    *,
    name: str,
    revision: str = "UNKNOWN",
    minimum_duration_seconds: float | None = None,
    maximum_duration_seconds: float | None = None,
) -> ProviderProfile:
    """Read a provider's capabilities off the object itself.

    Deliberately duck-typed against the generation client's protocols
    rather than importing them: this package must not depend on that
    one. What it costs is a `getattr` per question; what it buys is a
    seam that points in one direction only.

    A capability is claimed only when the provider positively says so. A
    method that is missing, or that raises, means the capability is
    absent — a provider that cannot answer "do you support this" is not
    one to route a request to on the strength of an assumption.
    """
    capabilities: set[str] = set()

    # Every provider in this system can do text-to-music; the ABC's one
    # abstract method is `generate`. Lyrics, vocal control and musical
    # controls travel in that same request model, so a provider that can
    # generate can carry them.
    if callable(getattr(provider, "generate", None)):
        capabilities.update(
            {
                Capability.TEXT_TO_MUSIC.value,
                Capability.LYRICS.value,
                Capability.VOCAL_CONTROL.value,
                Capability.MUSICAL_CONTROLS.value,
            }
        )

    if _asks(provider, "supports_reference_audio"):
        capabilities.add(Capability.REFERENCE_CONDITIONED.value)

    if _asks(provider, "supports_audio_to_audio"):
        capabilities.add(Capability.COVER.value)

    supports_edit = getattr(provider, "supports_edit", None)
    if callable(supports_edit):
        for kind, capability in (
            ("EXTEND", Capability.EXTEND.value),
            ("REPLACE_RANGE", Capability.REPLACE_RANGE.value),
        ):
            if _asks_edit(supports_edit, kind):
                capabilities.add(capability)

    return ProviderProfile(
        name=name,
        capabilities=frozenset(capabilities),
        revision=revision,
        minimum_duration_seconds=minimum_duration_seconds,
        maximum_duration_seconds=maximum_duration_seconds,
    )


def _asks(provider: Any, method: str) -> bool:
    """Ask a provider a yes/no question, however it chose to expose it.

    Both shapes are accepted, and that is not tolerance for sloppiness:
    the generation client's own protocol declares
    `supports_reference_audio` as a **property** and
    `supports_audio_to_audio()` as a **method**. A profiler that
    insisted on callables read the property as "not a method, so no",
    and every real provider came back unable to take a reference track
    — a capability silently lost, which is the exact failure this phase
    exists to prevent.
    """
    probe = getattr(provider, method, None)
    if probe is None:
        return False
    if isinstance(probe, bool):
        # A property; `getattr` already evaluated it.
        return probe
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        # A provider that raises when asked what it can do has not said
        # yes. Treating an exception as a capability would route a
        # request on the strength of a bug.
        return False


def _asks_edit(probe: Any, kind: str) -> bool:
    """`supports_edit` takes an enum this package cannot import.

    Tried by value first, then by any enum member whose value matches.
    A provider that will not answer for a kind does not support it.
    """
    for candidate in _edit_kind_candidates(probe, kind):
        try:
            if bool(probe(candidate)):
                return True
        except Exception:
            continue
    return False


def _edit_kind_candidates(probe: Any, kind: str) -> list[Any]:
    """The values worth trying as an edit kind.

    The provider's own enum member if it can be found through the
    function's annotations, then the bare string — which works because
    the repository's `AudioEditKind` is a `StrEnum` and compares equal
    to its value.
    """
    candidates: list[Any] = []
    annotations = getattr(probe, "__annotations__", {}) or {}
    for annotation in annotations.values():
        member = getattr(annotation, kind, None)
        if member is not None:
            candidates.append(member)
    candidates.append(kind)
    return candidates


__all__ = [
    "TASK_CAPABILITY",
    "Capability",
    "EquivalenceVerdict",
    "ProviderProfile",
    "RequestNeeds",
    "check_equivalence",
    "profile_provider",
]
