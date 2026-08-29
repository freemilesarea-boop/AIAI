"""Where a visit came from, decided once and deterministically.

Pure functions over what a browser can tell us: query parameters, a
referrer, a landing path. No database, no request object, no clock — so
the classification can be tested exhaustively and reused anywhere.

Three rules run through this module.

**Explicit beats inferred.** A UTM tag on a BOORDA campaign link is a
statement of intent by whoever built the link. A referrer host is a
guess about what a browser happened to send. When both are present the
tag wins, always.

**Never claim paid traffic without evidence.** `instagram.com` in the
referrer means somebody followed a link on Instagram; it does not mean
an ad was clicked. Only an ad click identifier (`gclid`, `fbclid`, …)
or an explicit `utm_medium` saying so produces a paid medium. The cost
of guessing wrong is an operator believing an ad worked.

**Direct is an absence, not a source.** No campaign and no referrer is
`direct / none`, and that is a fact about our knowledge rather than
about the visitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

#: Query parameters that must never be stored, whatever else happens.
#:
#: A landing URL can carry a password reset token, an OAuth code, a
#: session handle. Analytics has no use for any of it, and a table of
#: landing parameters is a table nobody audits — so the denylist is
#: applied before anything is written and matched on substrings, because
#: the next such parameter will be named something we did not predict.
SENSITIVE_PARAM_FRAGMENTS = frozenset(
    {
        "auth",
        "code",
        "credential",
        "key",
        "password",
        "passwd",
        "pwd",
        "secret",
        "session",
        "signature",
        "token",
    }
)

#: Parameters worth keeping. An allowlist as well as the denylist above:
#: the denylist stops what we can name, and this stops everything else.
CAMPAIGN_PARAMS = (
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
)

#: Ad click identifiers, and the network each implies.
CLICK_IDS: dict[str, tuple[str, str]] = {
    "gclid": ("google", "cpc"),
    "gbraid": ("google", "cpc"),
    "wbraid": ("google", "cpc"),
    "fbclid": ("facebook", "paid_social"),
}

#: Hosts that are BOORDA itself. A visit from one of our own pages is
#: navigation, not acquisition, and must never start a new attribution.
SELF_HOSTS = frozenset({"boorda.kr", "www.boorda.kr", "api.boorda.kr", "localhost"})

DIRECT_SOURCE = "direct"
DIRECT_MEDIUM = "none"

#: Referrer hosts we can classify with confidence, as (source, medium).
#:
#: Medium is the conservative reading in every case. A link from
#: youtube.com is a referral; it is not evidence of a video ad. Search
#: engines get `organic` because arriving from a search results page
#: without a click identifier is, by definition, not the paid result.
_REFERRER_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("google.",), "google", "organic"),
    (("youtube.com", "youtu.be"), "youtube", "referral"),
    (("instagram.com",), "instagram", "social"),
    (("facebook.com", "fb.com", "fb.me", "messenger.com"), "facebook", "social"),
    (("naver.com", "naver.me"), "naver", "organic"),
    (("daum.net", "kakao.com", "kakaocdn.net"), "daum", "organic"),
    (("bing.com",), "bing", "organic"),
    (("duckduckgo.com",), "duckduckgo", "organic"),
    (("x.com", "twitter.com", "t.co"), "x", "social"),
    (("tiktok.com",), "tiktok", "social"),
    (("threads.net", "threads.com"), "threads", "social"),
    (("reddit.com",), "reddit", "social"),
    (("linkedin.com", "lnkd.in"), "linkedin", "social"),
)


@dataclass(frozen=True)
class Attribution:
    """One classified touch. Every field is normalised and bounded."""

    source: str
    medium: str
    campaign: str | None = None
    content: str | None = None
    term: str | None = None

    @property
    def is_direct(self) -> bool:
        """Whether this touch tells us nothing about acquisition.

        The load-bearing property: direct traffic may update when a
        visitor was last seen, and may never overwrite a known source.
        """
        return self.source == DIRECT_SOURCE and self.medium == DIRECT_MEDIUM


DIRECT = Attribution(source=DIRECT_SOURCE, medium=DIRECT_MEDIUM)

#: Column widths in the database. Values are truncated to fit rather
#: than rejected: a campaign name someone typed too long is still worth
#: counting, and a 500-error on a marketing link is a worse outcome.
MAX_VALUE_LENGTH = 120
MAX_PATH_LENGTH = 200


def _clean(value: Any) -> str | None:
    """Normalise one supplied value, or drop it.

    Lowercased and trimmed so `Instagram`, `instagram ` and `INSTAGRAM`
    are one row in the report rather than three. Control characters go:
    these values are rendered in an operator's browser and written to
    logs, and neither should have to cope with a newline in a campaign
    name.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    text = "".join(ch for ch in text if ch.isprintable())
    text = text[:MAX_VALUE_LENGTH].strip()
    return text or None


def is_sensitive_param(name: str) -> bool:
    """Whether a parameter must never be stored.

    Substring matching on purpose. `token`, `access_token`, `id_token`
    and `csrf_token` are all tokens, and enumerating them is a race
    against whoever adds the next one.
    """
    lowered = name.strip().lower()
    return any(fragment in lowered for fragment in SENSITIVE_PARAM_FRAGMENTS)


def sanitise_params(params: dict[str, Any] | None) -> dict[str, str]:
    """The campaign parameters worth keeping, and nothing else.

    Allowlist first, denylist second. Both, because the allowlist is
    what we mean and the denylist is what we are afraid of, and the two
    protect against different mistakes.
    """
    if not params:
        return {}
    kept: dict[str, str] = {}
    for name in (*CAMPAIGN_PARAMS, *CLICK_IDS):
        if name in params and not is_sensitive_param(name):
            value = _clean(params[name])
            if value:
                kept[name] = value
    return kept


def normalise_path(path: str | None) -> str:
    """A landing path with any query string discarded.

    The query is dropped wholesale rather than filtered, because the
    parts of it we want are already extracted into campaign fields and
    the parts we do not want include every secret a URL can carry.
    """
    if not path:
        return "/"
    cleaned = str(path).split("?", 1)[0].split("#", 1)[0].strip()
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    printable = "".join(ch for ch in cleaned if ch.isprintable())
    return printable[:MAX_PATH_LENGTH] or "/"


def referrer_host(referrer: str | None) -> str | None:
    """The bare host of a referrer, or None when there is not one.

    Only the host is kept. A full referring URL is somebody else's page
    address and can itself carry query parameters we have no business
    storing.
    """
    if not referrer:
        return None
    raw = str(referrer).strip()
    if "//" not in raw:
        raw = f"//{raw}"
    try:
        host = (urlsplit(raw).hostname or "").lower()
    except ValueError:
        return None
    return host or None


def is_self_referral(host: str | None) -> bool:
    """Whether a referrer is BOORDA itself, including its subdomains."""
    if not host:
        return False
    return host in SELF_HOSTS or host.endswith(".boorda.kr")


def _from_referrer(host: str) -> Attribution:
    """Classify a referrer host, conservatively.

    An unrecognised host becomes `<host> / referral` rather than being
    forced into a category. "We do not know which bucket this is" is a
    useful answer; a wrong bucket is not.
    """
    for needles, source, medium in _REFERRER_RULES:
        if any(needle in host for needle in needles):
            return Attribution(source=source, medium=medium)
    return Attribution(source=host[:MAX_VALUE_LENGTH], medium="referral")


def classify(
    params: dict[str, Any] | None = None,
    referrer: str | None = None,
) -> Attribution:
    """Decide where one visit came from.

    Order of authority, highest first:

    1. `utm_source` — an explicit statement on a link we built.
    2. An ad click identifier — the network told us, in its own
       parameter, that this was a click on an ad.
    3. The referrer host — an inference, classified conservatively.
    4. Nothing — direct.
    """
    kept = sanitise_params(params)

    utm_source = kept.get("utm_source")
    utm_medium = kept.get("utm_medium")
    campaign = kept.get("utm_campaign")
    content = kept.get("utm_content")
    term = kept.get("utm_term")

    click_source = click_medium = None
    for name, (source, medium) in CLICK_IDS.items():
        if kept.get(name):
            click_source, click_medium = source, medium
            break

    if utm_source:
        # An explicit tag wins outright. Its medium falls back to the ad
        # network's when a click id is present — a link tagged with a
        # source but no medium, arriving with a gclid, is paid search.
        return Attribution(
            source=utm_source,
            medium=utm_medium or click_medium or "referral",
            campaign=campaign,
            content=content,
            term=term,
        )

    if click_source:
        return Attribution(
            source=click_source,
            medium=utm_medium or click_medium or "cpc",
            campaign=campaign,
            content=content,
            term=term,
        )

    host = referrer_host(referrer)
    if host and not is_self_referral(host):
        inferred = _from_referrer(host)
        return Attribution(
            source=inferred.source,
            medium=utm_medium or inferred.medium,
            campaign=campaign,
            content=content,
            term=term,
        )

    # A self-referral or no referrer at all. If a campaign name survived
    # without a source there is still nothing to attribute it to.
    if utm_medium:
        return Attribution(
            source=DIRECT_SOURCE,
            medium=utm_medium,
            campaign=campaign,
            content=content,
            term=term,
        )
    return DIRECT


#: The channel rows the console groups into, in display order.
#:
#: Labels are Korean because the console is; the identifiers underneath
#: are the stable ones. A channel is a (source, medium) pair collapsed
#: into the question an operator actually asks — "did the ads work" is
#: a different question from "did anyone find us on Google".
CHANNEL_RULES: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("direct", "직접 유입", None),
    ("google_ads", "Google 광고", ("cpc", "paid_search", "ppc")),
    ("google_organic", "Google 검색", None),
    ("youtube_ads", "YouTube 광고", ("paid_video", "cpc", "paid_social")),
    ("youtube_organic", "YouTube", None),
    ("instagram_ads", "Instagram 광고", ("paid_social", "cpc")),
    ("instagram_organic", "Instagram", None),
    ("facebook_ads", "Facebook 광고", ("paid_social", "cpc")),
    ("facebook_organic", "Facebook", None),
    ("naver", "네이버", None),
    ("email", "이메일", None),
    ("affiliate", "제휴", None),
    ("referral", "추천 링크", None),
    ("other", "기타", None),
)

CHANNEL_LABELS: dict[str, str] = {key: label for key, label, _ in CHANNEL_RULES}

#: Media that mean somebody paid for the click.
PAID_MEDIA = frozenset({"cpc", "ppc", "paid_search", "paid_social", "paid_video", "paidsearch"})


def channel_of(source: str | None, medium: str | None) -> str:
    """Collapse a source/medium pair into one reportable channel.

    Deterministic and total: every pair lands somewhere, and `other` is
    a real answer rather than a gap.
    """
    src = (source or "").strip().lower()
    med = (medium or "").strip().lower()
    paid = med in PAID_MEDIA

    if not src or src == DIRECT_SOURCE:
        return "email" if med == "email" else "direct"
    if med == "email":
        return "email"
    if med == "affiliate":
        return "affiliate"
    if src == "google":
        return "google_ads" if paid else "google_organic"
    if src == "youtube":
        return "youtube_ads" if paid else "youtube_organic"
    if src in {"instagram", "meta"}:
        return "instagram_ads" if paid else "instagram_organic"
    if src == "facebook":
        return "facebook_ads" if paid else "facebook_organic"
    if src == "naver":
        return "naver"
    if med == "referral":
        return "referral"
    return "other"


__all__ = [
    "CAMPAIGN_PARAMS",
    "CHANNEL_LABELS",
    "CHANNEL_RULES",
    "CLICK_IDS",
    "DIRECT",
    "DIRECT_MEDIUM",
    "DIRECT_SOURCE",
    "MAX_PATH_LENGTH",
    "MAX_VALUE_LENGTH",
    "PAID_MEDIA",
    "SELF_HOSTS",
    "SENSITIVE_PARAM_FRAGMENTS",
    "Attribution",
    "channel_of",
    "classify",
    "is_self_referral",
    "is_sensitive_param",
    "normalise_path",
    "referrer_host",
    "sanitise_params",
]
