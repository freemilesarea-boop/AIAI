"""Classifying where a visit came from.

The rules this pins down are the ones that quietly ruin an acquisition
report when they are wrong: claiming paid traffic without evidence,
letting a self-referral start a new source, and storing a secret that
happened to be in a landing URL.
"""

from __future__ import annotations

from luber_schemas.acquisition import (
    DIRECT,
    Attribution,
    channel_of,
    classify,
    is_self_referral,
    is_sensitive_param,
    normalise_path,
    referrer_host,
    sanitise_params,
)

# ── sanitisation ─────────────────────────────────────────────────────


def test_sensitive_parameters_are_never_stored() -> None:
    """A landing URL can carry a password reset token or an OAuth code.

    Analytics has no use for any of it, and a table nobody audits is
    exactly where such a value should not end up.
    """
    for name in (
        "token",
        "access_token",
        "id_token",
        "csrf_token",
        "code",
        "auth",
        "authorization",
        "password",
        "pwd",
        "session",
        "session_id",
        "api_key",
        "secret",
        "client_secret",
        "signature",
    ):
        assert is_sensitive_param(name), name
        assert is_sensitive_param(name.upper()), name


def test_only_campaign_parameters_survive() -> None:
    """Allowlist and denylist both, because they catch different bugs."""
    kept = sanitise_params(
        {
            "utm_source": "instagram",
            "utm_campaign": "summer",
            "gclid": "abc123",
            "access_token": "secret-value",
            "email": "someone@example.com",
            "ref": "whatever",
        }
    )

    assert kept == {"utm_source": "instagram", "utm_campaign": "summer", "gclid": "abc123"}
    assert "secret-value" not in str(kept)
    assert "someone@example.com" not in str(kept)


def test_values_are_normalised_so_one_campaign_is_one_row() -> None:
    kept = sanitise_params({"utm_source": "  Instagram  ", "utm_medium": "PAID_SOCIAL"})

    assert kept["utm_source"] == "instagram"
    assert kept["utm_medium"] == "paid_social"


def test_a_landing_path_keeps_nothing_of_its_query() -> None:
    """The parts worth keeping are already extracted into campaign
    fields; the rest includes every secret a URL can carry."""
    assert normalise_path("/plans?utm_source=x&token=secret") == "/plans"
    assert normalise_path("/song/123#t=30") == "/song/123"
    assert normalise_path(None) == "/"
    assert normalise_path("plans") == "/plans"


def test_an_overlong_value_is_truncated_rather_than_rejected() -> None:
    """A marketing link that 500s is worse than a truncated label."""
    kept = sanitise_params({"utm_campaign": "x" * 500})

    assert len(kept["utm_campaign"]) == 120


# ── referrers ────────────────────────────────────────────────────────


def test_only_the_referrer_host_is_kept() -> None:
    assert referrer_host("https://www.google.com/search?q=secret+thing") == "www.google.com"
    assert referrer_host("instagram.com") == "instagram.com"
    assert referrer_host(None) is None
    assert referrer_host("") is None


def test_boorda_never_refers_itself() -> None:
    """Navigation inside the product is not acquisition."""
    for host in ("boorda.kr", "www.boorda.kr", "api.boorda.kr", "anything.boorda.kr"):
        assert is_self_referral(host), host
    assert not is_self_referral("instagram.com")


def test_a_self_referral_stays_direct() -> None:
    assert classify(None, "https://boorda.kr/plans") == DIRECT


# ── classification ───────────────────────────────────────────────────


def test_no_campaign_and_no_referrer_is_direct() -> None:
    """Direct is an absence — a fact about our knowledge."""
    assert classify(None, None) == DIRECT
    assert DIRECT.is_direct


def test_a_search_engine_referrer_is_organic_not_paid() -> None:
    """Arriving from a results page without a click id is, by
    definition, not the paid result."""
    assert classify(None, "https://www.google.com/search?q=ai+music") == Attribution(
        source="google", medium="organic"
    )
    assert classify(None, "https://search.naver.com/search.naver?query=x").source == "naver"


def test_a_social_referrer_is_social_not_an_ad() -> None:
    """instagram.com means somebody followed a link. It is not evidence
    that an ad was clicked, and claiming otherwise makes an operator
    believe a campaign worked."""
    assert classify(None, "https://www.instagram.com/") == Attribution(
        source="instagram", medium="social"
    )
    assert classify(None, "https://www.youtube.com/watch?v=x") == Attribution(
        source="youtube", medium="referral"
    )
    assert classify(None, "https://m.facebook.com/").source == "facebook"


def test_an_unknown_host_is_a_referral_rather_than_a_guess() -> None:
    """ "We do not know which bucket" is a useful answer."""
    assert classify(None, "https://some-blog.example/post") == Attribution(
        source="some-blog.example", medium="referral"
    )


def test_an_explicit_utm_beats_the_referrer() -> None:
    """A tag on a link we built is a statement of intent; a referrer is
    a guess about what the browser happened to send."""
    result = classify(
        {"utm_source": "instagram", "utm_medium": "paid_social", "utm_campaign": "summer_launch"},
        "https://www.google.com/search?q=x",
    )

    assert result == Attribution(source="instagram", medium="paid_social", campaign="summer_launch")


def test_an_ad_click_id_is_evidence_of_an_ad() -> None:
    assert classify({"gclid": "abc"}, None) == Attribution(source="google", medium="cpc")
    assert classify({"wbraid": "abc"}, None).source == "google"
    assert classify({"fbclid": "abc"}, None) == Attribution(source="facebook", medium="paid_social")


def test_an_ad_click_id_supplies_the_medium_a_tag_omitted() -> None:
    """A link tagged with a source but no medium, arriving with a gclid,
    is paid search."""
    result = classify({"utm_source": "google", "gclid": "abc"}, None)

    assert result.source == "google"
    assert result.medium == "cpc"


def test_content_and_term_survive_when_supplied() -> None:
    result = classify(
        {
            "utm_source": "instagram",
            "utm_medium": "paid_social",
            "utm_campaign": "summer",
            "utm_content": "reel_01",
            "utm_term": "ai music",
        }
    )

    assert result.content == "reel_01"
    assert result.term == "ai music"


# ── channels ─────────────────────────────────────────────────────────


def test_paid_and_organic_are_different_channels() -> None:
    """The whole reason the split exists: "did the ads work" is a
    different question from "did anyone find us"."""
    assert channel_of("google", "cpc") == "google_ads"
    assert channel_of("google", "organic") == "google_organic"
    assert channel_of("instagram", "paid_social") == "instagram_ads"
    assert channel_of("instagram", "social") == "instagram_organic"
    assert channel_of("youtube", "paid_video") == "youtube_ads"
    assert channel_of("youtube", "referral") == "youtube_organic"


def test_every_pair_lands_somewhere() -> None:
    """Total by construction; `other` is an answer, not a gap."""
    assert channel_of("direct", "none") == "direct"
    assert channel_of(None, None) == "direct"
    assert channel_of("naver", "organic") == "naver"
    assert channel_of("mailchimp", "email") == "email"
    assert channel_of("partner", "affiliate") == "affiliate"
    assert channel_of("some-blog.example", "referral") == "referral"
    assert channel_of("whatever", "unrecognised") == "other"


def test_meta_is_read_as_instagram_for_paid_social() -> None:
    assert channel_of("meta", "paid_social") == "instagram_ads"
