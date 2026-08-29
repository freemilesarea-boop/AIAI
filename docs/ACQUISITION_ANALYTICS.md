# Acquisition attribution

First-party only. No Meta Pixel, no GA4, no Google Ads tag, no TikTok
pixel — nothing on the page talks to anyone but BOORDA, and the schema
is shaped so those can be added later without changing it.

## What is collected

Per visit, sent by the page to `POST /v1/acquisition/visit`:

| Field | Example | Why |
| --- | --- | --- |
| campaign parameters | `utm_source=instagram` | which link brought them |
| ad click ids | `gclid`, `fbclid` | evidence a click was on an ad |
| referrer **host** | `instagram.com` | classification when no UTM exists |
| landing path | `/plans` | which page a campaign points at |
| timestamps | first seen, last seen | ordering the touches |

**What is deliberately not collected:** IP address, user agent, screen
or device characteristics, any browser fingerprint, any cross-domain
identifier, and the full landing URL. There is nothing here that
identifies a person and nothing that works on anyone else's site.

## Sensitive parameters

A landing URL can carry a password-reset token or an OAuth code. Two
independent filters run before anything is stored:

* an **allowlist** — only the five `utm_*` parameters and the four ad
  click ids are ever read;
* a **denylist** — any parameter whose name contains `token`, `auth`,
  `code`, `key`, `secret`, `session`, `password`, `pwd`, `credential`
  or `signature` is dropped, matched on substrings so `access_token`
  and `id_token` are caught without being named.

The landing path is stored with its query string removed entirely.

## Visitor identity

A random UUID in a first-party cookie, `boorda_visitor`: `HttpOnly`
(scripts never read it), `SameSite=Lax` (survives the click from a
campaign link), `Secure` in production, 400 days — the longest lifetime
Chrome grants.

It is not derived from anything. Not the IP, not the user agent, not a
fingerprint. It is a number we made up, and it means nothing outside
BOORDA.

## Attribution rules

**First touch** is written once and never changes. Somebody acquired by
a Google search stays acquired by Google, however many campaigns they
click later — otherwise last month's report changes after the fact.

**Last non-direct touch** moves only when an attributable visit
happens. Direct traffic updates when the visitor was last seen and
nothing else: a person who arrives from an Instagram ad, leaves, and
returns by typing the address was still brought here by the ad, and
crediting that return to "direct" is how paid acquisition ends up
looking worthless.

**Signup takes a snapshot.** The visitor row keeps evolving; the
snapshot does not, so historical reporting is stable.

**Paid conversion is derived, never recorded.** It is the earliest
successful row in `billing_payments` for an attributed account — the
same source of truth the revenue dashboard uses. Nothing is written at
payment time, so the verified PayApp path is untouched and a retried
callback cannot double-count a conversion that was never a row.
Renewals are excluded from conversions by construction and remain in
revenue: a renewal is retention, not acquisition.

## What the date range means

Event-period, not cohort:

* 방문자 — session started in the period
* 가입자 — signup happened in the period
* 유료 전환 — *first* successful payment happened in the period
* 매출 — successful payment paid in the period

A visitor acquired in July who pays in August is in July's visitors and
August's conversions. Cohort lifetime value is a different report and is
not pretended at here.

## Existing accounts

Nothing was backfilled. Every account and payment that predates this
has no attribution record and is reported as
`unattributed_users` — never as direct. There is no evidence either
way, and a fabricated source is worse than a missing one because
somebody will budget against it.

## Account deletion

Closing an account is anonymisation; the `users` row survives, so
nothing here can block a closure and nothing needs to cascade.

On closure, `acquisition_visitors.user_id` is cleared: a browser still
carrying the cookie is anonymous again from that moment.

The `acquisition_attributions` snapshot is retained. It records that an
account arrived from a campaign, and once the account is anonymised that
row names nobody — no address, no display name, no credential. Keeping
it means a month's marketing figures do not silently change when
somebody leaves.

## Retention — POLICY_REQUIRED

**BOORDA has no approved retention period for acquisition data, and one
has not been invented here.**

The schema is built to support deletion whenever a period is agreed:
`acquisition_visitors.first_seen_at`, `acquisition_sessions.started_at`
and `acquisition_attributions.created_at` are all indexed, so a
time-based purge is a single ranged delete per table.

What needs deciding, by someone with the authority to decide it:

1. how long visitor and session rows are kept;
2. whether attribution snapshots outlive the accounts they describe;
3. what the privacy policy tells users about the above.

**BOORDA currently has no privacy policy page at all** (`apps/web` has
no `/privacy` route). Publishing one is out of scope for this phase and
is the blocking item before this data is used for anything beyond
internal reporting.

## Excluded traffic

Not recorded as acquisition: `/admin`, `/ops`, `/api` and `/_next`
paths — the console and operator tools are navigation by people already
here, and Next asset requests are not visits. The beacon skips them in
the browser and the endpoint refuses them again on the server.

No bot detection beyond that. A weak fingerprint-based filter would be
worse than none: it would produce numbers that look filtered without
being filtered.
