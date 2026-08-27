# PayApp recurring billing

BOORDA's paid plans are charged through [PayApp](https://docs.payapp.kr/dev_center01.html)
recurring payments. This document is the reasoning; the code is in
`packages/billing` (what a payment *means*), `packages/database`
(where it is stored) and `apps/api/src/luber_api/routes/billing.py`
(the endpoints).

Phase 6 built plans, allowances and entitlement. That architecture was
right and is unchanged — it simply had nothing telling it when a period
had been paid for. Phase 7 supplies that, and nothing else.

## The one thing to understand

**`rebillRegist` succeeding is not a payment.**

PayApp's registration call returns `state=1` when it has accepted a
*request* to set up a recurring payment. At that moment the customer has
not authenticated, no card has been charged, and PayApp has not tried to
charge one. The only fact that grants a paid plan is a validated
`pay_state=4` notification arriving at our feedback endpoint from
PayApp's servers.

Everything else in this document follows from that.

A subscription therefore has explicit states, not a boolean:

```
                    ┌──────────────────────────┐
    checkout ──────▶│ PENDING_INITIAL_PAYMENT  │  grants nothing
                    └───────────┬──────────────┘
                     pay_state=4│
                    ┌───────────▼──────────────┐
                    │         ACTIVE           │  grants the plan
                    └──┬────────────────────┬──┘
        pay_state=99   │                    │  user cancels
                    ┌──▼───────┐      ┌─────▼──────────┐
                    │ PAST_DUE │      │ CANCEL_PENDING │  still granted
                    └──┬───────┘      └─────┬──────────┘  to period end
          later success│                    │ period ends
                    ┌──▼───────┐      ┌─────▼────┐
                    │  ACTIVE  │      │ CANCELED │
                    └──────────┘      └──────────┘
```

`ACTIVE` and `CANCEL_PENDING` are the only states that entitle, **and
only while the paid period contains the current moment**. Both
conditions are required, which is what makes a missed renewal fail
closed: the row stays ACTIVE — that is exactly what reconciliation looks
for — but it grants nothing in the meantime.

## Amounts come from the server, always

The browser sends a plan id and a phone number. That is the entire
request:

```json
{ "plan_id": "pro", "phone": "010-1234-5678" }
```

The server resolves the plan to a price from `luber_schemas.plans`,
stores it on the checkout, sends it to PayApp, and later compares
PayApp's reported amount against it. There is no field anywhere in the
billing API through which a client could suggest a price, a generation
limit, a download entitlement or a `rebill_no`.

Before any entitlement is granted, the reported amount must equal the
plan price exactly — not "at least". An overpayment is refused for the
same reason as an underpayment: a figure that is not the price means
something is wrong, and guessing which is worse than stopping.

A mismatch records an `AMOUNT_MISMATCH` anomaly with **both** numbers and
grants nothing. The reported figure is never overwritten with the
expected one; the difference is the entire signal.

## Idempotency

PayApp documents that feedback may be delivered more than once, and
retries whenever the response is not exactly `SUCCESS` (we register with
`checkretry=y`, up to ten attempts).

Every notification is written to `billing_events` **in the same
transaction as its effect**:

```
insert event  →  apply effect  →  commit  →  answer SUCCESS
```

`UNIQUE(provider, kind, fingerprint)` makes a redelivery collide and
become a no-op. The ordering is what matters:

- Crash before the commit → both halves roll back, PayApp retries, the
  retry applies exactly once.
- Crash after the commit, before the response → PayApp retries, the retry
  collides on the fingerprint and returns `SUCCESS` having done nothing.
- Effect cannot be written → we answer 500, *not* `SUCCESS`. Telling
  PayApp we handled a payment we did not record would mean it never sends
  it again.

There is no ordering of failures that produces two charges or a silent
loss. A design that recorded the event *before* applying the effect
would have one: the crash would leave an event behind, and the retry
would be dismissed as a duplicate.

`billing_payments` carries a second constraint,
`UNIQUE(provider, provider_payment_id)`, so even two independent code
paths cannot write one PayApp payment twice.

Fingerprints are built from provider identifiers only, never from our own
process state, so a replay after a restart still collapses onto the same
row. Failure notifications may arrive without a `mul_no`; those fall back
to subscription + state + day, which means two genuinely distinct
failures for one subscription on one day collapse into one recorded
event. That is the correct trade — the alternative is a fingerprint that
changes on every redelivery and defeats the constraint entirely.

## Authenticating a notification

The feedback and failure endpoints cannot require a session: PayApp's
servers do not have one. Authenticity rests on PayApp's documented rule —
「userid, linkkey, linkval 값을 비교 확인하고 동일한 경우에만」 — all three
compared, all three in constant time.

A comparison that returned early on the first wrong byte would hand the
integration secret to a patient caller one byte at a time, and this
endpoint is reachable by anyone.

A failed check answers 403, never `SUCCESS`, and records an anomaly. If
it really was PayApp — a rotated key, a misconfiguration — we want it to
keep retrying and the anomalies to accumulate until somebody looks. The
endpoint is rate-limited per address, because an open endpoint that
writes an audit row per rejection is also a way to fill the database.

`var1` carries our correlation id. It is a *correlation aid*: it says
which of our rows to look at first and authorises nothing, because anyone
who can post to the endpoint can put anything in it. It is random and
carries no account information — encoding a user id would leak one, and a
guessable value would let a forged notification be aimed at a chosen
account.

Unknown fields are ignored, not rejected. PayApp documents that fields may
be added over time, and a parser that refused a notification over an
unfamiliar key would stop processing real payments the day PayApp shipped
a feature.

## The return URL proves nothing

`/billing/return` is reached by a user who paid, by a user who closed the
PayApp window, and by anyone who types it. The page therefore does not
read the query string at all — the test file mocks `useSearchParams` to
throw, so a change that started reading it fails.

It asks BOORDA's own server what happened and polls for up to two
minutes, because the browser redirect and the server-to-server
notification are independent and either may land first. If the
confirmation has not arrived by then it says so honestly — *아직 결제
확인을 받지 못했습니다*, never *결제에 실패했습니다*, because we do not know
that, and telling someone their payment failed when it did not invites a
second charge.

## Failed renewals

`failurl` receives `pay_state=99` when a recurring approval fails. Three
things must not happen, and each has cost somebody real money:

- no successful payment row
- no allowance period advance
- no change to the period already paid for

The subscription moves to `PAST_DUE`, which grants nothing — a failed
charge is the *absence* of a payment. A later successful charge recovers
it to `ACTIVE`, because a transient card decline must not end the
relationship.

**First-cycle failure is not notified.** PayApp: 「1회차 승인 실패는 Noti되지
않음」. Nothing in the system waits on `failurl` to learn that an initial
payment did not happen; an unpaid checkout resolves by timing out after
24 hours, and reconciliation closes it so the account's single checkout
slot is released.

## Cancellation

`rebillCancel` stops the *next* charge. It does not refund the last one —
PayApp: 「다음 결제 주기에 정기 결제가 발생되지 않습니다」.

So BOORDA preserves the period already paid for. Cancelling moves the
subscription to `CANCEL_PENDING`, which still entitles, and it becomes
`CANCELED` when the period ends. The dialog says exactly this. No refunds
are issued automatically.

The request takes **no arguments**. The account comes from the session and
the `rebill_no` is looked up from our own records, so there is no field
through which a caller could name somebody else's contract.

PayApp is called *before* the local state changes. If PayApp refuses,
nothing local moves — a subscription marked cancelled here while PayApp
still holds a live contract would charge the customer next month with the
product telling them they had cancelled.

## One subscription per account

`uq_one_open_checkout_per_user` is a partial unique index over open
checkout states. Two Subscribe clicks in the same instant cannot become
two recurring contracts: the second insert has nowhere to go. The
disabled button handles the honest double-click; the index handles the
two requests that arrive in the same millisecond, which a button cannot.

An operator-assigned plan (Phase 6's `set_plan` script) does *not* block
checkout. It has no provider contract, so it cannot be double-billed —
the rule exists to prevent two recurring contracts, not two rows.

## Plan changes

Not implemented, deliberately.

Doing one safely means either proration or two live recurring contracts,
and the second is how people get billed twice. V1 says: cancel the
current subscription, then choose a new plan. Payment correctness over
convenience.

## Reconciliation, and its documented limit

Webhook delivery alone is not enough. A notification that is never
delivered leaves no trace anywhere, so the only way to notice is to have
written down what we expected — `subscriptions.next_renewal_at` — and
come back to check.

`scripts/ops/billing_reconcile.py` finds:

| | |
|---|---|
| `MISSING_EXPECTED_RENEWAL` | ACTIVE, period ended, auto-renew on, no renewal payment |
| `ABANDONED_CHECKOUT` | registered with PayApp, never paid, older than 24h |
| `UNRESOLVED_PAST_DUE` | PAST_DUE past its period |

It never grants entitlement, activates a subscription or invents a
payment, and it does not re-flag a gap it has already flagged — a nightly
job that turned one missed renewal into thirty rows would make the
operator queue unreadable, which is the same as having no queue.

**The limit.** PayApp's published API documents `rebillRegist`,
`rebillCancel`, `rebillStop` and `rebillStart`, but **no command that
authoritatively answers "what is the status of this rebill_no"** or
"list the payments taken against it". So reconciliation cannot confirm
from the provider whether a charge happened; it can only notice that we
have no record of one and say so.

This is recorded rather than worked around. Inventing a status-query
endpoint and parsing whatever came back would produce a reconciliation
system that appears authoritative and is not. If PayApp publishes such an
API, `luber_database.billing_reconciliation` is the one module to change.

## Secrets

`PAYAPP_USERID`, `PAYAPP_LINKKEY` and `PAYAPP_LINKVAL` are server-side
only. They never appear in a frontend bundle, an API response, a log line
or a stored event — `redact()` drops them at the parsing boundary, and a
test asserts no billing response contains them.

Billing is off unless all three are set. A half-configured integration is
more dangerous than none: it could register real recurring contracts
while being unable to validate the notifications that confirm them.

`.env.example` holds placeholders. `.env` is gitignored.

## No card data

PayApp sends `card_num` and similar on card payments. They are dropped at
the parsing boundary and there is no column that could hold one. We are
not a card processor, and a masked PAN in a billing table is a compliance
liability with no product use. Settings says so rather than showing an
empty 결제 수단 row.

## Financial history is append-only

`billing_payments` rows are never updated in place and never deleted,
including when a subscription is cancelled. A correction is a new row
with its own event, so the record of what we believed and when survives
the correction — the only way to answer a chargeback honestly.

Losing a subscription is not losing your work: expiry drops the account
to Free and the Library is untouched. Phase 6's download restrictions
then apply.

## Local development

There is no documented PayApp sandbox in the material available, so none
is claimed. The provider client is a Protocol with a deterministic fake
(`luber_billing.payapp.fake`), installed **autouse** across the API
suite — there is no code path in this repository's tests that constructs
a real `HttpPayAppClient`, even with live credentials in the environment.

Callbacks need a publicly reachable URL. `localhost` is not one; testing
them locally requires a tunnel, and there is no way around that.

**No real charge has been made.** The integration is ready for an
operator-supervised real recurring-payment test, which is a decision for
a person, not for this codebase.
