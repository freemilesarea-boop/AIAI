# Plans and the monthly allowance

BOORDA V1 has four plans and one thing to enforce: how many songs an
account may finish in a period, and whether it may download them.

This document is about the decisions, not the code. The code is in
`packages/schemas/src/luber_schemas/plans.py` (the definition),
`packages/database/src/luber_database/allowance_repository.py` (the
ledger), and the two places the API reads them —
`apps/api/src/luber_api/routes/generations.py` and
`apps/api/src/luber_api/routes/plans.py`.

## The tiers

| | Free | Basic | Pro | Creator |
|---|---|---|---|---|
| Monthly price | ₩0 | ₩19,900 | ₩29,900 | ₩49,900 |
| Songs per period | 20 | 200 | 500 | 1,000 |
| MP3 / WAV download | — | ✓ | ✓ | ✓ |
| Commercial use | — | ✓ | ✓ | ✓ |

Every figure above is read from one file. Nothing else in the repository
— no component, no test, no fixture used by product code — states a
price or a limit of its own. A number duplicated into a component is a
number that disagrees with the server the first time it changes, and the
disagreement is always in someone's favour.

`priority_level` and `lab_access` are configured but unused. They exist
so the values are in one place when a scheduler or LAB gate eventually
reads them; nothing enforces them today, and the pricing page does not
advertise them.

## Songs, not credits

The unit is a song. Users do not convert between a currency we invented
and the thing they wanted.

Internally each generation carries a *cost* (`STANDARD_GENERATION_COST`,
currently 1) rather than the counter being a bare tally, so a model that
genuinely costs more can charge more without a migration. For V1 every
generation costs exactly one.

## Failed generations are never charged

A slot is taken **before** the job is queued and settled when the
generation finishes:

```
reserve(generation_id)   before enqueue
consume(generation_id)   on completion   → the slot is spent
release(generation_id)   on failure      → the slot is returned
```

Reserving after the work would mean the limit could be exceeded by
exactly the number of requests racing at that moment. Reserving before it
means a queue outage happens with the reservation already held — so
settlement lives inside `GenerationRepository.mark_completed` and
`mark_failed`, which every failure path already calls. No route has to
remember to refund, because no route does the refunding.

A reservation that is never settled keeps its slot. That is the correct
bias: a generation still running has genuinely spent the allowance, and a
crashed worker's row stays visible in the ledger rather than being
silently refunded.

## Why a slot index

Counting usage and then inserting is a race. Ten concurrent requests all
read 199, all insert, and the account gets 209 songs.

`SELECT … FOR UPDATE` on the subscription row would close it on
PostgreSQL and quietly do nothing on SQLite, which is what the test suite
runs on — so the bug would ship green.

Instead the slot is part of the key:

```sql
UNIQUE (user_id, period_start, slot_index)
```

Two requests racing for slot 199 cannot both hold it. The loser rolls
back, recounts, and tries the next slot; the loop ends when the next slot
would be past the limit. The invariant is held by the schema, where
forgetting it is not possible.

`packages/database/tests/test_allowance.py` runs ten concurrent
reservations against one remaining slot and asserts exactly one winner.
That test uses a file-backed SQLite database rather than the shared
in-memory one: `StaticPool` hands every session the same connection, so
one session's rollback tears down the others and the race never actually
happens.

The next slot index is one past the *highest ever issued*, including
released ones — not the used count. A release lowers the spend but does
not vacate its index, and reusing that index would collide with the rows
above it forever. Spend governs the limit; the index only has to be
unique.

## Periods

Calendar months in UTC, unless the subscription row carries its own
bounds that still contain the current moment — that is where a billing
provider's cycle will land when one exists.

The window rolls by being *recomputed*, not by a job: an account whose
period has passed is measured against the new month the next time anyone
asks. Nothing has to run on a schedule for a Free account's allowance to
reset, which is one fewer thing that can fail silently.

Changing plan keeps the period. Switching tiers mid-month must not hand
out a fresh allowance.

## Downloads

Downloads are a plan entitlement; listening is not. A Free account can
play everything it made and download none of it.

The check is in the audio route, **after** the ownership check and never
before it. A Free account asking for someone else's file is told the file
does not exist — answering "your plan is too small" would confirm that
the id names a real generation, which is precisely the leak the ownership
rule exists to prevent. There is a test for this ordering.

Entitlement is evaluated per request, not stamped onto the file. A song
made on Free becomes downloadable the moment the account upgrades; it was
always the user's song.

## There is no endpoint that grants a plan

This is the load-bearing absence, and there is a test that fails if
someone adds the convenient thing.

A route that lets a signed-in account choose its own tier is a way to
take Creator for nothing, whatever it is named, however it is documented,
and whichever environment variable is supposed to keep it out of
production — the switch that disables it is one misconfiguration away
from being on.

Plans are assigned by a script that needs shell access to the machine
holding the database:

```
uv run python scripts/development/set_plan.py --email you@example.com --plan pro
uv run python scripts/development/set_plan.py --email you@example.com --show
```

It writes a subscription row and nothing else. No payment is recorded, no
receipt is issued and no billing history is created, because none of
those things happened.

## What this phase deliberately does not do

- **No payment provider.** Toss, Stripe, PortOne, PayPal, Apple Pay and
  Google Pay are all absent. `/v1/plans` reports
  `checkout_available: false`, and the pricing page renders an honest
  unavailable state rather than a subscribe button that opens nothing.
- **No invented billing records.** There are no payment, invoice or
  receipt tables. Settings still shows 결제 수단 and 최근 결제 as 미정,
  because there is nothing truthful to put there.
- **No per-song usage history in the UI.** The ledger records which
  generation spent which slot, so the data exists; nothing renders it
  yet, and a summary that implied more than the ledger holds would be a
  claim.
- **No enforcement of `priority_level` or `lab_access`.** Configured,
  unused, and not advertised.

## Existing accounts

Additive. An account with no subscription row resolves to Free —
`plan_for(None)` returns the least-privileged tier by design — so every
user who predates this migration keeps working, none is silently
upgraded, and no backfill runs.

The ledger starts empty, which means songs made before Phase 6 do not
count against the current period. That is a one-time under-count in the
user's favour, and the alternative — retroactively charging an allowance
that did not exist when the songs were made — would be worse.

## The client cannot enforce any of this

Everything the interface does about plans is presentation:

- The Create form is disabled when the allowance is spent, because
  submitting into a guaranteed refusal wastes the user's time.
- The Library's WAV control links to `/plans` on a plan without
  downloads, because meeting a 402 with no explanation is worse than
  being told first.

Both are conveniences. A user who edits them away still gets refused, and
`apps/api/tests/test_plans_api.py` drives the real routes with a real
session to prove it.
