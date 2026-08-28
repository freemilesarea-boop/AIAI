# Operator console

The console at `/admin` and the API under `/v1/admin/*`. It exists so
that running BOORDA — reading revenue, answering a customer, promoting a
colleague — does not require a database client.

## Permissions

`users.role` holds `USER`, `ADMIN` or `SUPER_ADMIN`. It defaults to
`USER`, so migration 0022 granted nobody anything.

**Why a column and not an email check.** The shortcut an admin console
usually reaches for is comparing the signed-in address against a
constant. That is a permission that cannot be revoked, cannot be
audited, appears in source control, and transfers to whoever comes to
own that mailbox. There is no `if email == "..."` anywhere in this
codebase, and adding one would defeat the point of the column.

**Where the check lives.** `apps/api/src/luber_api/admin_security.py`,
in four dependencies and nowhere else:

| Dependency | Admits |
| --- | --- |
| `require_admin` | `ADMIN`, `SUPER_ADMIN` |
| `require_super_admin` | `SUPER_ADMIN` |
| `get_admin_repository` | as `require_admin`, and binds the actor |
| `get_super_admin_repository` | as `require_super_admin` |

Every admin route sits behind one of them, and
`test_every_admin_route_is_behind_the_dependency` walks the router to
prove it — including through the `include_router` wrappers, which have
no `path` of their own and would otherwise make the scan pass vacuously.

**Identity comes only from the session.** Nothing in the permission path
reads a user id, an email or a role from a request body, a header or a
query string. `RoleChangeRequest` carries a `user_id`, but that is the
*target* of a change; the actor is whoever the session cookie resolves
to.

**The frontend gate is a courtesy.** `AdminShell` refuses to render for
an account without a role, and the nav hides `/admin` — that is request
hygiene, so a customer who types the URL does not fire a page of 403s.
It is not the control. A browser that lies about its role reaches
endpoints that refuse it.

**An unrecognised role is not an administrator.** The column is a
string; if one ever holds `SUPERADMIN` from a hand-run statement,
`_role_of` reads it as `USER` and logs a warning.

## Bootstrapping the first operator

No admin route can grant the first role — an endpoint that hands out
permission without already requiring it hands it to anyone. So:

```
uv run python scripts/ops/grant_admin.py --list
uv run python scripts/ops/grant_admin.py --email you@example.com --dry-run
uv run python scripts/ops/grant_admin.py --email you@example.com --role SUPER_ADMIN
```

Its authorisation is possession of `DATABASE_URL` and shell access to
the machine holding the database. Whoever has that could write the
column by hand already, so the script adds no privilege — it adds the
audit row, the lockout guard, and no chance of a typo.

It promotes accounts that already exist and never creates one. A
bootstrap tool that could manufacture a login would be a way in.

## The lockout guard

`set_role` refuses any change that would leave zero super
administrators, including a super administrator demoting themselves.

A check before the write is not enough, and the test
`test_two_super_admins_cannot_both_demote_the_other` exists because the
first version of this guard was exactly that and both demotions
succeeded. Two operators acting at the same instant each read a count
that was true when they started and false by the time they committed.

So the guard is two things:

1. `SELECT ... FOR UPDATE` over the super-administrator rows, which
   serialises concurrent transactions where the database has row locks
   and makes the second one re-read what the first committed.
2. The same question as a condition in the `UPDATE ... WHERE`, evaluated
   by the database while it holds the write lock. The losing demotion
   updates zero rows, and a zero rowcount is raised as `LastSuperAdmin`.

Neither is sufficient alone: SQLite has no row locks, and on PostgreSQL
it is the lock that forces the re-read.

Self-demotion is allowed when another super administrator remains —
someone handing over should be able to step down. The API answers 409
and the console explains it in its own words, because a generic
"failed" leaves the operator retrying a refusal that will never succeed.

Recovering from an empty console means a hand-written database
statement. That is why the guard is a refusal rather than a warning.

## What the console cannot do

There is no route that deletes an account, cancels a subscription,
issues a refund, triggers a charge, or impersonates a customer.
`test_destructive_routes_do_not_exist` asserts their absence, which is a
strange thing to test until you consider that the way they would appear
is somebody adding one.

Each of those actions has real consequences for a real person and
belongs to a path with its own confirmation. `DELETE
/v1/admin/admins/{id}` removes a *role*; the account keeps existing.

Closing an account remains the customer's own action in Settings, and it
anonymises rather than erases — see `docs/ACCOUNT.md`.

## Analytics

Every figure is a `GROUP BY` or a `count()` in
`packages/database/src/luber_database/admin_analytics.py`. Nothing is
counted in Python. Pulling payment rows to sum them in the application
works until the day it does not, and that day is the day revenue matters
most.

**Days are Korean days.** Timestamps are stored in UTC and stay that
way; only the bucket boundary shifts by nine hours. A payment at 08:00
KST on the 28th is 23:00 UTC on the 27th, so bucketing on the raw UTC
date would file a morning's revenue under yesterday — and the operator
comparing the dashboard against a bank statement would find them
disagreeing by a day with nothing on the page to explain it.

The shift is done in SQL, which means expressing it for both dialects:

```python
@compiles(_KstDate)                    # SQLite, in the tests
def _kst_date_default(...):
    return f"date({inner}, '+9 hours')"

@compiles(_KstDate, "postgresql")      # production
def _kst_date_postgresql(...):
    return f"(({inner}) AT TIME ZONE 'Asia/Seoul')::date"
```

Range *filters* use `kst_day_bounds` to produce UTC instants, so the
`WHERE` clause has no function on the left-hand side and the index does
the work.

**What counts as revenue.** Only `BillingPayment` rows a verified PayApp
notification wrote as `SUCCEEDED`. A checkout is not revenue and a
failed charge is not revenue.

New versus renewal is computed from payment history — a payment is
"new" when it is the first successful one for its subscription — because
nothing in the billing path records which it was, and adding a column
would mean backfilling a guess for every payment already taken.

**Zero is an answer.** Generation is switched off in production
(`GENERATION_ENABLED=false`), so the generation charts are empty and
render as empty rather than disappearing. A dashboard whose panels
vanish at zero teaches its reader to distrust the ones that remain.

## Download tracking

`download_events` records deliveries, written by the audio route when
`download=true`. Streaming for the in-page player is not counted — an
audio element fetching ranges is one song, and a metric that says
otherwise measures HTTP behaviour rather than what people took away.

The row carries the plan in force at the time, denormalised on purpose,
so "was this download permitted" stays answerable after the account
changes tier.

The write happens after every authorisation check has passed and cannot
cost a customer their file: a failure is logged and swallowed. An
undercounted statistic is a smaller problem than a paid download that
returned 500.

## Audit log

`admin_audit_logs` records role grants and revocations, support status
changes, note writes and campaign creation. Append-only in practice:
nothing in the product updates or deletes a row, and there is no route
that writes one directly.

The audit row and the change it describes share a transaction, so a
refused action leaves no entry claiming it happened.

Entries record **actions, not content**. A note written on a support
ticket appears as "a note was written" with the ticket reference —
never the note's text. An audit log is not a second copy of the data it
describes.

Readable by any administrator, not only super administrators: a log only
some operators can see is a weaker deterrent than one they can all see.

## Support

Operators may move a ticket's status and attach an internal note. They
may not edit what the customer wrote — `TicketUpdateRequest` forbids
extra fields, so the attempt is a 422 rather than a silently ignored
key.

`admin_note` is never returned by any customer-facing route. It lives on
the ticket rather than in the reply thread precisely because the reply
thread is what the customer will eventually be shown.

**Replying to customers is not implemented.** There is no mail provider.

## Email campaigns — NOT IMPLEMENTED

BOORDA has no email provider configured, in the repository or in the
deployment. The console composes a campaign, resolves its audience to a
count on the server, and stores a `DRAFT`. There is no send route and no
send button, and every campaign response carries `delivery_note` saying
so.

A send button that stored a row and showed a success toast would be
worse than none: its cost lands on the operator who believes they
announced a price change and did not.

The recipient count is resolved server-side because it is the number an
operator would confirm against before a send — not something the browser
worked out from a page of results.

When a provider is added, the pieces already exist: the campaign row,
the audience resolution, and `CAMPAIGN_QUEUED` / `CAMPAIGN_SENT` /
`CAMPAIGN_FAILED`.

## Schema

Migration `0022_admin_console`:

- `users.role` (default `USER`) and `ix_users_role`
- `download_events`
- `admin_audit_logs`
- `admin_email_campaigns`
- `support_tickets.admin_note`
- aggregation indexes on `billing_payments(status, paid_at)`,
  `generations(status, created_at)`, `users(created_at)`

Every change is additive. No existing row is modified.

## Operational notes

- The console is served from the same origin as the product and uses the
  same session cookie. There is no second login and no separate
  credential to distribute.
- `/ops` is a different thing: a deployment-gated, non-production
  training console with token auth, refused in production by
  `console_available()`. It is not reusable here and is not part of this
  console.
- Nothing in this phase changes PayApp handling, entitlement
  calculation, the generation provider, or GPU/storage configuration.
