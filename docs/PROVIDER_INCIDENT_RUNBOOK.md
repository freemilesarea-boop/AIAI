# Provider incident runbook

For the person who has just been told "generations are failing".

Read `docs/PROVIDER_RESILIENCE.md` for how the machinery works. This
document is only about what to do.

---

## 0. First, one command

```
python -m luber_provider_resilience status
```

One call: every circuit, what it is doing, why, and what the service can
currently generate. It calls no provider and changes nothing, so it is
always safe to run.

If the console is available on this deployment, `/ops/inference/circuits`
shows the same thing plus the transition log. The console does **not**
exist in production — that is deliberate, and it is why every action
below is a CLI.

---

## 1. Which incident is this?

| What you see | What it means | Go to |
|---|---|---|
| A circuit is `OPEN` | The breaker stopped calling a provider | §2 |
| Every circuit `CLOSED`, generations still failing | Not an availability problem | §3 |
| A circuit is `HALF_OPEN` and staying there | Recovery is being attempted and not succeeding | §4 |
| `AUTH_FAILED` as the last category | Credentials, not capacity | §5 |
| A capability is `UNAVAILABLE`, others fine | One task path is down | §6 |
| You need traffic stopped *now* | Deliberate drain | §7 |
| Circuits disagree between workers | Should be impossible; see §8 | §8 |

---

## 2. A circuit is open

**This is the system working.** An open circuit means requests are
failing fast with `PROVIDER_BUSY` instead of each one waiting out a
timeout. Users are getting a quick, honest refusal.

```
python -m luber_provider_resilience show ace_step --task TEXT_TO_MUSIC --history 20
```

That prints the open reason, the evidence that opened it, when it will
next probe, and the recent transitions.

**Do not force it closed to "get things moving".** A forced-closed
circuit against a provider that is still down converts fast refusals
back into slow timeouts, and the queue that builds behind them is the
part that takes an hour to drain.

Then:

1. Read the open reason. `PROVIDER_TIMEOUT` means the engine did not
   answer. `PROVIDER_INTERNAL_ERROR` means it answered with a 5xx.
   `PROVIDER_RATE_LIMIT` means it is working and declining.
2. Check the provider itself — the ACE-Step server's own health and
   logs. The circuit is a symptom; the provider is where the cause is.
3. Do nothing else. When the provider recovers, the cooldown expires, a
   probe is admitted, two successes close the circuit, and traffic
   resumes without anybody touching it.

Force it closed **only** when you have confirmed the provider is healthy
and want to skip the remaining cooldown:

```
python -m luber_provider_resilience close ace_step --task TEXT_TO_MUSIC \
  --operator "$USER" --reason "engine restarted and verified by hand at 14:05"
```

That pins the circuit to MANUAL control. It stays pinned until reset —
which is the point (a manual decision should not be silently undone),
and also the trap:

```
python -m luber_provider_resilience reset ace_step --task TEXT_TO_MUSIC --operator "$USER"
```

**Hand it back to the policy when the incident is over.** A circuit left
pinned closed cannot protect anybody from the next outage.

---

## 3. Everything is closed and generations still fail

The circuits are telling you the truth: this is not an availability
problem. Look at Phase 29 and Phase 30 instead.

- Quality rejections do not count toward circuits and never open one. If
  generations are failing QC, the provider is answering fine and the
  audio is being judged unusable.
- `/ops/inference` (the Phase 30 health page) is where retry rate,
  first-candidate accept rate and collapse findings live.
- A per-generation QC trace carries both stories; the `resilience` block
  will show a single successful attempt.

---

## 4. Stuck in HALF_OPEN

A circuit that promotes to HALF_OPEN, fails its probe, reopens with a
longer cooldown, and repeats is a provider that is *partially* up. Check
`consecutive_opens` in `show` — that is the number driving the doubling
cooldown, and if it is climbing, each recovery attempt is failing.

If probes are never admitted at all (`active_probes` at the limit but
nothing progressing), a worker died holding a probe lease. The lease
expires on its own — `probe_lease`, five minutes — and the next request
is admitted. Waiting is the correct action.

---

## 5. `AUTH_FAILED`

Credentials, and no amount of waiting fixes it. It is classified
non-retryable precisely so it does not burn a generation's whole attempt
budget reproducing the same 401.

1. Fix the key in the deployment's environment.
2. Restart the workers so they pick it up.
3. `reset` the circuit rather than waiting out the cooldown, since you
   know exactly what changed.

Nothing in the circuit record, the console, or the logs contains the
key — only the category. That is intentional; do not add the provider's
error body to a trace to "help debugging".

---

## 6. One capability is down

```
python -m luber_provider_resilience readiness
```

Circuits are per provider *and task type*, so `ace_step:COVER` can be
open while `ace_step:TEXT_TO_MUSIC` serves normally. The service is
degraded, not down, and readiness names exactly which capability is
affected.

Requests for the broken capability are **refused, not downgraded**. The
system will not drop a reference track and generate from the prompt
alone to keep a success rate up. If somebody asks why a reference
request failed when "the provider is clearly up", this is the answer.

---

## 7. Draining a provider deliberately

Before a model swap, a maintenance window, or a suspicious deploy:

```
python -m luber_provider_resilience open ace_step --task TEXT_TO_MUSIC \
  --operator "$USER" --reason "draining for model swap, ticket LM-482"
```

In-flight generations finish. New ones are refused immediately with
`PROVIDER_BUSY`.

Afterwards, hand it back:

```
python -m luber_provider_resilience reset ace_step --task TEXT_TO_MUSIC --operator "$USER"
```

Both actions are recorded in the transition log with your name and your
reason, and both appear in the console beside the automatic ones.

---

## 8. Workers disagreeing about a circuit

They should not be able to. State is in the database and every
transition is a compare-and-set; a worker that loses a race re-reads and
adopts the winner's state.

If you genuinely see divergence, the likely causes are, in order: two
deployments pointed at different databases; a worker that has not been
restarted since a migration; or a circuit row edited by hand.

```
python -m luber_provider_resilience verify
```

checks stored circuits for states that should be impossible — a
duplicate circuit key, a state name the code does not know, an
automatically opened circuit with no cooldown (which would never be
probed), probe leases held by a circuit that is not HALF_OPEN, a
negative counter — and prints what it found. It changes nothing, and
exits non-zero if anything is wrong.

---

## 9. Things this system will not do for you

Worth knowing before you go looking for them.

- **It will not fail over in production.** One provider is configured.
  Failover exists and is tested, and has nowhere to go until a second
  equivalent provider exists. Enabling the setting does not create one —
  it logs a warning saying exactly this.
- **It will not retry its way through an outage.** The attempt budget is
  Phase 29's and is unchanged. Circuit breaking removes attempts; it
  never adds them.
- **It will not serve a different song.** No degradation path drops a
  reference, shortens a duration, or changes a task type to keep a
  request alive.
- **It does not run generation remotely.** Phase 27's remote execution
  covers training and checkpoints. Generation is local to the ARQ
  worker.
- **It will not tune itself.** Thresholds are code. Changing them is a
  deploy, not a console action.
