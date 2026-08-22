# Phase 31 audit — where resilience goes, and what it may not assume

Written before the code. Four questions decide this phase, and all four
are answerable by reading the repository rather than by designing
against an imagined architecture.

1. What is the real path from a request to a provider call?
2. Who already owns retry, and how does a new layer avoid multiplying it?
3. What can the current failure taxonomy actually distinguish?
4. How many production providers are there — really?

The fourth answer shapes the honesty of everything else.

---

## 1. The actual path

Traced through the code, not inferred:

```
POST /v1/generations                 apps/api/src/luber_api/routes/generations.py
  → row created, request_trace written
  → enqueue(generation_id)           apps/api/src/luber_api/jobs.py
        ↓ Redis queue, ARQ max_tries=2, max_jobs=1 per worker
  → generate(ctx, generation_id)     services/generation-worker/.../worker.py
  → GenerationService.execute()      packages/generation-client/.../service.py
        STARTING
      → _produce_audio(generation, stack)
          → _prepare_call(...)  ──────────────────────────────┐
            builds ONE closure bound to self._provider        │  ← the seam
          → CandidateGenerationController.run(generate=…)     │
              → guarded(seed) → call(seed) → provider.generate/edit/create_from_audio
              → workspace.adopt → judge() → select()
        POST_PROCESSING / UPLOADING
      → produce_delivery_assets()    Phase 22 finishing
      → record_inference_qc_trace()  Phase 29 trace
      → record observation           Phase 30 projection
        COMPLETED
```

**The provider is chosen once per worker job.**
`provider_from_settings(config)` runs in `worker.generate`, and
`GenerationService` holds the result as `self._provider` for the whole
execution. Nothing today can change provider mid-generation, and nothing
consults provider health before calling.

## 2. The insertion point

**`GenerationService._prepare_call`, and only there.**

That function builds a `call(seed)` closure for one of three task
shapes — text-to-music, edit, cover — and binds it to `self._provider`.
The Phase 29 controller then invokes that closure once per attempt
through `guarded(seed)`, which is already the single place every
provider exception is caught and translated.

So Phase 31 changes *which provider the closure is bound to, per
attempt*, and records the outcome. It adds no loop of its own.

Two consequences worth stating plainly.

**Phase 29's budget remains the only attempt budget.** The controller's
`maximum_total_provider_calls` already bounds every call. Failover rides
on attempts the controller was going to make anyway: when a candidate
fails with a `PROVIDER_*` finding, Phase 29 already plans
`RETRY_IDENTICAL_REQUEST`, and Phase 31 answers that retry with a
different provider instead of the same one. Nothing multiplies. There is
no `providers × transport retries × quality retries` explosion because
Phase 31 introduces no multiplier — it only redirects.

**A second parallel provider system is not needed and would be wrong.**
The router hands back a provider; the controller keeps owning the loop;
the service keeps owning delivery.

## 3. Retry ownership as it stands

| Layer | What it retries | Bound |
| --- | --- | --- |
| ARQ | The whole job, after a worker died | `max_tries = 2` |
| Phase 29 controller | A candidate, for a measurable defect or a provider failure | `maximum_total_provider_calls` (3 under STANDARD) |
| ACE-Step HTTP client | **Nothing** — there is no retry loop in it | — |

That last row is the important one. There is currently **no transport
retry layer at all**: a dropped connection surfaces immediately as a
`GenerationProviderError` and consumes a Phase 29 attempt.

Phase 31 will keep it that way. Adding a transport retry underneath
Phase 29 would create exactly the nesting the brief warns about, and the
controller already retries transport failures with the identical
request — which is what a transport retry *is*. What Phase 31 adds is
the vocabulary to tell the three layers apart in the trace, not a fourth
loop.

## 4. What the failure taxonomy can distinguish today

`GenerationProviderError(message, error_code: ErrorCode)` where
`ErrorCode` is the platform-wide enum. Phase 29's `candidates.py`
already splits it:

- **Retryable:** `GENERATION_TIMEOUT`, `PROVIDER_BUSY`, `INVALID_AUDIO`,
  `UNKNOWN_GENERATION_ERROR`
- **Non-retryable:** `MODEL_LOAD_FAILED`, `REFERENCE_AUDIO_UNAVAILABLE`,
  `OUT_OF_MEMORY`

**The gap:** the ACE-Step provider maps *every* `AceStepApiError` and
`httpx.HTTPError` to `MODEL_LOAD_FAILED` or `INVALID_AUDIO`. HTTP 429
(rate limit) and 401/403 (auth) are not distinguished from a generic
transport failure, even though `AceStepApiError` carries `status_code`.

That matters for Phase 31 specifically:

- A **rate limit** is the provider saying "later", and should make it
  temporarily unavailable without being counted as it being broken.
- An **auth/config failure** reproduces on every request and should not
  burn a quality-retry budget.
- Both currently look like `MODEL_LOAD_FAILED`, which is non-retryable —
  correct for auth, wrong for a 429.

So Phase 31 must classify at the point where the status code still
exists, and translate into a resilience category *without* changing what
`ErrorCode` the user-facing failure carries. Existing error semantics
stay; a new, separate classification sits beside them.

## 5. What must never affect provider health

Read off the existing status model and error codes:

- **User input failures.** `GenerationStatus` never reaches the provider
  for a request that failed validation — those are 422s at the API. But
  `REFERENCE_AUDIO_UNAVAILABLE` is a request-shaped failure that *does*
  surface from the provider, and it says something about the request,
  not the provider.
- **Cancellation.** `GenerationStatus.CANCELLED` exists and Phase 30
  already excludes it from every quality rate. Phase 31 must exclude it
  from circuit evidence for the same reason.
- **Quality findings.** `EARLY_COLLAPSE`, `NARROW_STEREO`,
  `HIGH_HARSHNESS_PROXY` and the rest are Phase 29 verdicts about audio
  the provider *successfully returned*. A provider that answers every
  time is available, whatever the audio sounds like. Availability and
  quality are different axes and the circuit is an availability device.

## 6. Phase 30 is evidence, not control

Phase 30 opens incidents from rolling-window comparisons against a
baseline, deliberately reluctant and deliberately slow: a minimum sample
count, an absolute *and* relative delta, three clean evaluations before
recovery.

Those properties are right for "should a human look at this" and wrong
for "should the next request go somewhere else". A circuit that waited
for Phase 30's evidence would keep calling a dead provider for a whole
window; a Phase 30 incident that opened a circuit would let a quality
regression — which is not an availability problem — stop traffic.

**Decision: no automatic coupling.** Phase 31 keeps its own fast,
narrow, availability-only evidence. Phase 30 incidents remain
observational. The dashboard will *show* circuit state next to
incidents, because an operator wants both on one screen, but nothing
flows automatically from one to the other.

## 7. How many providers are there, really

`provider_from_settings` knows two names:

- `ace_step` — the production provider.
- `mock` — `MockGenerationProvider`, a test double that returns a
  committed fixture. It identifies itself as `mock` and is used by every
  API and service test.

**So there is exactly one production provider, and production failover
is not available.** Inventing a second one to demonstrate failover would
be fabricating a capability the product does not have.

What follows from that:

- The **circuit breaker** is useful today, with one provider. Refusing
  fast when the provider is down beats timing out for four minutes per
  request and burning three attempts doing it.
- **Failover infrastructure** is built and tested against deterministic
  synthetic providers that exist only under `tests/`. They are never
  registered in `provider_from_settings`.
- The report will say production failover is unavailable until a second
  equivalent provider exists. It is not a partial implementation; it is
  a capability with no second party.

## 8. Capability model already present

Failover may only go somewhere that can represent the same request, and
the repository already describes capability precisely:

| Capability | Where it is declared |
| --- | --- |
| reference-conditioned generation | `MusicGenerationProvider.supports_reference_audio` (a **property**, not a method — see below) |
| edits (extend / replace-range) | `AudioEditingProvider.supports_edit(kind)` |
| cover / source-conditioned | `AudioToAudioProvider.supports_audio_to_audio()` |
| adherence range for covers | `validated_adherence_range()` |
| duration bounds, BPM, key | `luber_schemas` request validation |

The shapes are not uniform: the first is a property and the other two
are methods. Reading only one shape is a real hazard rather than a
tidiness complaint — the first implementation of the profiler accepted
callables only, read the property as "not a method, therefore no", and
came back saying every real provider was unable to accept a reference
track. A capability lost in silence is precisely what this phase exists
to prevent, so `_asks` handles both and a test builds the real provider
and asserts the answer.

Phase 31's equivalence check is built from these rather than from a new
parallel declaration. A provider that does not implement
`AudioToAudioProvider` cannot receive a cover, and the check is a
`isinstance` plus a `supports_*` call — facts, not configuration
somebody has to keep in sync.

## 9. Persistence and multi-worker coordination

Circuit state must survive a restart and be shared between workers.
`max_jobs = 1` per worker means several worker processes is the normal
way to scale, and a per-process circuit would let worker B keep calling
a provider worker A has already given up on.

The repository's durable store is PostgreSQL (SQLite in tests) through
`luber-database`. Phase 30 added two tables there and the pattern works.
Circuit state goes the same way: one row per circuit identity, with
transitions made by conditional `UPDATE` so two workers crossing the
threshold at once produce one coherent transition.

Redis is present but used only as ARQ's queue transport. Building the
circuit on it would add a second source of truth for durable state and
would lose it on a flush.

## 10. Health, readiness and a third thing

Two endpoints exist and both keep their meaning:

- `/health` — the process is up. Never touches a dependency.
- `/ready` — PostgreSQL and Redis are reachable. 503 when not.

Generation readiness is neither: the API can be alive, its dependencies
reachable, and every provider circuit open. That is a third question,
and it belongs on the operator surface rather than as a new public
endpoint — it is exactly the kind of internal state Phase 28's console
exists for, and putting it on `/ready` would make a provider outage look
like an infrastructure failure to a load balancer.

## 11. The remote-generation gap, stated plainly

**Remote generation is NOT implemented.**

Phase 27 built remote execution for **training and checkpoint
workflows** — `packages/training/src/luber_training/remote/`, driven by
the operator CLI over SSH. Generation runs entirely on the local ARQ
worker path traced in §1. No part of a generation has ever executed on a
remote machine.

Phase 31 therefore works with the local worker architecture as it is.
The design keeps two questions apart so a future phase can add remote
generation without touching resilience semantics:

- **Which provider** — `ProviderRouter`. Phase 31 owns this.
- **Where it executes** — an execution backend. Phase 27 owns this shape
  for training; generation has no such layer yet.

A future remote-generation executor plugs in under the provider the
router already chose. Conflating them — letting the router pick "remote
ACE-Step" as if it were a different provider — would make circuit
evidence about a *machine* look like evidence about a *model*, and the
first confusing outage would be one where a broken GPU host opened the
circuit on a perfectly healthy provider.

## 12. What Phase 31 will not do

- Not train, not touch weights, not modify ACE-Step upstream.
- Not add a transport retry loop under Phase 29.
- Not open a circuit from a Phase 30 quality incident.
- Not fail over silently, and not fail over at all unless the fallback
  can carry the same request.
- Not register a synthetic provider anywhere a deployment could reach.
- Not redefine `/health` or `/ready`.
- Not fall back to local when a selected remote path is unavailable —
  there is no remote generation path to fall back from.

---

*Insertion point:* `GenerationService._prepare_call` /
`_produce_audio`, `packages/generation-client/src/luber_generation_client/service.py`.

*Budget owner:* Phase 29's `CandidatePolicy.maximum_total_provider_calls`
remains the single total attempt budget.
