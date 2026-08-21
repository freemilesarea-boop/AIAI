# The operator training console

What Phases 25–27 built is operable only from a terminal. This is the
screen that makes it operable from a browser — and the whole design
question is what a browser is allowed to be trusted with.

Phase 25 decides *what* may be trained. Phase 26 decides whether a
checkpoint earned anything. Phase 27 gets a run to a machine LUBER does
not own. None of them has a face. An operator watching a rented GPU
burn money has to read a registry directory, and an operator who has
just lost contact with a worker has to remember which CLI verb
establishes what actually happened.

---

## 1. Who can reach it, and why it is not a role

The console shows dataset identities, checkpoint digests, worker hosts
and trainer logs, and it can spend money. So the first question is what
stops a product account from reaching it.

**Not a role, because there is no role.** `User` has an id, an email, a
password hash and a display name. Adding `is_admin` would invent a
privilege model the product has not designed, put it on the table every
signup writes to, and make the difference between an operator and a
customer one boolean that one bug can flip.

**The deployment, because a deployment cannot be escalated into.** There
is no request that turns the console on. It is off unless the process
was started with it on, it is refused outright when the environment is
production, and it is gated behind a shared operator token even then.

Concretely, four independent things must all hold:

| Where | Control |
|---|---|
| API process | `OPS_CONSOLE_ENABLED=true` **and** `ENVIRONMENT != production`, or the router is never registered |
| API request | `X-Luber-Operator-Token` matching `OPS_OPERATOR_TOKEN`, compared in constant time |
| Web process | `OPS_CONSOLE_ENABLED=true`, or `/ops/training` renders the application's 404 |
| Web request | Same-origin `/ops/api/…`, where the server attaches the token |

Three consequences are deliberate.

*Disabled reads as absent.* A console that is off answers 404, not 403.
A 403 confirms to an anonymous prober that this deployment has a
training console worth attacking.

*Production is refused, not gated.* `create_app` does not mount the
router when the environment is production, and the request dependency
refuses again in case something else mounts it. Two checks, because the
cost of one being wrong is a public training console.

*A missing token fails closed.* Enabled with no token configured is a
misconfiguration, not a convenience — and it is exactly the shape that
leaves a console open. It answers 503 with the reason.

### Where the token lives

Nowhere a browser can reach. There is no safe place for a credential in
a browser: `NEXT_PUBLIC_*` is in the bundle, `localStorage` is one XSS
away, a URL is in the proxy logs. So the browser calls a same-origin
`/ops/api/…` route with **no credential at all**, and the Next server
attaches the token as the request leaves for the API
(`apps/web/src/app/ops/api/[...path]/route.ts`).

What that buys precisely: the secret exists only in the Next server's
environment, and a page can trigger an operator action without ever
having held the thing that authorises it.

What it does not buy: the proxy is as reachable as the console is.
Anyone who can load `/ops/training` can reach `/ops/api`. That is why
the console is a non-production deployment switch and not a permission.

The proxy forwards an allowlist of headers. A browser's `Cookie` is not
among them: a product session has no business at an operator endpoint,
and forwarding it would be the beginning of one being consulted there.

---

## 2. Routes

| Route | What it answers |
|---|---|
| `/ops/training` | What exists, and what this deployment can actually do |
| `/ops/training/experiments` | Every hypothesis, filtered and searched |
| `/ops/training/experiments/new` | Record a hypothesis; starts nothing |
| `/ops/training/experiments/[id]` | One hypothesis and everything descended from it |
| `/ops/training/runs` | Every execution attempt |
| `/ops/training/runs/new` | Create a DRAFT run from two locked builds |
| `/ops/training/runs/[id]` | One run, in full — the most important screen |
| `/ops/training/workers` | The fleet, with liveness kept apart from record status |
| `/ops/training/workers/[id]` | One capability report, exactly as measured |
| `/ops/training/checkpoints` | Every artifact, with placeholders unmistakable |
| `/ops/training/checkpoints/[id]` | One checkpoint, its lineage, and whether anything judged it |
| `/ops/training/evaluations` | Evaluation runs and their verdicts |
| `/ops/training/evaluations/[id]` | What was compared, what regressed, and the qualification |

The API is `/v1/ops/training/…`; the browser reaches it through
`/ops/api/…`.

The console has its own shell. It does not appear in the customer's
navigation, and the customer's navigation does not appear in it — a
training console next to a "Create" button invites somebody to reach one
from the other, and the two have different audiences and different
consequences for a misclick.

---

## 3. Configuration

```bash
OPS_CONSOLE_ENABLED=true                    # off by default
OPS_OPERATOR_TOKEN=<a long random string>   # required when enabled
OPS_REGISTRY_ROOT=./training-registry       # the Phase 25 registry
OPS_ARTIFACTS_ROOT=./training-registry/training_runs
OPS_DATASET_BUILDS_ROOT=./data/dataset-builds
OPS_CURATION_BUILDS_ROOT=./data/curation-builds
OPS_WORKER_TRANSPORT=none                   # or "local"
OPS_WORKER_ROOT=                            # when transport is "local"
OPS_PAGE_SIZE_LIMIT=200
```

The web process needs `OPS_CONSOLE_ENABLED`, `OPS_OPERATOR_TOKEN` and
`API_PROXY_TARGET`.

### Roots are configuration; ids are input

Every path the console can reach is derived from a configured root by
joining an identifier that was checked against the entries actually
present. **No request carries a path.** An operator selects a dataset
build by name from what the deployment offers, and an identifier that
would escape its root is refused rather than sanitised — a name that
escapes is not a name with a typo in it.

### The transport is opt-in and never SSH

`OPS_WORKER_TRANSPORT` is `none` by default. Reaching a rented GPU needs
a host, a user, a key reference and a known-hosts file: operator
credentials that the CLI holds. Putting them into a process a browser
can reach would move the Phase 27 boundary for the sake of a button.

`local` points at a worker root **on the same machine** — the real
Phase 27 worker driven through `LocalWorkerClient`. It is enough to
exercise every remote path without a GPU or a key, and it is what the
tests use.

There is deliberately no `ssh` option.

---

## 4. The read model

The browser never reads a registry file. Every view is built in
`luber_api.ops.readmodel` from records the Phase 25 and 26 packages own,
and the translation enforces four things a direct read could not.

**Nothing is invented.** Every field traces to a record, a lock or a
measurement. Where a fact was never established the view carries `null`
and the UI renders UNKNOWN — never zero, never a dash, never a tick. An
unprobed Mac shows eleven UNKNOWN capability values and a list naming
them, because a machine with eleven gaps is a machine nobody has probed.

**Local and remote stay apart.** `RunStatus` and the worker's
`WorkerState` are separate fields on separate models. They can
legitimately disagree, and the disagreement *is* the information: a
worker reporting RUNNING while the registry says LOST is precisely the
case reconciliation exists for.

**Secrets have nowhere to go.** No response model has a field a
credential could occupy — a stronger guarantee than remembering to strip
one. Credential *references* are reduced to a boolean, log text is
redacted server-side, and free-form documents (environment locks, audit
metadata, run bundles) pass through a redactor that also shortens
filesystem paths to their last component.

**Reading is bounded.** Lists are paginated and filtered server-side,
metric series are thinned with the sampling disclosed, logs are read
from an offset. A registry with a thousand runs renders 25 rows and
about 550 DOM nodes.

---

## 5. Actions

Every action re-validates on the server. A disabled button is a courtesy
to the operator, not a control: the same request can arrive from a stale
tab, a double click, or curl.

| Action | What it does |
|---|---|
| Create experiment | Records a hypothesis. Starts nothing. |
| Create run | Writes a DRAFT from two selected builds, deriving every digest from the locks themselves. |
| Validate | Runs every Phase 25 gate, records the report, compiles the plan, runs control-plane preflight. |
| Dispatch | Starts a **dry run** only. See below. |
| Cancel | Stops a dry run; for anything else, records a request. |
| Reconcile | Asks the worker what is actually happening. Changes nothing there. |
| Create retry run | A new run citing its parent. The original is never edited. |

Create, Validate, Stage and Dispatch stay separate. One "TRAIN" button
would make the expensive, irreversible part — sending approved data to a
rented machine and starting a trainer — indistinguishable from writing a
record.

### Three refusals worth stating

**Remote dispatch is not offered.** It needs SSH credentials this
console does not hold. The button is disabled with that sentence, and
the endpoint refuses with the CLI command that does it.

**A cancellation the console cannot deliver is reported as a request.**
For the dry-run backend the cancel is real and immediate. For a remote
run the console records the intent in the audit log and the run stays
exactly as it was — a run shown CANCELLED is a GPU an operator believes
they have stopped paying for. The result panel says "Recorded, not
performed".

**Reconciliation reports; it does not tidy.** Where the worker's answer
maps onto a legal Phase 25 transition, the transition is made. Where it
does not — a worker saying RUNNING about a run the registry wrote off as
LOST — the finding is reported and the record is left alone, because the
state machine has no path back and forcing one would be the console
rewriting history to look neat.

### Gates and preflight are both required before dispatch

Gates decide whether the *data* may be trained on. Preflight decides
whether this *machine* can execute the plan. A run whose gates passed
can still be pointed at a worker that has never demonstrated CUDA, and
dispatching on the strength of the gates alone is how that gets
discovered by renting the hardware.

Note the consequence, which is Phase 25's design showing through: every
compiled plan requires CUDA, and `LocalDryRunBackend` deliberately runs
the same capability check rather than letting "it passed on dry-run"
become evidence that a development Mac can take a GPU job. So a dry run
can only be dispatched where a real run could be.

### There is no override

No route, button or flag runs a rights-blocked or leakage-blocked run
anyway. `grep` the operator surface for `force`, `override`, `skip` or
`bypass` and there is nothing — a test asserts it.

---

## 6. Failure, in the operator's language and the system's

Both, always. The humanised line is what stops an operator hunting
through logs for a rights failure; the raw code is what they search for
when they do go to the logs.

Two entries carry most of the weight.

**`WORKER_LOST` must not read as "training failed".** It means contact
was lost, which is a statement about the connection and not about the
trainer. An operator told "failed" retries; a retry against a trainer
that is still running puts two of them in one checkpoint directory, and
the artifacts they produce are individually well-formed and jointly
worthless. The console says so, marks the classification as not
definitive, and **does not offer a retry button** until the run has been
reconciled.

**`OOM` is claimed only where it was established.** Phase 27 raises it
on an explicit CUDA out-of-memory message and never on a SIGKILL, which
the kernel OOM killer and `kill -9` produce identically. The console
shows the worker's VRAM beside the batch size, gradient accumulation and
rank — and changes none of them. A different configuration is a new run.

A failed run's screen leads with the failure: headline, guidance, raw
code, last heartbeat, last metric, checkpoints written, worker state,
exit code, and the tail of stderr. An operator should not have to open a
log viewer to find out what happened.

---

## 7. Metrics, logs and telemetry

**Only metrics that exist are charted.** There is no placeholder panel.
The installed trainer computes no validation loss, so there is no
validation chart — an empty axis labelled "validation loss" reads as a
number that has not arrived rather than one that is never coming.

**Simulated values stay labelled.** A dry run's chart is a real chart of
numbers nothing measured, drawn dashed and marked SIMULATED.

**Sampling is disclosed.** A long run's series is thinned to an even
stride with the first and last points kept, and the caption says how
many of how many are drawn. Charts are inline SVG; no charting library
was added to draw a polyline.

**Logs are incremental**, using Phase 27's cursor: the browser sends
back the offset it was given and receives only what arrived since. The
first read of a large file starts at the tail and says so. Only the most
recent 2,000 lines stay in the DOM, with the dropped count visible.

**Redaction is server-side.** A component that hides a token has already
received it. Private key blocks, `Authorization` headers, secret-named
assignments and credentials in URLs are removed before the response is
built.

**Hardware telemetry** appears where a worker reported it, and says it
is absent where nothing did. The overview never says "GPU READY" — it
says how many probe-verified workers reported inside the liveness
window.

---

## 8. Polling

Bounded, and it stops:

- Terminal runs (`COMPLETED`, `FAILED`, `CANCELLED`) are not polled at
  all. `LOST` is polled slowly, because a worker coming back changes the
  picture.
- A hidden tab stops polling and refreshes when it becomes visible.
- One request at a time per resource; a slow response cannot let a
  second interval fire on top of it.
- Unmounting cancels.

A failed poll does not clear the previous reading. The panel shows the
last good data with a warning, because blanking what an operator was
reading is worse than showing it as of a moment ago.

Every page has a manual refresh. Nothing requires a browser reload.

---

## 9. Known limitations

These are real, and none of them is hidden in the UI.

1. **Not available in production.** By construction. The console has no
   operator role, no per-operator identity and no record of *who* acted
   — only that the console acted. Shipping it to production behind a
   shared token would be all three of those problems at once. The
   upgrade path is a real operator role on the session, at which point
   the deployment switch becomes a permission.

2. **No remote dispatch, cancellation delivery, or staging from the
   browser.** Those need credentials the console does not hold. The CLI
   does them.

3. **Remote state, remote preflight and artifact staging are visible
   only with a local worker transport.** Otherwise every one of those
   panels says why it cannot see anything, rather than showing nothing.

4. **No ETA, ever.** The installed trainer measures length in epochs and
   records no step total, so remaining time cannot be derived from what
   has been measured. The panel says that rather than guessing.

5. **Cost is recorded, never computed from a price list.** An estimate
   appears only where an hourly rate was recorded on the worker;
   otherwise the field is UNKNOWN and names what is missing.

6. **Staging reports presence, not integrity.** Re-hashing a staged
   dataset on every page view is not something a console gets to do; the
   digests were verified on arrival by Phase 27.

7. **Human review is displayed, not run.** The console shows that
   listening is owed and what for. It does not revive the 41-dimension
   Phase 20H rubric.

8. **Nothing promotes a model.** Promotion review may approve a
   checkpoint for staging. Activating a model in production is a runtime
   deployment decision made elsewhere, and there is no route here that
   does it.

9. **Checkpoint comparison is available through the API
   (`POST /checkpoints/compare`) but has no dedicated page yet.** The
   data is there; the screen is not.

10. **Narrow-viewport layout has not been verified in a real browser
    window.** Dense tables scroll inside their own containers and the
    page does not overflow at desktop widths, which was verified; the
    automation environment could not resize the window below that.

---

## 10. Fixtures

`scripts/development/seed_operator_fixture.py` writes a synthetic
registry containing every state the console can display — a running
remote job, a lost worker, an OOM, a rights-blocked run, a MOCK
checkpoint, and qualification verdicts of all three kinds — plus
optional bulk runs and workers for measuring a list page at scale.

Everything in it is simulated. It is a script an operator runs against a
directory they name, not a demo mode inside the product: a switch in the
running application is one misconfiguration away from synthetic data
appearing beside real records, and there is no misconfiguration that
makes a script somebody did not run write into a registry.

```bash
uv run python scripts/development/seed_operator_fixture.py \
  --root ./ops-fixture --runs 1000 --workers 100
```

The script prints the environment to point the API at, and the URL of
every state worth looking at.
