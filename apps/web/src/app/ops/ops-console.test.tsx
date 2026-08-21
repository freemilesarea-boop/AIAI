/**
 * Phase 28 — the operator training console, in the browser.
 *
 * Most of these assert restraint rather than capability. The console is
 * easy to make impressive and hard to make honest, so what is defended
 * here is the honesty: a figure nobody measured renders as UNKNOWN, a
 * dry run's numbers stay labelled SIMULATED, a placeholder checkpoint is
 * marked TEST ONLY, a lost worker never reads as "training failed", and
 * a preflight that could not establish something is visually distinct
 * from one that passed.
 *
 * The proxy route is tested too, because it is where the operator token
 * lives and the whole reason a browser never holds one.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Suspense, type ReactElement } from "react";

import { CheckpointTable } from "@/components/ops/CheckpointTable";
import { MetricChart } from "@/components/ops/MetricChart";
import { RunActions } from "@/components/ops/RunActions";
import { RunTimeline } from "@/components/ops/RunTimeline";
import { OpsStatus, Maybe } from "@/components/ops/primitives";
import RunDetailPage from "@/app/ops/training/runs/[id]/page";
import OverviewPage from "@/app/ops/training/page";
import WorkersPage from "@/app/ops/training/workers/page";
import WorkerDetailPage from "@/app/ops/training/workers/[id]/page";
import EvaluationDetailPage from "@/app/ops/training/evaluations/[id]/page";
import {
  actions,
  checkpoint,
  evaluationDetail,
  gates,
  heartbeat,
  metricSeries,
  mockCheckpoint,
  overview,
  runDetail,
  runSummary,
  timeline,
  unprobedWorker,
  worker,
} from "@/test/ops-factories";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/ops/training",
  useParams: () => ({ id: "run_1" }),
}));

/**
 * A fetch double keyed by URL fragment.
 *
 * Keyed rather than sequential because these pages fire several
 * independent requests — a run detail and its diagnostics, an overview
 * and its baseline — and a queue would make the test depend on the order
 * React happened to schedule them in.
 */
function mockFetch(routes: Record<string, unknown>, status = 200) {
  const spy = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const match = Object.keys(routes).find((fragment) => url.includes(fragment));
    if (!match) {
      return new Response(JSON.stringify({ detail: `no route for ${url}` }), { status: 404 });
    }
    return new Response(JSON.stringify(routes[match]), {
      status,
      headers: { "content-type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

/**
 * Pages read their route params with React's `use()`, which suspends
 * until the promise resolves. Without a boundary the tree renders
 * nothing and every assertion fails on an empty body — so page renders
 * go through here.
 */
async function renderPage(ui: ReactElement) {
  // `act` has to be awaited, or the suspended tree never resumes and
  // every assertion runs against an empty body.
  await act(async () => {
    render(<Suspense fallback={null}>{ui}</Suspense>);
  });
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/* ── unknown is never zero ───────────────────────────────────────────── */

describe("unmeasured values", () => {
  it("renders UNKNOWN rather than a zero or a blank", () => {
    render(
      <p>
        <Maybe value={null} />
      </p>,
    );
    expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
    expect(screen.getByTitle("Nobody has measured this")).toBeInTheDocument();
  });

  it("keeps an unprobed worker's GPU fields empty rather than filling them in", async () => {
    mockFetch({
      "/workers/wrk_mac": {
        worker: unprobedWorker(),
        heartbeat: heartbeat({
          available: false,
          unavailable_reason: "This worker has never reported.",
          liveness: "UNKNOWN",
        }),
        software_environment: {},
        recent_runs: [],
        audit_events: [],
        unknown_capabilities: ["gpu_model", "vram_total_mb", "cuda_available"],
      },
    });

    await renderPage(<WorkerDetailPage params={Promise.resolve({ id: "wrk_mac" })} />);

    await screen.findByText("Capability report");
    // Three GPU fields, all UNKNOWN, none rendered as 0.
    expect(screen.getAllByText("UNKNOWN").length).toBeGreaterThanOrEqual(3);
    expect(
      screen.getByText(/3 capability value\(s\) have never been measured/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Nothing here fills them in/)).toBeInTheDocument();
  });
});

/* ── status is never colour alone ────────────────────────────────────── */

describe("status encoding", () => {
  it("carries a word for every state, not only a colour", () => {
    render(
      <>
        <OpsStatus status="RUNNING" />
        <OpsStatus status="LOST" />
        <OpsStatus status="UNKNOWN" />
      </>,
    );
    expect(screen.getByText("RUNNING")).toBeInTheDocument();
    expect(screen.getByText("LOST")).toBeInTheDocument();
    expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
  });
});

/* ── overview ────────────────────────────────────────────────────────── */

describe("training overview", () => {
  it("shows counts by state and never claims GPU READY", async () => {
    mockFetch({
      "/overview": overview(),
      "/baseline": { production: [], all_models: [], note: "Nothing in this console changes a model stage." },
    });

    await renderPage(<OverviewPage />);

    await screen.findByText("System status");
    expect(screen.getByText("training capability")).toBeInTheDocument();
    expect(screen.queryByText(/GPU READY/i)).not.toBeInTheDocument();
    expect(screen.getByText(/probe-verified worker/)).toBeInTheDocument();
  });

  it("distinguishes an empty registry from a missing one", async () => {
    mockFetch({
      "/overview": overview({
        empty_reason:
          "This registry holds no experiments, runs, workers or checkpoints yet. Register a " +
          "model baseline and create an experiment to begin, or point OPS_REGISTRY_ROOT at the " +
          "registry you meant.",
        runs: { total: 0, by_state: {} },
      }),
      "/baseline": { production: [], all_models: [], note: "n/a" },
    });

    await renderPage(<OverviewPage />);
    expect(await screen.findByText("This registry is empty")).toBeInTheDocument();
    expect(screen.getByText(/OPS_REGISTRY_ROOT/)).toBeInTheDocument();
  });

  it("says when no model is marked production", async () => {
    mockFetch({
      "/overview": overview(),
      "/baseline": {
        production: [],
        all_models: [],
        note: "Nothing in this console changes a model stage.",
      },
    });

    await renderPage(<OverviewPage />);
    expect(await screen.findByText("No model is marked PRODUCTION")).toBeInTheDocument();
  });
});

/* ── run detail ──────────────────────────────────────────────────────── */

async function renderRun(detail: ReturnType<typeof runDetail>, diagnostics: string[] = []) {
  mockFetch({
    "/diagnostics": diagnostics,
    "/logs": {
      available: true,
      unavailable_reason: null,
      stream: "stdout",
      offset: 0,
      next_offset: 40,
      size_bytes: 40,
      eof: true,
      truncated: false,
      text: "step 1 loss 2.38\nstep 2 loss 2.36",
      from_tail: false,
    },
    "/runs/": detail,
  });
  await renderPage(<RunDetailPage params={Promise.resolve({ id: detail.run.run_id })} />);
  await screen.findByText("Lifecycle");
}

describe("run detail", () => {
  it("keeps the control plane's status and the worker's state apart", async () => {
    await renderRun(runDetail());

    const remote = screen.getByRole("region", { name: /Remote state/i });
    expect(
      within(remote).getByText(/This console holds no transport to a worker/),
    ).toBeInTheDocument();
    // The run is RUNNING; nothing has invented a worker state to match.
    expect(within(remote).queryByText("RUNNING")).not.toBeInTheDocument();
  });

  it("shows the plan hash and the config hash with copy controls", async () => {
    await renderRun(runDetail());

    const repro = screen.getByRole("region", { name: /Reproducibility/i });
    expect(within(repro).getByLabelText("Copy training plan hash")).toBeInTheDocument();
    expect(within(repro).getByLabelText("Copy training config hash")).toBeInTheDocument();
    expect(within(repro).getByLabelText("Copy LUBER commit")).toBeInTheDocument();
  });

  it("refuses to show an ETA and says why", async () => {
    await renderRun(runDetail());

    const progress = screen.getByRole("region", { name: /Progress/i });
    expect(within(progress).getByText(/measures length in epochs/)).toBeInTheDocument();
  });

  it("marks an unknown preflight check as distinct from a pass", async () => {
    await renderRun(runDetail());

    const control = screen.getByRole("region", { name: /Control-plane preflight/i });
    expect(
      within(control).getByText(/Could not be established — not a pass/),
    ).toBeInTheDocument();
  });

  it("shows a rights failure with its offending ids and no override", async () => {
    await renderRun(
      runDetail({
        run: runSummary({
          status: "FAILED",
          failure: {
            code: "RIGHTS_GATE_FAILED",
            headline: "Rights are not clear for every selected track",
            guidance:
              "One or more tracks in the curated manifest are not cleared for training. This " +
              "gate has no override: resolve the rights record, re-curate, and create a new run.",
            raw_message: "2 selected track(s) are not cleared for training",
            confident: true,
          },
        }),
        gates: gates(false),
        timeline: timeline("FAILED"),
      }),
    );

    expect(
      screen.getByText("Rights are not clear for every selected track"),
    ).toBeInTheDocument();
    expect(screen.getByText(/has no override/)).toBeInTheDocument();
    expect(screen.getByText(/trk-0002, trk-0003/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /force/i })).not.toBeInTheDocument();
  });

  it("reports OOM truthfully with the hardware beside it", async () => {
    await renderRun(
      runDetail({
        run: runSummary({
          status: "FAILED",
          failure: {
            code: "OOM",
            headline: "CUDA out of memory",
            guidance:
              "The trainer said so explicitly. Compare the worker's VRAM with the batch size, " +
              "gradient accumulation and rank below, then create a new run with a changed " +
              "configuration — nothing here edits a run in place.",
            raw_message: "CUDA out of memory. Tried to allocate 2.00 GiB",
            confident: true,
          },
        }),
        timeline: timeline("FAILED"),
      }),
      ["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"],
    );

    expect(screen.getByText("CUDA out of memory")).toBeInTheDocument();
    expect(screen.getByText(/nothing here edits a run in place/)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText(/torch.OutOfMemoryError/)).toBeInTheDocument(),
    );
    // The raw code stays visible for a log search.
    expect(screen.getAllByText("OOM").length).toBeGreaterThan(0);
  });

  it("never says a lost worker means training failed", async () => {
    await renderRun(
      runDetail({
        run: runSummary({
          status: "LOST",
          failure: {
            code: "WORKER_LOST",
            headline: "Worker connection lost",
            guidance:
              "The remote trainer may still be running. Reconcile before doing anything else: " +
              "launching a retry now can put two trainers in one checkpoint directory.",
            raw_message: "the worker stopped reporting; remote state is unknown",
            confident: false,
          },
        }),
        timeline: timeline("LOST"),
      }),
    );

    expect(screen.getByText("Worker connection lost")).toBeInTheDocument();
    // Said twice on purpose: once as the guidance, once as a callout
    // above the actions, because this is the sentence that stops a
    // second trainer being launched.
    expect(screen.getAllByText(/may still be running/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/Reconcile before doing anything else/)).toBeInTheDocument();
    expect(screen.queryByText(/Training failed/i)).not.toBeInTheDocument();
    expect(screen.getByText(/classification is not definitive/)).toBeInTheDocument();
  });

  it("shows a cost only where the figures were recorded", async () => {
    await renderRun(
      runDetail({
        cost: {
          provider: null,
          instance_type: null,
          hourly_rate: null,
          currency: null,
          wall_seconds: 120,
          gpu_seconds: null,
          estimated_cost: null,
          actual_cost: null,
          unknown: ["no hourly rate has been recorded for this worker"],
        },
      }),
    );

    const cost = screen.getByRole("region", { name: /Cost/i });
    expect(within(cost).getByText(/no hourly rate has been recorded/)).toBeInTheDocument();
    expect(within(cost).getAllByText("UNKNOWN").length).toBeGreaterThan(0);
  });
});

/* ── metrics ─────────────────────────────────────────────────────────── */

describe("metric charts", () => {
  it("labels simulated series and does not present them as measurements", () => {
    render(<MetricChart series={metricSeries({ sources: ["SIMULATED"] })} />);
    expect(screen.getByText("SIMULATED")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAccessibleName(/SIMULATED and measure nothing/);
  });

  it("discloses sampling rather than implying every point was drawn", () => {
    render(<MetricChart series={metricSeries({ sampled: true, total_points: 40_000 })} />);
    expect(screen.getByText(/of 40,000 points/)).toBeInTheDocument();
    expect(screen.getByText(/sampled to fit/)).toBeInTheDocument();
  });

  it("carries a text summary so the finding survives without the picture", () => {
    render(<MetricChart series={metricSeries()} />);
    expect(screen.getByRole("img")).toHaveAccessibleName(/Lowest .* highest .* latest/);
  });

  it("renders nothing at all for a series with no points", () => {
    const { container } = render(<MetricChart series={metricSeries({ points: [] })} />);
    expect(container).toBeEmptyDOMElement();
  });
});

/* ── timeline ────────────────────────────────────────────────────────── */

describe("run timeline", () => {
  it("shows the terminal states a run did not take", () => {
    render(<RunTimeline entries={timeline("LOST")} />);
    for (const state of ["COMPLETED", "FAILED", "CANCELLED", "LOST"]) {
      expect(screen.getByText(state)).toBeInTheDocument();
    }
  });

  it("does not mark a draft run as having been validated", () => {
    render(<RunTimeline entries={timeline("DRAFT")} />);
    const queued = screen.getByText("QUEUED").closest("span");
    expect(queued?.textContent).toContain("○");
  });
});

/* ── checkpoints ─────────────────────────────────────────────────────── */

describe("checkpoints", () => {
  it("marks a MOCK checkpoint TEST ONLY and refuses evaluation", () => {
    render(<CheckpointTable rows={[mockCheckpoint()]} />);
    expect(screen.getByText("TEST ONLY")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
    expect(
      screen.getByTitle(/contains no trained weights and can never become an evaluation candidate/),
    ).toBeInTheDocument();
  });

  it("does not mark a real adapter as test material", () => {
    render(<CheckpointTable rows={[checkpoint()]} />);
    expect(screen.queryByText("TEST ONLY")).not.toBeInTheDocument();
    expect(screen.getByText("Yes")).toBeInTheDocument();
  });
});

/* ── actions ─────────────────────────────────────────────────────────── */

describe("run actions", () => {
  it("says what an action will do rather than asking whether you are sure", async () => {
    const user = userEvent.setup();
    render(<RunActions runId="run_1" actions={actions()} onCompleted={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByText(/Request graceful cancellation of run run_1/)).toBeInTheDocument();
    expect(within(dialog).getByText(/it does not deliver the signal/)).toBeInTheDocument();
    expect(within(dialog).queryByText(/Are you sure/i)).not.toBeInTheDocument();
  });

  it("disables an action that is not legal and shows the reason", () => {
    render(<RunActions runId="run_1" actions={actions()} onCompleted={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Dispatch" })).toBeDisabled();
    expect(
      screen.getByText("Only a QUEUED run can be dispatched; this run is RUNNING."),
    ).toBeInTheDocument();
  });

  it("reports a cancellation that was recorded but not performed", async () => {
    const user = userEvent.setup();
    mockFetch({
      "/actions/cancel": {
        action: "cancel",
        performed: false,
        run_status: "RUNNING",
        detail:
          "Graceful cancellation requested and recorded in the audit log. The run remains " +
          "RUNNING: this console holds no transport to the worker.",
        outcome: "CANCEL_REQUESTED",
        created_id: null,
      },
    });

    render(<RunActions runId="run_1" actions={actions()} onCompleted={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Cancel the run" }),
    );

    // Deliberately not "Done": a run shown cancelled is a GPU an
    // operator believes they stopped paying for.
    expect(await screen.findByText("Recorded, not performed")).toBeInTheDocument();
    expect(screen.getByText(/remains RUNNING/)).toBeInTheDocument();
  });

  it("surfaces a server refusal instead of failing silently", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ detail: "Run run_1 is COMPLETED and cannot be cancelled." }),
            { status: 409, headers: { "content-type": "application/json" } },
          ),
      ),
    );

    render(<RunActions runId="run_1" actions={actions()} onCompleted={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Cancel the run" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Run run_1 is COMPLETED and cannot be cancelled.",
    );
  });

  it("sends one request when a confirmed action is double clicked", async () => {
    const user = userEvent.setup();
    const spy = mockFetch({
      "/actions/cancel": {
        action: "cancel",
        performed: true,
        run_status: "CANCELLED",
        detail: "The dry run was cancelled.",
        outcome: "CANCELLED",
        created_id: null,
      },
    });

    render(<RunActions runId="run_1" actions={actions()} onCompleted={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    const confirm = within(await screen.findByRole("alertdialog")).getByRole("button", {
      name: "Cancel the run",
    });
    await user.dblClick(confirm);

    await screen.findByText("Done");
    expect(spy).toHaveBeenCalledTimes(1);
  });
});

/* ── workers ─────────────────────────────────────────────────────────── */

describe("workers", () => {
  it("shows liveness separately from the registry's status field", async () => {
    mockFetch({
      "/workers": {
        items: [
          worker({ status: "ONLINE", liveness: "STALE", heartbeat_age_seconds: 480 }),
          unprobedWorker(),
        ],
        page: { total: 2, limit: 25, offset: 0, returned: 2 },
        available_classes: ["GPU_TRAINING_READY", "DEVELOPMENT_ONLY"],
        available_liveness: ["STALE", "UNKNOWN"],
      },
    });

    await renderPage(<WorkersPage />);

    await screen.findByText("Registered workers");
    const table = screen.getByRole("table");
    expect(within(table).getByText("STALE")).toBeInTheDocument();
    // The record still says ONLINE, and both are visible.
    expect(within(table).getAllByText("ONLINE").length).toBeGreaterThan(0);
    // The unprobed worker's liveness *and* its unmeasured capabilities
    // all read UNKNOWN — none of them as a zero.
    expect(within(table).getAllByText("UNKNOWN").length).toBeGreaterThanOrEqual(2);
  });

  it("never shows a credential reference, only that one is configured", async () => {
    mockFetch({
      "/workers/wrk_1": {
        worker: worker(),
        heartbeat: heartbeat(),
        software_environment: {},
        recent_runs: [],
        audit_events: [],
        unknown_capabilities: [],
      },
    });

    await renderPage(<WorkerDetailPage params={Promise.resolve({ id: "wrk_1" })} />);

    await screen.findByText("Credential configured");
    expect(screen.getByText(/never receives the reference name/)).toBeInTheDocument();
    expect(screen.queryByText(/operator-training-key/)).not.toBeInTheDocument();
  });
});

/* ── evaluation ──────────────────────────────────────────────────────── */

describe("evaluation and qualification", () => {
  it("shows a rejection with the gate that failed rather than a score", async () => {
    mockFetch({ "/evaluations/eval_1": evaluationDetail("REJECTED") });
    await renderPage(<EvaluationDetailPage params={Promise.resolve({ id: "eval_1" })} />);

    await screen.findByText("Qualification: REJECTED");
    const failed = screen.getByText("Failed (1)").parentElement!;
    expect(within(failed).getByText("lyric_intelligibility")).toBeInTheDocument();
    expect(screen.getByText(/regressed by a MAJOR margin/)).toBeInTheDocument();
    // A verdict, never a number: a borderline pass and a catastrophic
    // regression must not look adjacent.
    expect(screen.queryByText(/score/i)).not.toBeInTheDocument();
    expect(screen.getByText("Passed (2)")).toBeInTheDocument();
  });

  it("presents human review as an outcome, not a gap", async () => {
    mockFetch({ "/evaluations/eval_1": evaluationDetail("HUMAN_REVIEW_REQUIRED") });
    await renderPage(<EvaluationDetailPage params={Promise.resolve({ id: "eval_1" })} />);

    await screen.findByText("Human review required");
    expect(screen.getByText(/cannot be qualified by measurement alone/)).toBeInTheDocument();
    expect(screen.getByText("LIGHT_AB")).toBeInTheDocument();
    expect(screen.getByText(/does not run the session/)).toBeInTheDocument();
  });

  it("stops a qualified candidate at promotion review", async () => {
    mockFetch({ "/evaluations/eval_1": evaluationDetail("QUALIFIED") });
    await renderPage(<EvaluationDetailPage params={Promise.resolve({ id: "eval_1" })} />);

    await screen.findByText("Qualification: QUALIFIED");
    const promotion = screen.getByRole("region", { name: /Promotion review/i });
    expect(within(promotion).getByText("HOLD")).toBeInTheDocument();
    expect(
      within(promotion).getByText(/runtime deployment decision made elsewhere/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /activate|deploy|promote/i })).not.toBeInTheDocument();
  });
});

/* ── sectional failure ───────────────────────────────────────────────── */

describe("error recovery", () => {
  it("does not blank the console when one section fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/baseline")) {
          return new Response(JSON.stringify({ detail: "registry unreadable" }), { status: 500 });
        }
        return new Response(JSON.stringify(overview()), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }),
    );

    await renderPage(<OverviewPage />);

    // The failing panel reports; the rest of the page still renders.
    await screen.findByText("System status");
    expect(await screen.findByRole("alert")).toHaveTextContent("registry unreadable");
    expect(screen.getByText("training capability")).toBeInTheDocument();
  });
});
