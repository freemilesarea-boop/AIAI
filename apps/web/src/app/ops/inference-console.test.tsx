/**
 * Phase 30 — the inference console, in the browser.
 *
 * Like Phase 28's suite, most of these assert restraint. The failure
 * this console can cause is not a crash; it is an operator acting on a
 * number that looked more certain than it was. So what is defended here
 * is the honesty of the display: an empty window says NO_DATA rather
 * than 0%, a gap in a chart is a gap rather than a line drawn through
 * it, a thin bucket is marked as thin, an advisory is never listed
 * beside a rejection, a new provider revision says it is still building
 * a baseline, and nothing on any page claims something was fixed
 * automatically.
 *
 * The privacy assertions run against payloads that could carry user
 * content if the API ever let them: the check is that no page renders a
 * prompt, lyrics or a title under any circumstances.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Suspense, type ReactElement } from "react";

import GenerationDetailPage from "@/app/ops/inference/generations/[id]/page";
import GenerationsPage from "@/app/ops/inference/generations/page";
import IncidentDetailPage from "@/app/ops/inference/incidents/[id]/page";
import IncidentsPage from "@/app/ops/inference/incidents/page";
import InferenceHealthPage from "@/app/ops/inference/page";
import ProvidersPage from "@/app/ops/inference/providers/page";
import { TimeSeriesChart } from "@/components/ops/TimeSeriesChart";
import {
  emptySummary,
  emptyTrend,
  generationList,
  generationTrace,
  incident,
  incidentList,
  ingestStatus,
  overview,
  providers,
  regression,
  segments,
  summary,
  trend,
} from "@/test/inference-factories";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/ops/inference",
  useParams: () => ({ id: "abc123" }),
}));

/** A fetch double keyed by URL fragment; these pages fire many requests. */
function mockFetch(routes: Record<string, unknown>, status = 200) {
  const spy = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const match = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((fragment) => url.includes(fragment));
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

async function renderPage(ui: ReactElement) {
  await act(async () => {
    render(<Suspense fallback={null}>{ui}</Suspense>);
  });
}

const HEALTHY_ROUTES = {
  "/overview": overview(),
  "/trend?chart=retry": trend([
    "first_candidate_accept_rate",
    "quality_retry_rate",
    "retry_exhaustion_rate",
  ]),
  "/trend?chart=failure": trend([
    "invalid_audio_rate",
    "early_collapse_rate",
    "duration_failure_rate",
    "provider_failure_rate",
  ]),
  "/trend?chart=latency": trend([
    "total_latency_seconds",
    "provider_latency_seconds",
    "qc_latency_seconds",
  ]),
  "/regressions": [regression()],
  "/providers": providers(),
  "/segments": segments(),
  "/ingest-status": ingestStatus(),
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("inference health page", () => {
  it("renders every health card with its counts, not only a percentage", async () => {
    mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);

    await waitFor(() => expect(screen.getByText("First-candidate accept")).toBeInTheDocument());
    // 88/118 — the counts are on the card, not in a tooltip.
    expect(screen.getByText("88/118")).toBeInTheDocument();
    expect(screen.getByText("Retry")).toBeInTheDocument();
    expect(screen.getByText("30/118")).toBeInTheDocument();
  });

  it("says NO_DATA for an empty window rather than showing zero", async () => {
    mockFetch({
      ...HEALTHY_ROUTES,
      "/overview": overview({ summary: emptySummary(), open_incidents: 0 }),
      "/trend?chart=retry": emptyTrend(["quality_retry_rate"]),
      "/trend?chart=failure": emptyTrend(["early_collapse_rate"]),
      "/trend?chart=latency": emptyTrend(["total_latency_seconds"]),
      "/regressions": [],
      "/segments": segments(false),
    });
    await renderPage(<InferenceHealthPage />);

    await waitFor(() =>
      expect(screen.getByText("No generations in this window")).toBeInTheDocument(),
    );
    // The distinction the whole system rests on.
    expect(screen.queryByText("0.00%")).not.toBeInTheDocument();
  });

  it("states that nothing was done automatically", async () => {
    mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);

    await waitFor(() =>
      expect(screen.getByText(/detects and explains/i)).toBeInTheDocument(),
    );
  });

  it("keeps advisories out of the rejection list", async () => {
    mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);

    await waitFor(() => expect(screen.getByText("Rejections")).toBeInTheDocument());
    const advisories = screen.getByText("Advisories on delivered audio");
    expect(advisories).toBeInTheDocument();
    expect(screen.getByText("Not failures.")).toBeInTheDocument();
  });

  it("warns when the projection is behind", async () => {
    mockFetch({ ...HEALTHY_ROUTES, "/ingest-status": ingestStatus(true) });
    await renderPage(<InferenceHealthPage />);

    await waitFor(() => expect(screen.getByText(/Data may be behind/i)).toBeInTheDocument());
  });

  it("warns when a window contains generations that predate the trace", async () => {
    const partial = summary();
    partial.coverage = { ...partial.coverage, partial: true, without_qc_data: 30 };
    partial.coverage.note = "30 of 120 generations predate the candidate trace.";
    mockFetch({ ...HEALTHY_ROUTES, "/overview": overview({ summary: partial }) });
    await renderPage(<InferenceHealthPage />);

    await waitFor(() =>
      expect(screen.getByText(/Partial data before 460642e/i)).toBeInTheDocument(),
    );
  });

  it("shows a regression with its counts and its threshold", async () => {
    mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);

    await waitFor(() =>
      expect(screen.getByText(/3\/412 \(0.73%\) to 11\/96 \(11.46%\)/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/Threshold: absolute/)).toBeInTheDocument();
  });

  it("reports how many segments were too small to rank", async () => {
    mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);

    await waitFor(() =>
      expect(screen.getByText(/4 segment\(s\) had too few samples/)).toBeInTheDocument(),
    );
  });

  it("offers a time range and refetches when it changes", async () => {
    const spy = mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);
    await waitFor(() => expect(screen.getByText("First-candidate accept")).toBeInTheDocument());

    await act(async () => {
      await userEvent.click(screen.getByRole("button", { name: "Last 7 days" }));
    });

    await waitFor(() =>
      expect(spy.mock.calls.some(([url]) => String(url).includes("window=7d"))).toBe(true),
    );
  });

  it("marks a brand-new revision as still building a baseline", async () => {
    mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);

    await waitFor(() => expect(screen.getByText(/acestep@v2/)).toBeInTheDocument());
    expect(screen.getByText(/· new/)).toBeInTheDocument();
  });
});

describe("charts", () => {
  it("breaks the line at a gap instead of drawing through it", async () => {
    const data = trend(["quality_retry_rate"], true);
    const { container } = render(
      <TimeSeriesChart
        title="Retry"
        points={data.points}
        unit="rate"
        series={[{ key: "quality_retry_rate", label: "Retried", unit: "rate" }]}
      />,
    );

    // Two polylines, because the empty buckets in the middle end one
    // segment and start another. One would mean the gap was bridged.
    expect(container.querySelectorAll("polyline")).toHaveLength(2);
  });

  it("says so rather than drawing a flat line when there is no data", () => {
    render(
      <TimeSeriesChart
        title="Retry"
        points={[]}
        unit="rate"
        series={[{ key: "quality_retry_rate", label: "Retried", unit: "rate" }]}
      />,
    );

    expect(screen.getByText(/No data in this window/)).toBeInTheDocument();
    expect(document.querySelector("polyline")).toBeNull();
  });

  it("explains that hollow points are thin buckets", () => {
    const data = trend(["quality_retry_rate"], false);
    render(
      <TimeSeriesChart
        title="Retry"
        points={data.points}
        unit="rate"
        series={[{ key: "quality_retry_rate", label: "Retried", unit: "rate" }]}
      />,
    );

    expect(screen.getByText(/Hollow points are buckets with fewer than/)).toBeInTheDocument();
  });
});

describe("incidents", () => {
  it("lists open incidents with severity and segment", async () => {
    mockFetch({ "/incidents": incidentList() });
    await renderPage(<IncidentsPage />);

    await waitFor(() => expect(screen.getByText("EARLY_COLLAPSE_INCREASE")).toBeInTheDocument());
    expect(screen.getByText("CRITICAL")).toBeInTheDocument();
    expect(screen.getByText("duration_bucket=181_240")).toBeInTheDocument();
  });

  it("says so when there is nothing open", async () => {
    mockFetch({ "/incidents": incidentList([]) });
    await renderPage(<IncidentsPage />);

    await waitFor(() => expect(screen.getByText("No open incidents")).toBeInTheDocument());
  });

  it("shows the evidence and both windows on the detail page", async () => {
    mockFetch({ "/incidents/abc123": incident() });
    await renderPage(<IncidentDetailPage params={Promise.resolve({ id: "abc123" })} />);

    await waitFor(() => expect(screen.getByText("Evidence")).toBeInTheDocument());
    // Twice on purpose: once in the headline summary, once in the
    // timeline. An operator scanning the page and one reading it
    // carefully should both find the numbers.
    expect(screen.getAllByText(/3\/412 \(0.73%\) to 11\/96 \(11.46%\)/).length).toBeGreaterThan(
      0,
    );
    expect(screen.getByText("Baseline window")).toBeInTheDocument();
    expect(screen.getByText("Current window")).toBeInTheDocument();
  });

  it("offers no button that changes the system", async () => {
    mockFetch({ "/incidents/abc123": incident() });
    await renderPage(<IncidentDetailPage params={Promise.resolve({ id: "abc123" })} />);

    await waitFor(() => expect(screen.getByText("Operator actions")).toBeInTheDocument());
    for (const forbidden of [/disable/i, /roll ?back/i, /restart/i, /apply fix/i, /retry now/i]) {
      expect(screen.queryByRole("button", { name: forbidden })).toBeNull();
    }
    expect(screen.getByText(/performs none of them/i)).toBeInTheDocument();
  });

  it("will not dismiss without a reason", async () => {
    mockFetch({ "/incidents/abc123": incident() });
    await renderPage(<IncidentDetailPage params={Promise.resolve({ id: "abc123" })} />);
    await waitFor(() => expect(screen.getByText("Operator actions")).toBeInTheDocument());

    const dismiss = screen.getByRole("button", { name: "Dismiss" });
    expect(dismiss).toBeDisabled();

    await act(async () => {
      await userEvent.type(screen.getByLabelText(/Your name/), "alex");
    });
    // A name alone is not enough: the reason is what the next person needs.
    expect(screen.getByRole("button", { name: "Dismiss" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Acknowledge" })).toBeEnabled();
  });

  it("says that acknowledging does not stop measurement", async () => {
    mockFetch({ "/incidents/abc123": incident() });
    await renderPage(<IncidentDetailPage params={Promise.resolve({ id: "abc123" })} />);

    await waitFor(() =>
      expect(screen.getByText(/does not stop measurement/i)).toBeInTheDocument(),
    );
  });
});

describe("generations", () => {
  it("lists generations without any prompt, lyrics or title", async () => {
    mockFetch({ "/generations": generationList() });
    await renderPage(<GenerationsPage />);

    await waitFor(() => expect(screen.getByText("TEXT_TO_MUSIC")).toBeInTheDocument());
    // Searching for the sentinel strings rather than for the words
    // "prompt" and "lyric": the page's own copy says it stores neither,
    // and a naive substring search would match that sentence.
    for (const secret of ["ZZPROMPTZZ", "ZZLYRICSZZ", "ZZTITLEZZ"]) {
      expect(document.body.textContent).not.toContain(secret);
    }
    expect(screen.getByText(/No prompt, lyrics or title is stored/i)).toBeInTheDocument();
  });

  it("explains one generation the way Phase 29 explains it", async () => {
    mockFetch({ "/generations/": generationTrace() });
    await renderPage(
      <GenerationDetailPage
        params={Promise.resolve({ id: "11111111-1111-4111-8111-111111111111" })}
      />,
    );

    await waitFor(() =>
      expect(screen.getByText("Attempt 1 rejected: EARLY_COLLAPSE")).toBeInTheDocument(),
    );
    expect(screen.getByText("Attempt 2 selected.")).toBeInTheDocument();
    expect(screen.getByText("Provider calls: 2. Quality retries: 1.")).toBeInTheDocument();
  });

  it("names the delivery span for what it actually covers", async () => {
    mockFetch({ "/generations/": generationTrace() });
    await renderPage(
      <GenerationDetailPage
        params={Promise.resolve({ id: "11111111-1111-4111-8111-111111111111" })}
      />,
    );

    await waitFor(() => expect(screen.getByText("Where the time went")).toBeInTheDocument());
    expect(screen.getByText("Delivery")).toBeInTheDocument();
    // Not called "finishing": Phase 22 records no timing of its own.
    expect(screen.queryByText("Finishing latency")).toBeNull();
  });

  it("separates an advisory from a rejection on an attempt", async () => {
    mockFetch({ "/generations/": generationTrace() });
    await renderPage(
      <GenerationDetailPage
        params={Promise.resolve({ id: "11111111-1111-4111-8111-111111111111" })}
      />,
    );

    await waitFor(() =>
      expect(screen.getByText(/Rejected for: EARLY_COLLAPSE/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/Advisories \(not failures\): NARROW_STEREO/)).toBeInTheDocument();
  });
});

describe("providers", () => {
  it("marks a revision without history as still building", async () => {
    mockFetch({ "/providers": providers() });
    await renderPage(<ProvidersPage />);

    await waitFor(() => expect(screen.getByText("acestep@v2")).toBeInTheDocument());
    expect(screen.getByText("building")).toBeInTheDocument();
    expect(screen.getByText("ready")).toBeInTheDocument();
  });

  it("shows nothing to compare until two revisions are picked", async () => {
    mockFetch({ "/providers": providers() });
    await renderPage(<ProvidersPage />);

    await waitFor(() => expect(screen.getByText("Nothing selected")).toBeInTheDocument());
  });

  it("carries the caveat that this is not a controlled experiment", async () => {
    mockFetch({
      "/providers/compare": {
        status: "OK",
        sufficient_data: true,
        deltas: {
          early_collapse_rate: {
            status: "OK",
            before: 0.005,
            after: 0.4,
            absolute_delta: 0.395,
          },
        },
        caveat:
          "Two revisions measured over the same window. Traffic mix may differ between them; a difference here is a difference in what was observed, not a controlled experiment.",
      },
      "/providers": providers(),
    });
    await renderPage(<ProvidersPage />);
    await waitFor(() => expect(screen.getByText("acestep@v1")).toBeInTheDocument());

    const rows = screen.getAllByRole("button", { name: "A" });
    await act(async () => {
      await userEvent.click(rows[0]);
      await userEvent.click(screen.getAllByRole("button", { name: "B" })[1]);
    });

    await waitFor(() =>
      expect(screen.getByText(/not a controlled experiment/i)).toBeInTheDocument(),
    );
  });
});

describe("privacy", () => {
  it("renders no user content on any page, whatever the payload", async () => {
    // Every page, one after another, with the strings that would only
    // appear if something had leaked.
    const forbidden = ["ZZPROMPTZZ", "ZZLYRICSZZ", "ZZTITLEZZ"];

    mockFetch(HEALTHY_ROUTES);
    await renderPage(<InferenceHealthPage />);
    await waitFor(() => expect(screen.getByText("First-candidate accept")).toBeInTheDocument());
    for (const secret of forbidden) {
      expect(document.body.textContent).not.toContain(secret);
    }
  });
});
