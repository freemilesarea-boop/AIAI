/**
 * Phase 31 — the circuit panel, in the browser.
 *
 * What is defended here is honesty under partial failure. The panel is
 * read by somebody who has just been told generations are failing, and
 * the ways it can mislead them are specific: showing 0% for a circuit
 * nobody has measured, showing a failover mode that cannot move
 * anything as though it could, hiding a circuit for a provider that has
 * been removed, or offering a button that does not work where the
 * incident is.
 */

import { act, render, screen, within, type RenderResult } from "@testing-library/react";
import { Suspense, type ReactElement } from "react";

import CircuitsPage from "@/app/ops/inference/circuits/page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/ops/inference/circuits",
  useParams: () => ({}),
}));

const VERSIONS = {
  resilience_schema_version: "luber-provider-resilience/1",
  circuit_policy_version: "circuit-v1",
  routing_policy_version: "routing-v1",
  failover_policy_version: "failover-v1",
};

function circuit(overrides: Record<string, unknown> = {}) {
  return {
    circuit_key: "ace_step:TEXT_TO_MUSIC",
    provider: "ace_step",
    task_type: "TEXT_TO_MUSIC",
    state: "CLOSED",
    control: "AUTOMATIC",
    consecutive_failures: 0,
    consecutive_successes: 12,
    sample_count: 40,
    failure_count: 1,
    failure_rate: 0.025,
    opened_at: null,
    open_until: null,
    open_reason: null,
    consecutive_opens: 0,
    active_probes: 0,
    probe_successes: 0,
    last_failure_at: null,
    last_failure_category: null,
    last_success_at: "2026-08-22T11:59:00+00:00",
    last_transition_at: null,
    manual_reason: null,
    manual_operator: null,
    revision: 9,
    ...overrides,
  };
}

const ROUTES = {
  "/circuits": {
    ...VERSIONS,
    at: "2026-08-22T12:00:00+00:00",
    circuits: [
      circuit(),
      circuit({
        circuit_key: "ace_step:REFERENCE_CONDITIONED",
        task_type: "REFERENCE_CONDITIONED",
        state: "OPEN",
        consecutive_failures: 5,
        failure_rate: 1.0,
        sample_count: 5,
        failure_count: 5,
        open_reason: "5 consecutive failures (PROVIDER_TIMEOUT)",
        last_failure_category: "PROVIDER_TIMEOUT",
        open_until: "2026-08-22T12:00:30+00:00",
        last_transition_at: "2026-08-22T11:55:00+00:00",
      }),
      circuit({
        circuit_key: "ace_step:COVER",
        task_type: "COVER",
        sample_count: 0,
        failure_count: 0,
        failure_rate: null,
        consecutive_successes: 0,
      }),
    ],
    unconfigured_providers: [],
  },
  "/readiness": {
    ...VERSIONS,
    at: "2026-08-22T12:00:00+00:00",
    generation_available: true,
    degraded: true,
    summary: "serving with reduced capability; unavailable: REFERENCE_CONDITIONED",
    capabilities: [
      {
        capability: "TEXT_TO_MUSIC",
        status: "AVAILABLE",
        detail: "1 of 1 provider(s) serving",
        providers: [],
      },
      {
        capability: "REFERENCE_CONDITIONED",
        status: "UNAVAILABLE",
        detail: "5 consecutive failures (PROVIDER_TIMEOUT)",
        providers: [],
      },
    ],
    metrics: { circuits_open: 1, circuits_manual: 0 },
  },
  "/policy": {
    ...VERSIONS,
    resilience_enabled: true,
    failover_mode: "SAFE_EQUIVALENT_ONLY",
    failover_possible: false,
    routable_providers: ["ace_step"],
    circuit_policy: {
      consecutive_failure_threshold: 5,
      failure_rate_threshold: 0.5,
      minimum_samples: 10,
      open_duration_seconds: 30,
      maximum_open_duration_seconds: 600,
    },
  },
  "/transitions": {
    ...VERSIONS,
    transitions: [
      {
        id: "t1",
        circuit_key: "ace_step:REFERENCE_CONDITIONED",
        provider: "ace_step",
        task_type: "REFERENCE_CONDITIONED",
        previous_state: "CLOSED",
        current_state: "OPEN",
        occurred_at: "2026-08-22T11:55:00+00:00",
        reason: "5 consecutive failures (PROVIDER_TIMEOUT)",
        automatic: true,
        operator: null,
        circuit_policy_version: "circuit-v1",
      },
    ],
  },
};

function mockFetch(routes: Record<string, unknown>) {
  const spy = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const match = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((fragment) => url.includes(fragment));
    if (!match) {
      return new Response(JSON.stringify({ detail: `no route for ${url}` }), { status: 404 });
    }
    return new Response(JSON.stringify(routes[match]), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

async function renderPage(ui: ReactElement): Promise<RenderResult> {
  let result!: RenderResult;
  await act(async () => {
    result = render(<Suspense fallback={null}>{ui}</Suspense>);
  });
  return result;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("circuit panel", () => {
  it("shows each circuit with the reason it is open", async () => {
    mockFetch(ROUTES);
    await renderPage(<CircuitsPage />);

    const table = screen.getByRole("table", { name: /provider circuits/i });
    const row = within(table).getByText("REFERENCE_CONDITIONED").closest("tr")!;

    expect(within(row).getByText("OPEN")).toBeInTheDocument();
    expect(within(row).getByText(/5 consecutive failures/)).toBeInTheDocument();
  });

  it("reports no failure rate for a circuit nobody has measured", async () => {
    mockFetch(ROUTES);
    await renderPage(<CircuitsPage />);

    const table = screen.getByRole("table", { name: /provider circuits/i });
    const row = within(table).getByText("COVER").closest("tr")!;

    // Never 0%: nothing has been measured, and a zero would read as
    // "nothing is failing".
    expect(within(row).queryByText("0% of 0")).not.toBeInTheDocument();
    expect(within(row).getAllByTitle(/nobody has measured this/i).length).toBeGreaterThan(0);
  });

  it("says which capability is unavailable rather than only that something is wrong", async () => {
    mockFetch(ROUTES);
    await renderPage(<CircuitsPage />);

    expect(
      screen.getByText(/unavailable: REFERENCE_CONDITIONED/i),
    ).toBeInTheDocument();
    expect(screen.getByText("1 of 1 provider(s) serving")).toBeInTheDocument();
  });

  it("does not imply a redundancy that does not exist", async () => {
    mockFetch(ROUTES);
    await renderPage(<CircuitsPage />);

    // The mode is set, and it can still move nothing.
    expect(screen.getByText("SAFE_EQUIVALENT_ONLY")).toBeInTheDocument();
    expect(screen.getByText(/No request can be moved/i)).toBeInTheDocument();
  });

  it("offers no control that would change a circuit", async () => {
    mockFetch(ROUTES);
    const { container } = await renderPage(<CircuitsPage />);

    const labels = Array.from(container.querySelectorAll("button")).map(
      (button) => button.textContent?.toLowerCase() ?? "",
    );
    for (const forbidden of ["open circuit", "close circuit", "reset", "disable", "force"]) {
      expect(labels.some((label) => label.includes(forbidden))).toBe(false);
    }
    expect(screen.getByText(/luber_provider_resilience/)).toBeInTheDocument();
  });

  it("shows the transition log with who caused each change", async () => {
    mockFetch(ROUTES);
    await renderPage(<CircuitsPage />);

    const table = screen.getByRole("table", { name: /circuit transitions/i });
    expect(within(table).getByText("automatic")).toBeInTheDocument();
    expect(within(table).getByText(/5 consecutive failures/)).toBeInTheDocument();
  });

  it("names circuits for providers the deployment no longer configures", async () => {
    mockFetch({
      ...ROUTES,
      "/circuits": {
        ...ROUTES["/circuits"],
        unconfigured_providers: ["retired_provider"],
      },
    });
    await renderPage(<CircuitsPage />);

    expect(screen.getByText(/retired_provider/)).toBeInTheDocument();
  });

  it("renders nothing a user wrote", async () => {
    mockFetch(ROUTES);
    const { container } = await renderPage(<CircuitsPage />);

    const text = container.textContent ?? "";
    for (const needle of ["prompt", "lyrics", "Bearer", "sk-"]) {
      expect(text.toLowerCase()).not.toContain(needle.toLowerCase());
    }
  });
});
