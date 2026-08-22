/**
 * Phase 33 in the browser — the three statuses, kept apart.
 *
 * The whole value of a training preflight is that an operator can tell
 * READY from UNVERIFIED at a glance, before renting a GPU. So these
 * tests defend the distinction rather than the layout: UNVERIFIED must
 * not render as a pass, a BLOCKED result must show its machine-readable
 * reason, and a capacity figure must never lose the fact that nobody
 * measured it.
 */

import { act, render, screen, within } from "@testing-library/react";
import { Suspense, type ReactElement } from "react";

import RunDetailPage from "@/app/ops/training/runs/[id]/page";
import { OpsStatus } from "@/components/ops/primitives";
import { canaryRun, runDetail, trainingPreflight } from "@/test/ops-factories";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/ops/training",
  useParams: () => ({ id: "run_1" }),
}));

function mockFetch(routes: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const match = Object.keys(routes).find((fragment) => url.includes(fragment));
      if (!match) {
        return new Response(JSON.stringify({ detail: `no route for ${url}` }), { status: 404 });
      }
      return new Response(JSON.stringify(routes[match]), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }),
  );
}

async function renderRun(detail: ReturnType<typeof runDetail>) {
  mockFetch({
    "/diagnostics": [],
    "/logs": {
      available: false,
      unavailable_reason: "no logs",
      stream: "stdout",
      offset: 0,
      next_offset: 0,
      size_bytes: 0,
      eof: true,
      truncated: false,
      text: "",
      from_tail: false,
    },
    "/runs/": detail,
  });
  const element: ReactElement = (
    <RunDetailPage params={Promise.resolve({ id: detail.run.run_id })} />
  );
  // Pages read their route params with React's `use()`, which suspends.
  // `act` has to be awaited or the tree never resumes and every
  // assertion runs against an empty body.
  await act(async () => {
    render(<Suspense fallback={null}>{element}</Suspense>);
  });
  await screen.findByText("Lifecycle");
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("training preflight status encoding", () => {
  it("gives READY a passing mark rather than the neutral one READY means elsewhere", () => {
    const { container } = render(<OpsStatus status="READY" tone="good" />);
    expect(container.textContent).toContain("READY");
    expect(container.textContent).toContain("✓");
  });

  it("marks UNVERIFIED with the unknown symbol, never a tick", () => {
    const { container } = render(<OpsStatus status="UNVERIFIED" />);
    expect(container.textContent).toContain("?");
    expect(container.textContent).not.toContain("✓");
  });

  it("marks BLOCKED as a failure", () => {
    const { container } = render(<OpsStatus status="BLOCKED" />);
    expect(container.textContent).toContain("✕");
  });

  it("does not let an estimate look like a measurement", () => {
    const measured = render(<OpsStatus status="MEASURED" />).container.textContent;
    const estimated = render(<OpsStatus status="ESTIMATED" />).container.textContent;
    expect(measured).toContain("✓");
    expect(estimated).not.toContain("✓");
  });
});

describe("training preflight panel", () => {
  it("renders a READY preflight with its device, precision and plan digest", async () => {
    await renderRun(
      runDetail({
        training_preflight: trainingPreflight({
          status: "READY",
          intent: "CANARY",
          execution_device: "MPS",
          resolved_precision: "bf16",
          unverified: [],
          checks: [],
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Training preflight/i });
    expect(within(panel).getByText("READY")).toBeInTheDocument();
    expect(within(panel).getByText("MPS")).toBeInTheDocument();
    expect(within(panel).getByText("bf16")).toBeInTheDocument();
    expect(within(panel).getByLabelText("Copy plan digest")).toBeInTheDocument();
  });

  it("shows a BLOCKED preflight with its machine-readable reason", async () => {
    await renderRun(
      runDetail({
        training_preflight: trainingPreflight({
          status: "BLOCKED",
          blocking_reasons: [
            "DEVICE_UNAVAILABLE: hardware.device: fixture-mac does not offer CUDA",
          ],
          unverified: [],
          checks: [
            {
              name: "hardware.device",
              group: "hardware",
              status: "FAIL",
              detail: "fixture-mac does not offer CUDA",
              reason: "DEVICE_UNAVAILABLE",
              mandatory: true,
            },
          ],
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Training preflight/i });
    expect(within(panel).getByText("BLOCKED")).toBeInTheDocument();
    expect(within(panel).getAllByText(/DEVICE_UNAVAILABLE/).length).toBeGreaterThan(0);
  });

  it("keeps UNVERIFIED visually distinct from a pass and says so in words", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Training preflight/i });
    expect(within(panel).getByText("UNVERIFIED")).toBeInTheDocument();
    expect(within(panel).getByText(/Could not be established — not a pass/)).toBeInTheDocument();
    expect(within(panel).queryByText("READY")).not.toBeInTheDocument();
  });

  it("says an unknown capacity figure is unknown rather than showing a number", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Training preflight/i });
    expect(within(panel).getByText("training_memory_requirement_mb")).toBeInTheDocument();
    expect(within(panel).getAllByText("UNKNOWN").length).toBeGreaterThan(0);
  });

  it("labels unified memory as shared with the OS rather than as VRAM", async () => {
    await renderRun(
      runDetail({
        training_preflight: trainingPreflight({
          execution_device: "MPS",
          capacity: [
            {
              name: "device_memory_mb",
              source: "MEASURED",
              value_mb: 24576,
              detail: "unified memory",
              derivation: "",
              unified_memory: true,
            },
          ],
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Training preflight/i });
    expect(within(panel).getByText(/not VRAM/)).toBeInTheDocument();
  });

  it("says plainly when nobody has run one", async () => {
    await renderRun(
      runDetail({
        training_preflight: trainingPreflight({
          available: false,
          unavailable_reason: "No training preflight has been recorded for this run.",
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Training preflight/i });
    expect(within(panel).getByText(/No training preflight has been recorded/)).toBeInTheDocument();
  });
});

describe("canary panel", () => {
  it("shows the bounds a canary ran under, not just that it passed", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Bounded canary/i });
    expect(within(panel).getByText("PASSED")).toBeInTheDocument();
    expect(within(panel).getByText(/of at most/)).toBeInTheDocument();
    expect(
      within(panel).getByText(/proves the mechanism and nothing about the model/i),
    ).toBeInTheDocument();
  });

  it("reports a failed checkpoint with its problems", async () => {
    await renderRun(
      runDetail({
        canary: canaryRun({
          status: "FAILED",
          checkpoint_ok: false,
          checkpoint_problems: ["every adapter tensor is zero"],
          resume_ok: null,
          resume_detail: "",
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Bounded canary/i });
    expect(within(panel).getByText("FAILED")).toBeInTheDocument();
    expect(within(panel).getByText(/every adapter tensor is zero/)).toBeInTheDocument();
  });

  it("says plainly when no canary has been run", async () => {
    await renderRun(
      runDetail({
        canary: canaryRun({
          available: false,
          unavailable_reason: "No canary has been run for this run.",
          status: "NOT_RUN",
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Bounded canary/i });
    expect(within(panel).getByText(/No canary has been run/)).toBeInTheDocument();
  });
});
