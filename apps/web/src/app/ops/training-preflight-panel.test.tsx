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
import {
  canaryRun,
  capacity,
  memoryProfile,
  pilot,
  runDetail,
  trainingPreflight,
} from "@/test/ops-factories";

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

describe("memory and capacity panel", () => {
  it("shows a QUALIFIED verdict with the measured peak and the reserve", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getAllByText("QUALIFIED").length).toBeGreaterThan(0);
    expect(within(panel).getAllByText(/9751 MiB/).length).toBeGreaterThan(0);
    expect(within(panel).getAllByText(/SAMPLED_PEAK/).length).toBeGreaterThan(0);
  });

  it("never calls Apple unified memory VRAM", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getAllByText(/unified memory/i).length).toBeGreaterThan(0);
    // The only permitted use of the word is the disclaimer itself.
    const text = panel.textContent ?? "";
    expect(text.replace(/not VRAM/g, "")).not.toMatch(/VRAM/);
  });

  it("shows the sequence a profile was measured on, not just the peak", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getAllByText(/6000 latent frames/).length).toBeGreaterThan(0);
    expect(within(panel).getByText("REPRESENTATIVE")).toBeInTheDocument();
  });

  it("marks a derived figure as derived rather than measured", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getByText("DERIVED")).toBeInTheDocument();
    expect(within(panel).getByText(/x 1.25 safety margin/)).toBeInTheDocument();
  });

  it("shows INSUFFICIENT as a failure, not a caveat", async () => {
    await renderRun(
      runDetail({
        capacity: capacity({
          qualification: "INSUFFICIENT",
          domains: [
            {
              domain: "APPLE_UNIFIED",
              qualification: "INSUFFICIENT",
              peak_mib: 22000,
              peak_kind: "SAMPLED_PEAK",
              required_mib: 27500,
              reserved_mib: 3686,
              budget_mib: 20889,
              total_mib: 24576,
              detail: "27500 MiB required against a 20889 MiB budget",
            },
          ],
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getAllByText("INSUFFICIENT").length).toBeGreaterThan(0);
    expect(within(panel).queryByText("QUALIFIED")).not.toBeInTheDocument();
  });

  it("says plainly when nothing has been measured", async () => {
    await renderRun(
      runDetail({
        capacity: capacity({
          available: false,
          unavailable_reason: "No memory profile has been recorded for this run.",
          qualification: "UNVERIFIED",
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getByText(/No memory profile has been recorded/)).toBeInTheDocument();
  });

  it("says a profile of an unrealistic workload is not representative", async () => {
    await renderRun(
      runDetail({
        capacity: capacity({
          qualification: "UNVERIFIED",
          profile: memoryProfile({
            representativeness: "NOT_REPRESENTATIVE",
            latent_length: 64,
            latent_seconds: 2.6,
            representativeness_detail: "64 latent frames ≈ 3s of audio",
          }),
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getByText("NOT_REPRESENTATIVE")).toBeInTheDocument();
    expect(within(panel).getAllByText("UNVERIFIED").length).toBeGreaterThan(0);
  });

  it("carries the checkpoint and resume peaks separately", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getByText("9393 MiB")).toBeInTheDocument();
    expect(within(panel).getByText("5464 MiB")).toBeInTheDocument();
  });

  it("says a memory profile makes no quality claim", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Memory and capacity/i });
    expect(within(panel).getByText(/nothing about music quality/i)).toBeInTheDocument();
  });
});

describe("bounded pilot panel", () => {
  it("shows a valid signal with the steps it took and the ceiling", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText("COMPLETED_VALID_SIGNAL")).toBeInTheDocument();
    expect(within(panel).getByText("VALID_SIGNAL")).toBeInTheDocument();
    expect(within(panel).getByText(/48 of 48 \(ceiling 48\)/)).toBeInTheDocument();
  });

  it("makes no convergence or quality claim", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    const text = panel.textContent ?? "";
    expect(text).toMatch(/nothing about\s+convergence, music quality/);
    expect(text).not.toMatch(/converged|improved the model|higher quality/i);
  });

  it("labels a synthetic pilot as not evidence about real music", async () => {
    await renderRun(runDetail({ pilot: pilot({ dataset_kind: "SYNTHETIC_FIXTURE" }) }));

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText("SYNTHETIC_FIXTURE")).toBeInTheDocument();
    expect(within(panel).getByText(/not\s+evidence about real music/)).toBeInTheDocument();
  });

  it("marks real rights-cleared material distinctly from a fixture", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText("REAL_OPERATOR_AUTHORIZED")).toBeInTheDocument();
    expect(within(panel).queryByText(/not\s+evidence about real music/)).not.toBeInTheDocument();
  });

  it("shows NO_UPDATE as a failure rather than a quiet pass", async () => {
    await renderRun(
      runDetail({
        pilot: pilot({
          outcome: "FAILED_NUMERIC",
          signal: "NO_UPDATE",
          signal_detail: "the loss was finite throughout and no trainable parameter changed",
          failure: "NO_PARAMETER_UPDATE",
          failure_detail: "no trainable parameter changed",
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText("FAILED_NUMERIC")).toBeInTheDocument();
    expect(within(panel).getByText("NO_UPDATE")).toBeInTheDocument();
    expect(within(panel).getByText(/NO_PARAMETER_UPDATE/)).toBeInTheDocument();
  });

  it("labels the slope as derived rather than as a trend", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText("Slope (DERIVED)")).toBeInTheDocument();
  });

  it("shows both segments and the resume between them", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText(/resumed from epoch_24_loss_3.3100/)).toBeInTheDocument();
  });

  it("says every pilot artifact is never auto-promoted", async () => {
    await renderRun(runDetail());

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText(/NEVER_AUTO_PROMOTE/)).toBeInTheDocument();
  });

  it("says plainly when no pilot has been run", async () => {
    await renderRun(
      runDetail({
        pilot: pilot({
          available: false,
          unavailable_reason: "No pilot has been run for this run.",
          outcome: "NOT_RUN",
        }),
      }),
    );

    const panel = screen.getByRole("region", { name: /Bounded pilot/i });
    expect(within(panel).getByText(/No pilot has been run/)).toBeInTheDocument();
  });
});
