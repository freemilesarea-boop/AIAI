/**
 * Phase 32 — the compute-targets panel, in the browser.
 *
 * What is defended here is the panel's honesty about absence. An
 * operator plans against this page: a row that claims a GPU nobody
 * rented, or promises a Mac can take heavy training when the scheduler
 * refuses it, is planning material that is wrong.
 */

import { act, render, screen, within, type RenderResult } from "@testing-library/react";
import { Suspense, type ReactElement } from "react";

import ComputeTargetsPage from "@/app/ops/training/compute/page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/ops/training/compute",
  useParams: () => ({}),
}));

const PAYLOAD = {
  at: "2026-08-22T12:00:00+00:00",
  summary: "2 target(s) ready; none can take HEAVY_TRAINING — no CUDA worker is connected",
  local_training_concurrency: 1,
  capability_schema_version: "luber-hardware-capability/1",
  execution_placement_policy_version: "placement-v1",
  targets: [
    {
      name: "control-plane",
      location: "LOCAL",
      device: "MPS",
      status: "READY",
      detail: "",
      memory_mb: 24576,
      precisions: ["fp32", "fp16", "bf16"],
      workloads: ["INFERENCE", "LIGHT_FINE_TUNE", "CHECKPOINT_VALIDATION"],
      limitations: [],
      planned: false,
      capability_digest: "abc123",
    },
    {
      name: "control-plane",
      location: "LOCAL",
      device: "CPU",
      status: "READY",
      detail: "",
      memory_mb: 24576,
      precisions: ["fp32"],
      workloads: ["PREPROCESS", "EVALUATION"],
      limitations: [],
      planned: false,
      capability_digest: "abc123",
    },
    {
      name: "UNKNOWN",
      location: "REMOTE",
      device: "CUDA",
      status: "NOT_CONNECTED",
      detail: "no NVIDIA worker has been registered with this deployment",
      memory_mb: null,
      precisions: [],
      workloads: [],
      limitations: [],
      planned: false,
      capability_digest: null,
    },
  ],
};

function mockFetch(payload: unknown) {
  const spy = vi.fn(
    async () =>
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
  );
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

describe("compute targets panel", () => {
  it("shows a remote CUDA row even when there is no GPU", async () => {
    mockFetch(PAYLOAD);
    await renderPage(<ComputeTargetsPage />);

    const table = screen.getByRole("table", { name: /compute targets/i });
    const row = within(table).getByText("NOT_CONNECTED").closest("tr")!;

    expect(within(row).getByText("CUDA")).toBeInTheDocument();
    expect(
      within(row).getByText(/no NVIDIA worker has been registered/i),
    ).toBeInTheDocument();
  });

  it("never claims a local target can take heavy training", async () => {
    mockFetch(PAYLOAD);
    const { container } = await renderPage(<ComputeTargetsPage />);

    expect(container.textContent).not.toContain("HEAVY_TRAINING,");
    expect(screen.getByText(/none can take HEAVY_TRAINING/i)).toBeInTheDocument();
  });

  it("leaves an unmeasured precision blank rather than calling it none", async () => {
    mockFetch(PAYLOAD);
    await renderPage(<ComputeTargetsPage />);

    const table = screen.getByRole("table", { name: /compute targets/i });
    const row = within(table).getByText("NOT_CONNECTED").closest("tr")!;

    // "nobody measured" and "nothing works" are different claims.
    expect(within(row).getAllByTitle(/nobody has measured this/i).length).toBeGreaterThan(0);
  });

  it("states the local training concurrency limit and why", async () => {
    mockFetch(PAYLOAD);
    const { container } = await renderPage(<ComputeTargetsPage />);

    // Read from the whole panel rather than one element: the sentence
    // is deliberately broken up by emphasis, and asserting on a single
    // text node would be testing the markup rather than the claim.
    const text = container.textContent ?? "";
    expect(text).toContain("Concurrent local training jobs:");
    expect(text).toContain("refused when none is connected");
    expect(text).toContain("nobody chose");
  });

  it("marks a planned profile so it cannot be mistaken for hardware", async () => {
    mockFetch({
      ...PAYLOAD,
      targets: [
        { ...PAYLOAD.targets[0], name: "planned-mac-mini", planned: true },
        ...PAYLOAD.targets.slice(1),
      ],
    });
    await renderPage(<ComputeTargetsPage />);

    expect(screen.getByText("planned")).toBeInTheDocument();
  });

  it("renders nothing that identifies a host", async () => {
    mockFetch(PAYLOAD);
    const { container } = await renderPage(<ComputeTargetsPage />);

    const text = container.textContent ?? "";
    for (const needle of ["/Users/", "/home/", "ssh", "@"]) {
      expect(text).not.toContain(needle);
    }
  });
});
