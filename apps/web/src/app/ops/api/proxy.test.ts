/**
 * The proxy that holds the operator token.
 *
 * This is the reason a browser never has a credential for the operator
 * API, so it is worth its own file. Four properties:
 *
 * The console being off means the route does not exist — 404, matching
 * the API, so a deployment that has not enabled it exposes no operator
 * surface at all rather than an endpoint that answers 401.
 *
 * A half configuration serves nothing. Enabled with no token is the
 * shape that would otherwise send unauthenticated requests upstream.
 *
 * The token is attached server-side and never returned. A response that
 * echoed it back would undo the whole arrangement.
 *
 * A browser's `Cookie` is not forwarded. A product session has no
 * business reaching an operator endpoint, and forwarding it would be the
 * beginning of one being consulted there.
 */

import { GET, POST } from "@/app/ops/api/[...path]/route";

const TOKEN = "test-operator-token";

function context(path: string[]) {
  return { params: Promise.resolve({ path }) };
}

let fetchSpy: ReturnType<typeof vi.fn>;

beforeEach(() => {
  process.env.OPS_CONSOLE_ENABLED = "true";
  process.env.OPS_OPERATOR_TOKEN = TOKEN;
  process.env.API_PROXY_TARGET = "http://api.test";

  fetchSpy = vi.fn(
    async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", fetchSpy);
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete process.env.OPS_CONSOLE_ENABLED;
  delete process.env.OPS_OPERATOR_TOKEN;
  delete process.env.API_PROXY_TARGET;
});

describe("operator proxy", () => {
  it("is absent when the console is not enabled", async () => {
    delete process.env.OPS_CONSOLE_ENABLED;

    const response = await GET(new Request("http://web.test/ops/api/overview"), context(["overview"]));

    expect(response.status).toBe(404);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("refuses to serve when no token is configured", async () => {
    delete process.env.OPS_OPERATOR_TOKEN;

    const response = await GET(new Request("http://web.test/ops/api/overview"), context(["overview"]));

    expect(response.status).toBe(503);
    expect(await response.json()).toMatchObject({ detail: expect.stringContaining("OPS_OPERATOR_TOKEN") });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("attaches the token upstream and never returns it", async () => {
    const response = await GET(
      new Request("http://web.test/ops/api/runs?status=RUNNING"),
      context(["runs"]),
    );

    expect(response.status).toBe(200);
    const [target, init] = fetchSpy.mock.calls[0];
    expect(String(target)).toBe("http://api.test/v1/ops/training/runs?status=RUNNING");
    expect((init.headers as Headers).get("X-Luber-Operator-Token")).toBe(TOKEN);

    // Nothing carrying the secret comes back to the browser.
    expect(response.headers.get("X-Luber-Operator-Token")).toBeNull();
    expect(await response.text()).not.toContain(TOKEN);
  });

  it("does not forward a product session cookie", async () => {
    await GET(
      new Request("http://web.test/ops/api/overview", {
        headers: { cookie: "luber_session=secret-session-value", origin: "http://web.test" },
      }),
      context(["overview"]),
    );

    const headers = fetchSpy.mock.calls[0][1].headers as Headers;
    expect(headers.get("cookie")).toBeNull();
    // Origin is forwarded, so the API's own CSRF check still applies.
    expect(headers.get("origin")).toBe("http://web.test");
  });

  it("forwards a POST body and its content type", async () => {
    await POST(
      new Request("http://web.test/ops/api/runs/run_1/actions/cancel", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      }),
      context(["runs", "run_1", "actions", "cancel"]),
    );

    const [target, init] = fetchSpy.mock.calls[0];
    expect(String(target)).toBe("http://api.test/v1/ops/training/runs/run_1/actions/cancel");
    expect(init.method).toBe("POST");
    expect(init.body).toBe("{}");
  });

  it("says the API is unreachable rather than throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("ECONNREFUSED");
      }),
    );

    const response = await GET(new Request("http://web.test/ops/api/overview"), context(["overview"]));

    expect(response.status).toBe(502);
    expect(await response.json()).toMatchObject({
      detail: expect.stringContaining("not reachable"),
    });
  });
});
