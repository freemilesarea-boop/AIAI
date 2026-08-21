/**
 * The operator token lives here and nowhere the browser can reach.
 *
 * The console needs an operator token on every request, and there is no
 * safe way for a browser to hold one: a token in `NEXT_PUBLIC_*` is in
 * the JavaScript bundle, a token in `localStorage` is one XSS away, and
 * a token in a URL is in the proxy logs. So the browser calls this
 * same-origin path with no credential at all, and the server attaches
 * the token as the request leaves for the API.
 *
 * What that buys, precisely: the secret exists only in the Next server's
 * environment. A page can trigger an operator action and still never
 * have held the thing that authorises it.
 *
 * What it does not buy: this route is as reachable as the console is.
 * Anyone who can load `/ops/training` can reach `/ops/api`, which is why
 * the console is a non-production deployment switch rather than a
 * permission — see `docs/TRAINING_CONSOLE.md`.
 *
 * The route 404s when the console is off, matching what the API does, so
 * a deployment that has not enabled it exposes no operator surface at
 * all rather than an endpoint that answers 401.
 *
 * The first path segment names an operator namespace and is checked
 * against an allowlist. Phase 30 added a second console — inference
 * observability — and the alternative to a namespace was either filing
 * inference under `/v1/ops/training`, which would be a lie about the
 * system, or a second proxy route holding a second copy of the token.
 * An allowlist keeps one credential in one place and makes a third
 * namespace a deliberate edit rather than a reachable path.
 */

import { NextResponse } from "next/server";

/**
 * Where the FastAPI backend lives, as the rest of the app resolves it.
 *
 * Read per request rather than captured at import, matching how the
 * token is read. A value frozen at module load is one a test cannot set
 * and a running process cannot correct.
 */
function apiTarget(): string {
  return process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";
}

const OPERATOR_TOKEN_HEADER = "X-Luber-Operator-Token";

/**
 * Never `NEXT_PUBLIC_`. That prefix is what puts a value in the client
 * bundle, and this is the one value in the application that must not be
 * there.
 */
function operatorToken(): string | undefined {
  return process.env.OPS_OPERATOR_TOKEN;
}

function consoleEnabled(): boolean {
  return process.env.OPS_CONSOLE_ENABLED === "true";
}

const notFound = () => NextResponse.json({ detail: "Not found." }, { status: 404 });

/**
 * Headers worth forwarding upstream.
 *
 * An allowlist rather than a copy: forwarding the browser's `Cookie`
 * would send a product session to an operator endpoint that has no use
 * for one, and forwarding `Authorization` would let a caller try to
 * influence how the API authenticates a request this server is
 * authorising.
 */
const FORWARDED = ["content-type", "accept", "origin"];

/**
 * Operator namespaces this proxy will forward to.
 *
 * An allowlist rather than a pass-through: without it, `/ops/api/foo`
 * would reach `/v1/ops/foo`, and any operator surface added to the API
 * later would become browser-reachable by having been mounted rather
 * than by anybody deciding it should be.
 */
const NAMESPACES = new Set(["training", "inference"]);

async function proxy(request: Request, path: string[]): Promise<Response> {
  if (!consoleEnabled()) return notFound();

  const token = operatorToken();
  if (!token) {
    return NextResponse.json(
      {
        detail:
          "The operator console is enabled but OPS_OPERATOR_TOKEN is not set on the web " +
          "server, so requests cannot be authorised.",
      },
      { status: 503 },
    );
  }

  const [namespace, ...rest] = path;
  if (!NAMESPACES.has(namespace)) return notFound();

  const incoming = new URL(request.url);
  const target = new URL(`/v1/ops/${namespace}/${rest.join("/")}`, apiTarget());
  target.search = incoming.search;

  const headers = new Headers();
  for (const name of FORWARDED) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set(OPERATOR_TOKEN_HEADER, token);

  const body =
    request.method === "GET" || request.method === "HEAD" ? undefined : await request.text();

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
    });
  } catch {
    // The backend being down is an infrastructure fact, not a mystery.
    // Saying so beats a stack trace rendered as a blank panel.
    return NextResponse.json(
      { detail: "The LUBER API is not reachable from this server." },
      { status: 502 },
    );
  }

  // Streamed straight through, so a markdown report or a bundle download
  // keeps its own content type and disposition.
  const responseHeaders = new Headers();
  for (const name of ["content-type", "content-disposition"]) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  responseHeaders.set("cache-control", "no-store");

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export async function GET(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxy(request, path);
}

export async function POST(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxy(request, path);
}
