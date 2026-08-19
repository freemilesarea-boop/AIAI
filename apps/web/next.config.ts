import type { NextConfig } from "next";

/**
 * The browser talks to the API through this origin, never directly.
 *
 * Every product request goes to a same-origin `/api/...` path, which
 * Next rewrites to the FastAPI backend. That is what makes a
 * `SameSite=Lax` session cookie work: the browser sees one origin, so
 * the cookie is same-site on every request including POSTs, and no
 * credentialed cross-origin traffic is needed.
 *
 * The alternative — the browser calling :8000 directly — would force
 * `SameSite=None`, which requires HTTPS even locally and reintroduces
 * the CSRF exposure that Lax removes. That trade was rejected
 * deliberately; see docs/AUTHENTICATION_ARCHITECTURE.md.
 *
 * In production the same shape holds with one public origin in front of
 * both, so nothing about this arrangement is development-only.
 */
const apiTarget = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";

/**
 * Conservative headers only — ones that cannot break audio delivery,
 * range requests or the same-origin API proxy.
 *
 * Deliberately absent: Content-Security-Policy and HSTS. A CSP tight
 * enough to be worth having needs the app's real script/style/media
 * origins measured under a production build, and a loose one is
 * decoration. HSTS is meaningless until the deployment terminates TLS,
 * and setting it on a localhost origin would be a nuisance to undo.
 * Both are recorded as deployment work rather than guessed at here.
 */
const securityHeaders = [
  // Nothing in LUBER is meant to be framed, and clickjacking a player
  // that streams private audio is a real if modest risk.
  { key: "X-Frame-Options", value: "DENY" },
  // Stops a browser second-guessing a declared audio/wav or JSON type.
  { key: "X-Content-Type-Options", value: "nosniff" },
  // Song ids live in URLs; a full referrer would leak them to any
  // outbound link. Origin-only on cross-origin, full path same-origin.
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  // No feature here needs these, and denying them costs nothing.
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
];

const nextConfig: NextConfig = {
  transpilePackages: ["@luber/ui"],
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
  async rewrites() {
    return [
      // `/api/*` is reserved for the backend. Next's own routes live
      // under /app, so there is no collision to design around.
      { source: "/api/:path*", destination: `${apiTarget}/:path*` },
    ];
  },
};

export default nextConfig;
