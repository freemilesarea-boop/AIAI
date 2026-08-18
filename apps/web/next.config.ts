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

const nextConfig: NextConfig = {
  transpilePackages: ["@luber/ui"],
  async rewrites() {
    return [
      // `/api/*` is reserved for the backend. Next's own routes live
      // under /app, so there is no collision to design around.
      { source: "/api/:path*", destination: `${apiTarget}/:path*` },
    ];
  },
};

export default nextConfig;
