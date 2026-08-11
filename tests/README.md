# End-to-end tests

Cross-service E2E tests land in Phase 3+:

login → create → prompt/lyrics input → vocal selection → generate →
job queued → worker (mock in CI, real on GPU hosts) → complete →
playback → WAV download.

Service-level tests live next to each service
(`apps/api/tests`, `apps/web/src/**/*.test.tsx`, `packages/*/tests`,
`services/*/tests`).
