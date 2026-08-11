# Testing

## Layers

| Layer | Tooling | Location |
|---|---|---|
| Backend unit | pytest (+ pytest-asyncio, fakeredis, aiosqlite) | `apps/api/tests`, `packages/*/tests`, `services/*/tests` |
| Generation contract | pytest | `packages/generation-client/tests` |
| API integration | httpx ASGI client | `apps/api/tests` |
| Database | model/metadata tests now; migration tests against real PostgreSQL in CI | `packages/database/tests` + CI job |
| Frontend | Vitest + Testing Library | `apps/web/src/**/*.test.tsx` |
| E2E (Phase 3+) | browser-driven: create → generate → playback → WAV download | `tests/` |

## Commands

```bash
uv run pytest            # all backend tests
pnpm test                # frontend tests
uv run ruff check .      # lint (bare except forbidden: E722)
uv run mypy .            # strict typing (no `any` sprawl on TS side either)
pnpm lint && pnpm typecheck
```

## Rules

- CI never runs GPU models. `MockGenerationProvider` (Phase 1) returns
  a real fixture WAV and satisfies the same interface contract as
  production providers.
- Real-model tests are explicit, opt-in integration tests separated
  from the default CI suite (Phase 2).
- Mock results are never reported as real AI generation success.
- Tests are never deleted to make CI pass.
- Every phase runs its tests before being declared complete.
