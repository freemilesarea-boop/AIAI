"""Generation worker service.

Phase 0: queue plumbing only (a ``ping`` task proving the Redis/ARQ
loop works). Phase 1 wires MockGenerationProvider; Phase 2 adds the
real ACE-Step provider on GPU hosts. Model inference never runs inside
the API process.
"""
