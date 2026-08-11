# Model Provider Abstraction

`packages/generation-client` defines the only seam between the LUBER
platform and any music generation engine.

```python
class MusicGenerationProvider(ABC):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
```

## Rules

1. Business logic (API, services) imports **only** this interface.
2. Each engine gets its own provider implementation:
   - `MockGenerationProvider` — Phase 1; returns a real (tiny) fixture
     WAV; used by CI; never pretends to be a real model run.
   - `AceStepProvider` — implemented (Phase 2); drives a self-hosted
     ACE-Step 1.5 API server via its documented async REST interface
     (`release_task` → `query_result` poll → `/v1/audio` download),
     verified against the pinned upstream — no invented API fields.
     See `ACE_STEP_UPSTREAM_AUDIT.md` for the pin and protocol notes.
     Select with `GENERATION_PROVIDER=ace_step` + `ACE_STEP_*` env vars.
   - Future: Stable Audio family, licensed models, custom foundation
     model.
3. Providers return raw model output; post-processing/upload belong to
   the audio worker.
4. A provider failure surfaces as a standard `ErrorCode` — never a
   fabricated success.

## Prompt compilation & vocal conditioning (Phase 2+)

User input is never passed verbatim to a model. A `PromptCompiler`
adapter per provider converts
`{prompt, vocal_gender, language}` into model-specific conditioning:

- `female` → "female lead vocal, natural female singing voice"
- `male` → "male lead vocal, natural male singing voice"
- `instrumental` → "instrumental, no vocals"

Lyrics are passed to the vocal model separately with section tags
(`[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`,
`[Bridge]`, `[Break]`, `[Outro]`) preserved. Korean lyrics are never
force-translated.

Voice cloning of specific real artists is not offered. Artist-name
prompts are normalized into musical characteristics (e.g. "warm female
vocal, breathy delivery, dreamy indie pop"), not identity replication.

## Model configuration

Per-deployment YAML (Phase 2) selects the provider and model:

```yaml
provider: ace_step
model:
  dit: acestep-v15-xl-turbo
  lm: acestep-5Hz-lm-1.7B
generation:
  default_duration: 180
  max_duration: 360
  output_sample_rate: 48000
  output_format: wav
```

Every generation records `provider`, `model_name`, `model_version` for
reproducibility.
