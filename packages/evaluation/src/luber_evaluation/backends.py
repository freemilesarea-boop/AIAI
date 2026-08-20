"""Backends that produce real audio for an evaluation.

The synthetic backend in :mod:`luber_evaluation.runner` exercises the
machinery without a model. These two produce actual audio, and both are
built around one hazard: **a comparison is worthless if either side's
audio came from the wrong model.**

An ACE-Step server hosts one model at a time. Nothing in a request says
which weights answered it, so a single misconfigured base URL — or one
`run start` issued before the operator finished swapping checkpoints —
would attribute baseline audio to the candidate and produce a confident
verdict about nothing. Both backends therefore declare the model they
serve and refuse any case for a different one. The refusal is a
recorded failure, not an exception that stops the run: an operator who
pointed at the wrong server should see it in the results rather than in
a traceback halfway through.

`RenderedAudioBackend` ingests audio rendered elsewhere. It exists
because the GPU host and this tooling need not be the same machine, and
because re-analysing an existing set of renders must not require
re-generating them.

`AceStepEvaluationBackend` drives a running ACE-Step server through the
same provider the product uses. Not the evaluation-specific copy of it:
a second client would drift, and then the thing being measured would no
longer be the thing being shipped.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_evaluation.runner import GenerationBackend, GenerationOutcome
from luber_evaluation.schemas import EvaluationCase, ModelRef

#: Vocal selections a real backend can honour, mapped to the product's
#: own enum values. "unknown" is deliberately absent: a case that does
#: not state a voice is refused rather than assigned one.
VOCAL_GENDERS: frozenset[str] = frozenset({"female", "male", "instrumental"})

#: What a rendered file is called. Fixed so that an operator rendering
#: on a GPU box and an analysis run here cannot disagree about which
#: file belongs to which case and seed.
RENDER_NAME_TEMPLATE = "{case_id}__seed{seed}"
RENDER_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3")


def render_filename(case_id: str, seed: int, extension: str = ".wav") -> str:
    return RENDER_NAME_TEMPLATE.format(case_id=case_id, seed=seed) + extension


def _wrong_model(case: EvaluationCase, seed: int, serves: str, asked: str) -> GenerationOutcome:
    return GenerationOutcome(
        case_id=case.case_id,
        seed=seed,
        succeeded=False,
        error=(
            f"this backend serves model {serves!r} but was asked to generate for "
            f"{asked!r}; refusing rather than attributing audio to the wrong model"
        ),
    )


class RenderedAudioBackend(GenerationBackend):
    """Reads audio that was rendered ahead of time.

    Renders nothing and downloads nothing: given a directory it locates
    the file for each case and seed, and a missing file is a recorded
    failure rather than a substitution. Filling a gap with another
    seed's render, or with the other side's, would silently answer a
    different question than the one asked.
    """

    name = "rendered"
    produces_audio = True

    def __init__(self, root: Path, *, serves_model_id: str) -> None:
        self.root = Path(root)
        self.serves_model_id = serves_model_id

    def locate(self, case_id: str, seed: int) -> Path | None:
        for extension in RENDER_EXTENSIONS:
            candidate = self.root / render_filename(case_id, seed, extension)
            if candidate.is_file():
                return candidate
        return None

    def generate(
        self, case: EvaluationCase, seed: int, model: ModelRef, output_dir: Path
    ) -> GenerationOutcome:
        if model.model_id != self.serves_model_id:
            return _wrong_model(case, seed, self.serves_model_id, model.model_id)

        path = self.locate(case.case_id, seed)
        if path is None:
            expected = render_filename(case.case_id, seed)
            return GenerationOutcome(
                case_id=case.case_id,
                seed=seed,
                succeeded=False,
                error=f"no render found; expected {expected} under {self.root}",
            )
        return GenerationOutcome(
            case_id=case.case_id,
            seed=seed,
            succeeded=True,
            audio_path=path,
            # Duration is measured by the analyser from the file itself.
            # Reporting the requested duration here would make every
            # render look correct by construction.
            duration_seconds=None,
        )


@dataclass
class AceStepBackendConfig:
    """Where the server is, and what it is serving.

    ``serves_model_id`` and ``serves_checkpoint_id`` are operator
    assertions about weights this code cannot inspect over HTTP. They
    are recorded so that a later dispute has something to check, and
    enforced so a single URL cannot answer for both sides of one
    comparison.
    """

    base_url: str
    serves_model_id: str
    serves_checkpoint_id: str | None = None
    api_key_ref: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("evaluation-output"))
    inference_steps: int = 8
    request_timeout: float = 60.0
    poll_interval: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "serves_model_id": self.serves_model_id,
            "serves_checkpoint_id": self.serves_checkpoint_id,
            # A reference name, never a value. Phase 25's rule, kept.
            "api_key_ref": self.api_key_ref,
            "inference_steps": self.inference_steps,
        }


class AceStepEvaluationBackend(GenerationBackend):
    """Generates through a running ACE-Step server.

    Cannot run in CI, and does not pretend otherwise: there is no
    fallback path that produces audio when the server is absent. A
    backend that quietly degraded to something synthetic would be the
    one bug in this package guaranteed to go unnoticed.
    """

    name = "ace-step"
    produces_audio = True

    def __init__(self, config: AceStepBackendConfig, *, provider: Any = None) -> None:
        self.config = config
        self._provider = provider

    def provider(self) -> Any:
        """The product's own ACE-Step provider, constructed lazily.

        Imported inside the method so that using the synthetic backend
        never drags in an HTTP client, and so this module stays
        importable on a machine with no generation client installed.
        """
        if self._provider is None:
            from luber_generation_client.ace_step.provider import (
                AceStepProvider,
                AceStepProviderConfig,
            )

            self._provider = AceStepProvider(
                AceStepProviderConfig(
                    base_url=self.config.base_url,
                    model="acestep-v15-turbo",
                    request_timeout=self.config.request_timeout,
                    poll_interval=self.config.poll_interval,
                    output_dir=Path(self.config.output_dir),
                    inference_steps=self.config.inference_steps,
                )
            )
        return self._provider

    def _request(self, case: EvaluationCase, seed: int, output_dir: Path) -> Any:
        from luber_generation_client.provider import GenerationRequest
        from luber_schemas.enums import VocalGender

        spec = case.spec
        if spec.vocal_gender not in VOCAL_GENDERS:
            raise ValueError(
                f"case {case.case_id} does not state a vocal gender; a backend that chose "
                "one would make the two sides sing differently"
            )
        return GenerationRequest(
            title=case.case_id,
            prompt=spec.prompt or case.case_id,
            lyrics=spec.lyrics,
            vocal_gender=VocalGender(spec.vocal_gender),
            duration_seconds=round(spec.duration_seconds),
            seed=seed,
            language=spec.language if spec.language != "unknown" else None,
            instrumental=spec.vocal_gender == "instrumental",
            bpm=spec.bpm,
            key_scale=spec.key_scale,
            time_signature=spec.time_signature,
        )

    def generate(
        self, case: EvaluationCase, seed: int, model: ModelRef, output_dir: Path
    ) -> GenerationOutcome:
        if model.model_id != self.config.serves_model_id:
            return _wrong_model(case, seed, self.config.serves_model_id, model.model_id)

        started = time.monotonic()
        try:
            request = self._request(case, seed, output_dir)
        except ValueError as exc:
            return GenerationOutcome(
                case_id=case.case_id, seed=seed, succeeded=False, error=str(exc)
            )

        try:
            result = asyncio.run(self.provider().generate(request))
        except Exception as exc:
            # Recorded, not raised. One case failing is a measurement —
            # it is exactly what generation_failure_rate counts — and
            # aborting the run would discard every case already done.
            return GenerationOutcome(
                case_id=case.case_id,
                seed=seed,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
                timed_out="timeout" in str(exc).lower(),
                wall_seconds=time.monotonic() - started,
            )

        return GenerationOutcome(
            case_id=case.case_id,
            seed=seed,
            succeeded=True,
            audio_path=Path(result.audio_path),
            duration_seconds=float(result.duration_seconds),
            wall_seconds=time.monotonic() - started,
        )


__all__ = [
    "RENDER_EXTENSIONS",
    "RENDER_NAME_TEMPLATE",
    "VOCAL_GENDERS",
    "AceStepBackendConfig",
    "AceStepEvaluationBackend",
    "RenderedAudioBackend",
    "render_filename",
]
