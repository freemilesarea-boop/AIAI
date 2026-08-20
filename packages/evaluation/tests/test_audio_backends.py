"""Backends that touch real audio, and the guard that keeps them honest.

The synthetic backend is enough to test gates and verdicts. It is not
enough to test the path that actually measures music, so these tests
render short WAVs and run them through the same analyser Phase 22
decides from.

The mis-attribution guard gets its own tests because it is the one
failure that produces a confident, plausible, entirely wrong result: a
comparison where both sides came from the same weights looks exactly
like a comparison where nothing improved.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from luber_evaluation.backends import (
    AceStepBackendConfig,
    AceStepEvaluationBackend,
    RenderedAudioBackend,
    render_filename,
)
from luber_evaluation.metrics import MetricStatus
from luber_evaluation.runner import EvaluationRun, execute_side
from luber_evaluation.schemas import CandidateLineage, ModelRef
from luber_evaluation.suite import smoke_suite

RATE = 44_100


def write_wav(path: Path, seconds: float, *, amplitude: float = 0.3, rate: int = RATE) -> Path:
    """A short stereo tone. Real audio, small enough to analyse fast."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    time = np.arange(frames) / rate
    left = amplitude * np.sin(2 * np.pi * 220.0 * time)
    right = amplitude * np.sin(2 * np.pi * 220.0 * time + 0.4)
    stereo = np.stack([left, right], axis=1)
    data = np.clip(stereo, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((data * 32767).astype("<i2").tobytes())
    return path


def _run(evaluation_id: str = "eval_" + "0" * 16) -> EvaluationRun:
    suite = smoke_suite(seeds=(11,))
    return EvaluationRun(
        evaluation_id=evaluation_id,
        suite=suite,
        baseline=ModelRef(model_id="mdl_base", upstream_commit="abc"),
        candidate=ModelRef(model_id="cand_new", upstream_commit="abc", checkpoint_id="ckpt_new"),
        lineage=CandidateLineage(
            candidate_id="cand_new",
            checkpoint_id="ckpt_new",
            run_id="run_1",
            experiment_id="exp_1",
            base_model_id="mdl_base",
        ),
        seeds=(11,),
    )


def test_rendered_audio_is_measured_not_assumed(tmp_path: Path) -> None:
    """Real files, real analysis, and durations taken from the audio."""
    run = _run()
    renders = tmp_path / "candidate-renders"
    for case in run.suite.cases:
        # Deliberately short: the case asks for 60 seconds and gets 4.
        write_wav(renders / render_filename(case.case_id, 11), 4.0)

    backend = RenderedAudioBackend(renders, serves_model_id="cand_new")
    side = execute_side(run, run.candidate, backend, tmp_path / "out")

    assert all(outcome.succeeded for outcome in side.outcomes)
    # The measured length, not the requested one.
    assert all(abs((o.duration_seconds or 0) - 4.0) < 0.05 for o in side.outcomes)

    reliability = {m.metric_name: m.value for m in side.reliability_metrics(run.suite)}
    assert reliability["generation_success_rate"] == 1.0
    # Every render is far short of the requested duration, and that has
    # to show up rather than being taken on trust from the backend.
    assert reliability["wrong_duration_rate"] == 1.0

    measured = [m for m in side.metrics if m.status == MetricStatus.MEASURED.value]
    assert {m.metric_name for m in measured} >= {"peak_dbfs", "silence_ratio", "sample_rate"}
    assert all(m.source == "AUDIO_ANALYSIS" for m in measured)


def test_rendered_samples_carry_their_digest(tmp_path: Path) -> None:
    """No mystery WAVs: every sample is tied to the bytes on disk."""
    run = _run()
    renders = tmp_path / "renders"
    for case in run.suite.cases:
        write_wav(renders / render_filename(case.case_id, 11), 3.0)

    side = execute_side(
        run,
        run.candidate,
        RenderedAudioBackend(renders, serves_model_id="cand_new"),
        tmp_path / "out",
    )
    assert side.samples
    for sample in side.samples:
        assert sample.raw_sha256 and len(sample.raw_sha256) == 64
        assert sample.synthetic is False
        assert sample.checkpoint_id == "ckpt_new"


def test_a_missing_render_is_a_failure_not_a_substitution(tmp_path: Path) -> None:
    """A gap is recorded. Nothing is borrowed from another case or seed."""
    run = _run()
    renders = tmp_path / "renders"
    first = sorted(run.suite.cases, key=lambda c: c.case_id)[0]
    write_wav(renders / render_filename(first.case_id, 11), 3.0)

    side = execute_side(
        run,
        run.candidate,
        RenderedAudioBackend(renders, serves_model_id="cand_new"),
        tmp_path / "out",
    )
    failed = [o for o in side.outcomes if not o.succeeded]
    assert len(failed) == len(run.suite.cases) - 1
    assert all("no render found" in (o.error or "") for o in failed)
    assert all(o.audio_path is None for o in failed)


def test_a_backend_refuses_to_answer_for_a_model_it_does_not_serve(tmp_path: Path) -> None:
    """The guard against the most dangerous possible misconfiguration.

    A directory of baseline renders pointed at the candidate side would
    produce a comparison of the baseline against itself, reported as a
    finished evaluation.
    """
    run = _run()
    renders = tmp_path / "baseline-renders"
    for case in run.suite.cases:
        write_wav(renders / render_filename(case.case_id, 11), 3.0)

    backend = RenderedAudioBackend(renders, serves_model_id="mdl_base")
    side = execute_side(run, run.candidate, backend, tmp_path / "out")

    assert all(not outcome.succeeded for outcome in side.outcomes)
    assert all("refusing" in (o.error or "") for o in side.outcomes)
    # Recorded as failures, so the reliability gate sees them.
    reliability = {m.metric_name: m.value for m in side.reliability_metrics(run.suite)}
    assert reliability["generation_failure_rate"] == 1.0


def test_ace_step_backend_refuses_a_case_with_no_stated_voice(tmp_path: Path) -> None:
    """A backend never picks a voice a case did not specify."""
    run = _run()
    case = next(c for c in run.suite.cases if c.case_id == "SYN-KO-01")
    case.spec.vocal_gender = "unknown"

    backend = AceStepEvaluationBackend(
        AceStepBackendConfig(base_url="http://127.0.0.1:9", serves_model_id="cand_new"),
        provider=object(),
    )
    outcome = backend.generate(case, 11, run.candidate, tmp_path)
    assert not outcome.succeeded
    assert "vocal gender" in (outcome.error or "")


def test_ace_step_backend_refuses_the_wrong_model_before_any_request(tmp_path: Path) -> None:
    run = _run()
    case = run.suite.cases[0]
    backend = AceStepEvaluationBackend(
        AceStepBackendConfig(base_url="http://127.0.0.1:9", serves_model_id="mdl_base"),
        # A provider that would explode if it were ever reached.
        provider=object(),
    )
    outcome = backend.generate(case, 11, run.candidate, tmp_path)
    assert not outcome.succeeded
    assert "refusing" in (outcome.error or "")


def test_ace_step_config_records_a_key_reference_never_a_key() -> None:
    config = AceStepBackendConfig(
        base_url="https://gpu.example",
        serves_model_id="cand_new",
        api_key_ref="ACE_STEP_EVAL_KEY",
    )
    payload = config.to_dict()
    assert payload["api_key_ref"] == "ACE_STEP_EVAL_KEY"
    assert "api_key" not in payload
