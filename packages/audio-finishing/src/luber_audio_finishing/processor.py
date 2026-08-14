"""Rendering a finishing plan, and proving the render was safe.

The raw generation is the only copy of what the model produced, so the
processor never writes to its input and never accepts its own output as
one. Everything else here follows from those two rules.

Two passes, for a reason. The corrective filters change the peak by an
amount no formula predicts, and generated masters arrive at exactly
-1.0 dBFS with no headroom at all, so the level has to be measured after
filtering rather than guessed before it. The first pass renders the
chain at 32-bit float where nothing can clip; the second applies a
single measured gain and writes the deliverable.

That final gain is the smaller of two numbers: the one that matches the
finished loudness to the source's, and the one that keeps the true peak
under the ceiling. Peak safety wins when they disagree, which can leave
the finished file slightly quieter than the raw. That is deliberate. The
alternative is a limiter, and a limiter would change the dynamics of the
comparison the listening test is supposed to be about.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from luber_audio_finishing.analysis import AudioAnalysis, analyze_audio
from luber_audio_finishing.decision import (
    MAX_BALANCE_CORRECTION_DB,
    ActionKind,
    FinishingAction,
    FinishingDecisionEngine,
    FinishingPlan,
)
from luber_audio_finishing.risks import STEREO_IMBALANCE_DB
from luber_audio_finishing.version import (
    FINISHING_STAMP_KEY,
    FINISHING_VERSION,
    finishing_stamp,
    parse_finishing_stamp,
)

#: How far the finished loudness may be moved to match the source.
MAX_LOUDNESS_MATCH_DB = 3.0
#: True peak is allowed this much slack above the ceiling before the
#: render is rejected, covering inter-sample estimation differences
#: between the measuring and rendering passes.
PEAK_TOLERANCE_DB = 0.1
#: Every filter in the chain is a zero-delay biquad or a channel-matrix
#: operation, so any duration change beyond a rounding error means
#: something is wrong.
DURATION_TOLERANCE_SECONDS = 0.01
#: Residual imbalance smaller than this is left alone: it is inaudible,
#: and correcting it would add a stage to every stereo render.
MIN_BALANCE_CORRECTION_DB = 0.25


class FinishingError(Exception):
    """Raised when a file cannot be finished, or the result is unsafe."""


class AlreadyFinishedError(FinishingError):
    """Raised when the input is itself a finishing output.

    Finishing a finished file would apply a second round of corrections
    to audio whose measurements the first round already changed, and the
    result would depend on how many times it happened to be run.
    """


@dataclass(frozen=True)
class FinishingResult:
    """What was decided, what was rendered, and the proof it is safe."""

    finishing_version: str
    plan: FinishingPlan
    source_analysis: AudioAnalysis
    #: ``None`` when the plan was NO_ACTION and nothing was written.
    output_path: Path | None
    finished_analysis: AudioAnalysis | None
    #: The single gain applied after filtering, in dB.
    output_gain_db: float
    #: Gain that would have matched the source loudness exactly.
    loudness_match_gain_db: float
    #: How much of that was given up to stay under the peak ceiling.
    peak_safety_reduction_db: float
    #: Re-centring applied after the chain, in dB (positive favours left).
    balance_correction_db: float
    #: The exact ffmpeg filter graph, kept for audit.
    filter_graph: str

    @property
    def changed(self) -> bool:
        return self.output_path is not None


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise FinishingError(f"{binary} is required for audio finishing but is not on PATH")
    return path


def _run(command: list[str]) -> None:
    # Fixed argv, never a shell string. Every numeric parameter is
    # produced by this package, never by a caller or a file.
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()
        raise FinishingError(
            f"{Path(command[0]).name} failed (exit {result.returncode}): "
            f"{tail[-1] if tail else 'no output'}"
        )


def read_finishing_stamp(path: Path) -> str | None:
    """The engine version a file was finished with, if any."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None or not path.is_file():
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            f"format_tags={FINISHING_STAMP_KEY}",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return parse_finishing_stamp(result.stdout.strip() or None)


def _linear(gain_db: float) -> float:
    return float(10.0 ** (gain_db / 20.0))


def _biquad(action: FinishingAction, filter_name: str) -> str:
    gain = action.gain_db or 0.0
    frequency = action.frequency_hz or 1_000.0
    q = action.q or 0.7
    return f"{filter_name}=f={frequency:.2f}:width_type=q:width={q:.4f}:g={gain:.4f}"


def build_filter_graph(
    plan: FinishingPlan,
    *,
    output_gain_db: float | None = None,
    balance_correction_db: float | None = None,
) -> str:
    """The ffmpeg graph for a plan, in chain order.

    Order is the argument here. Subtractive EQ runs before additive EQ so
    a boost is applied to audio that has already lost its excess rather
    than compounding it. The mid/side section runs after all of it, and
    the level stage last of all, because both the balance trim and the
    output gain are decided from measurements of everything upstream.

    ``output_gain_db`` and ``balance_correction_db`` are threaded in here
    rather than spliced into a finished graph string: the graph becomes a
    filter_complex with named pads as soon as a stereo correction exists,
    and patching one by text substitution would break the moment its
    shape changed.
    """
    stages: list[str] = []

    balance = plan.action(ActionKind.BALANCE_CORRECTION)
    if balance is not None and balance.gain_db is not None:
        half = balance.gain_db / 2.0
        stages.append(f"pan=stereo|c0={_linear(half):.6f}*c0|c1={_linear(-half):.6f}*c1")

    for kind, filter_name in (
        (ActionKind.LOW_SHELF_CUT, "lowshelf"),
        (ActionKind.LOW_MID_CUT, "equalizer"),
        (ActionKind.PRESENCE_LIFT, "equalizer"),
        (ActionKind.HIGH_SHELF_LIFT, "highshelf"),
    ):
        action = plan.action(kind)
        if action is not None:
            stages.append(_biquad(action, filter_name))

    tail: list[str] = []
    if balance_correction_db is not None and abs(balance_correction_db) >= 1e-4:
        half = balance_correction_db / 2.0
        tail.append(f"pan=stereo|c0={_linear(half):.6f}*c0|c1={_linear(-half):.6f}*c1")
    if output_gain_db is not None and abs(output_gain_db) >= 1e-4:
        tail.append(f"volume={output_gain_db:.4f}dB")
    trim = ",".join(tail) if tail else None

    width = plan.action(ActionKind.STEREO_WIDTH)
    mono = plan.action(ActionKind.LOW_FREQUENCY_MONO)
    if width is None and mono is None:
        if trim is not None:
            stages.append(trim)
        return ",".join(stages)

    # Both stereo corrections happen in one mid/side section, and every
    # filter in it sits on the side channel only.
    #
    # The obvious alternative — a Linkwitz-Riley crossover that mono-ises
    # the low band and sums it back — was implemented first and measured
    # worse. Its two halves sum flat in magnitude but rotate phase
    # through 360 degrees, and that rotation reshapes transients: sample
    # peak rose by up to 3 dB with no change in loudness, which the level
    # stage then had to give back as 3 dB of gain reduction. Crest factor
    # on one baseline track went 14.3 -> 17.4 dB for an effect nobody can
    # hear. Filtering the side alone leaves the mid — which is the mono
    # sum, and most of the energy — bit-for-bit untouched, so peaks stay
    # where they were and the trim stays small.
    side_stages: list[str] = ["pan=mono|c0=0.5*c0-0.5*c1"]
    if mono is not None:
        # Two cascaded 2nd-order Butterworth sections: 24 dB/octave, so
        # side content is gone well before the musically useful range.
        crossover = mono.frequency_hz or 120.0
        side_stages.append(f"highpass=f={crossover:.2f}:poles=2,highpass=f={crossover:.2f}:poles=2")
    if width is not None and width.gain_db is not None:
        side_stages.append(f"volume={width.gain_db:.4f}dB")

    prefix = ",".join(stages) + "," if stages else ""
    suffix = f",{trim}" if trim is not None else ""
    return (
        f"[0:a]{prefix}asplit=2[ms_a][ms_b];"
        f"[ms_a]pan=mono|c0=0.5*c0+0.5*c1[mid];"
        f"[ms_b]{','.join(side_stages)}[side];"
        # L = M + S and R = M - S invert the encoding exactly, so an
        # untouched side channel reconstructs the input sample for sample.
        f"[mid][side]amerge=inputs=2,pan=stereo|c0=c0+c1|c1=c0-c1{suffix}[out]"
    )


def _is_filter_complex(graph: str) -> bool:
    return graph.startswith("[0:a]")


def _render(source: Path, destination: Path, graph: str, codec: str, stamp: str | None) -> None:
    ffmpeg = _require("ffmpeg")
    command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source)]
    if not graph:
        command += ["-map", "a:0"]
    elif _is_filter_complex(graph):
        command += ["-filter_complex", graph, "-map", "[out]"]
    else:
        command += ["-map", "a:0", "-filter:a", graph]
    command += ["-map_metadata", "-1", "-c:a", codec]
    if stamp is not None:
        command += ["-metadata", f"{FINISHING_STAMP_KEY}={stamp}"]
    command.append(str(destination))
    _run(command)


def _output_codec(analysis: AudioAnalysis) -> str:
    """Match the source's PCM width; never reduce it."""
    depth = analysis.technical.bit_depth
    if depth is None or depth >= 24:
        return "pcm_s24le"
    if depth <= 16:
        return "pcm_s16le"
    return "pcm_s24le"


def _verify(source: AudioAnalysis, finished: AudioAnalysis, ceiling_dbfs: float) -> None:
    problems: list[str] = []
    if finished.level.clipped_samples > 0:
        problems.append(f"{finished.level.clipped_samples} clipped samples")
    true_peak = finished.loudness.true_peak_dbfs
    if true_peak is not None and true_peak > ceiling_dbfs + PEAK_TOLERANCE_DB:
        problems.append(f"true peak {true_peak:.2f} dBFS exceeds the {ceiling_dbfs:.2f} ceiling")
    if finished.level.peak_dbfs > ceiling_dbfs + PEAK_TOLERANCE_DB:
        problems.append(
            f"sample peak {finished.level.peak_dbfs:.2f} dBFS exceeds "
            f"the {ceiling_dbfs:.2f} ceiling"
        )
    drift = abs(finished.technical.duration_seconds - source.technical.duration_seconds)
    if drift > DURATION_TOLERANCE_SECONDS:
        problems.append(f"duration moved by {drift:.4f} s")
    if finished.technical.sample_rate != source.technical.sample_rate:
        problems.append(
            f"sample rate changed {source.technical.sample_rate} -> "
            f"{finished.technical.sample_rate}"
        )
    if finished.technical.channels != source.technical.channels:
        problems.append(
            f"channel count changed {source.technical.channels} -> {finished.technical.channels}"
        )
    # The engine must never leave a file more off-centre than the amount
    # it treats as a defect in the first place. The mid/side stage can
    # shift balance as a side effect, so this is checked rather than
    # assumed.
    finished_balance = finished.stereo.lr_balance_db
    if finished_balance is not None and abs(finished_balance) > STEREO_IMBALANCE_DB:
        problems.append(f"output balance {finished_balance:+.2f} dB is off centre")
    if problems:
        raise FinishingError("finishing produced unsafe output: " + "; ".join(problems))


def finish_audio(
    source: Path,
    destination: Path,
    *,
    engine: FinishingDecisionEngine | None = None,
    finishing_version: str = FINISHING_VERSION,
) -> FinishingResult:
    """Analyse, plan, render and verify. The source is never modified.

    A NO_ACTION plan writes nothing and returns ``output_path=None``: the
    raw master is already the deliverable, and re-encoding it to produce
    an identical file would only invite the two to drift apart.
    """
    if not source.is_file():
        raise FinishingError(f"source audio does not exist: {source}")
    existing = read_finishing_stamp(source)
    if existing is not None:
        raise AlreadyFinishedError(
            f"{source.name} was already finished with engine {existing}; "
            "finish the raw generation master instead"
        )
    if destination.resolve() == source.resolve():
        raise FinishingError("finishing would overwrite its own source")

    source_analysis = analyze_audio(source)
    plan = (engine or FinishingDecisionEngine(finishing_version=finishing_version)).plan(
        source_analysis
    )
    graph = build_filter_graph(plan)

    if plan.is_no_action:
        return FinishingResult(
            finishing_version=plan.finishing_version,
            plan=plan,
            source_analysis=source_analysis,
            output_path=None,
            finished_analysis=None,
            output_gain_db=0.0,
            loudness_match_gain_db=0.0,
            peak_safety_reduction_db=0.0,
            balance_correction_db=0.0,
            filter_graph=graph,
        )

    codec = _output_codec(source_analysis)
    with tempfile.TemporaryDirectory(prefix="luber-finishing-") as workdir:
        intermediate = Path(workdir) / "filtered.wav"
        # 32-bit float: the corrective stage is allowed to exceed full
        # scale here, because the level stage below is what brings it back.
        _render(source, intermediate, graph, "pcm_f32le", stamp=None)
        filtered = analyze_audio(intermediate)

        match_gain = _loudness_match_gain(source_analysis, filtered)
        # Balance is decided before the headroom calculation because it
        # is itself a boost: re-centring raises one channel by half the
        # correction, and a peak budget that ignored it would let the
        # louder channel through the ceiling.
        balance = _residual_balance_correction(filtered)
        headroom = _peak_headroom(
            filtered, plan.output_ceiling_dbfs, match_gain + abs(balance) / 2.0
        )
        gain = match_gain + min(0.0, headroom)

        destination.parent.mkdir(parents=True, exist_ok=True)
        _render(
            source,
            destination,
            build_filter_graph(plan, output_gain_db=gain, balance_correction_db=balance),
            codec,
            stamp=finishing_stamp(plan.finishing_version),
        )

    finished = analyze_audio(destination)
    try:
        _verify(source_analysis, finished, plan.output_ceiling_dbfs)
    except FinishingError:
        destination.unlink(missing_ok=True)
        raise

    return FinishingResult(
        finishing_version=plan.finishing_version,
        plan=plan,
        source_analysis=source_analysis,
        output_path=destination,
        finished_analysis=finished,
        output_gain_db=gain,
        loudness_match_gain_db=match_gain,
        peak_safety_reduction_db=min(0.0, headroom),
        balance_correction_db=balance,
        filter_graph=build_filter_graph(plan, output_gain_db=gain, balance_correction_db=balance),
    )


def _residual_balance_correction(filtered: AudioAnalysis) -> float:
    """Re-centre the image against what the chain actually produced.

    Balance cannot be corrected from the source measurement alone,
    because the mid/side stage moves it. Scaling or high-passing the side
    changes L and R asymmetrically wherever mid and side correlate, and
    on one baseline track that shifted the balance by 1.19 dB — more than
    the 0.8 dB the engine treats as a defect worth fixing in the first
    place. Left uncorrected, the engine would introduce the very fault it
    flags. So the final balance is read off the filtered audio and
    corrected in the level stage, after everything that could disturb it.
    """
    balance = filtered.stereo.lr_balance_db
    if balance is None or abs(balance) < MIN_BALANCE_CORRECTION_DB:
        return 0.0
    return max(-MAX_BALANCE_CORRECTION_DB, min(MAX_BALANCE_CORRECTION_DB, -balance))


def _loudness_match_gain(source: AudioAnalysis, filtered: AudioAnalysis) -> float:
    """Gain that returns the filtered audio to the source's loudness.

    Zero when either measurement is missing: guessing would risk making
    the finished file louder, and a louder file wins A/B comparisons for
    reasons that have nothing to do with the processing.
    """
    original = source.loudness.integrated_lufs
    current = filtered.loudness.integrated_lufs
    if original is None or current is None:
        return 0.0
    return max(-MAX_LOUDNESS_MATCH_DB, min(MAX_LOUDNESS_MATCH_DB, original - current))


def _peak_headroom(filtered: AudioAnalysis, ceiling_dbfs: float, pending_gain_db: float) -> float:
    """dB of room left under the ceiling once pending boosts are applied.

    True peak is used when available because a sample-peak-safe file can
    still exceed the ceiling between samples after conversion.
    """
    peak = filtered.loudness.true_peak_dbfs
    if peak is None:
        peak = filtered.level.peak_dbfs
    return ceiling_dbfs - (peak + pending_gain_db)
