"""Whether a directory of preprocessed tensors can be trained on.

Preprocessing is the step where real audio becomes something the DiT
reads, and it is the step most likely to fail quietly: a track that
decoded to the wrong sample rate, a conditioning tensor that came out
empty, a NaN from a division nobody guarded. None of those raise at
preprocessing time. All of them make a training run meaningless.

So the tensors are opened and looked at before anything trains on them,
by the trainer's own interpreter, and the report says what was found
rather than whether it was good enough. The caller decides.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luber_training import _tensor_probe

DEFAULT_PROBE_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class SampleReport:
    """One preprocessed sample, as it actually is on disk."""

    name: str
    ok: bool
    readable: bool
    latent_length: int | None = None
    latent_channels: int | None = None
    encoder_length: int | None = None
    missing_fields: tuple[str, ...] = ()
    non_finite_fields: tuple[str, ...] = ()
    error: str = ""
    bytes: int = 0

    @property
    def exclusion_reason(self) -> str:
        """Why this sample cannot be trained on, or an empty string."""
        if not self.readable:
            return f"UNREADABLE: {self.error}"
        if self.missing_fields:
            return f"MISSING_FIELDS: {', '.join(self.missing_fields)}"
        if self.non_finite_fields:
            return f"NON_FINITE: {', '.join(self.non_finite_fields)}"
        if not self.latent_length or not self.encoder_length:
            return "EMPTY_SEQUENCE: a tensor with no sequence dimension"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "readable": self.readable,
            "latent_length": self.latent_length,
            "latent_channels": self.latent_channels,
            "encoder_length": self.encoder_length,
            "missing_fields": list(self.missing_fields),
            "non_finite_fields": list(self.non_finite_fields),
            "exclusion_reason": self.exclusion_reason,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class TensorReport:
    """What a whole preprocessed split looks like."""

    dataset_dir: str
    samples: tuple[SampleReport, ...] = ()
    probe_failed: str = ""

    @property
    def accepted(self) -> tuple[SampleReport, ...]:
        return tuple(sample for sample in self.samples if sample.ok)

    @property
    def rejected(self) -> tuple[SampleReport, ...]:
        return tuple(sample for sample in self.samples if not sample.ok)

    @property
    def finite_ratio(self) -> float | None:
        """Share of readable samples with no non-finite value anywhere.

        ``None`` when nothing was readable: a ratio over zero samples is
        not 1.0, it is unmeasured.
        """
        readable = [sample for sample in self.samples if sample.readable]
        if not readable:
            return None
        clean = sum(1 for sample in readable if not sample.non_finite_fields)
        return clean / len(readable)

    @property
    def max_latent_length(self) -> int | None:
        lengths = [s.latent_length for s in self.samples if s.latent_length]
        return max(lengths) if lengths else None

    @property
    def max_encoder_length(self) -> int | None:
        lengths = [s.encoder_length for s in self.samples if s.encoder_length]
        return max(lengths) if lengths else None

    @property
    def latent_statistics(self) -> dict[str, Any]:
        lengths = sorted(s.latent_length for s in self.samples if s.latent_length)
        if not lengths:
            return {}
        middle = len(lengths) // 2
        median = (
            lengths[middle] if len(lengths) % 2 else (lengths[middle - 1] + lengths[middle]) / 2
        )
        return {
            "count": len(lengths),
            "minimum": lengths[0],
            "median": median,
            "maximum": lengths[-1],
            "mean": round(sum(lengths) / len(lengths), 2),
        }

    @property
    def shapes_compatible(self) -> bool:
        """Whether every sample carries the same channel widths.

        Sequence length varies per track and that is expected. Channel
        width varying is not: it would mean two different preprocessing
        configurations were mixed in one directory.
        """
        widths = {s.latent_channels for s in self.accepted if s.latent_channels}
        return len(widths) <= 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "probe_failed": self.probe_failed,
            "sample_count": len(self.samples),
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "finite_ratio": self.finite_ratio,
            "max_latent_length": self.max_latent_length,
            "max_encoder_length": self.max_encoder_length,
            "latent_statistics": self.latent_statistics,
            "shapes_compatible": self.shapes_compatible,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _sample_from(payload: dict[str, Any]) -> SampleReport:
    return SampleReport(
        name=str(payload.get("name", "")),
        ok=bool(payload.get("ok", False)),
        readable=bool(payload.get("readable", False)),
        latent_length=payload.get("latent_length"),
        latent_channels=payload.get("latent_channels"),
        encoder_length=payload.get("encoder_length"),
        missing_fields=tuple(payload.get("missing_fields") or ()),
        non_finite_fields=tuple(payload.get("non_finite_fields") or ()),
        error=str(payload.get("error", "")),
        bytes=int(payload.get("bytes", 0)),
    )


def report_from_document(document: dict[str, Any]) -> TensorReport:
    """Build a report from a probe document, without running anything."""
    return TensorReport(
        dataset_dir=str(document.get("dataset_dir", "")),
        samples=tuple(_sample_from(item) for item in document.get("samples") or []),
    )


def inspect_tensors(
    dataset_dir: Path,
    *,
    python_executable: str | Path,
    trainer_root: Path | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> TensorReport:
    """Open every sample in *dataset_dir* with the trainer's interpreter."""
    directory = Path(dataset_dir)
    if not directory.is_dir():
        return TensorReport(dataset_dir=str(directory), probe_failed=f"{directory} does not exist")

    script = Path(_tensor_probe.__file__).resolve()
    request = json.dumps({"dataset_dir": str(directory)})
    try:
        completed = subprocess.run(
            [str(python_executable), str(script)],
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=None if trainer_root is None else str(trainer_root),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return TensorReport(dataset_dir=str(directory), probe_failed=f"{type(exc).__name__}: {exc}")

    for line in reversed((completed.stdout or "").strip().splitlines()):
        try:
            document = json.loads(line)
        except ValueError:
            continue
        if not isinstance(document, dict):
            continue
        if document.get("protocol_version") != _tensor_probe.TENSOR_PROBE_PROTOCOL_VERSION:
            return TensorReport(
                dataset_dir=str(directory),
                probe_failed=(
                    f"the probe reported protocol {document.get('protocol_version')!r} and "
                    f"this build reads {_tensor_probe.TENSOR_PROBE_PROTOCOL_VERSION!r}"
                ),
            )
        return report_from_document(document)

    tail = (completed.stderr or "").strip().splitlines()
    return TensorReport(
        dataset_dir=str(directory),
        probe_failed=(
            f"the probe exited {completed.returncode} without a document"
            + (f": {tail[-1]}" if tail else "")
        ),
    )


def render_markdown(reports: dict[str, TensorReport]) -> str:
    """A short report per split. Digests and counts, never paths."""
    lines = ["# Preprocessed tensors", ""]
    for name, report in reports.items():
        ratio = report.finite_ratio
        lines += [
            f"## {name}",
            "",
            f"- samples: **{len(report.samples)}**",
            f"- accepted: **{len(report.accepted)}**, rejected: **{len(report.rejected)}**",
            f"- finite ratio: **{'unmeasured' if ratio is None else f'{ratio:.3f}'}**",
            f"- latent length: {report.latent_statistics or 'none'}",
            f"- max encoder length: {report.max_encoder_length}",
            f"- channel widths consistent: {report.shapes_compatible}",
            "",
        ]
        for sample in report.rejected:
            lines.append(f"  - rejected `{sample.name}`: {sample.exclusion_reason}")
        if report.rejected:
            lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "SampleReport",
    "TensorReport",
    "inspect_tensors",
    "render_markdown",
    "report_from_document",
]
