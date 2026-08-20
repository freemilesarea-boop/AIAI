"""Translating LUBER's dataset into what ACE-Step's trainer reads.

Direction matters. The canonical manifest is **not** reshaped to suit
the trainer — it has its own semantics, its own versioning and other
consumers. Instead this adapter reads a curated manifest and emits the
`dataset.json` that ACE-Step's preprocessing expects, leaving both sides
free to change independently.

The trainer's own loader (`preprocess_discovery.load_sample_metadata`)
fills in defaults for anything missing: a caption derived from the
filename, and `"[Instrumental]"` lyrics. Those defaults are precisely
why every field is written explicitly. A vocal track silently labelled
instrumental because nobody supplied lyrics is a training-data error
that would be invisible in the loss curve.

The command compiler builds an **argv list**, never a shell string.
Experiment names and paths are operator-supplied text; concatenating
them into a command line is how a track called `x"; rm -rf ~` becomes an
incident. `subprocess` without `shell=True` makes the whole class of
problem structural rather than something to remember.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.config import TrainingStrategy
from luber_training.plan import TrainingPlan

#: Trainer subcommand LUBER compiles to. `vanilla` reproduces a path
#: upstream itself describes as bugged; `estimate` does not train.
TRAINER_SUBCOMMAND = "fixed"

#: Lyrics the trainer expects for material with no vocal.
INSTRUMENTAL_MARKER = "[Instrumental]"

class AdapterError(ValueError):
    """Raised when a record cannot be represented for the trainer."""


@dataclass
class TrainerSample:
    """One entry in ACE-Step's dataset.json."""

    filename: str
    caption: str
    lyrics: str
    genre: str = ""
    bpm: float | None = None
    keyscale: str = ""
    timesignature: str = ""
    duration: float = 0.0
    custom_tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "caption": self.caption,
            "lyrics": self.lyrics,
            "genre": self.genre,
            "bpm": self.bpm,
            "keyscale": self.keyscale,
            "timesignature": self.timesignature,
            "duration": self.duration,
            "custom_tag": self.custom_tag,
        }


def _caption(record: dict[str, Any]) -> str:
    """A conditioning caption built only from what is known.

    Nothing is invented. A track with no metadata beyond its duration
    gets a caption saying so, rather than one asserting a genre or mood
    nobody recorded — a fabricated caption is a fabricated training
    label, and the model would learn the fabrication.
    """
    sidecar = ((record.get("metadata") or {}).get("sidecar")) or {}
    music = record.get("music") or {}
    vocals = record.get("vocals") or {}
    language = ((record.get("metadata") or {}).get("language") or {}).get("language")

    parts: list[str] = []
    for descriptor in ("genre", "subgenre", "mood"):
        value = sidecar.get(descriptor)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if isinstance(language, str) and language and language != "unknown":
        parts.append(f"{language} language")
    vocal_class = vocals.get("vocal_class")
    if isinstance(vocal_class, str) and vocal_class in ("VOCAL", "INSTRUMENTAL"):
        parts.append(vocal_class.lower())
    bpm = music.get("bpm")
    if isinstance(bpm, (int, float)) and music.get("bpm_confidence"):
        parts.append(f"{round(float(bpm))} bpm")
    # Gated on confidence, exactly as the `keyscale` field is. The
    # caption is conditioning text the model learns from, so an
    # unconfident key here would teach a tonality nobody stands behind.
    key = music.get("key")
    mode = music.get("mode")
    if key and mode and music.get("key_confidence"):
        parts.append(f"{key} {mode}")

    return ", ".join(parts) if parts else "unlabelled music"


def _lyrics(record: dict[str, Any]) -> str:
    """Supplied lyrics, or the instrumental marker — never a guess.

    The marker is only used when the record positively says the track is
    instrumental. A track whose vocal class is UNCERTAIN and whose
    lyrics are absent gets an empty string: unknown, which the trainer
    can treat as it likes, rather than a claim that it has no vocal.
    """
    text = (record.get("text") or {}).get("lyrics")
    if isinstance(text, str) and text.strip():
        return text
    vocal_class = (record.get("vocals") or {}).get("vocal_class")
    if vocal_class == "INSTRUMENTAL":
        return INSTRUMENTAL_MARKER
    return ""


def to_trainer_sample(record: dict[str, Any]) -> TrainerSample:
    """One curated record as a trainer sample."""
    source = record.get("source") or {}
    filename = source.get("source_filename")
    if not isinstance(filename, str) or not filename.strip():
        raise AdapterError(
            f"track {record.get('track_id')!r} has no source filename; the trainer "
            "indexes samples by filename"
        )

    sidecar = ((record.get("metadata") or {}).get("sidecar")) or {}
    music = record.get("music") or {}
    analysis = record.get("analysis") or {}

    # Tempo and key are only forwarded when Phase 23 was confident. An
    # unconfident estimate recorded as fact would condition the model on
    # a number nobody stands behind.
    bpm = music.get("bpm")
    bpm_value = (
        float(bpm)
        if isinstance(bpm, (int, float)) and music.get("bpm_confidence") is not None
        else None
    )
    key_name, mode_name = music.get("key"), music.get("mode")
    keyscale = (
        f"{key_name} {mode_name}" if key_name and mode_name and music.get("key_confidence") else ""
    )

    genre_value = sidecar.get("genre")
    return TrainerSample(
        filename=filename,
        caption=_caption(record),
        lyrics=_lyrics(record),
        genre=genre_value.strip() if isinstance(genre_value, str) else "",
        bpm=bpm_value,
        keyscale=keyscale,
        timesignature="",
        duration=float(analysis.get("duration_seconds") or 0.0),
        custom_tag="",
    )


@dataclass
class TrainerDataset:
    """The `dataset.json` ACE-Step preprocessing reads."""

    samples: list[TrainerSample] = field(default_factory=list)
    custom_tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "custom_tag": self.custom_tag,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_dataset(curated_records: list[dict[str, Any]], *, custom_tag: str = "") -> TrainerDataset:
    """Adapt selected curated records into trainer input.

    Only records curation selected. Sorted by filename so two builds of
    the same selection produce an identical file.
    """
    from luber_training.gates import selected_records

    samples = [to_trainer_sample(record) for record in selected_records(curated_records)]
    samples.sort(key=lambda sample: sample.filename)
    return TrainerDataset(samples=samples, custom_tag=custom_tag)


def validate_dataset(dataset: TrainerDataset) -> list[str]:
    """Problems the trainer would hit, reported before it does."""
    problems: list[str] = []
    if not dataset.samples:
        problems.append("the dataset contains no samples")

    seen: set[str] = set()
    for sample in dataset.samples:
        if sample.filename in seen:
            problems.append(f"duplicate filename {sample.filename!r}: the loader indexes by name")
        seen.add(sample.filename)
        if not sample.caption.strip():
            problems.append(f"{sample.filename}: empty caption")
        if sample.duration <= 0:
            problems.append(f"{sample.filename}: duration is not positive")
    return problems


# ── command compilation ──────────────────────────────────────────────


def _token(value: Any) -> str:
    """A trainer argument as a single safe token.

    Values that are not obviously safe are still passed as one argv
    element — argv never goes through a shell, so quoting is not what
    protects us — but the check refuses control characters and newlines
    outright, since those corrupt logs and any future shell transport.
    """
    text = str(value)
    if any(character in text for character in "\n\r\x00"):
        raise AdapterError(f"argument {text!r} contains a control character")
    return text


@dataclass
class CompiledCommand:
    """A trainer invocation as argv, with no shell involved."""

    argv: list[str]
    working_directory: str
    #: Environment *names* the backend must set. Never values.
    required_env: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "working_directory": self.working_directory,
            "required_env": list(self.required_env),
        }

    def display(self) -> str:
        """A human-readable form for logs and reports.

        Quoted with `shlex.quote` for *display only*. Nothing executes
        this string, and it exists so an operator can read what will
        run without the argv list being turned into something a shell
        would interpret.
        """
        return " ".join(shlex.quote(part) for part in self.argv)


def compile_command(
    plan: TrainingPlan,
    *,
    trainer_root: str,
    python_executable: str = "python",
    model_variant: str = "turbo",
) -> CompiledCommand:
    """A TrainingPlan as a concrete ACE-Step invocation.

    Every flag emitted was read from the installed parser. A flag the
    trainer does not have cannot appear here, which is what makes the
    compiled command trustworthy rather than plausible.
    """
    config = plan.config
    if config.strategy not in (TrainingStrategy.LORA.value, TrainingStrategy.LOKR.value):
        raise AdapterError(
            f"strategy {config.strategy!r} has no entry point in the installed trainer"
        )

    argv: list[str] = [
        _token(python_executable),
        "train.py",
        TRAINER_SUBCOMMAND,
        "--dataset-dir",
        _token(plan.dataset_dir),
        "--output-dir",
        _token(plan.output_dir),
        "--checkpoint-dir",
        _token(plan.checkpoint_dir),
        "--model-variant",
        _token(model_variant),
        # ── device ──
        "--device",
        "cuda" if plan.requirements.requires_cuda else "cpu",
        "--precision",
        _token(config.precision),
        "--num-devices",
        _token(config.num_devices),
        "--strategy",
        "ddp" if config.num_devices > 1 else "auto",
        # ── training ──
        "--learning-rate",
        _token(config.learning_rate),
        "--batch-size",
        _token(config.batch_size),
        "--gradient-accumulation",
        _token(config.gradient_accumulation),
        "--epochs",
        _token(config.epochs),
        "--warmup-steps",
        _token(config.warmup_steps),
        "--weight-decay",
        _token(config.weight_decay),
        "--max-grad-norm",
        _token(config.max_grad_norm),
        "--seed",
        _token(config.seed),
        "--shift",
        _token(config.shift),
        "--num-inference-steps",
        _token(config.num_inference_steps),
        "--optimizer-type",
        _token(config.optimizer_type),
        "--scheduler-type",
        _token(config.scheduler_type),
        # ── adapter ──
        "--adapter-type",
        "lora" if config.strategy == TrainingStrategy.LORA.value else "lokr",
        # ── checkpointing and logging ──
        "--save-every",
        _token(config.checkpoint_every_epochs),
        "--log-every",
        _token(config.log_every_steps),
        "--log-heavy-every",
        _token(config.log_heavy_every_steps),
        "--sample-every-n-epochs",
        _token(config.sample_every_n_epochs),
        # ── data loading ──
        "--num-workers",
        _token(config.num_workers),
        "--prefetch-factor",
        _token(config.prefetch_factor),
    ]

    if config.strategy == TrainingStrategy.LORA.value:
        argv += [
            "--rank",
            _token(config.rank),
            "--alpha",
            _token(config.alpha),
            "--dropout",
            _token(config.dropout),
            "--bias",
            _token(config.bias),
            "--attention-type",
            _token(config.attention_type),
            "--target-modules",
            *[_token(module) for module in config.target_modules],
        ]

    # BooleanOptionalAction flags: the trainer defaults gradient
    # checkpointing on, so the negative form has to be emitted
    # explicitly rather than omitted.
    argv.append(
        "--gradient-checkpointing"
        if config.gradient_checkpointing
        else "--no-gradient-checkpointing"
    )
    argv.append("--offload-encoder" if config.offload_encoder else "--no-offload-encoder")
    argv.append("--pin-memory" if config.pin_memory else "--no-pin-memory")
    argv.append("--persistent-workers" if config.persistent_workers else "--no-persistent-workers")

    return CompiledCommand(
        argv=argv,
        working_directory=_token(trainer_root),
        required_env=(),
    )
