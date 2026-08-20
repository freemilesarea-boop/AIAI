"""Training configuration: exactly what the installed trainer accepts.

Every field here corresponds to a flag read out of
``acestep/training_v2/cli/args.py`` at commit ``6d467e4b``. Nothing is
aspirational, and three conventional fields are deliberately **absent**
because the trainer has no such flags:

* ``max_steps`` — training length is epochs only
* ``validation_interval`` — nothing computes a validation loss; the
  nearest flag generates audio samples
* ``checkpoint_interval`` in steps — ``--save-every`` counts **epochs**,
  which is why the field here is named ``checkpoint_every_epochs``

A field LUBER offers and the trainer ignores is worse than a missing
one: it looks like a setting, appears in the run record, and does
nothing. Unknown keys are a validation failure rather than a shrug.

The config hashes. Two configs that would produce the same training run
produce the same digest, and the digest is what a plan and a lock cite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

TRAINING_CONFIG_SCHEMA_VERSION = "luber-training-config/1"

#: The ACE-Step tree this schema was audited against. A config carries
#: it so a run can be checked against the trainer it was written for.
AUDITED_ACE_STEP_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"


class TrainingStrategy(StrEnum):
    """How the model is adapted.

    ``FULL`` is absent on purpose. The installed trainer has no
    full-fine-tune subcommand or flag, so offering the option would
    mean silently training an adapter and recording that a full
    fine-tune had happened.
    """

    LORA = "LORA"
    LOKR = "LOKR"


class Optimizer(StrEnum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw8bit"
    ADAFACTOR = "adafactor"
    PRODIGY = "prodigy"


class Scheduler(StrEnum):
    COSINE = "cosine"
    COSINE_RESTARTS = "cosine_restarts"
    LINEAR = "linear"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class Precision(StrEnum):
    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class AttentionTarget(StrEnum):
    SELF = "self"
    CROSS = "cross"
    BOTH = "both"


class BiasMode(StrEnum):
    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


#: Modules the trainer defaults to, and the only ones Phase 20 verified
#: exist on the DiT attention blocks.
DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class ConfigError(ValueError):
    """Raised when a configuration could not be executed as written."""


@dataclass(frozen=True)
class TrainingConfig:
    """One training configuration, hashable and strictly validated."""

    schema_version: str = TRAINING_CONFIG_SCHEMA_VERSION
    strategy: str = TrainingStrategy.LORA.value

    # ── training ─────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation: int = 4
    epochs: int = 10
    warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    optimizer_type: str = Optimizer.ADAMW.value
    scheduler_type: str = Scheduler.COSINE.value
    gradient_checkpointing: bool = True
    offload_encoder: bool = False

    # ── diffusion schedule ───────────────────────────────────────────
    #: Turbo uses 3.0 with 8 inference steps; base/sft use 1.0 and 50.
    shift: float = 3.0
    num_inference_steps: int = 8

    # ── adapter ──────────────────────────────────────────────────────
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    bias: str = BiasMode.NONE.value
    attention_type: str = AttentionTarget.BOTH.value

    # ── device ───────────────────────────────────────────────────────
    precision: str = Precision.AUTO.value
    num_devices: int = 1

    # ── checkpointing and logging ────────────────────────────────────
    #: Named for what it measures. `--save-every` counts epochs, and a
    #: field called `checkpoint_interval` would read as steps.
    checkpoint_every_epochs: int = 5
    log_every_steps: int = 10
    log_heavy_every_steps: int = 50
    sample_every_n_epochs: int = 0

    # ── data loading ─────────────────────────────────────────────────
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    #: Trainer tree this config was written against.
    ace_step_commit: str = AUDITED_ACE_STEP_COMMIT

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """SHA-256 over the canonical form. Same config, same hash."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_overrides(self, **kwargs: Any) -> TrainingConfig:
        unknown = sorted(set(kwargs) - set(self.__dataclass_fields__))
        if unknown:
            raise ConfigError(
                f"unrecognised training config field(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(self.__dataclass_fields__))}"
            )
        if "target_modules" in kwargs and kwargs["target_modules"] is not None:
            kwargs["target_modules"] = tuple(kwargs["target_modules"])
        return replace(self, **kwargs)


def validate(config: TrainingConfig) -> None:
    """Refuse a config the trainer would reject or silently misread."""
    if config.schema_version != TRAINING_CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"config schema version {config.schema_version!r} is not "
            f"{TRAINING_CONFIG_SCHEMA_VERSION!r}"
        )

    enums: tuple[tuple[str, str, type[StrEnum]], ...] = (
        ("strategy", config.strategy, TrainingStrategy),
        ("optimizer_type", config.optimizer_type, Optimizer),
        ("scheduler_type", config.scheduler_type, Scheduler),
        ("precision", config.precision, Precision),
        ("attention_type", config.attention_type, AttentionTarget),
        ("bias", config.bias, BiasMode),
    )
    for name, value, enum in enums:
        permitted = {member.value for member in enum}
        if value not in permitted:
            raise ConfigError(
                f"{name}={value!r} is not accepted by the trainer. "
                f"Permitted: {', '.join(sorted(permitted))}"
            )

    positive: tuple[tuple[str, float | int], ...] = (
        ("learning_rate", config.learning_rate),
        ("batch_size", config.batch_size),
        ("gradient_accumulation", config.gradient_accumulation),
        ("epochs", config.epochs),
        ("rank", config.rank),
        ("alpha", config.alpha),
        ("num_devices", config.num_devices),
        ("checkpoint_every_epochs", config.checkpoint_every_epochs),
        ("num_inference_steps", config.num_inference_steps),
    )
    for field_name, number in positive:
        if number <= 0:
            raise ConfigError(f"{field_name} must be positive, got {number}")

    non_negative: tuple[tuple[str, float | int], ...] = (
        ("warmup_steps", config.warmup_steps),
        ("weight_decay", config.weight_decay),
        ("dropout", config.dropout),
        ("sample_every_n_epochs", config.sample_every_n_epochs),
        ("num_workers", config.num_workers),
    )
    for field_name, number in non_negative:
        if number < 0:
            raise ConfigError(f"{field_name} must not be negative, got {number}")

    if not 0.0 <= config.dropout < 1.0:
        raise ConfigError(f"dropout must be in [0, 1), got {config.dropout}")
    if not config.target_modules:
        raise ConfigError("target_modules must name at least one module")
    if config.warmup_steps and config.epochs and config.warmup_steps > config.epochs * 1000:
        raise ConfigError(
            f"warmup_steps={config.warmup_steps} cannot be reached in {config.epochs} epochs"
        )
    if config.ace_step_commit != AUDITED_ACE_STEP_COMMIT:
        raise ConfigError(
            f"config targets ACE-Step {config.ace_step_commit[:12]} but this build was "
            f"audited against {AUDITED_ACE_STEP_COMMIT[:12]}; re-audit before training"
        )


def from_dict(payload: dict[str, Any]) -> TrainingConfig:
    """Build a config from JSON, rejecting anything unrecognised.

    Unknown keys fail rather than being ignored. Step 54 in one line:
    a parameter the trainer has never heard of must not travel quietly
    into a run record as though it had an effect.
    """
    if not isinstance(payload, dict):
        raise ConfigError("a training config must be a JSON object")
    known = set(TrainingConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ConfigError(
            f"unrecognised training config field(s): {', '.join(unknown)}. "
            f"The installed trainer accepts: {', '.join(sorted(known))}"
        )
    prepared = dict(payload)
    if "target_modules" in prepared and prepared["target_modules"] is not None:
        prepared["target_modules"] = tuple(prepared["target_modules"])
    config = TrainingConfig(**prepared)
    validate(config)
    return config


# ── presets ──────────────────────────────────────────────────────────
#
# Named for *intent*, never for hardware. Upstream ships presets called
# `vram_8gb` and `vram_24gb_plus`; those names make a promise about
# memory that nothing in this project has measured, so they are not
# reused and no VRAM figure appears here.


def smoke() -> TrainingConfig:
    """Infrastructure validation. Deliberately teaches nothing.

    One epoch at the smallest usable rank. The point is to prove that
    data loads, the adapter injects, a step runs and a checkpoint
    writes — not to move the model. A SMOKE run's checkpoint is
    expected to be worthless and must never be promoted.
    """
    return TrainingConfig(
        epochs=1,
        rank=4,
        alpha=8,
        warmup_steps=0,
        checkpoint_every_epochs=1,
        log_every_steps=1,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=0,
    )


def lora_small() -> TrainingConfig:
    return TrainingConfig(epochs=10, rank=16, alpha=32, warmup_steps=10)


def lora_standard() -> TrainingConfig:
    return TrainingConfig(epochs=30, rank=32, alpha=64, warmup_steps=100, checkpoint_every_epochs=5)


def lora_high_quality() -> TrainingConfig:
    return TrainingConfig(
        epochs=60,
        rank=64,
        alpha=128,
        warmup_steps=200,
        checkpoint_every_epochs=10,
        learning_rate=5e-5,
    )


PRESETS: dict[str, Any] = {
    "SMOKE": smoke,
    "LORA_SMALL": lora_small,
    "LORA_STANDARD": lora_standard,
    "LORA_HIGH_QUALITY": lora_high_quality,
}

#: What each preset is for, in words, since none of them can promise
#: hardware compatibility.
PRESET_INTENT: dict[str, str] = {
    "SMOKE": "prove the plumbing works; not a training run",
    "LORA_SMALL": "a first look at whether a dataset moves the model at all",
    "LORA_STANDARD": "the default for a real experiment",
    "LORA_HIGH_QUALITY": "longer, higher rank, lower learning rate",
}


def preset(name: str) -> TrainingConfig:
    key = name.strip().upper()
    if key not in PRESETS:
        raise ConfigError(f"unknown preset {name!r}. Available: {', '.join(sorted(PRESETS))}")
    config: TrainingConfig = PRESETS[key]()
    validate(config)
    return config
