"""Watch a controlled experiment from inside the trainer. No LUBER imports.

The sixth file in this shape, and it adds the two things Phase 36 needs
that a pilot did not.

**A validation loss.** The installed trainer has no validation loop at
all — it trains and it reports, and nothing anywhere measures held-out
loss. So this runs one: at each epoch boundary the module is put in eval
mode and the validation tensors are pushed through the *same* loss the
trainer optimises, under `no_grad`, with no optimizer anywhere near it.

That measurement is only worth having if it is comparable between
epochs, and by default it is not: the flow-matching loss draws fresh
noise and a fresh timestep every call, so two passes over identical
weights differ by more than a few epochs of training would move them. So
every validation pass reseeds the generator to the same value and
disables classifier-free-guidance dropout. Both are deliberate and both
are recorded: the number is a held-out loss under a fixed noise draw,
not a sample from the training objective, and the difference matters
when reading it.

**Checkpoint provenance.** The record is composed on the LUBER side,
validated there, and passed in whole; this file fills in the epoch, the
step and the path, and writes it beside each checkpoint the trainer
produces. Composing it here would put schema knowledge in the one
process that cannot import the schema.

Everything else is a pass-through wrapper, as before: each one calls the
original, records, and returns what the original returned. No ACE-Step
source is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import sys
import time
import traceback
from pathlib import Path
from typing import Any

#: Bump when the shape of the emitted document changes.
EXPERIMENT_PROBE_PROTOCOL_VERSION = "luber-experiment-probe/1"

#: Most points a probe will hold, training and validation each. An
#: experiment is capped in steps, so this can only be reached by a bug,
#: and a bug should truncate with a note rather than grow without bound.
MAX_POINTS = 100_000

_WRAPPER_PREFIXES: tuple[str, ...] = ("_forward_module.", "_original_module.", "module.")

#: How many trailing path components identify a parameter. Six, for the
#: reason `_pilot_probe` records: Fabric rewrites everything above a
#: parameter's own module path and leaves the tail alone.
_NAME_SUFFIX_DEPTH = 6


class StepCeilingExceeded(RuntimeError):
    """Raised in the trainer when an experiment passes its step ceiling.

    The last line of defence, not the first. The budget is computed and
    refused before launch; this exists because a computed budget is a
    statement about the trainer's arithmetic, and if that arithmetic is
    ever wrong the run stops here rather than continuing quietly.
    """


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalise(tensors: dict[str, Any]) -> dict[str, Any]:
    """Parameter names reduced to something both sides can agree on."""
    out: dict[str, Any] = {}
    for name, values in tensors.items():
        trimmed = name
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if trimmed.startswith(prefix):
                    trimmed = trimmed[len(prefix) :]
                    changed = True
        parts = trimmed.split(".")
        out[".".join(parts[-_NAME_SUFFIX_DEPTH:])] = values
    return out


class Validator:
    """Held-out loss, measured with the trainer's own objective.

    Holds the validation tensors in memory: an experiment's validation
    split is a handful of tracks, and re-reading them every epoch would
    add I/O to a measurement that is meant to be about the model.
    """

    #: Fields whose first dimension is the audio timeline, and which a
    #: bounded validation window therefore truncates together. The
    #: conditioning tensors are not here: they describe the prompt, not
    #: the audio, and cutting them would change the question.
    SEQUENCE_FIELDS: tuple[str, ...] = (
        "target_latents",
        "attention_mask",
        "context_latents",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        seed: int,
        max_samples: int = 64,
        latent_length: int | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.seed = int(seed)
        self.max_samples = int(max_samples)
        # A fixed leading window of each held-out track rather than the
        # whole thing. Measured: a full-length validation forward beside
        # a training step pushed this process to 29 GiB of MPS
        # allocations on a 24 GB machine and the run died at step 9. The
        # same window every pass keeps the series comparable; what it
        # costs is that the number describes a window, which every
        # report has to say.
        self.latent_length = int(latent_length) if latent_length else None
        self.samples: list[dict[str, Any]] = []
        self.names: list[str] = []
        self.load_error = ""

    def load(self) -> None:
        if self.samples or self.load_error:
            return
        try:
            import torch  # type: ignore[import-not-found]

            paths = sorted(Path(self.dataset_dir).glob("*.pt"))[: self.max_samples]
            if not paths:
                self.load_error = f"no tensors under {self.dataset_dir}"
                return
            for path in paths:
                payload = torch.load(str(path), map_location="cpu", weights_only=False)
                if not isinstance(payload, dict):
                    continue
                self.samples.append(payload)
                self.names.append(path.name)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    def evaluate(self, module: Any, *, epoch: int, step: int) -> dict[str, Any] | None:
        """One pass over the validation split. No gradients, no updates."""
        self.load()
        if self.load_error or not self.samples:
            return {
                "epoch": epoch,
                "step": step,
                "error": self.load_error or "no validation samples",
            }

        try:
            import torch
        except Exception as exc:
            return {"epoch": epoch, "step": step, "error": f"{type(exc).__name__}: {exc}"}

        model = getattr(module, "model", None)
        was_training = bool(getattr(model, "training", False)) if model is not None else False
        # Measured the hard way: without this the first validation pass
        # ran on top of everything the training step had left in the
        # caching allocator and the process hit 29 GiB on a 24 GB
        # machine. The blocks are free, they are simply still held, and
        # a validation forward at production sequence length needs them
        # back.
        _release_memory()
        # Classifier-free-guidance dropout is a training regulariser. On
        # four validation tracks it would null the conditioning of a
        # quarter of them, and the curve would then be measuring the
        # unconditional model on a random subset.
        original_cfg = getattr(module, "_cfg_ratio", None)

        losses: list[float] = []
        started = time.perf_counter()
        error = ""
        try:
            if model is not None:
                model.eval()
            if original_cfg is not None:
                module._cfg_ratio = 0.0
            with torch.no_grad():
                for payload in self.samples:
                    # Reseeded per sample so a sample's noise draw does
                    # not depend on how many samples preceded it, which
                    # would make the series depend on the split size.
                    torch.manual_seed(self.seed)
                    batch = {}
                    for key, value in payload.items():
                        if not torch.is_tensor(value):
                            continue
                        if self.latent_length and key in self.SEQUENCE_FIELDS:
                            value = value[: self.latent_length]
                        batch[key] = value.unsqueeze(0)
                    loss = module.training_step(batch)
                    value = _as_float(loss.item() if hasattr(loss, "item") else loss)
                    if value is not None:
                        losses.append(value)
                    # Between samples, not just around the pass. Each
                    # track has its own sequence length, so each forward
                    # asks the allocator for block sizes the last one
                    # did not use; holding four such sets at once is
                    # what pushed the pool past the device limit on a
                    # machine whose memory is shared with everything
                    # else running.
                    del batch, loss
                    _release_memory()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if original_cfg is not None:
                module._cfg_ratio = original_cfg
            if model is not None and was_training:
                model.train()
            # And again on the way out, so the next training step starts
            # from the same place it would have without a validation
            # pass in between.
            _release_memory()

        finite = [value for value in losses if math.isfinite(value)]
        return {
            "epoch": epoch,
            "step": step,
            "device_allocated_bytes": _device_allocated(),
            "latent_length": self.latent_length,
            "sample_count": len(self.samples),
            "measured_count": len(losses),
            "finite_count": len(finite),
            "loss": (sum(finite) / len(finite)) if finite else None,
            "minimum": min(finite) if finite else None,
            "maximum": max(finite) if finite else None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": error,
            "note": (
                "held-out loss under the trainer's own objective, with a fixed noise seed "
                "and CFG dropout disabled so passes are comparable"
                + (
                    f", measured on the first {self.latent_length} latent frames of each "
                    "track rather than the whole track"
                    if self.latent_length
                    else ""
                )
            ),
        }


def _release_memory() -> None:
    """Hand freed blocks back before and after a validation pass.

    The caching allocator holds what a training step released. That is
    the right default for training and the wrong one either side of a
    forward pass that needs several gigabytes of its own, on a device
    whose memory is shared with the whole machine.
    """
    try:
        import gc

        import torch

        gc.collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Freeing memory is best effort. Failing to free it is not a
        # reason to fail the measurement it was meant to make room for.
        return


def _device_allocated() -> int | None:
    """What the device reports holding right now, if it reports at all."""
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.current_allocated_memory())
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
    except Exception:
        return None
    return None


class Recorder:
    """Training points, validation points, fingerprints and the ceiling."""

    def __init__(
        self,
        *,
        step_ceiling: int,
        segment: str = "A",
        validator: Validator | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.started = time.perf_counter()
        self.step_ceiling = int(step_ceiling)
        self.segment = segment
        self.validator = validator
        self.provenance = provenance or {}
        self.points: list[dict[str, Any]] = []
        self.validation_points: list[dict[str, Any]] = []
        self.truncated = False
        self.before: dict[str, Any] | None = None
        self.after: dict[str, Any] | None = None
        self.grad_norms: list[float] = []
        self.model_ref: Any = None
        self.module_ref: Any = None
        self.ceiling_hit = False
        self.checkpoints: list[dict[str, Any]] = []
        self.last_epoch: int | None = None
        self.last_step: int = 0

    # ── training ─────────────────────────────────────────────────────
    def record_step(self, update: Any) -> None:
        step = _as_int(getattr(update, "step", None))
        loss = _as_float(getattr(update, "loss", None))
        if step is None:
            return
        self.last_step = step
        if len(self.points) >= MAX_POINTS:
            self.truncated = True
        else:
            self.points.append(
                {
                    "segment": self.segment,
                    "step": step,
                    "epoch": _as_int(getattr(update, "epoch", None)),
                    "loss": loss,
                    "learning_rate": _as_float(getattr(update, "learning_rate", None)),
                    "grad_norm": self.grad_norms[-1] if self.grad_norms else None,
                    "elapsed_seconds": round(time.perf_counter() - self.started, 3),
                    # Recorded every step because Phase 36's failure was
                    # cumulative allocator growth, and a trend is only
                    # visible if somebody wrote the numbers down before
                    # the process died.
                    "device_allocated_bytes": _device_allocated(),
                }
            )
        if self.step_ceiling and step > self.step_ceiling:
            self.ceiling_hit = True
            raise StepCeilingExceeded(
                f"the trainer reported step {step} and this experiment's ceiling is "
                f"{self.step_ceiling}"
            )

    def record_grad_norm(self, value: Any) -> None:
        number = _as_float(value)
        if number is not None:
            self.grad_norms.append(number)

    # ── validation ───────────────────────────────────────────────────
    def maybe_validate(self, update: Any) -> None:
        """Validate when the epoch turns over, and never mid-epoch."""
        if self.validator is None:
            return
        epoch = _as_int(getattr(update, "epoch", None))
        if epoch is None or epoch == self.last_epoch:
            return
        if self.last_epoch is not None:
            self.validate(epoch=self.last_epoch, step=max(0, self.last_step - 1))
        self.last_epoch = epoch

    def validate(self, *, epoch: int, step: int) -> None:
        if self.validator is None or self.module_ref is None:
            return
        if len(self.validation_points) >= MAX_POINTS:
            self.truncated = True
            return
        point = self.validator.evaluate(self.module_ref, epoch=epoch, step=step)
        if point is not None:
            point["segment"] = self.segment
            self.validation_points.append(point)

    # ── parameters ───────────────────────────────────────────────────
    def fingerprint(self, model: Any) -> dict[str, Any]:
        try:
            import torch

            values: dict[str, Any] = {}
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    data = parameter.detach().float()
                    values[name] = [
                        float(data.sum().item()),
                        float(data.norm().item()),
                        int(data.numel()),
                    ]
            return values
        except Exception:
            return {}

    def compare(self) -> dict[str, Any]:
        if not self.before or not self.after:
            return {
                "changed_tensor_count": None,
                "detail": "no fingerprint pair was taken, so nothing can be said",
            }
        before, after = _normalise(self.before), _normalise(self.after)
        shared = sorted(set(before) & set(after))
        if not shared:
            # Never zero. An empty intersection means the comparison
            # failed, not that nothing moved, and reporting zero would
            # be a false NO_UPDATE verdict.
            return {
                "changed_tensor_count": None,
                "detail": (
                    f"{len(before)} tensor(s) before and {len(after)} after share no "
                    "comparable name, so whether they changed is unknown"
                ),
            }
        changed, deltas = 0, []
        for name in shared:
            first, second = before[name], after[name]
            delta = max(abs(first[0] - second[0]), abs(first[1] - second[1]))
            deltas.append(delta)
            if delta > 0.0:
                changed += 1
        return {
            "changed_tensor_count": changed,
            "comparable_tensor_count": len(shared),
            "trainable_parameter_count": sum(item[2] for item in self.after.values()),
            "max_absolute_delta": max(deltas) if deltas else 0.0,
            "mean_absolute_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "trainable_before_digest": _digest(before),
            "trainable_after_digest": _digest(after),
            "detail": (
                f"{changed} of {len(shared)} comparable trainable tensor(s) changed, "
                "by per-tensor sum and L2 norm"
            ),
        }

    def gradient_summary(self) -> dict[str, Any]:
        finite = [value for value in self.grad_norms if math.isfinite(value)]
        return {
            "observed_steps": len(self.grad_norms),
            "finite_steps": len(finite),
            "nonzero_steps": sum(1 for value in finite if value != 0.0),
            "min_grad_norm": min(finite) if finite else None,
            "max_grad_norm": max(finite) if finite else None,
            "mean_grad_norm": (sum(finite) / len(finite)) if finite else None,
            "detail": (
                "gradient norms as Lightning Fabric computed them while clipping; "
                "summary statistics only, no tensors retained"
            ),
        }

    # ── provenance ───────────────────────────────────────────────────
    def write_provenance(self, checkpoint_dir: str, *, epoch: int | None, step: int | None) -> None:
        """Put the composed record beside a checkpoint the trainer wrote."""
        if not self.provenance:
            return
        try:
            directory = Path(checkpoint_dir)
            if not directory.is_dir():
                return
            payload = dict(self.provenance)
            payload["checkpoint_path"] = str(directory)
            payload["epoch"] = int(epoch if epoch is not None else payload.get("epoch") or 0)
            payload["step"] = int(step if step is not None else payload.get("step") or 0)
            payload["segment"] = self.segment
            target = directory / str(payload.get("_filename") or "luber_checkpoint_provenance.json")
            payload.pop("_filename", None)
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.checkpoints.append(
                {"path": str(directory), "epoch": payload["epoch"], "step": payload["step"]}
            )
        except OSError as exc:
            self.checkpoints.append({"path": checkpoint_dir, "error": str(exc)})


def _digest(values: dict[str, Any]) -> str:
    running = hashlib.sha256()
    for name in sorted(values):
        running.update(name.encode("utf-8"))
        running.update(json.dumps(values[name], sort_keys=True).encode("utf-8"))
    return running.hexdigest()


def install(recorder: Recorder) -> dict[str, str]:
    """Wrap the trainer's seams. Returns what could not be wrapped."""
    missing: dict[str, str] = {}

    try:
        from acestep.training_v2 import (  # type: ignore[import-not-found]
            fixed_lora_module,
        )

        original_inject = fixed_lora_module.inject_lora_into_dit

        def inject(*args: Any, **kwargs: Any) -> Any:
            result = original_inject(*args, **kwargs)
            model = result[0] if isinstance(result, tuple) else result
            recorder.model_ref = model
            recorder.before = recorder.fingerprint(model)
            return result

        fixed_lora_module.inject_lora_into_dit = inject
    except Exception as exc:
        missing["PARAMETER_FINGERPRINT_BEFORE"] = f"{type(exc).__name__}: {exc}"

    try:
        from lightning.fabric import Fabric  # type: ignore[import-not-found]

        original_clip = Fabric.clip_gradients

        def clip(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_clip(self, *args, **kwargs)
            recorder.record_grad_norm(result.item() if hasattr(result, "item") else result)
            return result

        Fabric.clip_gradients = clip
    except Exception as exc:
        missing["GRADIENT_NORM"] = f"{type(exc).__name__}: {exc}"

    # Checkpoint provenance, written the moment the trainer finishes
    # writing a checkpoint rather than afterwards from the outside: a
    # record written later can describe a directory that changed in
    # between.
    try:
        from acestep.training_v2 import trainer_fixed as _trainer_fixed

        # Patched on the trainer class, not on `trainer_helpers`: the
        # trainer imports `save_checkpoint` by name at module load, so
        # rebinding the helper's own attribute would leave the call site
        # pointing at the original and the wrapper would never run.
        original_save = _trainer_fixed.FixedLoRATrainer._save_checkpoint

        def save(
            self: Any,
            optimizer: Any,
            scheduler: Any,
            epoch: int,
            global_step: int,
            ckpt_dir: str,
        ) -> Any:
            result = original_save(self, optimizer, scheduler, epoch, global_step, ckpt_dir)
            recorder.write_provenance(str(ckpt_dir), epoch=epoch, step=global_step)
            return result

        _trainer_fixed.FixedLoRATrainer._save_checkpoint = save

        original_final = _trainer_fixed.FixedLoRATrainer._save_final

        def save_final(self: Any, output_dir: str) -> Any:
            result = original_final(self, output_dir)
            recorder.write_provenance(
                str(output_dir), epoch=recorder.last_epoch, step=recorder.last_step
            )
            return result

        _trainer_fixed.FixedLoRATrainer._save_final = save_final
    except Exception as exc:
        missing["CHECKPOINT_PROVENANCE"] = f"{type(exc).__name__}: {exc}"

    try:
        from acestep.training_v2 import trainer_fixed

        original_train = trainer_fixed.FixedLoRATrainer.train

        def train(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                for update in original_train(self, *args, **kwargs):
                    if getattr(update, "kind", "") == "step":
                        recorder.module_ref = getattr(self, "module", None)
                        recorder.record_step(update)
                        recorder.maybe_validate(update)
                    yield update
            finally:
                module = getattr(self, "module", None)
                recorder.module_ref = module or recorder.module_ref
                model = getattr(module, "model", None) or recorder.model_ref
                if model is not None and recorder.before is not None:
                    recorder.after = recorder.fingerprint(model)
                # One last validation on the finished weights, which is
                # the only pass that describes what the run produced.
                if recorder.last_epoch is not None:
                    recorder.validate(epoch=recorder.last_epoch, step=recorder.last_step)

        trainer_fixed.FixedLoRATrainer.train = train
    except Exception as exc:
        missing["LOSS_SERIES"] = f"{type(exc).__name__}: {exc}"

    return missing


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run one bounded experiment segment, in this process."""
    argv = [str(item) for item in (request.get("argv") or [])]
    if not argv:
        return {"outcome": "BLOCKED", "failure_reason": "no trainer argv was supplied"}

    validation_dir = request.get("validation_dir")
    validator = (
        Validator(
            str(validation_dir),
            seed=int(request.get("validation_seed") or 0),
            max_samples=int(request.get("validation_max_samples") or 64),
            latent_length=request.get("validation_latent_length"),
        )
        if validation_dir
        else None
    )
    recorder = Recorder(
        step_ceiling=int(request.get("step_ceiling") or 0),
        segment=str(request.get("segment") or "A"),
        validator=validator,
        provenance=request.get("provenance") or {},
    )
    missing = install(recorder)

    outcome = "COMPLETED"
    failure_reason = ""
    traceback_tail = ""
    started = time.perf_counter()
    try:
        saved_argv = list(sys.argv)
        sys.argv = argv
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exit_signal:
            code = exit_signal.code
            if code not in (0, None):
                outcome = "FAILED"
                failure_reason = f"the trainer exited {code}"
        finally:
            sys.argv = saved_argv
    except StepCeilingExceeded as exceeded:
        outcome = "STEP_CEILING_EXCEEDED"
        failure_reason = str(exceeded)
    except BaseException as exc:
        outcome = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"
        traceback_tail = traceback.format_exc()[-4000:]

    return {
        "protocol_version": EXPERIMENT_PROBE_PROTOCOL_VERSION,
        "outcome": outcome,
        "failure_reason": failure_reason,
        "traceback_tail": traceback_tail,
        "segment": recorder.segment,
        "loss_points": recorder.points,
        "validation_points": recorder.validation_points,
        "validation_error": validator.load_error if validator else "",
        "parameters": recorder.compare(),
        "gradients": recorder.gradient_summary(),
        "checkpoints": recorder.checkpoints,
        "not_observed": missing,
        "truncated": recorder.truncated,
        "ceiling_hit": recorder.ceiling_hit,
        "step_ceiling": recorder.step_ceiling,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    try:
        raw = sys.stdin.read()
        request = json.loads(raw) if raw.strip() else {}
    except ValueError:
        request = {}
    if not isinstance(request, dict):
        request = {}

    result = run(request)
    destination = request.get("result_path")
    if destination:
        try:
            with open(str(destination), "w", encoding="utf-8") as handle:
                json.dump(result, handle, sort_keys=True)
        except OSError as exc:
            print(json.dumps({"error": f"could not write the experiment record: {exc}"}))
            return 1
    print(
        json.dumps(
            {
                "outcome": result.get("outcome"),
                "steps": len(result["loss_points"]),
                "validations": len(result["validation_points"]),
            }
        )
    )
    return 0 if result.get("outcome") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
