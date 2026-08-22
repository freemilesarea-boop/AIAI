"""Watch a real pilot from inside the trainer. No LUBER imports.

The fourth file in this shape — after `_facts.py`, `_smoke.py`,
`_trainer_probe.py`, `_checkpoint_probe.py` and `_memory_probe.py` — and
for the same reason each time: the thing worth knowing is only visible
from inside the process that has torch.

What a pilot needs that the memory probe did not:

**A loss series.** Every optimizer step the trainer reports, with its
learning rate and the gradient norm Fabric computed while clipping.

**Proof the adapter moved.** A fingerprint of the trainable tensors
taken immediately after LoRA injection and again when training ends. Not
the weights — a per-tensor norm and sum, which is enough to say *which*
tensors changed and by how much, and small enough to put in a result
record.

**Proof the base model did not.** Checked by file digest on the LUBER
side rather than by hashing 2.4 billion parameters here: the question is
whether anything wrote to the weights, and a file digest answers it for
a fraction of the cost.

Everything is a pass-through wrapper. Each one calls the original,
records, and returns what the original returned; the trainer does
exactly what it would have done. No ACE-Step source is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import sys
import time
import traceback
from typing import Any

#: Bump when the shape of the emitted document changes.
PILOT_PROBE_PROTOCOL_VERSION = "luber-pilot-probe/1"

#: Most loss points a probe will hold. A pilot is capped at tens of
#: steps, so this can only be reached by a bug — and a bug should
#: produce a truncated record with a note rather than an unbounded one.
MAX_LOSS_POINTS = 10_000


class PilotStepCeilingExceeded(RuntimeError):
    """Raised in the trainer when a pilot passes its own step ceiling.

    The last line of defence, not the first. The budget is computed and
    refused before launch; this exists because a computed budget is a
    statement about the trainer's arithmetic, and if that arithmetic is
    ever wrong the run stops here rather than continuing quietly.
    """


class Recorder:
    """Loss points, parameter fingerprints and the step ceiling."""

    def __init__(self, *, step_ceiling: int, segment: str = "A") -> None:
        self.started = time.perf_counter()
        self.step_ceiling = int(step_ceiling)
        self.segment = segment
        self.points: list[dict[str, Any]] = []
        self.truncated = False
        self.before: dict[str, Any] | None = None
        self.after: dict[str, Any] | None = None
        self.grad_norms: list[float] = []
        self.model_ref: Any = None
        self.ceiling_hit = False

    # ── loss ─────────────────────────────────────────────────────────
    def record_step(self, update: Any) -> None:
        step = _as_int(getattr(update, "step", None))
        loss = _as_float(getattr(update, "loss", None))
        if step is None:
            return
        if len(self.points) >= MAX_LOSS_POINTS:
            self.truncated = True
            return
        # The gradient norm belongs to the step that just finished, so
        # it is taken from the most recent clip rather than averaged.
        norm = self.grad_norms[-1] if self.grad_norms else None
        self.points.append(
            {
                "step": step,
                "loss": loss if loss is not None else float("nan"),
                "epoch": _as_int(getattr(update, "epoch", None)),
                "learning_rate": _as_float(getattr(update, "lr", None)),
                "grad_norm": norm,
                "elapsed_seconds": round(time.perf_counter() - self.started, 3),
                "segment": self.segment,
            }
        )
        if len(self.points) > self.step_ceiling:
            self.ceiling_hit = True
            raise PilotStepCeilingExceeded(
                f"the pilot reported {len(self.points)} optimizer steps and its ceiling is "
                f"{self.step_ceiling}. The computed budget and the trainer disagree, so the "
                "run is stopped rather than continued"
            )

    def record_grad_norm(self, value: Any) -> None:
        norm = _as_float(value)
        if norm is not None:
            self.grad_norms.append(norm)

    # ── parameters ───────────────────────────────────────────────────
    def fingerprint(self, model: Any) -> dict[str, Any]:
        """A stable summary of every trainable tensor.

        Per tensor: its shape, the sum of its values and its L2 norm.
        Enough to tell which tensors moved and by how much; far too
        little to reconstruct a weight from, which is the point — this
        travels in a result record.
        """
        tensors: dict[str, list[float]] = {}
        total_parameters = 0
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            return {"error": "torch is not importable in this process"}

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                try:
                    flat = parameter.detach().to("cpu", torch.float32)
                    tensors[name] = [
                        float(flat.sum().item()),
                        float(flat.norm().item()),
                    ]
                    total_parameters += int(parameter.numel())
                except Exception:
                    continue

        digest = hashlib.sha256(
            json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "digest": digest,
            "tensor_count": len(tensors),
            "parameter_count": total_parameters,
            "tensors": tensors,
        }

    def compare(self) -> dict[str, Any]:
        """What changed between the two fingerprints.

        Names are normalised before comparison. Lightning Fabric wraps
        the module it sets up, and the wrapper prefixes every parameter
        name — so a naive comparison finds no name in common and reports
        that nothing changed. That is the single most dangerous wrong
        answer this probe could give: a healthy run would be classified
        `NO_UPDATE`, which reads as a broken trainer.

        So an empty intersection is reported as **unknown**, never as
        zero. A comparison that could not align its two sides has not
        established that nothing moved; it has established nothing.
        """
        if not self.before or not self.after:
            return {
                "detail": "one of the two fingerprints was never taken",
                "trainable_before_digest": (self.before or {}).get("digest"),
                "trainable_after_digest": (self.after or {}).get("digest"),
            }
        before = _normalise(self.before.get("tensors") or {})
        after = _normalise(self.after.get("tensors") or {})
        shared = sorted(set(before) & set(after))

        if not shared:
            return {
                "trainable_before_digest": self.before.get("digest"),
                "trainable_after_digest": self.after.get("digest"),
                "trainable_tensor_count": self.before.get("tensor_count"),
                "trainable_parameter_count": self.before.get("parameter_count"),
                "changed_tensor_count": None,
                "detail": (
                    f"the two fingerprints have no parameter name in common "
                    f"({len(before)} before, {len(after)} after), so whether anything "
                    "changed could not be established"
                ),
            }

        changed = 0
        deltas: list[float] = []
        for name in shared:
            first, second = before[name], after[name]
            delta = max(abs(first[0] - second[0]), abs(first[1] - second[1]))
            if delta > 0.0:
                changed += 1
                deltas.append(delta)
        return {
            "trainable_before_digest": self.before.get("digest"),
            "trainable_after_digest": self.after.get("digest"),
            "trainable_tensor_count": len(shared),
            "trainable_parameter_count": self.before.get("parameter_count"),
            "changed_tensor_count": changed,
            "max_absolute_delta": max(deltas) if deltas else 0.0,
            "mean_absolute_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "detail": (
                f"{changed} of {len(shared)} comparable trainable tensor(s) changed, by "
                "per-tensor sum and L2 norm"
            ),
        }

    def gradient_summary(self) -> dict[str, Any]:
        finite = [value for value in self.grad_norms if math.isfinite(value)]
        nonzero = [value for value in finite if value != 0.0]
        return {
            "observed_steps": len(self.grad_norms),
            "finite_steps": len(finite),
            "nonzero_steps": len(nonzero),
            "max_grad_norm": max(finite) if finite else None,
            "min_grad_norm": min(finite) if finite else None,
            "mean_grad_norm": (sum(finite) / len(finite)) if finite else None,
            "detail": (
                "gradient norms as Lightning Fabric computed them while clipping; summary "
                "statistics only, no tensors retained"
            ),
        }


#: Prefixes a wrapper adds to a parameter name without changing which
#: parameter it is. Lightning Fabric's `_FabricModule` is the live case;
#: `torch.nn.DataParallel` and DDP add the others.
_WRAPPER_PREFIXES: tuple[str, ...] = (
    "_forward_module.",
    "_original_module.",
    "module.",
)


#: How many trailing path components identify a parameter.
#:
#: Six is deep enough to be unique across a DiT's blocks — a LoRA weight
#: sits at something like `…blocks.11.attn.q_proj.lora_A.default.weight`
#: — and shallow enough to survive a wrapper rewriting everything above
#: it. Measured against the real model: the two sides share no full name
#: and every trailing suffix.
_NAME_SUFFIX_DEPTH = 6


def _normalise(tensors: dict[str, Any]) -> dict[str, Any]:
    """Parameter names reduced to something both sides can agree on.

    Lightning Fabric wraps the module it sets up and rewrites the path
    above every parameter, so a name taken before setup and a name taken
    after share no prefix at all. What they do share is the tail: the
    module tree *below* the wrapper is untouched.

    Prefixes are stripped first, and the result is keyed on its last few
    components. A collision would merge two tensors, so the depth is
    chosen to be deeper than the deepest repeated block path rather than
    as shallow as possible.
    """
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


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# ── instrumentation ──────────────────────────────────────────────────


def install(recorder: Recorder) -> dict[str, str]:
    """Wrap the trainer's seams. Returns what could not be wrapped."""
    missing: dict[str, str] = {}

    # The fingerprint before any step: taken the moment LoRA is attached,
    # which is the first point at which trainable parameters exist.
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

    # The gradient norm Fabric returns while clipping, once per step.
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

    # Every step the trainer reports, and the fingerprint after the last.
    try:
        from acestep.training_v2 import trainer_fixed

        original_train = trainer_fixed.FixedLoRATrainer.train

        def train(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                for update in original_train(self, *args, **kwargs):
                    if getattr(update, "kind", "") == "step":
                        recorder.record_step(update)
                    yield update
            finally:
                # In a `finally` so a crashed run still says whether the
                # adapter had moved by the time it crashed.
                module = getattr(self, "module", None)
                model = getattr(module, "model", None) or recorder.model_ref
                if model is not None and recorder.before is not None:
                    recorder.after = recorder.fingerprint(model)

        trainer_fixed.FixedLoRATrainer.train = train
    except Exception as exc:
        missing["LOSS_SERIES"] = f"{type(exc).__name__}: {exc}"

    return missing


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run one bounded pilot segment, in this process."""
    argv = [str(item) for item in (request.get("argv") or [])]
    if not argv:
        return {"outcome": "BLOCKED", "failure_reason": "no trainer argv was supplied"}

    recorder = Recorder(
        step_ceiling=int(request.get("step_ceiling") or 0),
        segment=str(request.get("segment") or "A"),
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
    except PilotStepCeilingExceeded as exceeded:
        outcome = "STEP_CEILING_EXCEEDED"
        failure_reason = str(exceeded)
    except BaseException as exc:
        # Broad on purpose: whatever the trainer did, the segment has to
        # come back with the loss points it collected and a reason,
        # rather than the probe dying alongside it.
        outcome = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"
        traceback_tail = traceback.format_exc()[-4000:]

    return {
        "protocol_version": PILOT_PROBE_PROTOCOL_VERSION,
        "outcome": outcome,
        "failure_reason": failure_reason,
        "traceback_tail": traceback_tail,
        "segment": recorder.segment,
        "loss_points": recorder.points,
        "parameters": recorder.compare(),
        "gradients": recorder.gradient_summary(),
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
        # A file, not stdout: the trainer prints a great deal, and a
        # loss series recovered from a log is a loss series a log line
        # could corrupt.
        try:
            with open(str(destination), "w", encoding="utf-8") as handle:
                json.dump(result, handle, sort_keys=True)
        except OSError as exc:
            print(json.dumps({"error": f"could not write the pilot record: {exc}"}))
            return 1
    print(json.dumps({"outcome": result.get("outcome"), "steps": len(result["loss_points"])}))
    return 0 if result.get("outcome") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
