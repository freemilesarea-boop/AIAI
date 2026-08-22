"""Open a checkpoint and say what is really in it. No LUBER imports.

Executed by the trainer's interpreter, like ``_trainer_probe.py`` and
for the same reason: reading an adapter needs `safetensors` and `torch`,
and LUBER's own environment has neither.

The distinction this file exists to enforce: **a file that exists is not
a checkpoint**. Phase 27 already hashes what the trainer wrote and
records its size, which catches a truncated transfer and nothing else. A
directory can hold a correctly-sized, correctly-hashed adapter whose
tensors are all zero, whose training state cannot be deserialised, or
whose step count is missing — and every one of those is a checkpoint
that will fail the first time somebody tries to use it, hours later.

So this loads the tensors, counts the ones that are not zero, reads the
training state back, and reports what it found. It writes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: Bump when the shape of the emitted document changes.
CHECKPOINT_PROBE_VERSION = "luber-checkpoint-probe/1"

#: What the trainer's `save_checkpoint` writes into a checkpoint dir.
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
TRAINING_STATE = "training_state.pt"


def _adapter(directory: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    weights = directory / ADAPTER_WEIGHTS
    out["weights_present"] = weights.is_file()
    out["weights_bytes"] = weights.stat().st_size if weights.is_file() else 0
    if not weights.is_file() or out["weights_bytes"] == 0:
        out["reopened"] = False
        out["error"] = f"{ADAPTER_WEIGHTS} is absent or empty"
        return out

    try:
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        tensors = load_file(str(weights))
    except Exception as exc:
        out["reopened"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    out["reopened"] = True
    out["tensor_count"] = len(tensors)
    total = 0
    non_zero = 0
    for tensor in tensors.values():
        total += int(tensor.numel())
        try:
            non_zero += int((tensor != 0).sum().item())
        except Exception:
            pass
    out["parameter_count"] = total
    out["non_zero_parameters"] = non_zero
    # A LoRA's B matrices are initialised to zero, so "some zeros" is
    # normal and "all zeros" means nothing was learned.
    out["all_zero"] = total > 0 and non_zero == 0

    config = directory / ADAPTER_CONFIG
    out["config_present"] = config.is_file()
    if config.is_file():
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
            out["config_readable"] = isinstance(payload, dict)
            if isinstance(payload, dict):
                out["adapter_r"] = payload.get("r")
                out["adapter_alpha"] = payload.get("lora_alpha")
                out["target_modules"] = sorted(payload.get("target_modules") or [])
        except Exception as exc:
            out["config_readable"] = False
            out["config_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _training_state(directory: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    state_path = directory / TRAINING_STATE
    out["present"] = state_path.is_file()
    if not state_path.is_file():
        return out

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        out["reopened"] = False
        out["error"] = f"torch is not importable here: {type(exc).__name__}: {exc}"
        return out

    payload = None
    try:
        payload = torch.load(str(state_path), map_location="cpu", weights_only=True)
        out["weights_only"] = True
    except Exception:
        try:
            # The optimizer state holds plain Python objects that the
            # restricted loader refuses. Falling back is recorded rather
            # than done quietly: it is a real difference in what was
            # trusted, and this file is one we just wrote ourselves.
            payload = torch.load(str(state_path), map_location="cpu", weights_only=False)
            out["weights_only"] = False
        except Exception as exc:
            out["reopened"] = False
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out

    out["reopened"] = True
    if isinstance(payload, dict):
        out["epoch"] = payload.get("epoch")
        out["global_step"] = payload.get("global_step")
        out["has_optimizer_state"] = "optimizer_state_dict" in payload
        out["has_scheduler_state"] = "scheduler_state_dict" in payload
        optimizer = payload.get("optimizer_state_dict")
        if isinstance(optimizer, dict):
            out["optimizer_tensor_groups"] = len(optimizer.get("state", {}) or {})
    return out


def run(directory: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "probe_version": CHECKPOINT_PROBE_VERSION,
        "directory": directory.name,
        "exists": directory.is_dir(),
    }
    if not directory.is_dir():
        out["error"] = "the checkpoint directory does not exist"
        return out
    out["file_count"] = sum(1 for path in directory.rglob("*") if path.is_file())
    out["adapter"] = _adapter(directory)
    out["training_state"] = _training_state(directory)
    out["ok"] = bool(
        out["adapter"].get("reopened")
        and not out["adapter"].get("all_zero")
        and (not out["training_state"].get("present") or out["training_state"].get("reopened"))
    )
    return out


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        print(json.dumps({"error": "a checkpoint directory is required"}, sort_keys=True))
        return 2
    print(json.dumps(run(Path(arguments[0])), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
