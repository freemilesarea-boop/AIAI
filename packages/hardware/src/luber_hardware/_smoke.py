"""A tiny training run, on whatever device is asked for.

Same shape as `_facts.py` and for the same reason: **no LUBER imports**,
so the interpreter that actually has torch can execute this file
directly and print one JSON document. LUBER's own environment has no
torch, so a test that could only run in-process would skip on the one
machine that can answer.

What this proves and what it does not.

It **proves the path**: that a forward, a backward, an optimizer step, a
scheduler step, a checkpoint save and a checkpoint load all work on a
given device, and that a checkpoint written on one device loads on
another. That is the compatibility question — can Apple silicon run the
*mechanics* of training at all — and it is answerable in seconds.

It **proves nothing about the model**. The network here is two small
linear layers on synthetic noise. It is not ACE-Step, it is not a DiT,
it has no music in it, and a number measured here may not be
extrapolated to a real run or to different hardware. The benchmarks
exist to catch "MPS is not actually being used" and "this path is
broken", not to rank machines.

Allocations are deliberately small and fixed. A memory probe that tried
to find the ceiling by allocating until something broke would swap a
24 GB machine into the ground, and the answer would be a number about
that day's memory pressure rather than about the hardware.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

#: Bump when the shape of the emitted result changes.
SMOKE_VERSION = "hardware-smoke/1"

#: The toy problem. Small enough to finish in under a second on a CPU,
#: large enough that a matmul is not entirely dominated by dispatch.
INPUT_FEATURES = 64
HIDDEN_FEATURES = 128
OUTPUT_FEATURES = 16
BATCH_SIZE = 32
TRAIN_STEPS = 8

#: The benchmark matmul. 512 by 512 is a few hundred megaflops — enough to
#: register above timer noise, small enough to be free on any machine
#: this project will ever touch.
MATMUL_SIZE = 512
MATMUL_ITERATIONS = 20

#: The bounded memory probe: 64 MiB of float32, allocated and released.
#: Chosen to be obviously safe on any machine, because the question is
#: "does allocation work and does the runtime report it", not "how much
#: is there".
MEMORY_PROBE_ELEMENTS = 16 * 1024 * 1024


def _tiny_model(torch: Any) -> Any:
    """Two linear layers. Deliberately not a music model."""
    from torch import nn  # type: ignore[import-not-found]

    return nn.Sequential(
        nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES),
        nn.ReLU(),
        nn.Linear(HIDDEN_FEATURES, OUTPUT_FEATURES),
    )


def _synthetic_batch(torch: Any, device: str, seed: int) -> tuple[Any, Any]:
    """Deterministic noise. No dataset, no audio, nothing on disk."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(BATCH_SIZE, INPUT_FEATURES, generator=generator)
    targets = torch.randn(BATCH_SIZE, OUTPUT_FEATURES, generator=generator)
    return inputs.to(device), targets.to(device)


def train_steps(torch: Any, device: str) -> dict[str, Any]:
    """Forward, backward, optimizer, scheduler — the whole mechanism.

    The loss values are returned rather than asserted on. Whether the
    toy problem converges is uninteresting; whether every step *ran* on
    this device is the entire point, and a caller can see from the
    numbers that something moved.
    """
    from torch import nn

    result: dict[str, Any] = {"device": device, "steps": TRAIN_STEPS}
    torch.manual_seed(0)

    model = _tiny_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    loss_fn = nn.MSELoss()

    losses: list[float] = []
    started = time.perf_counter()
    for step in range(TRAIN_STEPS):
        inputs, targets = _synthetic_batch(torch, device, seed=step)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        # Gradient accumulation in miniature: the clip is what touches
        # every gradient tensor, and it is a real operator that has to
        # exist on the device.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().to("cpu").item()))
    _synchronize(torch, device)

    result["seconds"] = time.perf_counter() - started
    result["first_loss"] = losses[0]
    result["last_loss"] = losses[-1]
    result["loss_changed"] = abs(losses[0] - losses[-1]) > 1e-9
    result["final_lr"] = float(scheduler.get_last_lr()[0])
    result["parameters_finite"] = all(
        bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()
    )
    result["ok"] = bool(result["loss_changed"] and result["parameters_finite"])
    return result


def checkpoint_roundtrip(torch: Any, device: str, load_on: list[str]) -> dict[str, Any]:
    """Save on one device, load on the others.

    `map_location` is the whole question. A checkpoint written from MPS
    holds MPS tensors, and loading it on a machine without Metal fails
    unless the loader is told where to put them. This is the test that
    says whether "train on the GPU box, inspect on the Mac" is a
    topology or a wish.
    """
    result: dict[str, Any] = {"saved_on": device, "loads": {}}
    torch.manual_seed(0)
    model = _tiny_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # One step, so the optimizer has real state to save. An optimizer
    # that has never stepped writes empty state and would make this
    # test pass without testing anything.
    inputs, targets = _synthetic_batch(torch, device, seed=99)
    loss = ((model(inputs) - targets) ** 2).mean()
    loss.backward()
    optimizer.step()

    # A low-rank pair standing in for a LoRA adapter: the same shape of
    # object — small, separate from the base weights, saved on its own.
    adapter = {
        "lora_A": torch.randn(8, INPUT_FEATURES, device=device),
        "lora_B": torch.zeros(OUTPUT_FEATURES, 8, device=device),
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.pt"
        started = time.perf_counter()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "adapter": adapter,
            },
            path,
        )
        result["save_seconds"] = time.perf_counter() - started
        result["bytes"] = path.stat().st_size

        for target in load_on:
            answer: dict[str, Any] = {}
            try:
                started = time.perf_counter()
                # `weights_only=True` where the runtime offers it: this
                # loads a file this process just wrote, but a loader
                # that unpickles arbitrary objects is a habit worth not
                # having in a file other machines will also read.
                payload = _load(torch, path, target)
                answer["seconds"] = time.perf_counter() - started

                restored = _tiny_model(torch).to(target)
                restored.load_state_dict(payload["model"])
                answer["model"] = True

                restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
                restored_optimizer.load_state_dict(payload["optimizer"])
                # Loading optimizer state is not proof it is usable —
                # the tensors have to be on the right device for the
                # next step to run. So take one.
                inputs, targets = _synthetic_batch(torch, target, seed=1)
                restored_optimizer.zero_grad(set_to_none=True)
                ((restored(inputs) - targets) ** 2).mean().backward()
                restored_optimizer.step()
                answer["optimizer"] = True

                adapter_in = payload["adapter"]
                moved = {key: value.to(target) for key, value in adapter_in.items()}
                answer["adapter"] = all(
                    str(value.device).startswith(target) for value in moved.values()
                )
                answer["ok"] = True
            except Exception as exc:
                answer["ok"] = False
                answer["error"] = f"{type(exc).__name__}: {exc}"
            result["loads"][target] = answer
    return result


def _load(torch: Any, path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch without `weights_only`.
        return torch.load(path, map_location=device)


def benchmark(torch: Any, device: str) -> dict[str, Any]:
    """A matmul and a forward/backward, timed. Sanity, not ranking.

    Every number here is about *this* machine on *this* day. It may not
    be extrapolated to different hardware, and there is nothing in the
    output shaped like a comparison because there is nothing to compare
    it with.
    """
    result: dict[str, Any] = {"device": device}
    left = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device=device)
    right = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device=device)

    # Warm up: the first call on an accelerator pays for kernel
    # compilation and allocation, and timing that would measure the
    # runtime's startup rather than the hardware.
    for _ in range(3):
        _ = left @ right
    _synchronize(torch, device)

    started = time.perf_counter()
    for _ in range(MATMUL_ITERATIONS):
        _ = left @ right
    _synchronize(torch, device)
    elapsed = time.perf_counter() - started
    result["matmul_ms"] = (elapsed / MATMUL_ITERATIONS) * 1000
    result["matmul_size"] = MATMUL_SIZE

    model = _tiny_model(torch).to(device)
    inputs, targets = _synthetic_batch(torch, device, seed=7)
    for _ in range(3):
        ((model(inputs) - targets) ** 2).mean().backward()
    _synchronize(torch, device)
    started = time.perf_counter()
    for _ in range(MATMUL_ITERATIONS):
        model.zero_grad(set_to_none=True)
        ((model(inputs) - targets) ** 2).mean().backward()
    _synchronize(torch, device)
    result["forward_backward_ms"] = ((time.perf_counter() - started) / MATMUL_ITERATIONS) * 1000
    return result


def memory_probe(torch: Any, device: str) -> dict[str, Any]:
    """Allocate a bounded block, ask the runtime what it thinks, release.

    64 MiB. Not a search for the ceiling: exhausting a machine's memory
    to find out how much it has would swap the control plane out and
    measure the day rather than the hardware.
    """
    result: dict[str, Any] = {
        "device": device,
        "allocated_mb": MEMORY_PROBE_ELEMENTS * 4 // (1024 * 1024),
    }
    try:
        block = torch.zeros(MEMORY_PROBE_ELEMENTS, dtype=torch.float32, device=device)
        _synchronize(torch, device)
        result["allocated"] = True
        result["reported_mb"] = _reported_memory_mb(torch, device)
        del block
        _empty_cache(torch, device)
        result["released"] = True
    except Exception as exc:
        result["allocated"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _reported_memory_mb(torch: Any, device: str) -> Any:
    """What the runtime says it is holding, where it will say."""
    try:
        if device == "mps":
            return int(torch.mps.current_allocated_memory() // (1024 * 1024))
        if device == "cuda":
            return int(torch.cuda.memory_allocated() // (1024 * 1024))
    except Exception:
        return None
    return None


def _empty_cache(torch: Any, device: str) -> None:
    try:
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _synchronize(torch: Any, device: str) -> None:
    """Wait for the device.

    Accelerator work is queued. Timing without this measures how fast
    Python can enqueue, which on MPS looks like an implausibly fast GPU
    and is the single easiest benchmark mistake to make.
    """
    try:
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def available_devices(torch: Any) -> list[str]:
    devices = ["cpu"]
    try:
        backends = getattr(torch.backends, "mps", None)
        if backends is not None and backends.is_available():
            devices.append("mps")
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:
        pass
    return devices


def run(devices: list[str] | None = None) -> dict[str, Any]:
    """Every check, on every requested device."""
    out: dict[str, Any] = {"smoke_version": SMOKE_VERSION}
    try:
        import torch
    except Exception as exc:
        out["torch_installed"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    out["torch_installed"] = True
    out["torch_version"] = str(torch.__version__)
    present = available_devices(torch)
    chosen = [item for item in (devices or present) if item in present]
    out["devices"] = chosen
    out["skipped"] = [item for item in (devices or []) if item not in present]

    results: dict[str, Any] = {}
    for device in chosen:
        entry: dict[str, Any] = {}
        try:
            entry["training"] = train_steps(torch, device)
            entry["checkpoint"] = checkpoint_roundtrip(torch, device, load_on=present)
            entry["benchmark"] = benchmark(torch, device)
            entry["memory"] = memory_probe(torch, device)
            entry["ok"] = bool(entry["training"].get("ok"))
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results[device] = entry
    out["results"] = results
    out["ok"] = all(item.get("ok") for item in results.values()) if results else False
    return out


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1:] or None), sort_keys=True))
