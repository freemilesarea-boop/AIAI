"""Read preprocessed tensors under the trainer's own interpreter.

Runs where torch lives, which is not where LUBER lives, so this file
imports nothing from LUBER and communicates by JSON on stdin and
stdout. Same contract as the other probes in this package.

It answers one question per sample: can this be opened, does it carry
the fields the DiT reads, are the values finite, and what shape is it.
Nothing is repaired and nothing is skipped — a sample that cannot be
read is reported as unreadable rather than quietly dropped, because a
dataset that shrank between preprocessing and training is exactly the
kind of thing that has to be noticed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TENSOR_PROBE_PROTOCOL_VERSION = "luber-tensor-probe/1"

#: What `PreprocessedDataModule` hands the module every step.
REQUIRED_FIELDS: tuple[str, ...] = (
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
)


def _inspect(path: Path) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    record: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # the reason is the result, not a crash
        record.update(readable=False, error=f"{type(exc).__name__}: {exc}")
        return record

    record["readable"] = True
    if not isinstance(payload, dict):
        record.update(ok=False, error=f"expected a dict of tensors, found {type(payload).__name__}")
        return record

    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    record["missing_fields"] = missing

    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    non_finite: list[str] = []
    for name in REQUIRED_FIELDS:
        value = payload.get(name)
        if value is None or not torch.is_tensor(value):
            continue
        shapes[name] = list(value.shape)
        dtypes[name] = str(value.dtype)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            non_finite.append(name)

    record["shapes"] = shapes
    record["dtypes"] = dtypes
    record["non_finite_fields"] = non_finite
    latents = shapes.get("target_latents") or []
    encoder = shapes.get("encoder_hidden_states") or []
    record["latent_length"] = latents[0] if latents else None
    record["latent_channels"] = latents[1] if len(latents) > 1 else None
    record["encoder_length"] = encoder[0] if encoder else None
    record["ok"] = not missing and not non_finite and bool(latents) and bool(encoder)
    return record


def main() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    directory = Path(request["dataset_dir"])
    samples = sorted(directory.glob("*.pt"))
    records = [_inspect(path) for path in samples]

    document = {
        "protocol_version": TENSOR_PROBE_PROTOCOL_VERSION,
        "dataset_dir": str(directory),
        "sample_count": len(records),
        "samples": records,
    }
    result_path = request.get("result_path")
    if result_path:
        Path(result_path).write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
