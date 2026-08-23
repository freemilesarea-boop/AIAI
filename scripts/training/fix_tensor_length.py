#!/usr/bin/env python3
"""Phase 36 — rewrite preprocessed tensors to one common sequence length.

Run by the *trainer's* interpreter; imports nothing from LUBER.

Why this exists. A training split of 24 tracks has 24 different latent
lengths, and Metal's caching allocator keeps a working set per shape.
Measured on an M4 Pro / 24 GB: a run over those 24 shapes reached 29.2
GiB of MPS allocations and died at step 9 of every attempt, while Phase
35B's four-track pilot at the *same* maximum length peaked at 9.4 GiB.
The variable that changed was the number of distinct shapes, not the
longest one — freeing 20 GB elsewhere on the machine did not move the
failure by a single step.

So every sample is cut to the same leading window. Truncation only: the
window is chosen at or below the shortest track so nothing is padded and
no frame of audio is invented. What it costs is that training sees the
first N frames of each track rather than all of it, which the experiment
record and the docs both state.

    ~/ace-step-1.5/.venv/bin/python scripts/training/fix_tensor_length.py \
        --input tensors/train --output tensors/train_fixed --length 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Fields whose first dimension is the audio timeline. The conditioning
#: tensors are not here: they describe the prompt, and cutting them
#: would change what the model is asked, not how much it is shown.
SEQUENCE_FIELDS: tuple[str, ...] = ("target_latents", "attention_mask", "context_latents")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True, help="latent frames to keep")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch  # type: ignore[import-not-found]

    samples = sorted(args.input.glob("*.pt"))
    if not samples:
        print(f"ABORT: no tensors under {args.input}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    shortest = None
    written = 0
    for path in samples:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            print(f"ABORT: {path.name} is not a tensor dict", file=sys.stderr)
            return 2
        length = int(payload["target_latents"].shape[0])
        shortest = length if shortest is None else min(shortest, length)
        if length < args.length:
            # Refused rather than padded. A padded frame is content the
            # recording does not have, and a loss computed over it is a
            # loss over silence somebody inserted.
            print(
                f"ABORT: {path.name} has {length} frames and the window is {args.length}; "
                "padding would train on frames the recording does not have",
                file=sys.stderr,
            )
            return 2
        for field in SEQUENCE_FIELDS:
            if field in payload and torch.is_tensor(payload[field]):
                payload[field] = payload[field][: args.length].contiguous()
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata["luber_window_latent_frames"] = args.length
            metadata["luber_original_latent_frames"] = length
        torch.save(payload, str(args.output / path.name))
        written += 1

    print(f"{written} sample(s) -> {args.output} at {args.length} latent frames")
    print(f"shortest input was {shortest} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
