#!/usr/bin/env python3
"""Record operator rights decisions against the approval manifest.

The operator is the only party who can move a candidate from UNVERIFIED
to CONFIRMED. This script writes that decision down together with who
attested it, on what basis, and when — so a later training run can cite
the record rather than an assumption.

It writes only to the approval manifest under ``~/.luber``. Source audio
is never touched: not modified, moved, renamed, or copied.

    uv run python scripts/dataset/apply_decisions.py \\
        --group "AI 음원" --decision CONFIRM_TRAINING_RIGHTS \\
        --origin AI_GENERATED --basis AI_SERVICE_OUTPUT_OWNED \\
        --rights-holder "operator (LUBER)" \\
        --document-reference "operator attestation 2026-08-12"

Commercial-reference groups cannot be confirmed through this tool. That
is deliberate: a bulk CLI flag is the wrong way to clear a commercial
catalogue, and refusing it here removes the easiest way to do the wrong
thing by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset.rights import OriginType, RightsBasis, TrainingRightsStatus  # noqa: E402

DECISIONS = ("CONFIRM_TRAINING_RIGHTS", "REFERENCE_ONLY", "EXCLUDE")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path.home() / ".luber" / "rights_approval_manifest.json"
    )
    parser.add_argument("--group", required=True, help="Directory group to decide on")
    parser.add_argument("--decision", required=True, choices=DECISIONS)
    parser.add_argument(
        "--origin",
        default="UNKNOWN",
        choices=[o.value for o in OriginType],
        help="How the audio was produced",
    )
    parser.add_argument("--basis", default="NONE", choices=[b.value for b in RightsBasis])
    parser.add_argument("--rights-holder", default="")
    parser.add_argument("--document-reference", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--lyrics-rights",
        action="store_true",
        help="Attest lyric rights (only meaningful where lyrics exist)",
    )
    parser.add_argument(
        "--performer-rights", action="store_true", help="Attest performer/vocal rights"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        print(f"ABORT: no approval manifest at {args.manifest}", file=sys.stderr)
        return 2

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    target = unicodedata.normalize("NFC", args.group)

    if args.decision == "CONFIRM_TRAINING_RIGHTS":
        if args.origin == OriginType.SELF_MODEL_OUTPUT.value:
            print(
                "ABORT: self-model output can never be confirmed for training",
                file=sys.stderr,
            )
            return 3
        if args.basis == RightsBasis.NONE.value:
            print("ABORT: confirming rights requires --basis", file=sys.stderr)
            return 3
        for field in ("rights_holder", "document_reference"):
            if not str(getattr(args, field)).strip():
                print(
                    f"ABORT: confirming rights requires --{field.replace('_', '-')}",
                    file=sys.stderr,
                )
                return 3

    groups = [
        g
        for g in manifest["group_summary"]
        if unicodedata.normalize("NFC", str(g["directory"])) == target
    ]
    if not groups:
        available = sorted(str(g["directory"]) for g in manifest["group_summary"])
        print(f"ABORT: no group named {target!r}. Available: {available}", file=sys.stderr)
        return 2

    if args.decision == "CONFIRM_TRAINING_RIGHTS" and any(
        g["commercial_reference"] for g in groups
    ):
        print(
            "ABORT: this group is classified as commercial reference material. "
            "Bulk-confirming a commercial catalogue is not supported here.",
            file=sys.stderr,
        )
        return 3

    confirmed = args.decision == "CONFIRM_TRAINING_RIGHTS"
    status = (
        TrainingRightsStatus.CONFIRMED.value
        if confirmed
        else TrainingRightsStatus.DENIED.value
        if args.decision == "EXCLUDE"
        else TrainingRightsStatus.UNVERIFIED.value
    )
    rights_record = {
        "origin_type": args.origin,
        "training_rights_status": status,
        "basis": args.basis,
        "rights_holder": args.rights_holder,
        "document_reference": args.document_reference,
        "confirmed_on": datetime.now(UTC).date().isoformat(),
        "audio_use_confirmed": confirmed,
        "lyrics_rights_confirmed": bool(args.lyrics_rights) and confirmed,
        "performer_rights_confirmed": bool(args.performer_rights) and confirmed,
        "commercial_training_allowed": confirmed,
        "notes": args.notes,
    }

    for group in groups:
        group["group_decision"] = args.decision
        group["rights_record"] = rights_record

    touched = 0
    for candidate in manifest["candidates"]:
        directory = unicodedata.normalize(
            "NFC", str(Path(candidate["sanitized_relative_name"]).parent)
        )
        if directory != target:
            continue
        candidate["decision"] = args.decision
        candidate["training_rights_status"] = status
        candidate["rights_record"] = rights_record
        touched += 1

    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"group      : {target}")
    print(f"decision   : {args.decision}")
    print(f"status     : {status}")
    print(f"origin     : {args.origin}")
    print(f"basis      : {args.basis}")
    print(f"holder     : {args.rights_holder or '—'}")
    print(f"document   : {args.document_reference or '—'}")
    print(f"candidates : {touched} updated")
    print("\nNo source audio was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
