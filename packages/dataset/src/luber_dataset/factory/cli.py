"""Command line for the dataset factory.

    python -m luber_dataset.factory --input ~/Music --output ./build --workers 8

Four subcommands, matching the four things an operator actually does:
``build`` a dataset, ``freeze`` one they approve of, ``export`` training
manifests from it, and ``verify`` that a frozen dataset still matches
what is on disk.

``--dry-run`` scans and reports without writing a manifest. It exists for
the first contact with an unfamiliar library, where the useful question
is "what is in here and how much of it is usable" and the expensive
answer is not yet wanted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from luber_dataset.factory import manifest as manifest_io
from luber_dataset.factory.config import (
    FACTORY_VERSION,
    FactoryConfig,
    QualityThresholds,
    SplitConfig,
)
from luber_dataset.factory.export import ExportPolicy, export
from luber_dataset.factory.pipeline import FactoryResult, run, worker_count
from luber_dataset.factory.scanner import scan


def _load_quality_config(path: Path | None, base: QualityThresholds) -> QualityThresholds:
    """Overlay a JSON file onto the default thresholds.

    Unknown keys are an error rather than a no-op: a misspelled
    threshold that silently keeps the default is a configuration change
    the operator believes they made and did not.
    """
    if path is None:
        return base
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    known = set(QualityThresholds.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise SystemExit(
            f"{path}: unrecognised threshold(s): {', '.join(unknown)}\n"
            f"Known thresholds: {', '.join(sorted(known))}"
        )
    for key, value in payload.items():
        if key == "disqualifying_flags" and isinstance(value, list):
            payload[key] = tuple(value)
    return replace(base, **payload)


def _build_config(args: argparse.Namespace) -> FactoryConfig:
    config = FactoryConfig()
    quality = _load_quality_config(
        Path(args.quality_config) if getattr(args, "quality_config", None) else None,
        config.quality,
    )
    split = SplitConfig(
        train=config.split.train,
        validation=config.split.validation,
        test=config.split.test,
        seed=getattr(args, "seed", None) or config.split.seed,
    )
    return config.with_overrides(
        quality=quality,
        split=split,
        workers=getattr(args, "workers", 0) or 0,
        include_rights_unknown=getattr(args, "include_rights_unknown", False),
        min_training_tier=getattr(args, "min_tier", None) or config.min_training_tier,
    )


def _print_summary(result: FactoryResult) -> None:
    summary = result.summary
    print("\n── dataset build ──")
    for key in (
        "total_files",
        "canonical_tracks",
        "valid_audio",
        "invalid_audio",
        "exact_duplicates",
        "near_duplicates",
        "quality_A",
        "quality_B",
        "quality_C",
        "rejected",
        "training_eligible",
        "rights_unknown",
        "review_required",
        "duration_total_hours",
    ):
        print(f"  {key:24s} {summary.get(key)}")
    print(f"  {'splits':24s} {summary.get('split_counts')}")
    print(f"  {'cache':24s} {summary.get('cache')}")
    if not result.source_integrity_ok:
        print("\n  !! SOURCE AUDIO CHANGED DURING THE RUN:")
        for path in result.changed_sources:
            print(f"     {path}")
    else:
        print(f"  {'source integrity':24s} every source hash unchanged")
    if result.leaked_groups:
        print(f"\n  !! split leakage in groups: {result.leaked_groups}")
    if result.worker_failures:
        print(f"\n  {len(result.worker_failures)} file(s) failed analysis:")
        for path, error in result.worker_failures[:10]:
            print(f"     {Path(path).name}: {error}")


def cmd_build(args: argparse.Namespace) -> int:
    config = _build_config(args)
    input_root = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()

    if args.dry_run:
        scanned = scan(input_root, max_files=args.max_files)
        print(f"\n── dry run: {input_root} ──")
        print(f"  audio files found   {len(scanned.files)}")
        print(f"  skipped             {len(scanned.skipped)}")
        by_extension: dict[str, int] = {}
        for item in scanned.files:
            by_extension[item.source_extension] = by_extension.get(item.source_extension, 0) + 1
        for extension, count in sorted(by_extension.items()):
            print(f"    {extension:8s} {count}")
        print(f"  workers would be    {worker_count(config.workers)}")
        print("\n  Nothing was written.")
        return 0

    result = run(
        input_root,
        output_root,
        config,
        resume=not args.no_resume,
        force_reanalyze=args.force_reanalyze,
        max_files=args.max_files,
    )

    manifest_io.write_manifest(output_root, result.records)
    manifest_io.write_rejections(output_root, result.rejections)
    manifest_io.write_duplicates(output_root, result.duplicates)
    manifest_io.write_review_queue(output_root, result.review_queue)
    manifest_io.write_summary(output_root, result.summary)
    _print_summary(result)
    print(f"\n  written to {output_root}")

    # A source that changed under a read-only pipeline is a defect, and
    # the exit code has to say so or automation will not notice.
    return 0 if result.source_integrity_ok and not result.leaked_groups else 1


def cmd_freeze(args: argparse.Namespace) -> int:
    output_root = Path(args.output).expanduser()
    records = manifest_io.read_manifest(output_root / manifest_io.MANIFEST_NAME)
    lock = manifest_io.freeze(output_root, records, _build_config(args), dataset_id=args.dataset_id)
    print(json.dumps(lock.to_dict(), indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    output_root = Path(args.output).expanduser()
    records = manifest_io.read_manifest(output_root / manifest_io.MANIFEST_NAME)
    policy = ExportPolicy(
        allow_rights_unknown=args.include_rights_unknown,
        allow_review_required=args.include_review_required,
        min_tier=args.min_tier or "B",
    )
    result = export(records, Path(args.export_dir).expanduser(), policy)
    print(json.dumps({"policy": policy.to_dict(), **result.to_dict()}, indent=2))
    if args.include_rights_unknown:
        print(
            "\n  NOTE: rights-unknown tracks were included by explicit request.",
            file=sys.stderr,
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    output_root = Path(args.output).expanduser()
    records = manifest_io.read_manifest(output_root / manifest_io.MANIFEST_NAME)
    problems = manifest_io.verify_lock(output_root / manifest_io.LOCK_NAME, records)
    if problems:
        print("dataset does not match its lock:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("dataset matches its lock")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m luber_dataset.factory",
        description=(
            "Deterministic, resumable dataset preparation. Source audio is "
            "read-only and is verified unchanged after every run."
        ),
    )
    parser.add_argument("--version", action="version", version=FACTORY_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True, help="dataset build directory")
    common.add_argument("--seed", type=int, help="split seed; determines the assignment")
    common.add_argument(
        "--min-tier", choices=["A", "B", "C"], help="lowest quality tier admitted to training"
    )
    common.add_argument(
        "--include-rights-unknown",
        action="store_true",
        help="admit tracks whose training rights are UNKNOWN (never hard-blocked audio)",
    )

    build = sub.add_parser("build", parents=[common], help="scan, analyse and write a manifest")
    build.add_argument("--input", required=True, help="root directory of source audio")
    build.add_argument("--workers", type=int, default=0, help="0 chooses from the cpu count")
    build.add_argument("--quality-config", help="JSON file overriding quality thresholds")
    build.add_argument("--no-resume", action="store_true", help="ignore the analysis cache")
    build.add_argument(
        "--force-reanalyze", action="store_true", help="recompute everything and rewrite the cache"
    )
    build.add_argument("--max-files", type=int, help="stop after this many files")
    build.add_argument(
        "--dry-run", action="store_true", help="report what would be processed; write nothing"
    )
    build.set_defaults(func=cmd_build)

    freeze = sub.add_parser("freeze", parents=[common], help="write dataset_lock.json")
    freeze.add_argument("--dataset-id", required=True)
    freeze.set_defaults(func=cmd_freeze)

    export_cmd = sub.add_parser("export", parents=[common], help="write training manifests")
    export_cmd.add_argument("--export-dir", required=True)
    export_cmd.add_argument("--include-review-required", action="store_true")
    export_cmd.set_defaults(func=cmd_export)

    verify = sub.add_parser("verify", parents=[common], help="check a build against its lock")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
