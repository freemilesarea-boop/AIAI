"""Command line for the dataset factory.

    python -m luber_dataset.factory --input ~/Music --output ./build --workers 8

Subcommands match the things an operator actually does: ``build`` a
dataset, ``freeze`` one they approve of, ``export`` training manifests
from it, and ``verify`` that a frozen dataset still matches what is on
disk. Phase 24 adds the intelligence layer on top — ``profile`` to
understand a dataset, ``curate`` to plan a training selection from it,
``compare`` two datasets, and ``verify-curation`` to check a curation
against its lock.

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
from typing import Any

from luber_dataset.factory import manifest as manifest_io
from luber_dataset.factory.config import (
    FACTORY_VERSION,
    FactoryConfig,
    QualityThresholds,
    SplitConfig,
)
from luber_dataset.factory.export import ExportPolicy, export
from luber_dataset.factory.intelligence import curation as curation_module
from luber_dataset.factory.intelligence import drift as drift_module
from luber_dataset.factory.intelligence import human_report
from luber_dataset.factory.intelligence import profile as profile_module
from luber_dataset.factory.intelligence import reports as curation_reports
from luber_dataset.factory.intelligence import targets as target_module
from luber_dataset.factory.intelligence.schemas import TrackView
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


# ── Phase 24: intelligence and curation ──────────────────────────────


def _load_profile(args: argparse.Namespace) -> target_module.TargetProfile:
    """A named built-in, or a JSON file, or the neutral default."""
    if getattr(args, "profile_file", None):
        payload = json.loads(Path(args.profile_file).expanduser().read_text(encoding="utf-8"))
        return target_module.load(payload)
    if getattr(args, "profile", None):
        return target_module.by_name(args.profile)
    return target_module.neutral()


def _curation_config(args: argparse.Namespace) -> curation_module.CurationConfig:
    evaluation_only: tuple[str, ...] = ()
    if getattr(args, "evaluation_only", None):
        path = Path(args.evaluation_only).expanduser()
        evaluation_only = tuple(
            sorted(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            )
        )
    return curation_module.CurationConfig(
        seed=getattr(args, "seed", None) or 42,
        min_music_confidence=getattr(args, "min_music_confidence", None) or 0.55,
        max_sampling_weight=getattr(args, "max_sampling_weight", None) or 4.0,
        evaluation_only=evaluation_only,
    )


def _read_manifest_path(args: argparse.Namespace) -> Path:
    if getattr(args, "manifest", None):
        return Path(args.manifest).expanduser()
    return Path(args.output).expanduser() / manifest_io.MANIFEST_NAME


def _profile_from_manifest(
    path: Path, config: curation_module.CurationConfig, *, eligible_only: bool
) -> profile_module.DatasetProfile:
    records, _ = curation_module.read_manifest(path)
    views = [TrackView(record, min_confidence=config.min_music_confidence) for record in records]
    if eligible_only:
        views = [view for view in views if view.training_eligible]
    return profile_module.build(
        views, population="training_eligible" if eligible_only else "corpus"
    )


def cmd_profile(args: argparse.Namespace) -> int:
    """Describe a dataset without curating it."""
    config = _curation_config(args)
    profile = _profile_from_manifest(
        _read_manifest_path(args), config, eligible_only=args.eligible_only
    )
    print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_curate(args: argparse.Namespace) -> int:
    manifest_path = _read_manifest_path(args)
    output = Path(args.curation_output).expanduser()
    target = _load_profile(args)
    config = _curation_config(args)

    dataset_lock = manifest_path.parent / manifest_io.LOCK_NAME
    result = curation_module.curate(
        manifest_path,
        target=target,
        config=config,
        dataset_lock_path=dataset_lock if dataset_lock.is_file() else None,
    )

    summary = curation_reports.build_summary(result)
    if args.dry_run:
        # Report the plan; write nothing. The Phase 23 manifest is not
        # touched either way, but a dry run must not leave a curated
        # artifact somebody could mistake for an approved one.
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        print("\n  Dry run: no curated artifacts were written.")
        return 0

    curation_reports.write_curated_manifest(output, result)
    weights_digest = curation_reports.write_sampling_weights(output, result)
    curation_reports.write_summary(output, result)
    curation_reports.write_wishlist(output, result)

    existing_review: list[dict[str, Any]] = []
    review_source = manifest_path.parent / "dataset_review_queue.jsonl"
    if review_source.is_file():
        existing_review = [
            json.loads(line)
            for line in review_source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    curation_reports.write_review_queue(output, result, existing_review)

    output.mkdir(parents=True, exist_ok=True)
    (output / curation_reports.REPORT_NAME).write_text(
        human_report.render(result), encoding="utf-8"
    )

    if args.curation_id:
        curation_reports.freeze(
            output, result, curation_id=args.curation_id, weights_digest=weights_digest
        )

    print("\n── curation ──")
    for key in (
        "tracks_input",
        "training_eligible_input",
        "tracks_selected",
        "hours_selected",
    ):
        print(f"  {key:26s} {summary.get(key)}")
    print(f"  {'actions':26s} {summary.get('actions')}")
    print(f"  {'excluded_by_reason':26s} {summary.get('excluded_by_reason')}")
    critical = [f for f in result.findings if f.severity == "CRITICAL"]
    print(f"  {'critical findings':26s} {len(critical)}")
    for finding in critical[:5]:
        print(f"     {finding.code}: {finding.detail}")
    print(f"\n  written to {output}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    config = _curation_config(args)
    before = _profile_from_manifest(Path(args.manifest_a).expanduser(), config, eligible_only=True)
    after = _profile_from_manifest(Path(args.manifest_b).expanduser(), config, eligible_only=True)
    report = drift_module.compare(
        before, after, label_a=args.label_a or "A", label_b=args.label_b or "B"
    )
    output = Path(args.curation_output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / curation_reports.DIFF_JSON_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = drift_module.render_markdown(report)
    (output / curation_reports.DIFF_MD_NAME).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Re-render the human report from a manifest, writing nothing else."""
    result = curation_module.curate(
        _read_manifest_path(args), target=_load_profile(args), config=_curation_config(args)
    )
    print(human_report.render(result))
    return 0


def cmd_verify_curation(args: argparse.Namespace) -> int:
    """Recompute the curation and check it against its lock.

    Recomputing rather than re-reading is the point: a lock that only
    checked the curated file against itself would pass after the source
    manifest changed underneath it.
    """
    output = Path(args.curation_output).expanduser()
    result = curation_module.curate(
        _read_manifest_path(args), target=_load_profile(args), config=_curation_config(args)
    )
    problems = curation_reports.verify(output / curation_reports.LOCK_NAME, result)

    for track_id in result.selection.selected_ids:
        decision = result.selection.decisions[track_id]
        if not decision.selected:
            problems.append(f"{track_id} is selected but its action is {decision.action}")
    plan = result.sampling_plan
    if not plan.bounded:
        problems.append("sampling weights fall outside their declared bounds")
    for track_id in set(result.config.evaluation_only) & set(result.selection.selected_ids):
        problems.append(f"evaluation-only track {track_id} entered the training selection")

    if problems:
        print("curation does not verify:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("curation matches its lock")
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

    # ── Phase 24 ─────────────────────────────────────────────────────
    curation_common = argparse.ArgumentParser(add_help=False)
    curation_common.add_argument(
        "--manifest", help="Phase 23 dataset_manifest.jsonl (defaults to --output/…)"
    )
    curation_common.add_argument(
        "--profile", help=f"built-in target profile: {', '.join(sorted(target_module.BUILT_IN))}"
    )
    curation_common.add_argument("--profile-file", help="JSON target profile")
    curation_common.add_argument(
        "--evaluation-only",
        help="file of track ids that must never enter training, one per line",
    )
    curation_common.add_argument(
        "--min-music-confidence",
        type=float,
        help="confidence a tempo or key needs before it counts as known (default 0.55)",
    )
    curation_common.add_argument("--seed", type=int, help="curation seed")
    curation_common.add_argument(
        "--curation-output", default="curated-build", help="where curated artifacts are written"
    )

    profile_cmd = sub.add_parser(
        "profile", parents=[curation_common], help="describe a dataset's distributions"
    )
    profile_cmd.add_argument("--output", help="dataset build directory")
    profile_cmd.add_argument(
        "--eligible-only",
        action="store_true",
        help="profile only training-eligible tracks",
    )
    profile_cmd.set_defaults(func=cmd_profile)

    curate_cmd = sub.add_parser(
        "curate", parents=[curation_common], help="plan a training selection"
    )
    curate_cmd.add_argument("--output", help="dataset build directory")
    curate_cmd.add_argument(
        "--max-sampling-weight", type=float, help="cap on any sampling weight (default 4.0)"
    )
    curate_cmd.add_argument(
        "--curation-id", help="freeze the result under this id, writing curation_lock.json"
    )
    curate_cmd.add_argument(
        "--dry-run", action="store_true", help="report the plan; write no curated artifacts"
    )
    curate_cmd.set_defaults(func=cmd_curate)

    compare_cmd = sub.add_parser(
        "compare", parents=[curation_common], help="drift between two manifests"
    )
    compare_cmd.add_argument("--manifest-a", required=True)
    compare_cmd.add_argument("--manifest-b", required=True)
    compare_cmd.add_argument("--label-a")
    compare_cmd.add_argument("--label-b")
    compare_cmd.add_argument("--output", help="dataset build directory")
    compare_cmd.set_defaults(func=cmd_compare)

    report_cmd = sub.add_parser(
        "report", parents=[curation_common], help="render the human curation report"
    )
    report_cmd.add_argument("--output", help="dataset build directory")
    report_cmd.set_defaults(func=cmd_report)

    verify_curation = sub.add_parser(
        "verify-curation", parents=[curation_common], help="check a curation against its lock"
    )
    verify_curation.add_argument("--output", help="dataset build directory")
    verify_curation.set_defaults(func=cmd_verify_curation)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
