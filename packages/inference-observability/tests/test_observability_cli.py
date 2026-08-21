"""The operator's way in, and the promise that it changes nothing.

Named for its package rather than `test_cli`: pytest resolves test
modules by basename without `__init__.py`, so two `test_cli.py` files in
one repository collide at collection and one of them never runs. Phase
29's QC package already has one.

Every command here reads. Two write — `ingest` and `backfill` — and both
write a projection keyed on the generation, so running either twice
changes no count. Nothing in this surface can start a generation, move a
threshold or disable a provider, and the parser is asserted for that
rather than only documented.
"""

from __future__ import annotations

import pytest

from luber_inference_observability.cli import build_parser


def parser():
    return build_parser()


def commands() -> set[str]:
    action = next(item for item in parser()._subparsers._group_actions if item.choices is not None)
    return set(action.choices)


def test_every_command_the_brief_asks_for_exists():
    assert {
        "ingest",
        "backfill",
        "summary",
        "regressions",
        "incidents",
        "incident",
        "report",
        "verify",
    } <= commands()


def test_no_command_can_change_the_system():
    """Detection only. A verb that acted would be a Phase 31 decision
    made quietly in a Phase 30 CLI."""
    forbidden = {
        "disable",
        "enable",
        "rollback",
        "restart",
        "set-threshold",
        "switch-policy",
        "generate",
        "retry",
    }
    assert not (commands() & forbidden)


def test_ingest_and_backfill_are_distinct_verbs():
    """Backfill re-reads everything; ingest resumes from a watermark.
    One verb with a flag would make the expensive one the default."""
    assert "ingest" in commands()
    assert "backfill" in commands()


def test_a_window_can_only_be_one_the_engine_supports():
    parsed = parser().parse_args(["summary", "--window", "24h"])
    assert parsed.window == "24h"
    with pytest.raises(SystemExit):
        parser().parse_args(["summary", "--window", "3y"])


def test_an_arbitrary_interval_is_supported():
    parsed = parser().parse_args(
        ["summary", "--start", "2026-08-01T00:00:00+00:00", "--end", "2026-08-02T00:00:00+00:00"]
    )
    assert parsed.start and parsed.end


def test_a_baseline_gap_is_configurable_and_defaults_to_an_hour():
    parsed = parser().parse_args(["regressions"])
    assert parsed.baseline_gap_hours == 1
    assert parsed.baseline_days == 7


def test_dismissing_an_incident_requires_a_reason():
    with pytest.raises(SystemExit):
        parser().parse_args(["incident", "dismiss", "abc", "--operator", "alex"])
    parsed = parser().parse_args(
        ["incident", "dismiss", "abc", "--operator", "alex", "--reason", "known load test"]
    )
    assert parsed.reason == "known load test"


def test_acknowledging_requires_naming_the_operator():
    with pytest.raises(SystemExit):
        parser().parse_args(["incident", "acknowledge", "abc"])
    assert (
        parser().parse_args(["incident", "acknowledge", "abc", "--operator", "alex"]).operator
        == "alex"
    )


def test_a_revision_comparison_needs_both_sides():
    with pytest.raises(SystemExit):
        parser().parse_args(["providers", "--left", "a"])
    parsed = parser().parse_args(["providers", "--left", "a", "--right", "b"])
    assert parsed.left == "a" and parsed.right == "b"


def test_a_deployment_comparison_needs_a_moment():
    with pytest.raises(SystemExit):
        parser().parse_args(["deployment"])
    assert parser().parse_args(["deployment", "--at", "2026-08-21T12:00:00+00:00"]).hours == 24


def test_segment_ranking_has_a_minimum_sample_default():
    parsed = parser().parse_args(["segments"])
    assert parsed.minimum_samples >= 1
    assert "," in parsed.group_by
