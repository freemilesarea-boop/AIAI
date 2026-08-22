"""The operator surface: it reports, and it changes nothing.

Two properties are held here. Every verb is read-only — there is no code
path in the CLI that trains, downloads or writes a configuration — and
the report distinguishes what was measured from what was merely
computed, because a document where those look alike is worse than no
document.
"""

from __future__ import annotations

import json

from hardware_fixtures import apple_machine, cuda_machine, gpu_target, mac_target

from luber_hardware.cli import build_parser, main
from luber_hardware.report import Evidence, build_report

# ── the report ───────────────────────────────────────────────────────


def test_every_finding_says_how_it_was_established():
    report = build_report(apple_machine(), targets=[mac_target()])

    assert report.findings
    known = {item.value for item in Evidence}
    assert all(finding.evidence in known for finding in report.findings)


def test_a_machine_without_cuda_says_so_and_says_why_it_matters():
    report = build_report(apple_machine(), targets=[mac_target()])
    cuda = next(item for item in report.findings if item.subject == "NVIDIA CUDA")

    assert "NOT AVAILABLE" in cuda.verdict
    assert "synthetic or not run" in cuda.detail


def test_a_blocked_placement_is_a_verified_finding_not_an_unchecked_one():
    """ "Cannot be placed" was computed from a real probe by the policy
    the scheduler uses. Marking it NOT_RUN would claim nobody looked."""
    report = build_report(apple_machine(), targets=[mac_target()])
    heavy = next(item for item in report.findings if item.subject == "placement: HEAVY_TRAINING")

    assert heavy.verdict.startswith("BLOCKED")
    assert heavy.evidence == Evidence.VERIFIED.value


def test_a_missing_smoke_is_reported_as_not_run_rather_than_omitted():
    report = build_report(apple_machine(), targets=[mac_target()])
    smoke = next(item for item in report.findings if item.subject == "tiny training smoke")

    assert smoke.verdict == "NOT RUN"
    assert smoke.evidence == Evidence.NOT_RUN.value


def test_a_smoke_result_becomes_verified_findings():
    report = build_report(
        apple_machine(),
        targets=[mac_target()],
        smoke={
            "torch_installed": True,
            "results": {
                "mps": {
                    "ok": True,
                    "training": {"steps": 8, "first_loss": 1.0, "last_loss": 0.5},
                    "checkpoint": {"loads": {"cpu": {"ok": True}, "mps": {"ok": True}}},
                    "benchmark": {
                        "matmul_size": 512,
                        "matmul_ms": 0.25,
                        "forward_backward_ms": 0.3,
                    },
                }
            },
        },
    )

    training = next(item for item in report.findings if item.subject == "tiny training on mps")
    assert training.verdict == "PASS"
    assert training.evidence == Evidence.VERIFIED.value
    assert "not ACE-Step" in training.detail

    benchmark = next(item for item in report.findings if "benchmark" in item.subject)
    assert "this machine only" in benchmark.detail


def test_the_memory_budget_never_offers_the_whole_machine():
    report = build_report(apple_machine(), targets=[mac_target()])
    budget = next(item for item in report.findings if item.subject == "memory budget")

    assert "24576 MB" in budget.verdict
    assert "usable" in budget.verdict
    assert "held back" in budget.detail
    assert "UNKNOWN" in budget.detail


def test_the_report_carries_no_host_identity():
    """Structural — the capability model has no field for one — and
    asserted against the rendered document anyway."""
    report = build_report(cuda_machine(), targets=[gpu_target()])
    both = report.to_json() + report.to_markdown()

    for needle in ("hostname", "/Users/", "/home/", "serial", "MAC address"):
        assert needle.lower() not in both.lower()


def test_the_report_renders_in_both_forms():
    report = build_report(apple_machine(), targets=[mac_target()])

    payload = json.loads(report.to_json())
    assert payload["capability_schema_version"]
    assert payload["findings"]

    markdown = report.to_markdown()
    assert markdown.startswith("# Hardware compatibility report")
    assert "| Subject | Verdict | Evidence | Detail |" in markdown
    assert "remote generation does not" in markdown, "the Phase 27 gap stays stated"


# ── the CLI ──────────────────────────────────────────────────────────


def test_every_verb_exists():
    parser = build_parser()
    for verb in ("probe", "compatibility", "placement", "readiness", "verify", "report"):
        assert parser.parse_args([verb]).command == verb


def test_planned_hardware_is_opt_in():
    """A status view that silently included a machine nobody owns would
    be planning material pretending to be a status report."""
    parser = build_parser()

    assert parser.parse_args(["readiness"]).include_planned is False
    assert parser.parse_args(["readiness", "--include-planned"]).include_planned is True


def test_local_fallback_is_opt_in():
    parser = build_parser()

    assert parser.parse_args(["placement"]).allow_local_fallback is False


def test_probe_runs_and_prints_without_touching_anything(capsys):
    assert main(["probe"]) == 0
    out = capsys.readouterr().out

    assert "devices:" in out


def test_probe_reports_json_on_request(capsys):
    assert main(["probe", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["capability_schema_version"]
    assert "devices" in payload


def test_a_missing_interpreter_fails_loudly_rather_than_falling_back(capsys):
    """Falling back to this process would answer about the wrong
    machine's Python, which is the one mistake this flag exists to
    prevent."""
    assert main(["probe", "--python", "/nonexistent/python"]) == 2

    assert "probe failed" in capsys.readouterr().err


def test_placement_exits_non_zero_when_nothing_can_run_it(capsys):
    """An exit code an operator can branch on."""
    code = main(["placement", "--workload", "HEAVY_TRAINING"])
    out = capsys.readouterr().out

    assert code == 1
    assert "BLOCKED" in out


def test_readiness_prints_the_remote_row_even_with_no_gpu(capsys):
    assert main(["readiness"]) == 0

    assert "REMOTE_CUDA: NOT_CONNECTED" in capsys.readouterr().out
