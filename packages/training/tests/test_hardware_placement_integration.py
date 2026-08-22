"""Phase 32 in the training package: a plan that knows its device.

Before this phase LUBER had exactly one device decision, and it was a
ternary in the command compiler:

    "--device", "cuda" if plan.requirements.requires_cuda else "cpu"

Two branches, for a trainer whose own parser has accepted
`auto | cuda | cuda:N | mps | xpu | cpu` since the pinned commit. LUBER
could not express an Apple run because it had no word for one — not
because the trainer could not do it.

These tests hold that seam: the device reaches the command, survives the
round trip to a remote worker, moves the plan digest, and cannot
disagree with `requires_cuda` without somebody being told.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from training_fixtures import gpu_worker, mac_worker  # noqa: F401

from luber_hardware import (
    ComputeDevice,
    ExecutionTarget,
    PlacementRequest,
    WorkloadClass,
    capability_from_facts,
    place,
)
from luber_training.backends import capability_check
from luber_training.config import TrainingConfig, preset
from luber_training.entities import TrainingDatasetRef
from luber_training.ids import EntityKind, new_id
from luber_training.plan import TrainingPlan, default_requirements
from luber_training.trainer_adapter import compile_command

ACE_STEP_ARGS = Path.home() / "ace-step-1.5" / "acestep" / "training_v2" / "cli" / "args.py"


def a_plan(device: str | None = None, config: TrainingConfig | None = None) -> TrainingPlan:
    config = config or preset("LORA_STANDARD")
    return TrainingPlan(
        plan_id=new_id(EntityKind.PLAN),
        run_id=new_id(EntityKind.RUN),
        experiment_id=new_id(EntityKind.EXPERIMENT),
        base_model_id=new_id(EntityKind.MODEL),
        base_model_upstream_commit="6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        dataset_ref=TrainingDatasetRef(
            dataset_id="ds-1",
            dataset_lock_sha256="a" * 64,
            curation_id="cur-1",
            curation_lock_sha256="b" * 64,
            curated_manifest_sha256="c" * 64,
            manifest_artifact_ref="curation://cur-1/curated_manifest",
        ),
        config=config,
        execution_backend="remote-gpu",
        requirements=default_requirements(config, device=device),
    )


def _device_flag(argv: list[str]) -> str:
    return argv[argv.index("--device") + 1]


# ── the device reaches the trainer ───────────────────────────────────


def test_a_plan_placed_on_apple_silicon_compiles_to_device_mps():
    """The flag that could not previously be produced."""
    command = compile_command(a_plan(ComputeDevice.MPS.value), trainer_root="/tmp/trainer")

    assert _device_flag(list(command.argv)) == "mps"


def test_a_plan_with_no_placement_compiles_exactly_as_it_did_before():
    """Phase 32 is additive. A plan that names no device behaves as it
    did when there was no way to name one."""
    command = compile_command(a_plan(None), trainer_root="/tmp/trainer")

    assert _device_flag(list(command.argv)) == "cuda"


@pytest.mark.parametrize(
    ("device", "flag"),
    [
        (ComputeDevice.CUDA.value, "cuda"),
        (ComputeDevice.MPS.value, "mps"),
        (ComputeDevice.CPU.value, "cpu"),
    ],
)
def test_every_device_produces_the_token_the_trainer_expects(device: str, flag: str):
    command = compile_command(a_plan(device), trainer_root="/tmp/trainer")

    assert _device_flag(list(command.argv)) == flag


@pytest.mark.skipif(not ACE_STEP_ARGS.is_file(), reason="the pinned ACE-Step tree is not installed")
def test_the_trainer_really_does_accept_mps():
    """Read from the installed parser, not from memory.

    The whole Apple story rests on this one line of upstream help text.
    If a future pin drops `mps`, this fails here rather than in an
    operator's first local run.
    """
    source = ACE_STEP_ARGS.read_text(encoding="utf-8")

    assert "--device" in source
    assert "mps" in source


# ── requirements cannot disagree with themselves ─────────────────────


def test_placing_on_apple_silicon_clears_the_cuda_requirement():
    """Stated once. Two fields saying the same thing is how they come to
    disagree."""
    requirements = default_requirements(TrainingConfig(), device=ComputeDevice.MPS.value)

    assert requirements.requires_cuda is False
    assert requirements.execution_device == ComputeDevice.MPS.value
    assert requirements.contradictions() == ()


def test_a_contradictory_requirement_is_refused_by_the_preflight(mac_worker):  # noqa: F811
    """`requires_cuda=True` beside `execution_device="MPS"` is two
    statements that cannot both hold. Picking one silently is how a run
    trains somewhere nobody chose."""
    plan = a_plan(ComputeDevice.MPS.value)
    broken = dataclasses.replace(plan.requirements, requires_cuda=True)
    plan = dataclasses.replace(plan, requirements=broken)

    check = capability_check(plan, mac_worker)

    assert not check.ok
    assert any("one of the two is wrong" in problem for problem in check.problems)


def test_an_unknown_device_name_is_refused_rather_than_defaulted(mac_worker):  # noqa: F811
    plan = a_plan(None)
    broken = dataclasses.replace(plan.requirements, execution_device="METAL")
    plan = dataclasses.replace(plan, requirements=broken)

    check = capability_check(plan, mac_worker)

    assert not check.ok
    assert any("is not a compute device" in problem for problem in check.problems)


# ── reproducibility ──────────────────────────────────────────────────


def test_the_device_moves_the_plan_digest():
    """MPS and CUDA do not train identically, so a plan that names one
    is not the same plan as one that names the other. Two runs with the
    same digest must be the same experiment."""
    cuda = a_plan(ComputeDevice.CUDA.value)
    mps = dataclasses.replace(
        cuda, requirements=default_requirements(cuda.config, device=ComputeDevice.MPS.value)
    )

    assert cuda.digest() != mps.digest()


def test_precision_moves_the_plan_digest():
    standard = a_plan(ComputeDevice.CUDA.value)
    other = dataclasses.replace(
        standard, config=dataclasses.replace(standard.config, precision="fp32")
    )

    assert standard.digest() != other.digest()


def test_the_same_placement_hashes_the_same_twice():
    """A digest that changed per compile would defeat the whole point.

    Compared against a rebuild of the *same* plan rather than a second
    fresh one: run and experiment ids are part of a plan's identity and
    are meant to differ. What must not differ is the same plan compiled
    twice.
    """
    first = a_plan(ComputeDevice.MPS.value)
    second = dataclasses.replace(first, plan_id=new_id(EntityKind.PLAN))

    assert first.digest() == second.digest(), "plan_id is excluded from the digest"


def test_no_hardware_measurement_enters_the_plan_digest():
    """Free disk and torch patch versions are properties of a machine at
    a moment, not of the training being requested. Hashing them would
    make every plan unique and the digest worthless."""
    payload = a_plan(ComputeDevice.MPS.value).canonical_dict()
    serialised = json.dumps(payload)

    for leaked in ("free_disk", "memory_total_mb", "capability_digest", "torch_version"):
        assert leaked not in serialised


# ── the round trip a remote worker performs ──────────────────────────


def test_the_device_survives_serialisation_to_a_worker():
    """`remote/worker.py` rebuilds the plan from JSON, field by field.

    A device dropped there would compile to `--device cuda` on a plan
    placed somewhere else, and the worker would train on hardware the
    control plane did not choose.
    """
    plan = a_plan(ComputeDevice.MPS.value)
    payload = json.loads(json.dumps(plan.to_dict()))

    assert payload["requirements"]["execution_device"] == ComputeDevice.MPS.value


# ── placement and the training preflight agree ───────────────────────


def test_placement_and_the_worker_preflight_reach_the_same_verdict(gpu_worker):  # noqa: F811
    """Two layers, one answer.

    Placement decides *where* a workload goes; Phase 27's preflight
    decides whether the chosen worker can hold the plan. A heavy
    training plan placed on CUDA must also pass the preflight, or the
    two are answering the same question differently.
    """
    capability = capability_from_facts(
        {
            "system": "Linux",
            "architecture": "x86_64",
            "torch_installed": True,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_memory_mb": 81920,
            "dtype_support": {"cpu": {"fp32": True}, "cuda": {"fp32": True, "bf16": True}},
        },
        label="gpu",
        location="REMOTE",
    )
    decision = place(
        PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value),
        [ExecutionTarget("gpu", capability, location="REMOTE", worker_id=gpu_worker.worker_id)],
    )
    assert decision.placed
    assert decision.compute_device == ComputeDevice.CUDA.value

    plan = a_plan(decision.compute_device)
    check = capability_check(plan, gpu_worker)

    assert check.ok, check.problems
