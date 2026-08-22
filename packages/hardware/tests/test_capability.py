"""The capability record: what it fingerprints, and what it cannot hold.

Two properties. The digest identifies a *configuration* rather than a
machine — two identically set up hosts must agree, because the question
it answers is "would this run the same way here". And there is nowhere
in the record for a host identity to go, which is a stronger guarantee
than remembering to strip one.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from hardware_fixtures import ALL_PRECISION, NO_BF16, apple_machine, cuda_machine

from luber_hardware import (
    PYTORCH_MPS_FALLBACK_ENV,
    ComputeDevice,
    DevicePrecisionSupport,
    MachineCapability,
    capability_from_facts,
    devices,
)
from luber_hardware.capability import DIGEST_FIELDS

# ── the digest ───────────────────────────────────────────────────────


def test_two_identical_machines_share_a_digest():
    """Deliberate. The digest says "would this run the same way here",
    not "which box is this"."""
    assert apple_machine().digest() == apple_machine().digest()


def test_a_different_capability_changes_the_digest():
    assert (
        apple_machine(precision=ALL_PRECISION).digest() != apple_machine(precision=NO_BF16).digest()
    )
    assert apple_machine(memory_mb=24576).digest() != apple_machine(memory_mb=16384).digest()
    assert apple_machine().digest() != cuda_machine().digest()


def test_a_name_does_not_change_the_digest():
    """A label is what an operator called a target. It says nothing
    about what the machine can do, so it must not move the fingerprint."""
    first = apple_machine(label="mac-1")
    second = apple_machine(label="somebody-elses-name")

    assert first.digest() == second.digest()


def test_the_digest_covers_capability_and_not_the_moment():
    """Free disk and utilisation move minute to minute. A digest that
    included them would change constantly and mean nothing."""
    for name in DIGEST_FIELDS:
        assert name in {field.name for field in dataclasses.fields(MachineCapability)}

    for excluded in ("kernel_release", "python_version", "label", "location", "notes"):
        assert excluded not in DIGEST_FIELDS


# ── what cannot be represented ───────────────────────────────────────


def test_the_record_has_no_field_for_a_host_identity():
    """Structural. A report cannot leak a hostname if no field holds one."""
    names = {field.name for field in dataclasses.fields(MachineCapability)}

    for forbidden in (
        "hostname",
        "host",
        "username",
        "user",
        "home",
        "serial",
        "serial_number",
        "mac_address",
        "ip_address",
        "ssh_key",
    ):
        assert forbidden not in names


def test_facts_that_arrive_with_junk_are_normalised_rather_than_trusted():
    """The facts may have come from another interpreter's stdout. A
    probe that crashed on a surprising value would be worse than one
    that reports it unknown."""
    capability = capability_from_facts(
        {
            "cpu_count": "fourteen",
            "memory_total_mb": None,
            "mps_available": "yes",
            "cuda_available": 1,
            "torch_installed": True,
        }
    )

    assert capability.cpu_count is None
    assert capability.mps_available is None, "a string is not a measurement"
    assert capability.cuda_available is None, "1 is not True"


def test_an_interpreter_without_torch_gets_a_note_explaining_the_gap():
    capability = capability_from_facts({"system": "Darwin", "apple_silicon": True})

    assert not capability.torch_installed
    assert capability.devices() == (ComputeDevice.CPU.value,)
    assert any("Probe the interpreter that runs training" in note for note in capability.notes)


def test_the_default_label_is_a_platform_class_not_a_product():
    """ "Apple Silicon (arm64)" is true of this machine and of every
    machine like it. Calling it a Mac mini because one is planned would
    put a guess in an operator's dashboard."""
    capability = capability_from_facts(
        {"system": "Darwin", "architecture": "arm64", "apple_silicon": True}
    )

    assert capability.label == "Apple Silicon (arm64)"
    assert "mini" not in capability.label.lower()


# ── memory semantics ─────────────────────────────────────────────────


def test_unified_memory_is_reported_for_mps_because_that_is_what_it_allocates_from():
    """And it is the same number as the system's, which is exactly why
    it must never be compared with a GPU's VRAM as though they meant the
    same thing."""
    capability = apple_machine(memory_mb=24576)

    assert capability.accelerator_memory_mb(ComputeDevice.MPS.value) == 24576
    assert capability.accelerator_memory_mb(ComputeDevice.CPU.value) == 24576


def test_cuda_memory_is_the_cards_own():
    capability = cuda_machine(memory_mb=81920)

    assert capability.accelerator_memory_mb(ComputeDevice.CUDA.value) == 81920
    assert capability.accelerator_memory_mb(ComputeDevice.CPU.value) == 131072


# ── serialisation ────────────────────────────────────────────────────


def test_a_serialised_capability_carries_its_versions_and_its_digest():
    payload = json.loads(json.dumps(cuda_machine().to_dict()))

    assert payload["capability_schema_version"]
    assert payload["precision_policy_version"]
    assert payload["capability_digest"]
    assert payload["devices"] == ["CUDA", "CPU"]


def test_an_unmeasured_precision_serialises_as_null_not_false():
    capability = MachineCapability(precision_support={"CPU": DevicePrecisionSupport(fp32=True)})
    payload = capability.to_dict()["precision_support"]["CPU"]

    assert payload["fp32"] is True
    assert payload["bf16"] is None, "nobody measured is not the same as does not work"


# ── the fallback environment variable ────────────────────────────────


def test_nothing_in_this_package_sets_the_mps_fallback_variable():
    """`PYTORCH_ENABLE_MPS_FALLBACK=1` makes an unsupported operation run
    on the CPU **silently**, which is the opposite of what
    `MpsFallbackPolicy.STRICT` is for. The constant is named in
    `devices.py` so the repository has one spelling of it and one place
    to check; nothing may assign it.

    Checked against the source rather than the environment: an operator
    is free to export it in their own shell, and this test is about what
    LUBER does, not about what a machine happens to have set.
    """
    source_root = Path(devices.__file__).resolve().parent
    offenders: list[str] = []

    for path in sorted(source_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or PYTORCH_MPS_FALLBACK_ENV not in line:
                if "PYTORCH_ENABLE_MPS_FALLBACK" not in line or stripped.startswith("#"):
                    continue
            if "environ[" in line or "setdefault" in line or "putenv" in line:
                offenders.append(f"{path.name}: {stripped}")

    assert offenders == [], f"the fallback must never be enabled from code: {offenders}"
