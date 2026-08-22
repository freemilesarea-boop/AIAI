"""Memory profiles for machines that may or may not exist.

Every CUDA figure here is a **fixture**: a literal describing hardware
nobody in this project owns, built so the qualification logic can be
tested without one. Nothing below was measured, and nothing below may be
read as a claim about an NVIDIA card.

The Apple figures are shaped like the ones the real probe returns but
are equally invented; the real measurements live in the profile
artifacts a run produces, never in a test file.
"""

from __future__ import annotations

from typing import Any

from luber_hardware import ComputeDevice
from luber_training.memory import (
    DomainPeak,
    MemoryDomain,
    MemoryProfileIdentity,
    MemorySnapshot,
    PeakKind,
    ProfileOutcome,
    ProfileStage,
    Representativeness,
    RuntimeIdentity,
    TrainingMemoryProfile,
)

GIB = 1024 * 1024 * 1024

ACE_STEP_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"

#: A production-length sequence: 240 s at 25 latent frames a second.
PRODUCTION_LATENT = 6000


def an_identity(**overrides: Any) -> MemoryProfileIdentity:
    base: dict[str, Any] = {
        "device": ComputeDevice.MPS.value,
        "precision": "bf16",
        "optimizer": "adamw",
        "strategy": "LORA",
        "micro_batch_size": 1,
        "gradient_accumulation": 4,
        "gradient_checkpointing": True,
        "lora_rank": 4,
        "lora_alpha": 8,
        "target_modules": ("k_proj", "o_proj", "q_proj", "v_proj"),
        "attention_type": "both",
        "latent_length": PRODUCTION_LATENT,
        "encoder_length": 256,
        "model_variant": "turbo",
        "base_model_upstream_commit": ACE_STEP_COMMIT,
        "ace_step_commit": ACE_STEP_COMMIT,
        "num_devices": 1,
        "offload_encoder": False,
    }
    base.update(overrides)
    return MemoryProfileIdentity(**base)


def a_snapshot(stage: str, **overrides: Any) -> MemorySnapshot:
    base: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": 1.0,
        "host_rss_bytes": 512 * 1024 * 1024,
        "host_available_bytes": 12 * GIB,
        "system_total_bytes": 24 * GIB,
    }
    base.update(overrides)
    return MemorySnapshot(**base)


def a_profile(
    *,
    identity: MemoryProfileIdentity | None = None,
    peak_bytes: int = 6 * GIB,
    domain: str = MemoryDomain.APPLE_UNIFIED.value,
    kind: str = PeakKind.SAMPLED_PEAK.value,
    outcome: str = ProfileOutcome.COMPLETED.value,
    representativeness: str = Representativeness.REPRESENTATIVE.value,
    total_bytes: int = 24 * GIB,
    torch_version: str = "2.10.0",
    profile_id: str = "fixture-profile",
) -> TrainingMemoryProfile:
    """A completed profile whose peak the caller chooses.

    Two domains, because a real one always has two: the host, and
    whichever accelerator the device names. The host peak is deliberately
    small — on Apple silicon the MPS driver allocation does not appear
    in process RSS, which the real measurement confirmed.
    """
    resolved = identity or an_identity()
    return TrainingMemoryProfile(
        profile_id=profile_id,
        plan_digest="d" * 64,
        identity=resolved,
        runtime=RuntimeIdentity(
            python_version="3.12.11",
            torch_version=torch_version,
            ace_step_commit=resolved.ace_step_commit,
            platform_class="Darwin arm64",
            device_class=resolved.device,
        ),
        outcome=outcome,
        representativeness=representativeness,
        representativeness_detail="fixture profile",
        snapshots=[
            a_snapshot(ProfileStage.BASELINE.value),
            a_snapshot(ProfileStage.FORWARD_COMPLETE.value),
        ],
        peaks=[
            DomainPeak(
                domain=MemoryDomain.HOST.value,
                kind=PeakKind.SAMPLED_PEAK.value,
                peak_bytes=1 * GIB,
                baseline_bytes=64 * 1024 * 1024,
                total_bytes=24 * GIB,
                sample_count=20,
            ),
            DomainPeak(
                domain=domain,
                kind=kind,
                peak_bytes=peak_bytes,
                baseline_bytes=0,
                total_bytes=total_bytes,
                sample_count=20,
            ),
        ],
        optimizer_steps=1,
        started_at="2026-08-22T12:00:00+00:00",
        finished_at="2026-08-22T12:05:00+00:00",
        wall_seconds=300.0,
    )


def a_cuda_profile(**overrides: Any) -> TrainingMemoryProfile:
    """A profile for a machine nobody owns. Fixture-only, by construction."""
    identity = an_identity(device=ComputeDevice.CUDA.value)
    return a_profile(
        identity=identity,
        domain=MemoryDomain.CUDA_DEVICE.value,
        kind=PeakKind.RUNTIME_PEAK.value,
        total_bytes=80 * GIB,
        profile_id="fixture-cuda-profile",
        **overrides,
    )


def requested_from(identity: MemoryProfileIdentity, **overrides: Any) -> dict[str, Any]:
    payload = identity.to_dict()
    payload.update(overrides)
    return payload


__all__ = [
    "ACE_STEP_COMMIT",
    "GIB",
    "PRODUCTION_LATENT",
    "a_cuda_profile",
    "a_profile",
    "a_snapshot",
    "an_identity",
    "requested_from",
]
