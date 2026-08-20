"""The refusals: rights, leakage, paths, shells, secrets, and scale.

Everything here is about something that must *not* happen. Each test
names the consequence rather than the mechanism, because the mechanisms
change and the consequences are why the mechanisms exist:

audio nobody licensed reaching a rented machine; benchmark material
entering a training set and quietly destroying the benchmark; a filename
becoming a path outside the run root; a track title becoming shell
syntax on a host that holds the dataset; a private key ending up in a
staging directory or a log.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from remote_fixtures import make_plan
from training_fixtures import curated_record, manifest_record

from luber_training.remote.capabilities import parse_gpu_query, parse_telemetry
from luber_training.remote.client import (
    ClientError,
    SshWorkerClient,
    WorkerEndpoint,
    safe_identifier,
)
from luber_training.remote.identity import (
    LeaseError,
    Liveness,
    LivenessPolicy,
    RunLease,
    WorkerIdentity,
    host_fingerprint,
    load_or_create_identity,
)
from luber_training.remote.manifest import (
    ArtifactEntry,
    ArtifactRole,
    ManifestError,
    RemoteArtifactManifest,
    disk_requirement,
    plan_transfer,
)
from luber_training.remote.paths import RemoteRoots, RunLayout, UnsafePathError, validate_relative
from luber_training.remote.protocol import (
    REMOTE_PROTOCOL_VERSION,
    Envelope,
    ProtocolError,
    check_protocol,
    run_status_for,
)
from luber_training.remote.secrets import (
    EnvironmentSecretResolver,
    FileSecretResolver,
    NullSecretResolver,
    SecretError,
    forget_secrets,
    redact,
    redact_mapping,
)
from luber_training.remote.staging import (
    LeakageViolation,
    RightsViolation,
    StagingInputs,
    build_staging,
    verify_staging,
)
from luber_training.remote.transport import ContentCache, IntegrityError, LocalArtifactTransport

# ── protocol ─────────────────────────────────────────────────────────


def test_an_unknown_protocol_is_refused_rather_than_attempted() -> None:
    assert check_protocol(REMOTE_PROTOCOL_VERSION) == REMOTE_PROTOCOL_VERSION
    with pytest.raises(ProtocolError) as excinfo:
        check_protocol("luber-remote/99", peer="worker")
    assert "Upgrade one side" in str(excinfo.value)


def test_an_unversioned_peer_is_refused() -> None:
    with pytest.raises(ProtocolError):
        check_protocol(None)
    with pytest.raises(ProtocolError):
        Envelope.from_dict({"ok": True, "command": "status"})


def test_an_unknown_worker_state_maps_to_lost_not_to_something_hopeful() -> None:
    assert run_status_for("SOMETHING_NEW") == "LOST"
    # A requested cancellation has not happened yet.
    assert run_status_for("CANCELLING") == "RUNNING"


# ── rights and leakage, before any transfer ──────────────────────────


def _dataset(
    tmp_path: Path,
    *,
    permitted: bool = True,
    split: str = "TRAIN",
    track_id: str = "trk_001",
) -> StagingInputs:
    """A minimal curated build holding one selected track.

    Built from Phase 25's own record helpers rather than a hand-written
    shape. A fixture that only resembled a curated record would let a
    gate pass here and fail on real data, which is the one way these
    tests could be worse than useless.
    """
    dataset_dir = tmp_path / "dataset-build"
    curation_dir = tmp_path / "curation-build"
    audio_root = tmp_path / "library"
    for path in (dataset_dir, curation_dir, audio_root):
        path.mkdir(parents=True, exist_ok=True)

    audio = audio_root / f"{track_id}.wav"
    audio.write_bytes(b"RIFF" + bytes(2048))
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()

    record = manifest_record(
        track_id,
        permission="TRUE" if permitted else "UNKNOWN",
        rights_status="USER_OWNED" if permitted else "UNKNOWN",
        training_eligible=permitted,
        split=split,
        sha256=digest,
    )
    record["source"]["relative_path"] = f"{track_id}.wav"
    curated = curated_record(record, action="KEEP")

    (curation_dir / "curated_manifest.jsonl").write_text(
        json.dumps(curated, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, payload in (
        ("curation_lock.json", {"curation_id": "cur_test"}),
        ("sampling_weights.json", {"weights": {track_id: 1.0}}),
    ):
        (curation_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (dataset_dir / "dataset_lock.json").write_text(
        json.dumps({"dataset_id": "ds_test"}), encoding="utf-8"
    )
    (dataset_dir / "dataset_manifest.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )

    return StagingInputs(
        dataset_build_dir=dataset_dir,
        curation_build_dir=curation_dir,
        audio_root=audio_root,
    )


def test_unauthorised_audio_never_reaches_staging(tmp_path: Path) -> None:
    """The gate runs before the first file is opened for transfer."""
    inputs = _dataset(tmp_path, permitted=False)
    staging_root = tmp_path / "remote_staging"

    with pytest.raises(RightsViolation) as excinfo:
        build_staging(plan=make_plan(), inputs=inputs, staging_root=staging_root)
    assert "not authorised for training" in str(excinfo.value)

    # Nothing was written, so nothing a later transfer could pick up.
    assert not (staging_root / make_plan().run_id).exists()


def test_evaluation_material_never_reaches_staging(tmp_path: Path) -> None:
    inputs = _dataset(tmp_path, track_id="trk_p20_benchmark")
    inputs = StagingInputs(
        dataset_build_dir=inputs.dataset_build_dir,
        curation_build_dir=inputs.curation_build_dir,
        audio_root=inputs.audio_root,
        evaluation_only_ids=frozenset({"trk_p20_benchmark"}),
    )
    staging_root = tmp_path / "remote_staging"

    with pytest.raises(LeakageViolation) as excinfo:
        build_staging(plan=make_plan(), inputs=inputs, staging_root=staging_root)
    assert "destroy the benchmark's meaning" in str(excinfo.value)
    assert not (staging_root / make_plan().run_id).exists()


def test_a_held_out_split_never_reaches_staging(tmp_path: Path) -> None:
    inputs = _dataset(tmp_path, split="TEST")
    with pytest.raises(LeakageViolation):
        build_staging(plan=make_plan(), inputs=inputs, staging_root=tmp_path / "remote_staging")


def test_staging_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    inputs = _dataset(tmp_path)
    plan = make_plan()

    first = build_staging(plan=plan, inputs=inputs, staging_root=tmp_path / "stage-a")
    second = build_staging(plan=plan, inputs=inputs, staging_root=tmp_path / "stage-b")

    # Same inputs, same identity — on different paths, at different times.
    assert first.manifest.digest() == second.manifest.digest()
    assert first.staging_manifest_sha256 == second.staging_manifest_sha256
    assert first.selected_tracks == 1

    assert verify_staging(first.staging_dir, plan=plan) == []

    # Same length, different bytes: the case a size check would miss.
    tampered = first.staging_dir / "dataset" / "trk_001.wav"
    original = tampered.read_bytes()
    tampered.write_bytes(b"XXXX" + original[4:])
    problems = verify_staging(first.staging_dir, plan=plan)
    assert any("digest is" in problem for problem in problems)


def test_staging_refuses_audio_that_changed_since_curation(tmp_path: Path) -> None:
    """The approved dataset is the one that was hashed, not the one named."""
    inputs = _dataset(tmp_path)
    (inputs.audio_root / "trk_001.wav").write_bytes(b"RIFF" + bytes(4096))

    from luber_training.remote.staging import StagingError

    with pytest.raises(StagingError) as excinfo:
        build_staging(plan=make_plan(), inputs=inputs, staging_root=tmp_path / "stage")
    assert "not the dataset that was approved" in str(excinfo.value)


# ── path safety ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "candidate",
    [
        "../etc/passwd",
        "a/../../b",
        "/absolute/path",
        "~/secrets",
        "C:/windows/system32",
        "dataset/..\\..\\escape",
        "dataset/\x00null",
        "",
        "   ",
        "con.txt",
        "trailing ",
        "trailing.",
        "x" * 300,
    ],
)
def test_a_path_that_could_escape_is_refused(candidate: str) -> None:
    with pytest.raises(UnsafePathError):
        validate_relative(candidate)


def test_ordinary_paths_are_accepted() -> None:
    assert validate_relative("dataset/track-001.wav") == "dataset/track-001.wav"
    assert validate_relative("./trainer/dataset.json") == "trainer/dataset.json"
    assert validate_relative("a//b") == "a/b"


def test_a_symlinked_parent_cannot_carry_a_write_outside_the_run(tmp_path: Path) -> None:
    """String validation is not enough once symlinks exist."""
    run_root = tmp_path / "runs" / "run_x"
    run_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_root / "dataset").symlink_to(outside, target_is_directory=True)

    layout = RunLayout(root=run_root)
    with pytest.raises(UnsafePathError) as excinfo:
        layout.resolve("dataset/escaped.wav")
    assert "outside" in str(excinfo.value)


def test_a_manifest_entry_with_a_traversing_path_is_refused() -> None:
    with pytest.raises(UnsafePathError):
        ArtifactEntry(
            artifact_id="a",
            role=ArtifactRole.DATASET_AUDIO.value,
            target_path="../../escape.wav",
            sha256="0" * 64,
            size_bytes=1,
        )


def test_a_transport_refuses_to_write_outside_its_root(tmp_path: Path) -> None:
    transport = LocalArtifactTransport(tmp_path / "remote")
    source = tmp_path / "local.bin"
    source.write_bytes(b"data")
    with pytest.raises(UnsafePathError):
        transport.upload(source, "../escaped.bin", expected_sha256="0" * 64)


# ── shell injection ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hostile",
    [
        'run_x"; rm -rf ~; echo "',
        "run_x; cat /etc/passwd",
        "run_x $(whoami)",
        "run_x `id`",
        "run_x && curl evil.example",
        "run_x\nrm -rf /",
        "run_x|tee /tmp/x",
    ],
)
def test_a_hostile_identifier_never_reaches_a_command_line(hostile: str) -> None:
    with pytest.raises(ClientError) as excinfo:
        safe_identifier(hostile, what="run id")
    assert "reaches a command line" in str(excinfo.value)


def test_remote_commands_are_quoted_even_after_validation() -> None:
    """Belt and braces: validation *and* quoting on the way out."""
    client = SshWorkerClient(
        WorkerEndpoint(worker_root="/opt/luber worker", host="gpu.example", user="ops")
    )
    command = client._remote_command("status", ["--run-id", "run_abc"])
    # The space in the root is quoted rather than splitting the argument.
    assert "'/opt/luber worker'" in command
    assert ";" not in command and "&&" not in command


def test_ssh_never_disables_host_key_verification() -> None:
    client = SshWorkerClient(
        WorkerEndpoint(worker_root="/opt/luber", host="gpu.example", user="ops")
    )
    options = " ".join(client.ssh_options())
    assert "StrictHostKeyChecking=yes" in options
    assert "StrictHostKeyChecking=no" not in options
    assert "accept-new" not in options
    assert "/dev/null" not in options
    # No prompt can be reached, so a misconfiguration fails rather than
    # silently falling back to a password.
    assert "BatchMode=yes" in options
    assert "PasswordAuthentication=no" in options


def test_an_ssh_endpoint_records_references_never_values() -> None:
    endpoint = WorkerEndpoint(
        worker_root="/opt/luber",
        host="gpu.example",
        ssh_key_ref="gpu-prod-key",
        known_hosts_ref="gpu-known-hosts",
    )
    payload = endpoint.to_dict()
    assert payload["ssh_key_ref"] == "gpu-prod-key"
    assert not any("BEGIN" in str(value) for value in payload.values())


# ── secrets ──────────────────────────────────────────────────────────


def test_a_secret_is_resolved_by_name_from_the_environment() -> None:
    forget_secrets()
    resolver = EnvironmentSecretResolver({"LUBER_SECRET_GPU_TOKEN": "s3cret-value-here"})
    assert resolver.available("gpu-token")
    assert resolver.resolve("gpu-token") == "s3cret-value-here"
    with pytest.raises(SecretError):
        resolver.resolve("missing-one")


def test_a_secret_reference_cannot_name_an_arbitrary_variable() -> None:
    resolver = EnvironmentSecretResolver({"PATH": "/usr/bin"})
    assert not resolver.available("PATH")
    with pytest.raises(SecretError):
        resolver.resolve("../../etc/passwd")


def test_a_key_readable_by_others_is_refused(tmp_path: Path) -> None:
    keys = tmp_path / "keys"
    keys.mkdir()
    key = keys / "gpu-prod-key"
    key.write_text("PRIVATE KEY MATERIAL", encoding="utf-8")
    key.chmod(0o644)

    resolver = FileSecretResolver(keys)
    with pytest.raises(SecretError) as excinfo:
        resolver.resolve_path("gpu-prod-key")
    assert "beyond its owner" in str(excinfo.value)

    key.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert resolver.resolve_path("gpu-prod-key") == key


def test_a_secret_directory_inside_the_repository_is_refused(tmp_path: Path) -> None:
    """Secrets in a working tree get committed eventually."""
    repository = tmp_path / "repo"
    (repository / "secrets").mkdir(parents=True)
    with pytest.raises(SecretError) as excinfo:
        FileSecretResolver(repository / "secrets", repository_root=repository)
    assert "get committed eventually" in str(excinfo.value)


def test_a_resolved_secret_is_scrubbed_from_anything_written() -> None:
    forget_secrets()
    resolver = EnvironmentSecretResolver({"LUBER_SECRET_GPU_TOKEN": "abcdef123456"})
    resolver.resolve("gpu-token")

    assert "abcdef123456" not in redact("connecting with abcdef123456 to host")
    payload = redact_mapping(
        {"detail": "token abcdef123456 failed", "api_key": "leaked", "ssh_key_ref": "gpu-prod-key"}
    )
    assert "abcdef123456" not in payload["detail"]
    # Redacted by field name, whatever it holds.
    assert payload["api_key"] == "«redacted»"
    # A reference is a name, and names are meant to be visible.
    assert payload["ssh_key_ref"] == "gpu-prod-key"
    forget_secrets()


def test_the_null_resolver_fails_loudly_rather_than_finding_something() -> None:
    with pytest.raises(SecretError) as excinfo:
        NullSecretResolver().resolve("anything")
    assert "not expected to need a credential" in str(excinfo.value)


# ── nvidia parsing ───────────────────────────────────────────────────


def test_gpu_query_parsing_survives_everything_a_driver_does() -> None:
    output = (
        "0, NVIDIA A100-SXM4-80GB, 81920, 1234, 535.104.05, 8.0, GPU-abc\n"
        "1, NVIDIA A100-SXM4-80GB, 81920, [N/A], 535.104.05, 8.0, GPU-def\n"
        "\n"
    )
    devices = parse_gpu_query(output)
    assert len(devices) == 2
    assert devices[0].memory_total_mb == 81920
    # An unreadable field is unknown, not zero.
    assert devices[1].memory_used_mb is None
    assert devices[1].uuid == "GPU-def"


@pytest.mark.parametrize("output", [None, "", "\n\n", "garbage without commas", "not,a,number"])
def test_unusable_nvidia_output_yields_no_devices(output: str | None) -> None:
    assert parse_gpu_query(output) == []


def test_an_old_driver_missing_columns_still_reports_what_it_knows() -> None:
    devices = parse_gpu_query("0, NVIDIA L4, 24576\n")
    assert len(devices) == 1
    assert devices[0].name == "NVIDIA L4"
    assert devices[0].driver_version is None


def test_telemetry_is_parsed_separately_from_identity() -> None:
    readings = parse_telemetry("0, 87, 40000, 81920, 62, 310.5\n")
    assert readings[0].utilization_pct == 87.0
    assert readings[0].temperature_c == 62.0
    assert readings[0].power_w == 310.5


def test_a_machine_with_no_gpu_reports_nothing_rather_than_zero() -> None:
    from luber_training.remote.capabilities import probe

    report = probe()
    if report.gpus:
        pytest.skip("this machine has NVIDIA hardware")
    assert report.gpu_count is None
    assert report.vram_total_mb is None
    assert report.classify() == "DEVELOPMENT_ONLY"
    assert any("nvidia-smi" in reason for reason in report.unknown)


def test_the_capability_signature_ignores_volatile_readings() -> None:
    from luber_training.remote.capabilities import CapabilityReport, GpuDevice

    first = CapabilityReport(
        architecture="x86_64", gpu_model="NVIDIA A100", gpu_count=1, vram_total_mb=81920
    )
    first.gpus = [GpuDevice(index=0, uuid="GPU-abc", memory_used_mb=1000)]
    second = CapabilityReport(
        architecture="x86_64", gpu_model="NVIDIA A100", gpu_count=1, vram_total_mb=81920
    )
    second.gpus = [GpuDevice(index=0, uuid="GPU-abc", memory_used_mb=70000)]
    second.free_disk_mb = 12
    # Used memory and free disk changed; the machine did not.
    assert first.signature() == second.signature()

    rebuilt = CapabilityReport(architecture="x86_64", gpu_model="NVIDIA L4", gpu_count=1)
    assert rebuilt.signature() != first.signature()


# ── identity, heartbeat, lease ───────────────────────────────────────


def test_a_worker_id_survives_a_restart_and_notices_a_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    first, changed = load_or_create_identity(
        path, worker_name="gpu-01", backend_type="remote-gpu", fingerprint="abc"
    )
    assert not changed

    again, changed = load_or_create_identity(
        path, worker_name="gpu-01", backend_type="remote-gpu", fingerprint="abc"
    )
    assert again.worker_id == first.worker_id
    assert not changed

    # Same name, different machine underneath.
    _, changed = load_or_create_identity(
        path, worker_name="gpu-01", backend_type="remote-gpu", fingerprint="rebuilt"
    )
    assert changed


def test_a_fingerprint_ignores_the_name_a_provider_assigned() -> None:
    @dataclass
    class Report:
        architecture: str = "x86_64"
        os_name: str = "Linux"
        cpu_count: int = 32
        system_ram_mb: int = 128000
        total_disk_mb: int = 500000
        gpu_model: str = "NVIDIA A100"
        gpu_count: int = 1
        gpus: list[Any] = field(default_factory=list)

    first = Report()
    second = Report()
    assert host_fingerprint(first) == host_fingerprint(second)

    second.cpu_count = 8
    assert host_fingerprint(first) != host_fingerprint(second)


def test_liveness_is_patient_before_it_declares_a_worker_gone() -> None:
    from datetime import UTC, datetime, timedelta

    policy = LivenessPolicy()
    reference = datetime.now(UTC)

    def beat(seconds_ago: float) -> str:
        return (reference - timedelta(seconds=seconds_ago)).isoformat()

    assert policy.evaluate(beat(30), reference=reference) == Liveness.ONLINE.value
    # One missed beat is not a dead machine.
    assert policy.evaluate(beat(120), reference=reference) == Liveness.ONLINE.value
    assert policy.evaluate(beat(400), reference=reference) == Liveness.STALE.value
    assert policy.evaluate(beat(1000), reference=reference) == Liveness.OFFLINE.value
    assert policy.evaluate(None) == Liveness.UNKNOWN.value


def test_a_lease_stops_two_workers_taking_one_run() -> None:
    lease = RunLease.create(run_id="run_x", worker_id="wrk_a", training_plan_sha256="p" * 64)
    lease.check_compatible(run_id="run_x", worker_id="wrk_a", plan_sha256="p" * 64)

    with pytest.raises(LeaseError) as excinfo:
        lease.check_compatible(run_id="run_x", worker_id="wrk_b", plan_sha256="p" * 64)
    assert "on two machines" in str(excinfo.value)


def test_a_lease_stops_one_run_id_meaning_two_configurations() -> None:
    lease = RunLease.create(run_id="run_x", worker_id="wrk_a", training_plan_sha256="p" * 64)
    with pytest.raises(LeaseError) as excinfo:
        lease.check_compatible(run_id="run_x", worker_id="wrk_a", plan_sha256="q" * 64)
    assert "two different training configurations" in str(excinfo.value)


# ── cache ────────────────────────────────────────────────────────────


def test_a_cache_entry_is_verified_before_it_is_trusted(tmp_path: Path) -> None:
    """The filename is a claim; the digest is the fact."""
    import hashlib

    cache = ContentCache(tmp_path / "cache")
    payload = b"dataset audio bytes"
    digest = hashlib.sha256(payload).hexdigest()
    source = tmp_path / "audio.wav"
    source.write_bytes(payload)

    cache.store(source, digest)
    assert cache.has(digest)

    # Corrupt the cached copy behind the cache's back.
    cache.path_for(digest).write_bytes(b"tampered")
    assert not cache.has(digest)
    # And the poisoned entry is gone rather than being served again.
    assert not cache.path_for(digest).exists()


def test_storing_content_under_the_wrong_digest_is_refused(tmp_path: Path) -> None:
    cache = ContentCache(tmp_path / "cache")
    source = tmp_path / "audio.wav"
    source.write_bytes(b"bytes")
    with pytest.raises(IntegrityError):
        cache.store(source, "0" * 64)


# ── planning at scale ────────────────────────────────────────────────


def _large_manifest(entries: int, *, size: int = 8_000_000) -> RemoteArtifactManifest:
    """A manifest describing many large artifacts. No bytes allocated.

    Ten thousand entries at eight megabytes each describes an eighty
    gigabyte transfer without writing eighty gigabytes — which is the
    point: the planner has to be exercised at a scale the disk cannot be.
    """
    manifest = RemoteArtifactManifest(run_id="run_scale", training_plan_sha256="p" * 64)
    for index in range(entries):
        manifest.entries.append(
            ArtifactEntry(
                artifact_id=f"a{index}",
                role=ArtifactRole.DATASET_AUDIO.value,
                target_path=f"dataset/{index // 100:04d}/track-{index:06d}.wav",
                sha256=f"{index:064x}",
                size_bytes=size,
                track_id=f"trk_{index:06d}",
            )
        )
    return manifest


def test_ten_thousand_artifacts_plan_in_linear_time() -> None:
    manifest = _large_manifest(10_000)

    started = time.monotonic()
    digest = manifest.digest()
    plan = plan_transfer(manifest)
    elapsed = time.monotonic() - started

    assert len(digest) == 64
    assert plan.total_entries == 10_000
    assert plan.unique_contents == 10_000
    assert plan.upload_bytes == 80_000_000_000
    # Generous, and still far under anything quadratic would take.
    assert elapsed < 10.0


def test_a_cache_turns_a_large_transfer_into_a_small_one() -> None:
    manifest = _large_manifest(10_000)
    already_there = frozenset(f"{index:064x}" for index in range(9_000))

    plan = plan_transfer(manifest, present_digests=already_there)
    assert plan.cached_entries == 9_000
    assert plan.upload_entries == 1_000
    assert plan.cache_hit_ratio == pytest.approx(0.9)


def test_duplicate_content_transfers_once() -> None:
    manifest = RemoteArtifactManifest(run_id="run_dup", training_plan_sha256="p" * 64)
    for index in range(3):
        manifest.entries.append(
            ArtifactEntry(
                artifact_id=f"a{index}",
                role=ArtifactRole.DATASET_AUDIO.value,
                target_path=f"dataset/copy-{index}.wav",
                sha256="a" * 64,
                size_bytes=1_000_000,
            )
        )
    assert manifest.total_bytes == 3_000_000
    assert manifest.transfer_bytes() == 1_000_000


def test_a_disk_requirement_states_what_it_cannot_know() -> None:
    plan = plan_transfer(_large_manifest(100))
    requirement = disk_requirement(plan)
    assert requirement.required_bytes == int(plan.upload_bytes * 1.5)
    assert any("checkpoint size" in reason for reason in requirement.unknown)

    measured = disk_requirement(plan, checkpoint_bytes=2_000_000_000)
    assert measured.unknown == []
    assert measured.required_bytes > requirement.required_bytes


def test_two_files_cannot_claim_one_target_path() -> None:
    manifest = RemoteArtifactManifest(run_id="run_x", training_plan_sha256="p" * 64)
    manifest.add(
        ArtifactEntry(
            artifact_id="a",
            role=ArtifactRole.DATASET_AUDIO.value,
            target_path="dataset/song.wav",
            sha256="a" * 64,
            size_bytes=10,
        )
    )
    with pytest.raises(ManifestError) as excinfo:
        manifest.add(
            ArtifactEntry(
                artifact_id="b",
                role=ArtifactRole.DATASET_AUDIO.value,
                target_path="dataset/song.wav",
                sha256="b" * 64,
                size_bytes=10,
            )
        )
    assert "both claim" in str(excinfo.value)


def test_an_edited_manifest_is_detected_when_it_is_read_back(tmp_path: Path) -> None:
    manifest = _large_manifest(3)
    path = tmp_path / "artifact_manifest.json"
    manifest.write(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        RemoteArtifactManifest.read(path)
    assert "edited or truncated" in str(excinfo.value)


# ── layout ───────────────────────────────────────────────────────────


def test_roots_are_declared_by_the_worker_not_assumed() -> None:
    roots = RemoteRoots.under("/workspace/luber")
    assert roots.run_root == "/workspace/luber/runs"
    assert roots.get("checkpoint_root") == "/workspace/luber/checkpoints"
    with pytest.raises(UnsafePathError):
        roots.get("somewhere_else")


def test_every_run_path_lives_under_the_run_root(tmp_path: Path) -> None:
    layout = RunLayout.for_run(tmp_path / "runs", "run_abc")
    layout.ensure()
    for path in (
        layout.plan_json,
        layout.manifest_json,
        layout.status_json,
        layout.stdout_log,
        layout.metrics_jsonl,
        layout.checkpoints_dir,
        layout.temp_dir,
    ):
        assert layout.root in path.parents or path.parent == layout.root


def test_a_worker_identity_round_trips() -> None:
    identity = WorkerIdentity(
        worker_id="wrk_abc",
        worker_name="gpu-01",
        backend_type="remote-gpu",
        host_fingerprint="f" * 64,
    )
    assert WorkerIdentity.from_dict(identity.to_dict()).worker_id == "wrk_abc"
    assert identity.protocol_version == REMOTE_PROTOCOL_VERSION


def test_umask_does_not_leave_a_worker_root_world_writable(tmp_path: Path) -> None:
    """A sanity check on the environment the worker creates for itself."""
    layout = RunLayout.for_run(tmp_path / "runs", "run_abc")
    layout.ensure()
    mode = layout.root.stat().st_mode
    assert not (mode & stat.S_IWOTH), oct(mode)


def test_no_absolute_operator_paths_leak_into_a_manifest(tmp_path: Path) -> None:
    """A manifest built on one machine must be replayable on another."""
    inputs = _dataset(tmp_path)
    result = build_staging(plan=make_plan(), inputs=inputs, staging_root=tmp_path / "stage")
    canonical = json.dumps(result.manifest.canonical_dict())
    assert str(tmp_path) not in canonical
    assert os.path.expanduser("~") not in canonical
