"""Remote GPU execution: the bridge from orchestration to hardware.

Phase 25 decides *what* may be trained. This decides *where* and gets it
there. The boundary between those two is the organising idea of the
package, and it is a boundary of authority rather than of code:

**The control plane** — the operator's machine — owns every judgement.
Which dataset is permitted, which experiment is worth running, whether
rights are valid, whether a checkpoint deserves evaluation. Those
questions are answered before anything reaches a rented machine.

**The worker** — the GPU box — owns no judgement at all. It receives
approved artifacts, verifies they are what was sent, runs the exact
command it was given, and reports truthfully. It cannot decide that a
dataset is acceptable, because it is never asked.

That split is why a compromised or misconfigured worker can waste time
but cannot make a decision nobody sanctioned.

Three commitments run through every module:

*Ambiguity is never resolved by launching.* A launch whose reply was
lost may have started a trainer. The response is always to reconcile —
ask the worker what actually happened — and only start something when
the worker positively says there is nothing there.

*Silence is not failure.* A worker that stops answering makes a run
LOST, not FAILED. We know we cannot see the trainer; we do not know it
stopped, and giving up on a rented GPU over a dropped connection is an
expensive way to be wrong.

*Nothing is trusted across the boundary.* Every artifact carries a
digest, every path is validated before it becomes a filesystem write,
every remote argument is quoted, and every checkpoint is re-hashed
locally before the registry hears about it.
"""

from luber_training.remote.backend import (
    DispatchResult,
    ReconcileReport,
    RemoteBackendError,
    RemoteGpuBackend,
    failure_code_for,
)
from luber_training.remote.capabilities import (
    CapabilityReport,
    GpuDevice,
    GpuTelemetry,
    WorkerClassification,
    parse_gpu_query,
    parse_telemetry,
    probe,
    sample_telemetry,
    to_worker_class,
)
from luber_training.remote.client import (
    ClientError,
    ClientRetryPolicy,
    LocalWorkerClient,
    RemoteWorkerClient,
    SshWorkerClient,
    WorkerEndpoint,
    WorkerUnreachable,
    safe_identifier,
)
from luber_training.remote.collect import (
    CollectedCheckpoint,
    CollectionError,
    CollectionReport,
    collect_checkpoint,
    collect_run,
    plan_remote_retention,
    register_collected,
)
from luber_training.remote.execution import (
    ExecutionState,
    ProcessRecord,
    TrainerProcess,
    classify_failure,
)
from luber_training.remote.identity import (
    Heartbeat,
    LeaseError,
    Liveness,
    LivenessPolicy,
    RunLease,
    WorkerIdentity,
    host_fingerprint,
)
from luber_training.remote.manifest import (
    ArtifactEntry,
    ArtifactRole,
    DiskRequirement,
    ManifestError,
    RemoteArtifactManifest,
    TransferPlan,
    disk_requirement,
    plan_transfer,
)
from luber_training.remote.paths import (
    RemoteRoots,
    RunLayout,
    UnsafePathError,
    resolve_within,
    validate_relative,
)
from luber_training.remote.preflight import (
    Check,
    CheckStatus,
    PreflightReport,
    PreflightStatus,
    Severity,
    run_preflight,
)
from luber_training.remote.protocol import (
    REMOTE_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    Envelope,
    ProtocolError,
    ReconcileOutcome,
    RemoteCommand,
    WorkerHealth,
    WorkerState,
    check_protocol,
    run_status_for,
)
from luber_training.remote.result import (
    ArtifactLocation,
    RemoteCheckpoint,
    RemoteCheckpointStatus,
    RemoteResult,
    build_result,
    discover_checkpoints,
)
from luber_training.remote.secrets import (
    EnvironmentSecretResolver,
    FileSecretResolver,
    NullSecretResolver,
    SecretError,
    SecretResolver,
    redact,
    redact_mapping,
)
from luber_training.remote.staging import (
    LeakageViolation,
    RightsViolation,
    StagingError,
    StagingInputs,
    StagingResult,
    build_staging,
    revalidate_before_transfer,
    verify_staging,
)
from luber_training.remote.streams import (
    LogChunk,
    LogCursor,
    MetricStream,
    deduplicate,
    merge_into,
    metric_identity,
    read_log,
)
from luber_training.remote.transport import (
    ArtifactTransport,
    ContentCache,
    IntegrityError,
    LocalArtifactTransport,
    RemoteFile,
    TransferResult,
    TransportError,
    verified_copy,
)
from luber_training.remote.worker import RemoteWorker, WorkerConfig, WorkerError

__all__ = [
    "REMOTE_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "ArtifactEntry",
    "ArtifactLocation",
    "ArtifactRole",
    "ArtifactTransport",
    "CapabilityReport",
    "Check",
    "CheckStatus",
    "ClientError",
    "ClientRetryPolicy",
    "CollectedCheckpoint",
    "CollectionError",
    "CollectionReport",
    "ContentCache",
    "DiskRequirement",
    "DispatchResult",
    "Envelope",
    "EnvironmentSecretResolver",
    "ExecutionState",
    "FileSecretResolver",
    "GpuDevice",
    "GpuTelemetry",
    "Heartbeat",
    "IntegrityError",
    "LeakageViolation",
    "LeaseError",
    "Liveness",
    "LivenessPolicy",
    "LocalArtifactTransport",
    "LocalWorkerClient",
    "LogChunk",
    "LogCursor",
    "ManifestError",
    "MetricStream",
    "NullSecretResolver",
    "PreflightReport",
    "PreflightStatus",
    "ProcessRecord",
    "ProtocolError",
    "ReconcileOutcome",
    "ReconcileReport",
    "RemoteArtifactManifest",
    "RemoteBackendError",
    "RemoteCheckpoint",
    "RemoteCheckpointStatus",
    "RemoteCommand",
    "RemoteFile",
    "RemoteGpuBackend",
    "RemoteResult",
    "RemoteRoots",
    "RemoteWorker",
    "RemoteWorkerClient",
    "RightsViolation",
    "RunLayout",
    "RunLease",
    "SecretError",
    "SecretResolver",
    "Severity",
    "SshWorkerClient",
    "StagingError",
    "StagingInputs",
    "StagingResult",
    "TrainerProcess",
    "TransferPlan",
    "TransferResult",
    "TransportError",
    "UnsafePathError",
    "WorkerClassification",
    "WorkerConfig",
    "WorkerEndpoint",
    "WorkerError",
    "WorkerHealth",
    "WorkerIdentity",
    "WorkerState",
    "WorkerUnreachable",
    "build_result",
    "build_staging",
    "check_protocol",
    "classify_failure",
    "collect_checkpoint",
    "collect_run",
    "deduplicate",
    "discover_checkpoints",
    "disk_requirement",
    "failure_code_for",
    "host_fingerprint",
    "merge_into",
    "metric_identity",
    "parse_gpu_query",
    "parse_telemetry",
    "plan_remote_retention",
    "plan_transfer",
    "probe",
    "read_log",
    "redact",
    "redact_mapping",
    "register_collected",
    "resolve_within",
    "revalidate_before_transfer",
    "run_preflight",
    "run_status_for",
    "safe_identifier",
    "sample_telemetry",
    "to_worker_class",
    "validate_relative",
    "verified_copy",
    "verify_staging",
]
