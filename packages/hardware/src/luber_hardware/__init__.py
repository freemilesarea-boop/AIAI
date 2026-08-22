"""Hardware capability, device resolution and execution placement.

Two questions, kept apart from everything that answers them.

**What can this machine do?** A probe that reports measurements and
`UNKNOWN`, never a default — and that can ask an interpreter other than
its own, because the control plane's Python has no torch while the
trainer's does.

**Where should this work run?** A placement decision carrying its
location, its device, its precision, its reasoning and its limits, so
the answer survives being read a month later.

The package imports nothing from LUBER. It never trains, never loads a
model and never opens a network connection. `torch` is consulted when it
happens to be importable and is not required: reporting honestly that
this interpreter cannot see an accelerator is one of the answers.
"""

from luber_hardware.capability import (
    UNKNOWN,
    DevicePrecisionSupport,
    MachineCapability,
    capability_from_facts,
)
from luber_hardware.devices import (
    PYTORCH_MPS_FALLBACK_ENV,
    ComputeDevice,
    ComputePreference,
    ExecutionLocation,
    MpsFallbackPolicy,
    Precision,
    torch_device_string,
)
from luber_hardware.memory import (
    DEFAULT_HEADROOM_FRACTION,
    DEFAULT_RESERVED_FLOOR_MB,
    LOCAL_TRAINING_CONCURRENCY,
    MemoryAssessment,
    MemoryBudget,
    MemoryVerdict,
    assess,
    budget_for,
)
from luber_hardware.placement import (
    DEFAULT_POLICY,
    POLICY_DEVICES,
    ExecutionPlacementDecision,
    ExecutionTarget,
    PlacementOutcome,
    PlacementPolicy,
    PlacementRequest,
    place,
)
from luber_hardware.precision import (
    AUTO_BY_DEVICE,
    FABRIC_PRECISION,
    PrecisionDecision,
    resolve_precision,
    supported_precisions,
)
from luber_hardware.probe import (
    ProbeError,
    collect_facts,
    probe_machine,
    probe_this_process,
)
from luber_hardware.profiles import (
    PLANNED_CUDA_WORKER_LABEL,
    PLANNED_MAC_MINI_24GB_LABEL,
    planned_cuda_worker,
    planned_mac_mini_24gb,
)
from luber_hardware.readiness import (
    ComputeTargetView,
    TargetStatus,
    TrainingExecutionReadiness,
    readiness,
)
from luber_hardware.resolver import (
    AUTO_ORDER,
    CUDA_ONLY_OPERATIONS,
    MPS_UNSUPPORTED_OPERATIONS,
    DeviceDecision,
    DeviceOutcome,
    DeviceRequest,
    DeviceResolver,
)
from luber_hardware.versions import (
    CAPABILITY_SCHEMA_VERSION,
    EXECUTION_PLACEMENT_POLICY_VERSION,
    PRECISION_POLICY_VERSION,
    version_block,
)
from luber_hardware.workloads import (
    DEVICE_BOUND,
    TRAINING_WORKLOADS,
    WorkloadClass,
    is_training,
    uses_device,
)

__all__ = [
    "AUTO_BY_DEVICE",
    "AUTO_ORDER",
    "CAPABILITY_SCHEMA_VERSION",
    "CUDA_ONLY_OPERATIONS",
    "DEFAULT_HEADROOM_FRACTION",
    "DEFAULT_POLICY",
    "DEFAULT_RESERVED_FLOOR_MB",
    "DEVICE_BOUND",
    "EXECUTION_PLACEMENT_POLICY_VERSION",
    "FABRIC_PRECISION",
    "LOCAL_TRAINING_CONCURRENCY",
    "MPS_UNSUPPORTED_OPERATIONS",
    "PLANNED_CUDA_WORKER_LABEL",
    "PLANNED_MAC_MINI_24GB_LABEL",
    "POLICY_DEVICES",
    "PRECISION_POLICY_VERSION",
    "PYTORCH_MPS_FALLBACK_ENV",
    "TRAINING_WORKLOADS",
    "UNKNOWN",
    "ComputeDevice",
    "ComputePreference",
    "ComputeTargetView",
    "DeviceDecision",
    "DeviceOutcome",
    "DevicePrecisionSupport",
    "DeviceRequest",
    "DeviceResolver",
    "ExecutionLocation",
    "ExecutionPlacementDecision",
    "ExecutionTarget",
    "MachineCapability",
    "MemoryAssessment",
    "MemoryBudget",
    "MemoryVerdict",
    "MpsFallbackPolicy",
    "PlacementOutcome",
    "PlacementPolicy",
    "PlacementRequest",
    "Precision",
    "PrecisionDecision",
    "ProbeError",
    "TargetStatus",
    "TrainingExecutionReadiness",
    "WorkloadClass",
    "assess",
    "budget_for",
    "capability_from_facts",
    "collect_facts",
    "is_training",
    "place",
    "planned_cuda_worker",
    "planned_mac_mini_24gb",
    "probe_machine",
    "probe_this_process",
    "readiness",
    "resolve_precision",
    "supported_precisions",
    "torch_device_string",
    "uses_device",
    "version_block",
]
