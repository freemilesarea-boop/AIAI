/**
 * What the circuit view returns.
 *
 * Mirrors `apps/api/src/luber_api/ops/resilience_schemas.py`. Note what
 * is absent: there is no field here for a prompt, a title, or a set of
 * lyrics, because there is none there either. A circuit is about a
 * provider name and a count.
 *
 * `failureRate` is `number | null` for the same reason the API makes it
 * optional: a circuit nobody has exercised has no rate, and rendering
 * that as 0% would read as "nothing is failing" when the truth is
 * "nothing has been measured".
 */

export interface VersionBlock {
  resilience_schema_version: string;
  circuit_policy_version: string;
  routing_policy_version: string;
  failover_policy_version: string;
}

export interface Circuit {
  circuit_key: string;
  provider: string;
  task_type: string;
  /** CLOSED | OPEN | HALF_OPEN */
  state: string;
  /** AUTOMATIC | MANUAL */
  control: string;
  consecutive_failures: number;
  consecutive_successes: number;
  sample_count: number;
  failure_count: number;
  failure_rate: number | null;
  opened_at: string | null;
  open_until: string | null;
  open_reason: string | null;
  consecutive_opens: number;
  active_probes: number;
  probe_successes: number;
  last_failure_at: string | null;
  last_failure_category: string | null;
  last_success_at: string | null;
  last_transition_at: string | null;
  manual_reason: string | null;
  manual_operator: string | null;
  revision: number;
}

export interface CircuitList extends VersionBlock {
  at: string;
  circuits: Circuit[];
  unconfigured_providers: string[];
}

export interface Transition {
  id: string;
  circuit_key: string;
  provider: string;
  task_type: string;
  previous_state: string;
  current_state: string;
  occurred_at: string;
  reason: string;
  automatic: boolean;
  operator: string | null;
  circuit_policy_version: string;
}

export interface TransitionList extends VersionBlock {
  transitions: Transition[];
}

export interface ProviderReadiness {
  provider: string;
  revision: string;
  circuit_state: string;
  control: string;
  open_reason: string | null;
  open_until: string | null;
}

export interface CapabilityReadiness {
  capability: string;
  /** AVAILABLE | DEGRADED | UNAVAILABLE | NOT_CONFIGURED */
  status: string;
  detail: string;
  providers: ProviderReadiness[];
}

export interface Readiness extends VersionBlock {
  at: string;
  generation_available: boolean;
  degraded: boolean;
  summary: string;
  capabilities: CapabilityReadiness[];
  metrics: Record<string, number>;
}

export interface Policy extends VersionBlock {
  resilience_enabled: boolean;
  failover_mode: string;
  failover_possible: boolean;
  routable_providers: string[];
  circuit_policy: Record<string, number | string>;
}
