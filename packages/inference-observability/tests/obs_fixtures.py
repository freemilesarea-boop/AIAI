"""Observations built to make one thing true, and nothing else.

Written as a builder rather than as frozen blobs so a test can say what
it is about — "a week where 95% of first candidates were accepted", "an
hour where collapse is everywhere" — instead of mutating a fixture and
hoping the reader can tell what changed.

Every timestamp is relative to `NOW`, which is fixed. A detector whose
verdict depended on when the suite ran would be a detector nobody could
reproduce, and reproducibility is most of what makes a finding
arguable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from luber_inference_observability import InferenceObservation

#: A fixed origin. Windows in these tests are built backwards from it.
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def observation(
    index: int,
    at: datetime,
    *,
    accepted: bool = True,
    retries: int = 0,
    critical: tuple[str, ...] = (),
    soft: tuple[str, ...] = (),
    status: str = "COMPLETED",
    revision: str = "acestep@v1",
    duration_bucket: str = "61_120",
    task_type: str = "TEXT_TO_MUSIC",
    total_latency: float = 60.0,
    provider_latency: float = 55.0,
    qc_latency: float = 1.0,
    exhausted: bool = False,
    qc_data: bool = True,
    failure_code: str | None = None,
) -> InferenceObservation:
    return InferenceObservation(
        # A real generation id is a UUID and the projection's key column
        # is one, so the fixture produces one too. Derived from the
        # index rather than random, because a store keyed on the id must
        # be reproducible across runs.
        generation_id=str(uuid.UUID(int=index)),
        occurred_at=at,
        provider="ace_step",
        model_name="acestep",
        model_version=revision.split("@")[-1],
        provider_revision=revision,
        task_type=task_type,
        duration_bucket=duration_bucket,
        requested_duration_seconds=90.0,
        language="ko",
        instrumental="NO",
        generation_status=status,
        generation_failure_code=failure_code,
        qc_data_available=qc_data,
        qc_schema_version="luber-inference-qc/1" if qc_data else "UNKNOWN",
        qc_engine_version="qc-v1" if qc_data else "UNKNOWN",
        retry_policy_version="retry-v1" if qc_data else "UNKNOWN",
        qc_policy="STANDARD" if qc_data else "UNKNOWN",
        candidate_count=(retries + 1) if qc_data else None,
        provider_call_count=(retries + 1) if qc_data else None,
        quality_retry_count=retries if qc_data else None,
        selected_on_attempt=(0 if accepted else retries) if qc_data and not exhausted else None,
        first_candidate_accepted=accepted if qc_data else None,
        retry_exhausted=exhausted if qc_data else None,
        candidate_rejections=retries if qc_data else None,
        critical_findings=tuple(sorted(critical)),
        soft_findings=tuple(sorted(soft)),
        provider_latency_seconds=provider_latency,
        qc_latency_seconds=qc_latency,
        total_latency_seconds=total_latency,
    )


def healthy_week(
    count: int = 1000,
    *,
    accept_rate: float = 0.95,
    retry_rate: float = 0.03,
    collapse_rate: float = 0.005,
    start_index: int = 0,
    revision: str = "acestep@v1",
    total_latency: float = 60.0,
    end: datetime | None = None,
) -> list[InferenceObservation]:
    """A week of ordinary traffic, spread evenly and ending before *end*.

    Deterministic rather than random: a baseline that differed between
    runs would make a threshold test a coin flip.
    """
    finish = (end or NOW) - timedelta(hours=2)
    start = finish - timedelta(days=7)
    step = (finish - start) / max(1, count)
    rows: list[InferenceObservation] = []
    for index in range(count):
        accepted = (index % max(1, round(1 / max(1e-9, 1 - accept_rate)))) != 0
        retried = retry_rate > 0 and (index % max(1, round(1 / retry_rate))) == 0
        collapsed = collapse_rate > 0 and (index % max(1, round(1 / collapse_rate))) == 0
        rows.append(
            observation(
                start_index + index,
                start + step * index,
                accepted=accepted and not collapsed,
                retries=1 if retried else 0,
                critical=("EARLY_COLLAPSE",) if collapsed else (),
                revision=revision,
                total_latency=total_latency,
            )
        )
    return rows


def recent_window(
    count: int,
    *,
    minutes: int = 55,
    start_index: int = 100_000,
    end: datetime | None = None,
    **kwargs: object,
) -> list[InferenceObservation]:
    """*count* observations inside the last hour."""
    finish = end or NOW
    start = finish - timedelta(minutes=minutes)
    step = timedelta(minutes=minutes) / max(1, count)
    return [
        observation(start_index + index, start + step * index, **kwargs)  # type: ignore[arg-type]
        for index in range(count)
    ]
