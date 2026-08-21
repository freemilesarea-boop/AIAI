"""What may be stored, and the promise that ingestion runs exactly once.

The privacy tests here are the last line rather than the only one: the
projection has no column a prompt could occupy and the response models
have no field for one. These check the layer in between — that the
projector, handed a generation carrying real user content, produces a
record that does not.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from obs_fixtures import NOW

from luber_inference_observability import InMemoryObservationStore, verify
from luber_inference_observability.dimensions import (
    GroupingTooWide,
    duration_bucket,
    task_type,
)
from luber_inference_observability.events import FORBIDDEN_FIELDS, DataQuality, validate
from luber_inference_observability.ingest import from_generation, project
from luber_inference_observability.storage import from_mapping, to_mapping

PROMPT = "ZZPROMPTZZ a very private prompt"
LYRICS = "ZZLYRICSZZ 오늘 밤 너를 생각해"
TITLE = "ZZTITLEZZ Midnight Window"
SECRETS = (PROMPT, LYRICS, TITLE, "ZZPROMPTZZ", "ZZLYRICSZZ", "ZZTITLEZZ")


class FakeGeneration:
    """A generation row carrying everything analytics must not keep."""

    def __init__(self, **overrides):
        self.id = overrides.get("id", "11111111-1111-4111-8111-111111111111")
        self.title = TITLE
        self.prompt = PROMPT
        self.lyrics = LYRICS
        self.user_id = "22222222-2222-4222-8222-222222222222"
        self.status = overrides.get("status", "COMPLETED")
        self.provider = "ace_step"
        self.model_name = "acestep"
        self.model_version = "v1"
        self.duration_requested = overrides.get("duration_requested", 200)
        self.language = "ko"
        self.instrumental = False
        self.bpm = 120
        self.key_scale = "A minor"
        self.edit_kind = overrides.get("edit_kind")
        self.reference_audio_id = overrides.get("reference_audio_id")
        self.error_code = overrides.get("error_code")
        self.created_at = overrides.get("created_at", NOW)
        self.started_at = overrides.get("started_at", NOW)
        self.completed_at = overrides.get("completed_at", NOW + timedelta(seconds=90))
        self.inference_qc_trace = overrides.get("inference_qc_trace", TRACE)
        self.finishing_trace = overrides.get(
            "finishing_trace", json.dumps({"outcome": "FINISHED", "finishing_version": "p14-v1"})
        )
        # The field that actually holds the prompt in production.
        self.request_trace = json.dumps({"original_prompt": PROMPT, "original_lyrics": LYRICS})


TRACE = json.dumps(
    {
        "qc_schema_version": "luber-inference-qc/1",
        "qc_engine_version": "qc-v1",
        "retry_policy_version": "retry-v1",
        "request_sha256": "d" * 64,
        "policy": {"name": "STANDARD"},
        "attempts": [
            {
                "candidate_id": "c0",
                "attempt_index": 0,
                "status": "REJECTED",
                "findings": [{"code": "EARLY_COLLAPSE", "severity": "CRITICAL"}],
            },
            {
                "candidate_id": "c1",
                "attempt_index": 1,
                "status": "ELIGIBLE",
                "findings": [{"code": "NARROW_STEREO", "severity": "INFO"}],
            },
        ],
        "selected_candidate_id": "c1",
        "outcome": "SELECTED",
        "exhausted": False,
        "timings": {
            "candidate_phase_seconds": 82.0,
            "provider_seconds": 80.0,
            "qc_seconds": 2.0,
        },
        "budget": {"provider_calls_used": 2, "candidates_generated": 2, "retry_rounds": 1},
    }
)


# ── privacy ──────────────────────────────────────────────────────────


def test_a_projected_observation_carries_no_user_content():
    observation = from_generation(FakeGeneration())
    rendered = observation.to_json()
    for secret in SECRETS:
        assert secret not in rendered


def test_the_observation_has_no_field_a_prompt_could_occupy():
    """A stronger guarantee than stripping: there is nowhere to put it."""
    fields = set(from_generation(FakeGeneration()).to_dict())
    assert not fields & FORBIDDEN_FIELDS


def test_the_request_trace_is_never_read():
    """It looks like a diagnostic blob and holds the prompt in full."""
    generation = FakeGeneration()
    generation.request_trace = json.dumps({"original_prompt": "SHOULD NOT APPEAR"})
    assert "SHOULD NOT APPEAR" not in from_generation(generation).to_json()


def test_the_verifier_would_notice_a_forbidden_field():
    store = InMemoryObservationStore([from_generation(FakeGeneration())])
    report = verify(list(store))
    assert report["ok"] is True
    assert "FORBIDDEN_FIELD_PRESENT" not in report["issues"]


def test_a_stored_row_round_trips_without_gaining_anything():
    original = from_generation(FakeGeneration())
    restored = from_mapping(to_mapping(original))

    assert restored.generation_id == original.generation_id
    assert restored.critical_findings == original.critical_findings
    assert restored.quality_retry_count == original.quality_retry_count
    for secret in SECRETS:
        assert secret not in restored.to_json()


# ── what the projection does record ──────────────────────────────────


def test_the_projection_records_the_candidate_phase():
    observation = from_generation(FakeGeneration())

    assert observation.qc_data_available is True
    assert observation.quality_retry_count == 1
    assert observation.provider_call_count == 2
    assert observation.first_candidate_accepted is False
    assert observation.selected_on_attempt == 1
    assert observation.critical_findings == ("EARLY_COLLAPSE",)
    # The advisory is kept apart from the rejection.
    assert observation.soft_findings == ("NARROW_STEREO",)


def test_delivery_latency_is_the_measured_span_and_named_for_it():
    """Phase 22 measures nothing, so this covers four stages and says so
    in its name rather than pretending to be finishing time."""
    observation = from_generation(FakeGeneration())
    assert observation.total_latency_seconds == 90.0
    assert observation.delivery_latency_seconds == 8.0
    assert not hasattr(observation, "finishing_latency_seconds")


def test_a_generation_without_a_trace_is_recorded_as_lacking_one():
    observation = from_generation(FakeGeneration(inference_qc_trace=None))
    assert observation.qc_data_available is False
    assert observation.quality_retry_count is None
    assert observation.first_candidate_accepted is None


def test_an_unparseable_trace_does_not_stop_ingestion():
    observation = from_generation(FakeGeneration(inference_qc_trace="{not json"))
    assert observation.qc_data_available is False


def test_the_luber_revision_is_only_written_when_it_is_known():
    assert from_generation(FakeGeneration()).luber_revision == "UNKNOWN"
    assert from_generation(FakeGeneration(), luber_revision="abc123").luber_revision == "abc123"


# ── dimensions ───────────────────────────────────────────────────────


def test_task_type_is_derived_from_columns_a_client_cannot_forge():
    assert task_type(edit_kind=None, has_reference=False) == "TEXT_TO_MUSIC"
    assert task_type(edit_kind=None, has_reference=True) == "REFERENCE_CONDITIONED"
    assert task_type(edit_kind="COVER", has_reference=True) == "COVER"
    assert task_type(edit_kind="EXTEND", has_reference=False) == "EXTEND"
    # A kind this build does not know is UNKNOWN, not folded into the
    # oldest bucket where it would silently inflate it.
    assert task_type(edit_kind="SOMETHING_NEW", has_reference=False) == "UNKNOWN"


def test_duration_buckets_are_labelled_the_way_they_read():
    assert duration_bucket(30.0) == "0_30"
    assert duration_bucket(31.0) == "31_60"
    assert duration_bucket(200.0) == "181_240"
    assert duration_bucket(1000.0) == "360_PLUS"
    assert duration_bucket(None) == "UNKNOWN"


def test_a_request_digest_is_not_a_grouping_dimension():
    """Grouping by it produces one bucket per request."""
    from luber_inference_observability.dimensions import validate_grouping

    try:
        validate_grouping(("request_sha256",))
    except GroupingTooWide as exc:
        assert "drilldown" in str(exc)
    else:
        raise AssertionError("request_sha256 must not be groupable")


def test_grouping_is_bounded():
    from luber_inference_observability.dimensions import validate_grouping

    try:
        validate_grouping(("provider", "duration_bucket", "language"))
    except GroupingTooWide as exc:
        assert "at most 2" in str(exc)
    else:
        raise AssertionError("a three-way split must be refused")


# ── ingestion ────────────────────────────────────────────────────────


def test_projecting_the_same_generation_twice_yields_one_row():
    generation = FakeGeneration()
    store = InMemoryObservationStore()
    for _ in range(3):
        observations, _ = project([generation])
        store.upsert(observations)
    assert store.count() == 1


def test_a_later_projection_replaces_an_earlier_one():
    """Phase 29 writes its trace as the run proceeds, so a generation
    observed mid-flight and again at the end must not count twice."""
    store = InMemoryObservationStore()
    midflight = FakeGeneration(status="GENERATING", completed_at=None)
    store.upsert(project([midflight])[0])

    finished = FakeGeneration(status="COMPLETED")
    store.upsert(project([finished])[0])

    assert store.count() == 1
    assert store.get(str(finished.id)).generation_status == "COMPLETED"


def test_one_malformed_row_does_not_stop_a_backfill():
    broken = FakeGeneration()
    broken.created_at = None
    broken.started_at = None

    observations, result = project([FakeGeneration(), broken, FakeGeneration(id="other")])

    assert result.scanned == 3
    assert result.written == 2
    assert result.failed == 1
    assert result.errors
    assert len(observations) == 2


def test_the_watermark_is_the_newest_observation():
    _, result = project([FakeGeneration()])
    assert result.watermark is not None


# ── data quality ─────────────────────────────────────────────────────


def test_impossible_counters_are_flagged_rather_than_crashing():
    observation = from_generation(FakeGeneration())
    observation.provider_call_count = 1
    observation.candidate_count = 5

    issues = validate(observation)

    assert DataQuality.CALLS_BELOW_CANDIDATES in issues


def test_a_negative_latency_is_flagged():
    observation = from_generation(FakeGeneration())
    observation.total_latency_seconds = -5.0
    assert DataQuality.NEGATIVE_LATENCY in validate(observation)


def test_a_completed_generation_with_no_candidates_is_flagged():
    observation = from_generation(FakeGeneration())
    observation.candidate_count = 0
    assert DataQuality.NO_CANDIDATES_ON_COMPLETION in validate(observation)


def test_a_naive_timestamp_is_reported_by_the_verifier():
    observation = from_generation(FakeGeneration())
    observation.occurred_at = datetime(2026, 8, 21, 12, 0)
    report = verify([observation])
    assert "NAIVE_TIMESTAMP" in report["issues"]


def test_a_duplicate_observation_is_reported():
    one = from_generation(FakeGeneration())
    report = verify([one, one])
    assert "DUPLICATE_OBSERVATION" in report["issues"]
