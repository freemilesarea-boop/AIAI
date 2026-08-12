"""Runner, store, metrics, scoring, reference import, and report tests."""

import json
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from bench.listening import build_listening_payload, render_ab_page, render_listening_page
from bench.metrics import (
    CORRUPTED_AUDIO,
    EXCESSIVE_SILENCE,
    SILENT_OUTPUT,
    TOO_SHORT,
    measure_wav,
    real_time_factor,
)
from bench.reference import ReferenceImportError, import_reference, sha256_file
from bench.report import render_report, seed_variance, summarize
from bench.runner import benchmark_id, free_disk_gb
from bench.scoring import (
    RUBRIC_DIMENSIONS,
    VOCAL_ONLY_DIMENSIONS,
    ScoreValidationError,
    make_blind_pair,
    meets_targets,
    validate_artifact_tags,
    validate_scores,
)
from bench.store import GenerationRecord, ResultStore, ScoreRecord, ScoreStore

REPO_ROOT = Path(__file__).resolve().parents[3]


def _wav(path: Path, *, seconds=1.0, rate=48000, channels=2, amplitude=0.5, silent_tail=0.0):
    frames = int(rate * seconds)
    peak = int(32767 * amplitude)
    data = bytearray()
    quiet_from = int(frames * (1 - silent_tail))
    for i in range(frames):
        value = 0 if i >= quiet_from else int(peak * (1 if (i // 50) % 2 else -1))
        data += struct.pack("<h", value) * channels
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(data))
    return path


def _record(**kw) -> GenerationRecord:
    base = dict(
        benchmark_id="P-1__cfg__d30__srand",
        benchmark_version="v1",
        prompt_id="P-1",
        genre="KPOP",
        language="ko",
        vocal_gender="female",
        duration_requested=30,
        lyrics_structure="simple",
        prompt_style="simple",
        prompt="p",
        compiled_prompt=None,
        lyrics="l",
        model="acestep-v15-turbo",
        model_version="1.5.0",
        lm_enabled=False,
        thinking_enabled=False,
        inference_steps=8,
        seed=None,
        configuration_id="cfg",
        runtime_backend="mps-mlx",
    )
    base.update(kw)
    return GenerationRecord(**base)  # type: ignore[arg-type]


# ── benchmark id / manifest ───────────────────────────────────────────


def test_benchmark_id_is_stable_and_distinguishes_seeds():
    a = benchmark_id("KPOP-01", "cfg", 30, None)
    assert a == benchmark_id("KPOP-01", "cfg", 30, None)
    assert a != benchmark_id("KPOP-01", "cfg", 30, 1)
    assert a != benchmark_id("KPOP-01", "cfg", 60, None)
    assert a != benchmark_id("KPOP-02", "cfg", 30, None)
    assert a != benchmark_id("KPOP-01", "other", 30, None)


def test_shipped_pilot_manifest_is_valid_and_meets_composition():
    from bench.dataset import load_dataset

    root = REPO_ROOT / "benchmarks" / "music_quality"
    manifest = json.loads((root / "manifests" / "pilot_baseline.json").read_text())
    dataset = load_dataset(root / "prompts" / "BENCHMARK_V1.json")

    units = manifest["units"]
    assert len(units) >= 24, "Phase 5 requires at least 24 real generations"

    ids = [
        benchmark_id(u["prompt_id"], manifest["configuration_id"], u["duration"], u.get("seed"))
        for u in units
    ]
    assert len(ids) == len(set(ids)), "manifest contains duplicate benchmark ids"

    for unit in units:
        dataset.by_id(unit["prompt_id"])  # raises if unknown

    def group(name):
        return [u for u in units if u.get("group") == name]

    assert len(group("korean_vocal")) >= 8
    assert len(group("english_vocal")) >= 4
    assert len(group("instrumental")) >= 4
    assert len(group("long_form")) >= 4
    assert len(group("seed_variance")) >= 4
    assert all(u["duration"] >= 180 for u in group("long_form"))
    # Long-form must include vocal songs, not only instrumentals.
    assert any(
        dataset.by_id(u["prompt_id"]).vocal_gender != "instrumental" for u in group("long_form")
    )
    # Seed variance needs >= 3 seeds on the same prompt to mean anything.
    seeds_per_prompt: dict[str, set] = {}
    for u in group("seed_variance"):
        seeds_per_prompt.setdefault(u["prompt_id"], set()).add(u["seed"])
    assert any(len(s) >= 3 for s in seeds_per_prompt.values())
    assert all(u.get("seed") is not None for u in group("seed_variance"))


# ── result store / resume ─────────────────────────────────────────────


def test_result_store_roundtrip_and_persistence(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    store.append(_record(status="COMPLETED", seed=7, output_sha256="a" * 64))
    reloaded = ResultStore(tmp_path / "r.jsonl").load()
    assert len(reloaded) == 1
    assert reloaded[0]["seed"] == 7, "seed must persist for reproducibility"
    assert reloaded[0]["output_sha256"] == "a" * 64


def test_resume_skips_completed_but_retries_failures(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    store.append(_record(benchmark_id="done", status="COMPLETED"))
    store.append(_record(benchmark_id="broke", status="FAILED"))
    assert store.completed_ids() == {"done"}
    assert store.all_ids() == {"done", "broke"}


def test_store_tolerates_a_truncated_final_line(tmp_path):
    path = tmp_path / "r.jsonl"
    store = ResultStore(path)
    store.append(_record(benchmark_id="ok", status="COMPLETED"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"benchmark_id": "half')
    assert store.completed_ids() == {"ok"}


def test_korean_survives_the_store_roundtrip(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    store.append(_record(lyrics="[Verse]\n오늘 밤 너를 생각해"))
    assert "오늘 밤" in store.load()[0]["lyrics"]


# ── audio metrics / failure flags ─────────────────────────────────────


def test_metrics_extracted_from_real_wav(tmp_path):
    m = measure_wav(_wav(tmp_path / "a.wav", seconds=2.0), requested_duration=2.0)
    assert m.decoded is True
    assert m.sample_rate == 48000
    assert m.channels == 2
    assert m.bit_depth == 16
    assert m.duration_seconds == pytest.approx(2.0, abs=0.01)
    assert 0.4 < m.peak < 0.6
    assert m.rms > 0
    assert m.flags == []


def test_silent_output_flagged(tmp_path):
    m = measure_wav(_wav(tmp_path / "s.wav", amplitude=0.0), requested_duration=1.0)
    assert SILENT_OUTPUT in m.flags


def test_excessive_silence_flagged(tmp_path):
    m = measure_wav(_wav(tmp_path / "t.wav", seconds=4.0, silent_tail=0.7), requested_duration=4.0)
    assert EXCESSIVE_SILENCE in m.flags


def test_duration_mismatch_flagged(tmp_path):
    m = measure_wav(_wav(tmp_path / "d.wav", seconds=1.0), requested_duration=30.0)
    assert TOO_SHORT in m.flags


def test_corrupted_and_missing_audio_flagged(tmp_path):
    junk = tmp_path / "j.wav"
    junk.write_bytes(b"not audio at all")
    assert CORRUPTED_AUDIO in measure_wav(junk).flags
    assert CORRUPTED_AUDIO in measure_wav(tmp_path / "ghost.wav").flags
    empty = tmp_path / "e.wav"
    empty.write_bytes(b"")
    assert CORRUPTED_AUDIO in measure_wav(empty).flags


def test_metrics_serialize_to_json(tmp_path):
    m = measure_wav(_wav(tmp_path / "a.wav", amplitude=0.0))
    json.dumps(m.to_dict())  # -inf dBFS must not break JSON


def test_real_time_factor():
    assert real_time_factor(60.0, 30.0) == 2.0
    assert real_time_factor(10.0, 0.0) is None


def test_free_disk_reports_a_positive_number():
    assert free_disk_gb(Path.home()) > 0


# ── scoring / rubric ──────────────────────────────────────────────────


def _all_scores(value=8):
    return {d: value for d in RUBRIC_DIMENSIONS}


def test_valid_scores_accepted():
    assert validate_scores(_all_scores()) == _all_scores()


def test_out_of_range_and_non_integer_scores_rejected():
    for bad in (0, 11, -1):
        with pytest.raises(ScoreValidationError, match="outside 1-10"):
            validate_scores({**_all_scores(), "harmony": bad})
    with pytest.raises(ScoreValidationError, match="must be an integer"):
        validate_scores({**_all_scores(), "harmony": 7.5})  # type: ignore[dict-item]
    with pytest.raises(ScoreValidationError, match="must be an integer"):
        validate_scores({**_all_scores(), "harmony": True})


def test_missing_and_unknown_dimensions_rejected():
    partial = _all_scores()
    del partial["harmony"]
    with pytest.raises(ScoreValidationError, match="missing rubric dimensions"):
        validate_scores(partial)
    with pytest.raises(ScoreValidationError, match="unknown rubric dimensions"):
        validate_scores({**_all_scores(), "vibes": 9})


def test_instrumental_omits_vocal_dimensions():
    scores = {d: 8 for d in RUBRIC_DIMENSIONS if d not in VOCAL_ONLY_DIMENSIONS}
    assert validate_scores(scores, instrumental=True) == scores
    with pytest.raises(ScoreValidationError, match="vocal dimensions scored"):
        validate_scores(_all_scores(), instrumental=True)


def test_artifact_tag_validation():
    assert validate_artifact_tags(["VOCAL_ROBOTIC", "MIX_MUDDY"]) == ["VOCAL_ROBOTIC", "MIX_MUDDY"]
    with pytest.raises(ScoreValidationError, match="unknown artifact tags"):
        validate_artifact_tags(["SOUNDS_BAD"])


def test_quality_targets_are_not_trivially_met():
    assert meets_targets({d: 7.9 for d in RUBRIC_DIMENSIONS})["overall_musical_quality"] is False
    assert meets_targets({d: 8.0 for d in RUBRIC_DIMENSIONS})["overall_musical_quality"] is True


# ── blind A/B ─────────────────────────────────────────────────────────


def test_blind_pair_is_deterministic_and_order_independent():
    a = make_blind_pair("x", "y")
    assert a == make_blind_pair("x", "y") == make_blind_pair("y", "x")


def test_blind_pair_randomizes_position_across_items():
    positions = {
        make_blind_pair(f"cfgA-{i}", f"cfgB-{i}").track_a.startswith("cfgA") for i in range(40)
    }
    # Both orderings must occur, otherwise position leaks the system.
    assert positions == {True, False}


def test_blind_pair_reveal_maps_back_to_ids():
    pair = make_blind_pair("left", "right")
    assert pair.reveal("A") == pair.track_a
    assert pair.reveal("B") == pair.track_b
    assert pair.reveal("tie") == "tie"
    with pytest.raises(ScoreValidationError, match="invalid choice"):
        pair.reveal("maybe")


def test_blind_pair_rejects_self_comparison():
    with pytest.raises(ScoreValidationError, match="against itself"):
        make_blind_pair("same", "same")


# ── listening tool ────────────────────────────────────────────────────


def test_listening_payload_hides_configuration_in_blind_mode():
    records = [json.loads(_record(status="COMPLETED", generation_id="g1").to_json())]
    blind = build_listening_payload(records, blind=True)
    assert len(blind) == 1
    assert "revealed" not in blind[0]
    assert "acestep" not in json.dumps(blind)
    revealed = build_listening_payload(records, blind=False)
    assert revealed[0]["revealed"]["model"] == "acestep-v15-turbo"


def test_listening_payload_skips_incomplete_generations():
    records = [json.loads(_record(status="FAILED", generation_id="g1").to_json())]
    assert build_listening_payload(records) == []


def test_listening_payload_drops_vocal_dimensions_for_instrumentals():
    records = [
        json.loads(
            _record(status="COMPLETED", generation_id="g1", vocal_gender="instrumental").to_json()
        )
    ]
    dims = build_listening_payload(records)[0]["dimensions"]
    assert not set(dims) & VOCAL_ONLY_DIMENSIONS


def test_listening_pages_render():
    records = [json.loads(_record(status="COMPLETED", generation_id="g1").to_json())]
    html = render_listening_page(build_listening_payload(records), blind=True)
    assert "BLIND MODE" in html and "<audio" in html
    ab = render_ab_page([{"pair_id": "p", "a_url": "/a", "b_url": "/b"}])
    assert "Track A" in ab and "Track B" in ab


# ── score store ───────────────────────────────────────────────────────


def test_score_store_roundtrip(tmp_path):
    store = ScoreStore(tmp_path / "s.jsonl")
    store.append(
        ScoreRecord(
            benchmark_id="b1",
            evaluator="dev",
            scored_at="2026-08-12T00:00:00Z",
            blind=True,
            scores=_all_scores(7),
            artifact_tags=["MIX_MUDDY"],
            notes="테스트",
        )
    )
    loaded = store.load()
    assert loaded[0]["scores"]["harmony"] == 7
    assert loaded[0]["artifact_tags"] == ["MIX_MUDDY"]
    assert loaded[0]["notes"] == "테스트"


# ── reference import ──────────────────────────────────────────────────


def _reference_meta(path: Path) -> dict:
    return {
        "source": "example-system",
        "version": "1.0",
        "prompt": "a prompt",
        "lyrics": "some lyrics",
        "date": "2026-08-12",
        "sha256": sha256_file(path),
    }


def test_reference_import_requires_full_provenance(tmp_path):
    audio = _wav(tmp_path / "ref.wav")
    meta = _reference_meta(audio)
    for field in ("source", "version", "prompt", "lyrics", "date", "sha256"):
        broken = dict(meta)
        broken[field] = ""
        with pytest.raises(ReferenceImportError, match="provenance metadata"):
            import_reference(audio, broken, reference_id="r1")


def test_reference_import_accepts_valid_local_file(tmp_path):
    audio = _wav(tmp_path / "ref.wav")
    track = import_reference(audio, _reference_meta(audio), reference_id="r1")
    assert track.reference_system == "example-system"
    assert track.sha256 == sha256_file(audio)
    assert track.file_size > 0


def test_reference_import_rejects_hash_mismatch(tmp_path):
    audio = _wav(tmp_path / "ref.wav")
    meta = {**_reference_meta(audio), "sha256": "b" * 64}
    with pytest.raises(ReferenceImportError, match="does not match"):
        import_reference(audio, meta, reference_id="r1")


def test_reference_import_rejects_missing_file_and_bad_date(tmp_path):
    audio = _wav(tmp_path / "ref.wav")
    with pytest.raises(ReferenceImportError, match="not found"):
        import_reference(tmp_path / "ghost.wav", _reference_meta(audio), reference_id="r")
    with pytest.raises(ReferenceImportError, match="ISO format"):
        import_reference(audio, {**_reference_meta(audio), "date": "12/08/2026"}, reference_id="r")


def test_reference_module_provides_no_network_capability():
    """Comparison audio must be user-supplied; no fetching of any kind."""
    source = (
        REPO_ROOT / "benchmarks" / "music_quality" / "scripts" / "bench" / "reference.py"
    ).read_text()
    for forbidden in ("urllib", "requests", "httpx", "selenium", "playwright", "socket"):
        assert forbidden not in source, f"reference importer must not use {forbidden}"


# ── report generation ─────────────────────────────────────────────────


def test_summarize_counts_failures():
    records = [
        json.loads(_record(benchmark_id="a", status="COMPLETED").to_json()),
        json.loads(_record(benchmark_id="b", status="FAILED").to_json()),
    ]
    s = summarize(records)
    assert s.total == 2 and s.completed == 1 and s.failed == 1
    assert s.success_rate == 0.5


def test_seed_variance_reports_best_median_worst():
    records = [
        json.loads(_record(benchmark_id=f"s{i}", prompt_id="P-1", seed=i).to_json())
        for i in range(3)
    ]
    scores = [
        {"benchmark_id": "s0", "scores": {"overall_musical_quality": 4}},
        {"benchmark_id": "s1", "scores": {"overall_musical_quality": 6}},
        {"benchmark_id": "s2", "scores": {"overall_musical_quality": 9}},
    ]
    variance = seed_variance(records, scores)["P-1"]
    assert variance == {"n": 3, "best": 9, "median": 6, "worst": 4, "spread": 5}


def test_report_renders_with_scores(tmp_path):
    records = [
        json.loads(
            _record(
                benchmark_id="a",
                status="COMPLETED",
                generation_seconds=40.0,
                real_time_factor=1.3,
                metrics={"flags": []},
            ).to_json()
        )
    ]
    scores = [{"benchmark_id": "a", "scores": _all_scores(6), "artifact_tags": ["MIX_MUDDY"]}]
    report = render_report(
        records=records,
        scores=scores,
        baseline_id="LUBER_BASELINE_P5_V1",
        benchmark_version="v1",
        ace_step_version="1.5.0",
        ace_step_commit="6d467e4",
        hardware="- test",
    )
    assert "LUBER_BASELINE_P5_V1" in report
    assert "Executive Summary" in report
    assert "Seed Variance" in report
    assert "MIX_MUDDY" in report
    # A 6.0 average must be reported as missing the 8.0 gate.
    assert "MISS" in report


def test_report_renders_without_any_scores():
    records = [json.loads(_record(benchmark_id="a", status="COMPLETED").to_json())]
    report = render_report(
        records=records,
        scores=[],
        baseline_id="B",
        benchmark_version="v1",
        ace_step_version="1.5.0",
        ace_step_commit="c",
        hardware="- test",
    )
    assert "Executive Summary" in report


# ── git hygiene ───────────────────────────────────────────────────────


def test_no_benchmark_audio_is_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files", "benchmarks/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    offenders = [f for f in tracked if f.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    assert offenders == [], f"benchmark audio must not be committed: {offenders}"
