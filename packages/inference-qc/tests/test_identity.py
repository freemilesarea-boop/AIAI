"""What makes two requests the same, and what makes two attempts differ.

Both halves have a way of going quietly wrong. A digest that included a
timestamp would make every trace incomparable while looking correct; a
seed derivation that used `random` would make a run unreproducible while
looking identical from the outside. Neither failure announces itself, so
both are pinned here.
"""

from __future__ import annotations

from luber_inference_qc import derive_seed, request_digest
from luber_inference_qc.identity import DIGEST_FIELDS, SEED_MODULUS

REQUEST = {
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘 밤",
    "duration_seconds": 30,
    "language": "ko",
    "instrumental": False,
}


# ── the digest ───────────────────────────────────────────────────────


def test_the_same_request_always_hashes_the_same():
    assert request_digest(REQUEST) == request_digest(dict(REQUEST))


def test_key_order_does_not_change_the_digest():
    reversed_order = dict(reversed(list(REQUEST.items())))
    assert request_digest(reversed_order) == request_digest(REQUEST)


def test_changing_something_the_model_hears_changes_the_digest():
    for field in ("prompt", "lyrics", "duration_seconds"):
        altered = {**REQUEST, field: "different" if isinstance(REQUEST[field], str) else 60}
        assert request_digest(altered) != request_digest(REQUEST), field


def test_a_field_that_is_not_part_of_identity_is_ignored():
    """A title and a submission time do not change what was generated."""
    noisy = {**REQUEST, "title": "Midnight Window", "submitted_at": "2026-08-21T10:00:00Z"}
    assert request_digest(noisy) == request_digest(REQUEST)


def test_the_seed_is_not_part_of_the_request_identity():
    """Two attempts differing only in seed are attempts at one request.

    This is the property the whole trace rests on: if the seed were in
    the digest, a retry would look like a different request and nothing
    could be said about the pair.
    """
    assert "seed" not in DIGEST_FIELDS
    assert request_digest({**REQUEST, "seed": 42}) == request_digest(REQUEST)


def test_an_absent_field_is_absent_rather_than_defaulted():
    """Otherwise two genuinely different requests would hash the same."""
    without = {key: value for key, value in REQUEST.items() if key != "bpm"}
    assert request_digest({**without, "bpm": 120}) != request_digest(without)


def test_extra_material_the_request_object_does_not_hold_is_included():
    """A reference track is a path locally and a digest everywhere else."""
    plain = request_digest(REQUEST)
    with_reference = request_digest(REQUEST, extra={"reference_sha256": "abc123"})
    assert with_reference != plain


def test_a_request_object_hashes_the_same_as_the_equivalent_dict():
    class Request:
        prompt = REQUEST["prompt"]
        lyrics = REQUEST["lyrics"]
        duration_seconds = REQUEST["duration_seconds"]
        language = REQUEST["language"]
        instrumental = REQUEST["instrumental"]
        title = "ignored"

    assert request_digest(Request()) == request_digest(REQUEST)


# ── the seed ─────────────────────────────────────────────────────────


def test_the_first_attempt_uses_the_seed_that_was_asked_for():
    assert derive_seed(1234, 0, "digest") == 1234


def test_a_later_attempt_is_derived_and_reproducible():
    first = derive_seed(1234, 1, "digest")
    assert first != 1234
    assert derive_seed(1234, 1, "digest") == first
    assert 0 <= first < SEED_MODULUS


def test_attempts_do_not_collide_with_each_other():
    seeds = {derive_seed(1234, index, "digest") for index in range(1, 8)}
    assert len(seeds) == 7


def test_the_same_base_seed_on_a_different_request_diverges():
    """Otherwise every request sharing a seed would retry identically."""
    assert derive_seed(1234, 1, "digest-a") != derive_seed(1234, 1, "digest-b")


def test_no_seed_stays_no_seed():
    """Inventing one would make every retry of a seedless request identical.

    A provider left to its own randomisation produces a different song
    each time, which is exactly what a retry is for. Handing it a derived
    seed would take that away in the name of reproducibility nobody asked
    for.
    """
    assert derive_seed(None, 0, "digest") is None
    assert derive_seed(None, 3, "digest") is None


def test_a_derived_seed_is_not_adjacent_to_the_one_it_replaces():
    """Adjacent seeds are not guaranteed to produce unrelated samples."""
    assert abs(derive_seed(1234, 1, "digest") - 1234) > 1
