"""Shape diversity in the capacity model — the Phase 36 OOM, as a regression.

Phase 36 qualified this machine from a profile measured at one tensor
shape, then trained a dataset with 24 different latent lengths. Metal
keeps an allocator working set per shape: the run reached 29 GiB where
the profile said 9.4, and died at the same step four times. Sequence
*length* matched perfectly the whole way through.

So the capacity model now carries how many distinct shapes a dataset
holds, and a one-shape profile no longer qualifies a many-shape run.
"""

import pytest
from memory_fixtures import a_profile

from luber_training.capacity_policy import (
    EXACT_FIELDS,
    MEMORY_RELEVANT_FIELDS,
    Applicability,
    applicability,
)


def _requested(profile, **overrides) -> dict:
    payload = profile.identity.to_dict()
    payload.update(overrides)
    return payload


class TestTheFieldExists:
    def test_shape_count_is_memory_relevant(self):
        assert "latent_shape_count" in MEMORY_RELEVANT_FIELDS
        why = MEMORY_RELEVANT_FIELDS["latent_shape_count"]
        assert "shape" in why

    def test_it_is_compared_exactly_not_permissively(self):
        """There is no ordering to exploit: more shapes is simply different."""
        assert "latent_shape_count" in EXACT_FIELDS

    def test_a_profile_records_one_shape_by_default(self):
        """A profiler generates a single shape, so a measurement is 1."""
        assert a_profile().identity.latent_shape_count == 1


class TestTheRegression:
    def test_a_one_shape_profile_does_not_qualify_a_many_shape_dataset(self):
        """Phase 36's exact mistake, refused."""
        profile = a_profile()
        verdict, detail = applicability(profile, _requested(profile, latent_shape_count=24))
        assert verdict == Applicability.CONFIGURATION_MISMATCH.value
        assert "latent_shape_count" in detail
        assert "shape" in detail

    def test_matching_sequence_length_is_not_enough_on_its_own(self):
        """The field that looked identical throughout the Phase 36 failure."""
        profile = a_profile()
        requested = _requested(profile, latent_shape_count=24)
        assert requested["latent_length"] == profile.identity.latent_length
        verdict, _ = applicability(profile, requested)
        assert verdict != Applicability.APPLICABLE.value

    @pytest.mark.parametrize("count", [2, 5, 24, 128])
    def test_any_count_above_one_is_refused_by_a_one_shape_profile(self, count):
        profile = a_profile()
        verdict, _ = applicability(profile, _requested(profile, latent_shape_count=count))
        assert verdict == Applicability.CONFIGURATION_MISMATCH.value

    def test_a_fixed_shape_dataset_still_qualifies(self):
        """Phase 37's representation: one length, so the profile applies."""
        profile = a_profile()
        verdict, _ = applicability(profile, _requested(profile, latent_shape_count=1))
        assert verdict == Applicability.APPLICABLE.value

    def test_a_caller_that_says_nothing_is_not_silently_qualified_as_many(self):
        """Absent means unstated, and unstated falls back to the profile."""
        profile = a_profile()
        requested = _requested(profile)
        requested.pop("latent_shape_count", None)
        verdict, _ = applicability(profile, requested)
        assert verdict == Applicability.APPLICABLE.value
