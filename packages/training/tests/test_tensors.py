"""Reading preprocessed tensors, and refusing the ones that cannot train.

Preprocessing fails quietly: a NaN from an unguarded division, an empty
conditioning tensor, a file that decoded to nothing. None of those raise
when they happen and all of them make a training run meaningless, so the
report has to name them rather than average over them.
"""

from luber_training.tensors import TensorReport, render_markdown, report_from_document


def _sample(name: str, **overrides) -> dict:
    base = {
        "name": name,
        "ok": True,
        "readable": True,
        "latent_length": 6000,
        "latent_channels": 64,
        "encoder_length": 769,
        "missing_fields": [],
        "non_finite_fields": [],
        "bytes": 1024,
    }
    base.update(overrides)
    return base


def _report(samples) -> TensorReport:
    return report_from_document({"dataset_dir": "/tensors", "samples": samples})


class TestItReportsWhatIsThere:
    def test_a_clean_split_is_all_accepted(self):
        report = _report([_sample("a.pt"), _sample("b.pt", latent_length=3000)])
        assert len(report.accepted) == 2
        assert not report.rejected
        assert report.finite_ratio == 1.0
        assert report.max_latent_length == 6000
        assert report.max_encoder_length == 769

    def test_latent_statistics_describe_the_spread(self):
        report = _report([_sample("a.pt", latent_length=n) for n in (3000, 4000, 5000, 6000)])
        stats = report.latent_statistics
        assert stats["minimum"] == 3000
        assert stats["maximum"] == 6000
        assert stats["median"] == 4500

    def test_mixed_channel_widths_are_flagged_as_incompatible(self):
        """Sequence length varies per track. Channel width must not."""
        report = _report([_sample("a.pt"), _sample("b.pt", latent_channels=128)])
        assert not report.shapes_compatible


class TestItRefusesRatherThanAverages:
    def test_an_unreadable_sample_is_rejected_with_its_reason(self):
        report = _report([_sample("a.pt"), _sample("b.pt", ok=False, readable=False, error="boom")])
        assert [s.name for s in report.rejected] == ["b.pt"]
        assert "UNREADABLE" in report.rejected[0].exclusion_reason

    def test_a_missing_field_is_rejected(self):
        report = _report([_sample("a.pt", ok=False, missing_fields=["encoder_hidden_states"])])
        assert "MISSING_FIELDS" in report.rejected[0].exclusion_reason

    def test_a_non_finite_value_is_rejected_and_lowers_the_ratio(self):
        report = _report(
            [_sample("a.pt"), _sample("b.pt", ok=False, non_finite_fields=["target_latents"])]
        )
        assert "NON_FINITE" in report.rejected[0].exclusion_reason
        assert report.finite_ratio == 0.5

    def test_an_empty_sequence_is_rejected(self):
        report = _report([_sample("a.pt", ok=False, latent_length=0)])
        assert "EMPTY_SEQUENCE" in report.rejected[0].exclusion_reason

    def test_a_finite_ratio_over_nothing_is_unmeasured_not_perfect(self):
        """Zero readable samples is not a clean bill of health."""
        report = _report([_sample("a.pt", ok=False, readable=False, error="boom")])
        assert report.finite_ratio is None

    def test_a_probe_that_never_ran_says_so(self):
        report = TensorReport(dataset_dir="/tensors", probe_failed="no interpreter")
        assert report.probe_failed
        assert report.finite_ratio is None
        assert report.max_latent_length is None


class TestTheReport:
    def test_it_names_every_rejected_sample_and_why(self):
        rendered = render_markdown(
            {
                "train": _report(
                    [_sample("a.pt"), _sample("b.pt", ok=False, readable=False, error="boom")]
                )
            }
        )
        assert "b.pt" in rendered
        assert "UNREADABLE" in rendered

    def test_an_unmeasured_ratio_is_not_printed_as_a_number(self):
        rendered = render_markdown(
            {"train": _report([_sample("a.pt", ok=False, readable=False, error="boom")])}
        )
        assert "unmeasured" in rendered
