"""A LoRA resume that restores nothing must not look like a resume.

Phase 38 was recorded as a 336-step continuation. It was not: segment B
began from a fresh adapter and trained 168 steps, and the preserved
adapter is that one. PEFT saves `...lora_A.weight` while the model holds
`...lora_A.<adapter>.weight`, and the trainer loads with `strict=False`,
so every key missed, nothing was restored, and nothing failed. What did
load was epoch, step and optimizer state, which is why the run reported a
successful resume and the counter carried on from 168.

These tests encode that defect so it cannot recur unnoticed.
"""

import pytest


class TestResumeCannotSilentlyStartFresh:
    """The Phase 38 defect, encoded so it cannot recur unnoticed.

    Phase 38 was recorded as a 336-step continuation. Segment B did not
    resume segment A: PEFT saves `...lora_A.weight` while the model holds
    `...lora_A.<adapter>.weight`, and the trainer loads with
    `strict=False`, so every key missed, nothing was restored, and
    nothing failed. Both segments began from identical weights and the
    preserved adapter is 168 steps, not 336.

    These tests stand in fake `acestep`, `torch` and `safetensors`
    modules — none is installed in this package's environment, which is
    deliberate: `luber_training` imports no torch.
    """

    def test_the_adapter_name_component_is_stripped(self):
        from luber_training._experiment_probe import _strip_adapter_name

        assert (
            _strip_adapter_name("base_model.model.q_proj.lora_A.default.weight")
            == "base_model.model.q_proj.lora_A.weight"
        )
        assert (
            _strip_adapter_name("layers.0.cross_attn.k_proj.lora_B.other.weight")
            == "layers.0.cross_attn.k_proj.lora_B.weight"
        )

    def test_a_file_key_and_a_model_key_agree_once_stripped(self):
        from luber_training._experiment_probe import _strip_adapter_name

        # This is the mismatch that caused the defect: textually different,
        # the same tensor.
        file_key = "base_model.model.q_proj.lora_A.weight"
        model_key = "base_model.model.q_proj.lora_A.default.weight"
        assert file_key != model_key
        assert _strip_adapter_name(model_key) == file_key

    def test_a_name_without_an_adapter_component_is_untouched(self):
        from luber_training._experiment_probe import _strip_adapter_name

        for name in ("q_proj.weight", "layers.0.mlp.gate_proj.weight", "lora_A.weight"):
            assert _strip_adapter_name(name) == name

    @pytest.fixture
    def resume(self, monkeypatch):
        """A trainer whose resume restores nothing, exactly as the real one."""
        import sys
        import types

        class T:
            """The tensor operations the verifier uses, and no others."""

            def __init__(self, value, shape=(1,)):
                self.value = float(value)
                self.shape = shape

            def detach(self):
                return self

            def to(self, *a, **k):
                return self

            def clone(self):
                return T(self.value, self.shape)

            def __sub__(self, other):
                return T(self.value - other.value, self.shape)

            def abs(self):
                return T(abs(self.value), self.shape)

            def max(self):
                return self.value

            def __float__(self):
                return self.value

        saved = {
            "base_model.model.q_proj.lora_A.weight": T(0.25),
            "base_model.model.q_proj.lora_B.weight": T(0.5),
        }

        class Decoder:
            def __init__(self, loaded):
                # `loaded` False reproduces the defect: fresh zeros.
                v = 0.5 if loaded else 0.0
                self._p = {
                    "base_model.model.q_proj.lora_A.default.weight": T(0.25 if loaded else 0.0),
                    "base_model.model.q_proj.lora_B.default.weight": T(v),
                }

            def named_parameters(self):
                return list(self._p.items())

            def load_state_dict(self, state, strict=True):
                for name, tensor in state.items():
                    if name in self._p:
                        self._p[name] = tensor

        torch_mod = types.ModuleType("torch")
        torch_mod.float32 = "float32"
        st = types.ModuleType("safetensors")
        st_torch = types.ModuleType("safetensors.torch")
        st_torch.load_file = lambda path: dict(saved)
        st.torch = st_torch
        acestep = types.ModuleType("acestep")
        tv2 = types.ModuleType("acestep.training_v2")
        tf = types.ModuleType("acestep.training_v2.trainer_fixed")

        def original_resume(trainer, resume_path, optimizer, scheduler):
            # The real one loads optimizer/step state and misses every
            # adapter key. It yields updates and returns (epoch, step).
            yield "info"
            return (7, 168)

        tf.resume_checkpoint = original_resume
        tv2.trainer_fixed = tf
        acestep.training_v2 = tv2
        for name, mod in {
            "torch": torch_mod,
            "safetensors": st,
            "safetensors.torch": st_torch,
            "acestep": acestep,
            "acestep.training_v2": tv2,
            "acestep.training_v2.trainer_fixed": tf,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)

        from luber_training import _experiment_probe

        return _experiment_probe, tf, Decoder

    def _run(self, probe, tf, decoder):
        trainer = type(
            "Tr", (), {"module": type("M", (), {"model": type("Mo", (), {"decoder": decoder})()})()}
        )()
        gen = tf.resume_checkpoint(trainer, "/some/dir", None, None)
        try:
            while True:
                next(gen)
        except StopIteration:
            return

    def test_a_failed_resume_is_repaired_and_then_verified(self, resume):
        probe, tf, Decoder = resume
        report = probe.install_resume_verification("/adapter.safetensors")
        decoder = Decoder(loaded=False)  # the real trainer leaves it like this
        self._run(probe, tf, decoder)
        assert report["verified"] is True
        assert report["repaired_tensors"] == 2
        # the weights the trainer failed to restore are now actually present
        assert (
            float(dict(decoder.named_parameters())["base_model.model.q_proj.lora_B.default.weight"])
            == 0.5
        )

    def test_an_already_correct_resume_verifies_without_complaint(self, resume):
        probe, tf, Decoder = resume
        report = probe.install_resume_verification("/adapter.safetensors")
        self._run(probe, tf, Decoder(loaded=True))
        assert report["verified"] is True
        assert report["mismatched"] == 0
        assert report["absent_from_model"] == 0

    def test_weights_the_file_does_not_cover_are_reported_as_absent(self, resume, monkeypatch):
        import sys

        probe, tf, Decoder = resume
        # A file naming a tensor the model does not have at all: the
        # verifier must refuse rather than shrug.
        st_torch = sys.modules["safetensors.torch"]
        original = st_torch.load_file
        st_torch.load_file = lambda path: {**original(path), "layers.99.q_proj.lora_A.weight": None}
        probe.install_resume_verification("/adapter.safetensors")
        with pytest.raises(probe.ResumeNotVerified, match="absent from the model"):
            self._run(probe, tf, Decoder(loaded=True))

    def test_a_decoderless_trainer_refuses_rather_than_assuming(self, resume):
        probe, tf, _ = resume
        probe.install_resume_verification("/adapter.safetensors")
        trainer = type(
            "Tr", (), {"module": type("M", (), {"model": type("Mo", (), {"decoder": None})()})()}
        )()
        gen = tf.resume_checkpoint(trainer, "/some/dir", None, None)
        with pytest.raises(probe.ResumeNotVerified, match="no decoder"):
            while True:
                next(gen)
