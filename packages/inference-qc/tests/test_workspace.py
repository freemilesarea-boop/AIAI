"""Where candidate audio lives, and why it is not a temp directory.

`tempfile` cleans up when the process exits, which is exactly the case
this has to survive: the worker was killed, the queue retried the job,
and the candidate that was already generated and already paid for should
be reused rather than bought again.
"""

from __future__ import annotations

import qc_fixtures as fx

from luber_inference_qc.workspace import CandidateWorkspace, sha256_file


def _workspace(tmp_path, generation_id="gen-0001") -> CandidateWorkspace:
    return CandidateWorkspace(tmp_path / "candidates", generation_id)


def test_a_candidate_is_copied_in_and_hashed(tmp_path, audio_dir):
    workspace = _workspace(tmp_path)
    source = fx.healthy(audio_dir / "out.wav")

    path, digest = workspace.adopt(source, 0)

    assert path.is_file()
    assert digest == sha256_file(source)
    # Copied, not moved: the provider owns its output path and may have
    # its own cleanup.
    assert source.is_file()


def test_a_surviving_candidate_is_recovered(tmp_path, audio_dir):
    workspace = _workspace(tmp_path)
    _, digest = workspace.adopt(fx.healthy(audio_dir / "out.wav"), 0)

    # A new process, the same directory.
    assert _workspace(tmp_path).recover(0, digest) is not None


def test_a_file_that_changed_is_not_recovered(tmp_path, audio_dir):
    """A half-written file that survived a crash is worse than no file,
    because it looks like one."""
    workspace = _workspace(tmp_path)
    path, digest = workspace.adopt(fx.healthy(audio_dir / "out.wav"), 0)
    path.write_bytes(b"interrupted")

    assert workspace.recover(0, digest) is None


def test_a_candidate_with_no_recorded_digest_is_not_recovered(tmp_path, audio_dir):
    workspace = _workspace(tmp_path)
    workspace.adopt(fx.healthy(audio_dir / "out.wav"), 0)
    assert workspace.recover(0, None) is None


def test_an_empty_file_is_not_recovered(tmp_path):
    workspace = _workspace(tmp_path)
    workspace.ensure()
    workspace.path_for(0).write_bytes(b"")
    assert workspace.recover(0, "a" * 64) is None


def test_a_missing_attempt_is_not_recovered(tmp_path):
    assert _workspace(tmp_path).recover(3, "a" * 64) is None


def test_candidates_are_named_by_attempt_rather_than_by_candidate_id(tmp_path):
    """Resume knows which attempt it wants; it does not know the id a
    previous process minted."""
    workspace = _workspace(tmp_path)
    assert workspace.path_for(0).name == "attempt-00.wav"
    assert workspace.path_for(12).name == "attempt-12.wav"


def test_one_generation_cannot_see_another(tmp_path, audio_dir):
    first = _workspace(tmp_path, "gen-0001")
    second = _workspace(tmp_path, "gen-0002")
    _, digest = first.adopt(fx.healthy(audio_dir / "out.wav"), 0)

    assert second.recover(0, digest) is None
    assert first.root != second.root


def test_cleanup_removes_the_generations_directory_and_nothing_else(tmp_path, audio_dir):
    neighbour = _workspace(tmp_path, "gen-0002")
    neighbour.adopt(fx.healthy(audio_dir / "other.wav"), 0)

    workspace = _workspace(tmp_path)
    workspace.adopt(fx.healthy(audio_dir / "out.wav"), 0)
    workspace.cleanup()

    assert not workspace.root.exists()
    assert neighbour.path_for(0).is_file()


def test_cleanup_can_keep_the_winner(tmp_path, audio_dir):
    """The window between selection and delivery: the winner's bytes are
    still needed and everything else is not."""
    workspace = _workspace(tmp_path)
    workspace.adopt(fx.healthy(audio_dir / "a.wav"), 0)
    workspace.adopt(fx.healthy(audio_dir / "b.wav"), 1)

    workspace.cleanup(keep=1)

    assert not workspace.path_for(0).exists()
    assert workspace.path_for(1).is_file()


def test_cleaning_a_workspace_that_was_never_created_is_not_an_error(tmp_path):
    _workspace(tmp_path).cleanup()


def test_discarding_one_candidate_leaves_the_others(tmp_path, audio_dir):
    workspace = _workspace(tmp_path)
    workspace.adopt(fx.healthy(audio_dir / "a.wav"), 0)
    workspace.adopt(fx.healthy(audio_dir / "b.wav"), 1)

    workspace.discard(0)

    assert not workspace.path_for(0).exists()
    assert workspace.path_for(1).is_file()
