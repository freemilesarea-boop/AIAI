"""Where candidate audio lives while a generation is deciding.

Not a system temp directory, and that is the whole point. `tempfile`
cleans up when the process exits, which is exactly the case resume has
to survive: the worker was killed, the queue retried the job, and the
candidate that was already generated should be reused rather than paid
for twice.

So candidates go in a directory named after the generation, under a root
the deployment configures. It survives a process death, it is removed on
a terminal outcome, and it is scoped to one generation so a stale
directory can never be read as another run's candidate.

Two properties matter more than they look.

**Files are named by attempt index, not by candidate id.** Resume knows
which attempt it is looking for; it does not know the id a previous
process minted. The id lives in the trace, which is durable.

**Reuse verifies the hash.** A file that survived a crash may have
survived it half-written. The recorded digest is what makes reusing it
safe, and a mismatch regenerates rather than trusts.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where generations keep candidates when the deployment names no other
#: place. Beside the run rather than in /tmp, so a crashed worker's
#: candidate is still there when the queue retries.
DEFAULT_WORKSPACE_DIRNAME = "generation-candidates"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidateWorkspace:
    """One generation's candidate files."""

    def __init__(self, root: Path, generation_id: str) -> None:
        self.root = Path(root) / str(generation_id)
        self.generation_id = str(generation_id)

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def path_for(self, attempt_index: int) -> Path:
        return self.root / f"attempt-{attempt_index:02d}.wav"

    def adopt(self, source: Path, attempt_index: int) -> tuple[Path, str]:
        """Copy a provider's output in, and hash it.

        Copied rather than moved: the provider owns its output path and
        may have its own cleanup or its own reasons to keep it. A move
        would reach into another component's temporary state.
        """
        self.ensure()
        destination = self.path_for(attempt_index)
        shutil.copy2(source, destination)
        return destination, sha256_file(destination)

    def recover(self, attempt_index: int, expected_sha256: str | None) -> Path | None:
        """A previous attempt's file, if it is still there and still itself.

        ``None`` when the file is absent, or present and does not match
        the digest recorded for it — a half-written file that survived a
        crash is worse than no file, because it looks like one.
        """
        path = self.path_for(attempt_index)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        if expected_sha256 is None:
            return None
        actual = sha256_file(path)
        if actual != expected_sha256:
            logger.warning(
                "discarding a recovered candidate whose digest changed",
                extra={
                    "generation_id": self.generation_id,
                    "attempt_index": attempt_index,
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual,
                },
            )
            return None
        return path

    def discard(self, attempt_index: int) -> None:
        """Remove one candidate's audio. Metadata is untouched."""
        self.path_for(attempt_index).unlink(missing_ok=True)

    def cleanup(self, *, keep: int | None = None) -> None:
        """Remove the workspace, optionally keeping one attempt.

        ``keep`` exists for the window between selection and delivery:
        the winner's bytes are still needed, and everything else is not.
        Never removes anything outside this generation's directory.
        """
        if not self.root.is_dir():
            return
        if keep is None:
            shutil.rmtree(self.root, ignore_errors=True)
            return
        survivor = self.path_for(keep)
        for path in self.root.glob("attempt-*.wav"):
            if path != survivor:
                path.unlink(missing_ok=True)


__all__ = ["DEFAULT_WORKSPACE_DIRNAME", "CandidateWorkspace", "sha256_file"]
