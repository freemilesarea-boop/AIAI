#!/usr/bin/env python3
"""Build a synthetic training registry for exercising the operator console.

Phase 28's console has to be usable before LUBER has ever rented a GPU,
so this writes a registry containing every state it can display: a
running remote job, a lost worker, an out-of-memory failure, a run
blocked on rights, a MOCK checkpoint, and qualification verdicts of all
three kinds.

**None of it is a measurement.** Every hardware figure is a value stated
in this file, every metric is fabricated, and every checkpoint is a few
bytes of placeholder. It exists so the console can be *driven*, not so
anything can be concluded. That is why this is a script an operator runs
deliberately against a directory they name, rather than a demo mode
inside the product: a switch in the running application is one
misconfiguration away from synthetic data appearing beside real records,
and there is no misconfiguration that makes a script somebody did not
run write into a registry.

Point the API at what it writes and the console has something to show::

    uv run python scripts/development/seed_operator_fixture.py --root ./ops-fixture
    OPS_CONSOLE_ENABLED=true OPS_OPERATOR_TOKEN=... \\
      OPS_REGISTRY_ROOT=./ops-fixture/registry \\
      OPS_ARTIFACTS_ROOT=./ops-fixture/runs \\
      OPS_DATASET_BUILDS_ROOT=./ops-fixture/builds/datasets \\
      OPS_CURATION_BUILDS_ROOT=./ops-fixture/builds/curations \\
      uv run uvicorn luber_api.main:app --port 8000

The scale flags exist because a list page that is pleasant with eight
runs can be unusable with a thousand, and a console for a project that
will have a thousand should be measured at that size before it does.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The scenario builder lives with the API tests, so the console is
#: driven by exactly the fixture its tests assert against. A second
#: builder here would drift, and the one that drifted would be the one
#: an operator was looking at.
sys.path.insert(0, str(REPO_ROOT / "apps" / "api" / "tests"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="./ops-fixture",
        help="directory to build the synthetic registry in",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="extra runs, for measuring a list page at scale",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="extra workers, for measuring the fleet page at scale",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete an existing fixture directory first",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if root.exists():
        if not args.force:
            print(
                f"{root} already exists. Pass --force to replace it, or choose another "
                "--root. Refusing to write into a directory that may not be a fixture.",
                file=sys.stderr,
            )
            return 2
        # Only ever a directory this script was pointed at, and only with
        # --force. It could be a real registry, and that is the operator's
        # call to make explicitly rather than a default.
        shutil.rmtree(root)

    # Imported after the path insert above, and invisible to mypy for
    # the same reason: the module lives in a test directory that is not
    # on the type checker's search path. Importing it is deliberate —
    # see the note beside `sys.path.insert`.
    from ops_fixtures import build_scenario  # type: ignore[import-not-found]

    scenario = build_scenario(root, bulk_runs=args.runs, bulk_workers=args.workers)

    print(f"Synthetic operator registry written to {root}")
    print("\nEverything in it is SIMULATED. No GPU was used and nothing was trained.\n")
    print("Point the API at it:")
    print(f"  OPS_REGISTRY_ROOT={root}/registry")
    print(f"  OPS_ARTIFACTS_ROOT={root}/runs")
    print(f"  OPS_DATASET_BUILDS_ROOT={root}/builds/datasets")
    print(f"  OPS_CURATION_BUILDS_ROOT={root}/builds/curations")
    print("\nStates to look at:")
    for label, run_id in sorted(scenario.run_ids.items()):
        print(f"  {label:<16} /ops/training/runs/{run_id}")
    print(f"  mock checkpoint  /ops/training/checkpoints/{scenario.checkpoint_ids['mock']}")
    for label, evaluation_id in sorted(scenario.evaluation_ids.items()):
        print(f"  {label:<16} /ops/training/evaluations/{evaluation_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
