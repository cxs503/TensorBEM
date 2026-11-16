"""Solve the ``A_inf`` anchor rungs of the extended-band mesh and emit them as JSON.

Gate ``C3a_extended`` is only free of free parameters when the high-frequency added
mass comes from an independent boundary-value solve on the *same* mesh as the sweep
(image method, ``phi = 0`` on the calm waterplane).  The extended sweep is solved on
``60 x 16`` (3840 panels), so the anchor ladder needs that rung; without it the gate
silently falls back to a band-fitted offset, i.e. a parameter fitted to the data it
judges.

``scripts/case_cummins_wigley_extended_anchor.py`` solves the same rungs but writes
them to ``anchor_extended_points/anchor_<i>.npz``; the cluster gives each job its own
checkout, so this wrapper prints the rung as one JSON line instead and
``--collect`` rebuilds ``anchor_ladder_extended.json`` in the committed schema.

Usage::

    PYTHONPATH=src python3 scripts/cummins_extended_anchor_job.py --workers 2
    PYTHONPATH=src python3 scripts/cummins_extended_anchor_job.py --collect job.log
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

RUNGS = ((44, 14), (60, 16))
LADDER = (
    WORKTREE
    / "benchmarks"
    / "verified"
    / "casepool_bem_cummins_wigley"
    / "anchor_ladder_extended.json"
)


def _solve(rung):
    n_sections, n_levels = rung
    payload = case.added_mass_inf_anchor(int(n_sections), int(n_levels))
    return payload


def solve_rungs(workers: int) -> int:
    with mp.Pool(processes=max(1, workers)) as pool:
        results = pool.map(_solve, list(RUNGS))
    for rung, payload in zip(RUNGS, results):
        print(
            f"ANCHOR {rung[0]}x{rung[1]} " + json.dumps(payload, sort_keys=True),
            flush=True,
        )
    return 0


def collect(paths) -> pathlib.Path:
    rungs = []
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            if not line.startswith("ANCHOR "):
                continue
            payload = json.loads(line.split(" ", 2)[2])
            rungs.append(payload)
    if not rungs:
        raise SystemExit("no ANCHOR lines found in the given logs")
    rungs.sort(key=lambda rung: (int(rung["n_sections"]), int(rung["n_levels"])))
    meshes = {(int(r["n_sections"]), int(r["n_levels"])) for r in rungs}
    if len(meshes) != len(rungs):
        raise SystemExit(f"duplicate rungs in the logs: {sorted(meshes)}")
    needed = {rung for rung in RUNGS}
    if not needed.issubset(meshes):
        raise SystemExit(
            f"the ladder is missing rungs {sorted(needed - meshes)}; the extended "
            "sweep needs its own mesh rung"
        )
    convergence = {}
    for dof in case.DOF_NAMES:
        values = [rung["added_mass_inf"][dof] for rung in rungs]
        step = (
            abs(values[-1] - values[-2]) / abs(values[-1]) if len(values) > 1 else None
        )
        convergence[dof] = {"values": values, "top_rung_relative_change": step}
    LADDER.write_text(
        json.dumps(
            {
                "mission": "m23070",
                "purpose": "A_inf anchor ladder on the extended-band meshes",
                "rungs": rungs,
                "convergence": convergence,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return LADDER


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--collect", nargs="+", default=None)
    args = parser.parse_args(argv)
    if args.collect:
        print(collect(args.collect))
        return 0
    return solve_rungs(args.workers)


if __name__ == "__main__":
    sys.exit(main())