"""Solve a range of extended-band nodes in parallel and emit them as JSON.

Why this wrapper exists
-----------------------
``scripts/case_cummins_wigley_extended_band.py`` solves **one** node per process
and writes it to ``extended_points/point_<k>.npz``.  That is the right shape for a
sharded queue, but the cluster hands every job its own checkout, so per-node files
written by different jobs never meet on one filesystem.  Rather than trust that
the scheduler's artifact fetcher reassembles 55 separate files, this wrapper
solves a node range inside a single job with a process pool and prints every node
as one self-describing JSON line on stdout:

    NODE <index> <omega> <json>

The JSON line carries exactly the fields ``assemble()`` needs (coefficients,
solver residual, panel count, panel scale, mesh), so
``scripts/assemble_cummins_extended_json.py`` can rebuild ``sweep_extended.npz``
bit-for-bit from the job log.  ``json`` round-trips ``float`` losslessly
(``repr`` precision), so nothing is truncated by going through text.

Usage::

    PYTHONPATH=src python3 scripts/cummins_extended_sweep_job.py --workers 8
    PYTHONPATH=src python3 scripts/cummins_extended_sweep_job.py --start 0 --stop 20 --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

import pathlib

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "scripts"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

EXTENDED_MESH = (60, 16)
EXTENDED_BAND = (0.2, 20.0)
EXTENDED_N_OMEGA = 55


def grid():
    return case.sweep_grid(EXTENDED_BAND, EXTENDED_N_OMEGA)


def _solve(task):
    """Solve one node; module level so the pool can pickle it."""
    index, omega = task
    n_sections, n_levels = EXTENDED_MESH
    t0 = time.perf_counter()
    om, mass, damper, residual, panels, scale = case._solve_one(
        (n_sections, n_levels, omega)
    )
    record = {
        "index": int(index),
        "omega": float(om),
        "added_mass": [[float(v) for v in row] for row in mass],
        "damping": [[float(v) for v in row] for row in damper],
        "solver_relative_residual": float(residual),
        "panel_count": int(panels),
        "max_panel_scale": float(scale),
        "n_sections": int(n_sections),
        "n_levels": int(n_levels),
        "wall_s": float(time.perf_counter() - t0),
    }
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    nodes = grid()
    stop = nodes.size if args.stop is None else int(args.stop)
    todo = [(k, float(nodes[k])) for k in range(int(args.start), stop)]
    print(
        f"extended sweep {EXTENDED_MESH[0]}x{EXTENDED_MESH[1]}: "
        f"{len(todo)} nodes [{args.start}, {stop}) of {nodes.size}, "
        f"{args.workers} workers",
        flush=True,
    )
    t0 = time.perf_counter()
    with mp.Pool(processes=max(1, int(args.workers))) as pool:
        for record in pool.imap_unordered(_solve, todo):
            print(
                f"NODE {record['index']:02d} {record['omega']:.6f} "
                + json.dumps(record, sort_keys=True),
                flush=True,
            )
    print(
        f"done {len(todo)} nodes in {time.perf_counter() - t0:.1f} s wall",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())