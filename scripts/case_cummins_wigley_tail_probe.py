"""Tail probe for CASE-BEM-CUMMINS-WIGLEY: 13 / 15 / 17 rad/s on a resolved mesh.

The committed band stops at 12 rad/s, where gate ``C2`` is already at its limit on
the 480-panel mesh (``k*sqrt(A) = 0.97``).  To measure whether the damping of
surge/sway/yaw reaches algebraic decay (``p > 2.05``), the probe solves three
frequencies **above** that band top on the 2464-panel ``44 x 14`` hull, whose
panel scale ``sqrt(A_max) = 2.90 cm`` keeps ``k*sqrt(A) <= 0.86`` up to 18.4 rad/s
-- i.e. the three probe points are inside the C2 validity window, not beyond it.

    PYTHONPATH=src python3 scripts/case_cummins_wigley_tail_probe.py --shard 0 --shards 3
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case

MESH = (44, 14)
OMEGAS = (13.0, 15.0, 17.0)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley" / "tail_probe"


def point_path(index: int) -> pathlib.Path:
    return OUT / f"probe_{int(index):02d}.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.shard is None:
        parser.error("pass --shard i (with --shards n)")
    total = max(1, int(args.shards))
    OUT.mkdir(parents=True, exist_ok=True)
    for index in range(int(args.shard), len(OMEGAS), total):
        omega = float(OMEGAS[index])
        om, mass, damper, residual, panels, scale = case._solve_one(
            (MESH[0], MESH[1], omega)
        )
        target = point_path(index)
        np.savez(
            target,
            index=np.array(index),
            omega=np.array(float(om)),
            added_mass=np.asarray(mass, dtype=float),
            damping=np.asarray(damper, dtype=float),
            solver_relative_residual=np.array(float(residual)),
            panel_count=np.array(int(panels)),
            max_panel_scale=np.array(float(scale)),
            n_sections=np.array(int(MESH[0])),
            n_levels=np.array(int(MESH[1])),
        )
        print(f"omega={om:.3f} -> {target} (residual {residual:.2e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())