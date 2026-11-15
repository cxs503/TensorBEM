"""Extended-band radiation sweep for CASE-BEM-CUMMINS-WIGLEY (0.2 .. 20 rad/s).

Why this script exists
----------------------
Gate ``C3a`` (Kramers-Kronig closure with ``A_inf`` supplied by the image solve,
zero free parameters) judges only the samples whose *truncation budget* is small
enough for the omitted band above ``s_max`` to be unable to produce the residual
on its own.  On the committed 0.2..12 rad/s medium-mesh sweep that budget is
**unbounded** for surge / sway / yaw: their measured damping slope at the band
top (0.28 .. 1.09) has not entered algebraic decay, so nothing can be certified
for those three DOF -- the residual there is empirical agreement, not evidence.

The only way out of that corner is a wider band, and gate ``C2`` fixes what a
wider band costs: ``k*sqrt(A_panel) <= 1`` at the new top.  At ``omega = 20``
rad/s (``k = 40.8 1/m``) the panel scale must be <= 2.45 cm, which the structured
Wigley mesh reaches at ``60 x 16`` (3840 panels, ``sqrt(A_max) = 2.32 cm``,
admissible up to ``omega = 20.5``).  This script solves exactly that sweep;
nothing about the committed coarse/medium artifacts is touched.

Usage (one shard per cluster job; the shards are independent by construction)::

    PYTHONPATH=src python3 scripts/case_cummins_wigley_extended_band.py --shard 0 --shards 55
    PYTHONPATH=src python3 scripts/case_cummins_wigley_extended_band.py --assemble

Each shard writes ``point_<k>.npz`` for the k-th node of the pre-declared grid,
so a lost shard is re-solved alone and the assembled sweep is bit-reproducible
from the per-node files.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case

#: Panel-resolved mesh for the extended band: ``C2`` holds up to 20.54 rad/s.
EXTENDED_MESH = (60, 16)
#: Pre-declared extended band and node count.  ``N_OMEGA`` keeps the node density
#: of the committed sweep (23 nodes/decade over 0.2..12 -> 55 nodes over
#: 0.2..20 give 18.7/decade), because gate ``C3`` integrates the *piecewise
#: linear* interpolant of ``B(omega)`` and the resonance peaks must stay resolved.
EXTENDED_BAND = (0.2, 20.0)
EXTENDED_N_OMEGA = 55

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
POINTS = OUT / "extended_points"
SWEEP_FILE = OUT / "sweep_extended.npz"


def grid() -> np.ndarray:
    """The pre-declared extended frequency grid (identical in every shard)."""
    return case.sweep_grid(EXTENDED_BAND, EXTENDED_N_OMEGA)


def point_path(index: int) -> pathlib.Path:
    return POINTS / f"point_{int(index):03d}.npz"


def solve_point(index: int) -> pathlib.Path:
    """Solve one grid node on the extended mesh and persist it."""
    nodes = grid()
    if not 0 <= int(index) < nodes.size:
        raise SystemExit(f"index {index} outside the {nodes.size}-node grid")
    n_sections, n_levels = EXTENDED_MESH
    omega = float(nodes[int(index)])
    om, mass, damper, residual, panels, scale = case._solve_one(
        (n_sections, n_levels, omega)
    )
    POINTS.mkdir(parents=True, exist_ok=True)
    target = point_path(index)
    np.savez(
        target,
        index=np.array(int(index)),
        omega=np.array(float(om)),
        added_mass=np.asarray(mass, dtype=float),
        damping=np.asarray(damper, dtype=float),
        solver_relative_residual=np.array(float(residual)),
        panel_count=np.array(int(panels)),
        max_panel_scale=np.array(float(scale)),
        n_sections=np.array(int(n_sections)),
        n_levels=np.array(int(n_levels)),
    )
    return target


def assemble() -> pathlib.Path:
    """Collect the per-node files into one ``Sweep`` artifact (all nodes required)."""
    nodes = grid()
    missing = [k for k in range(nodes.size) if not point_path(k).exists()]
    if missing:
        raise SystemExit(
            f"cannot assemble: {len(missing)}/{nodes.size} nodes missing "
            f"(first: {missing[:5]})"
        )
    omega, mass, damper, residual = [], [], [], []
    panels = set()
    scales = []
    for k in range(nodes.size):
        payload = np.load(point_path(k), allow_pickle=False)
        omega.append(float(payload["omega"]))
        mass.append(np.asarray(payload["added_mass"], dtype=float))
        damper.append(np.asarray(payload["damping"], dtype=float))
        residual.append(float(payload["solver_relative_residual"]))
        panels.add(int(payload["panel_count"]))
        scales.append(float(payload["max_panel_scale"]))
    if len(panels) != 1:
        raise SystemExit(f"shards disagree on the panel count: {sorted(panels)}")
    sweep = case.Sweep(
        label="extended",
        n_sections=int(EXTENDED_MESH[0]),
        n_levels=int(EXTENDED_MESH[1]),
        panel_count=int(panels.pop()),
        max_panel_scale=float(max(scales)),
        omega=np.array(omega, dtype=float),
        added_mass=np.array(mass, dtype=float),
        damping=np.array(damper, dtype=float),
        solver_relative_residual=np.array(residual, dtype=float),
        elapsed=0.0,
    )
    case.save_sweep(sweep, SWEEP_FILE)
    return SWEEP_FILE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    nodes = grid()
    if args.list:
        print(
            f"extended mesh {EXTENDED_MESH[0]}x{EXTENDED_MESH[1]}, "
            f"{nodes.size} nodes over {EXTENDED_BAND} rad/s"
        )
        for k, value in enumerate(nodes):
            print(f"{k:3d} {value:9.5f}")
        return 0
    if args.assemble:
        print(assemble())
        return 0
    if args.shard is None:
        parser.error("pass --shard i (with --shards n), --assemble or --list")
    total = max(1, int(args.shards))
    start = int(args.shard)
    if not 0 <= start < total:
        parser.error(f"--shard must be in [0, {total})")
    for index in range(start, nodes.size, total):
        print(f"solving node {index} omega = {nodes[index]:.5f}", flush=True)
        path = solve_point(index)
        print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())