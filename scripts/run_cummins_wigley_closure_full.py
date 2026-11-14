"""Full extended-band pipeline for CASE-BEM-CUMMINS-WIGLEY (round 5).

Runs the *solved* extended-band panel sweep on the Wigley III hull and
evaluates all five predeclared gates (G1-G5) on the closure grid.

Stages (each writes JSON under ``--outdir`` so a sharded/cluster run can be
resumed and every number is traceable to one job):

1. ``extended``   solve the 28-node :data:`EXTENDED_GRID` [0.10, 20] rad/s on
                  ``--level`` (the round-4 hand-off band; the round-2
                  ``OMEGA_GRID`` [0.30, 12.5] is its strict subset, so the
                  narrow/wide A/B control is the *same* run, no re-solve).
2. ``resonance``  locate the heave resonance ``omega_n`` from the extended
                  sweep (``heave_resonance_band``).
3. ``cluster``    build :func:`closure_grid(omega_n)` (extended band + local
                  cluster at ``RESONANCE_STEP`` = 0.0125 rad/s, 6 samples per
                  half-power width) and solve the nodes not already cached.
4. ``gates``      evaluate G1-G5 on the full closure grid and write
                  ``results.json`` (with the narrow-band A/B control and the
                  supplementary KK statistics).

Nothing here changes the predeclared gates: G1-G5 keep the definitions in
:mod:`tensorbem.benchmarks.cummins_wigley_kk_causality`.  G3b / rms / interior
statistics are *supplementary*: a passing supplement does not turn a failing
gate into a pass.

Usage::

    PYTHONPATH=src python3 scripts/run_cummins_wigley_closure_full.py \\
        --level medium --outdir artifacts/case_bem_cummins_wigley_ext
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kk_causality as case


def _point_path(outdir: pathlib.Path, level: str, omega: float) -> pathlib.Path:
    return outdir / "points" / f"{level}_omega_{omega:07.3f}.json"


def _solve_one(outdir: pathlib.Path, level: str, omega: float) -> dict:
    """Solve one (level, omega) and cache it; returns the payload."""
    path = _point_path(outdir, level, float(omega))
    if path.exists():
        return json.loads(path.read_text())
    started = time.perf_counter()
    payload = case.solve_point(level, float(omega))
    path.write_text(json.dumps(payload, indent=1))
    print(f"[point] {level} omega={omega:.3f} wall={payload['wall_seconds']:.1f}s "
          f"residual={payload['relative_residual']:.3e} -> {path} "
          f"({time.perf_counter() - started:.1f}s)", flush=True)
    return payload


def _load_points(outdir: pathlib.Path, level: str) -> dict[float, dict]:
    return case.load_points(outdir / "points", level)


def _gates_block(sweep: dict, reference: dict, label: str) -> dict:
    """G1-G5 on one band, with the supplementary KK statistics kept."""
    analysis = case.gates_from_sweep(sweep, reference=reference)
    gates = {}
    for key, value in analysis["gates"].items():
        gates[key] = {k: v for k, v in value.items() if k != "per_dof"}
    return {
        "label": label,
        "omega": list(analysis["omega"]),
        "points": len(analysis["omega"]),
        "verdict": analysis["verdict"],
        "failed": analysis["failed"],
        "gates": gates,
        "free_decay": {k: v for k, v in analysis["free_decay"].items()
                       if k != "half_power"},
        "half_power": analysis["free_decay"]["half_power"],
        "per_dof": {
            name: {
                "added_mass_inf": study["added_mass_inf"],
                "G2": study["G2"]["rel_err"],
                "G3": {k: study["G3"][k] for k in
                       ("rel_err", "rms", "interior_max", "argmax_omega")},
                "G3b": study["G3b"],
                "G5": {k: study["G5"][k] for k in ("rel_err", "band")},
            }
            for name, study in analysis["studies"].items()
        },
        "cross_solver": {k: v for k, v in analysis["cross_solver"].items()
                         if k != "rows"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="medium")
    parser.add_argument("--outdir", default="artifacts/case_bem_cummins_wigley_ext")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--skip-extended", action="store_true",
                        help="reuse cached extended points (no re-solve)")
    args = parser.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    (outdir / "points").mkdir(parents=True, exist_ok=True)
    reference = case.load_capytaine_reference()
    geometry = case.geometry_gate()

    # ---- stage 1: extended band (28 nodes) ----
    ext_grid = list(case.EXTENDED_GRID)
    if not args.skip_extended:
        print(f"[stage1] solving {len(ext_grid)} extended-band nodes on {args.level}",
              flush=True)
        for w in ext_grid:
            _solve_one(outdir, args.level, w)
    points = _load_points(outdir, args.level)
    have = sorted(points)
    print(f"[stage1] cached {len(have)} frequencies", flush=True)

    # ---- stage 2: locate heave resonance ----
    ext_sweep = case.sweep_from_points(points, ext_grid)
    res = case.heave_resonance_band(ext_sweep)
    print(f"[stage2] heave resonance: {json.dumps(res, indent=1)}", flush=True)
    if not res.get("inside_band", False):
        print("[stage2] WARNING: resonance not inside extended band; "
              "G4 will be undecidable", flush=True)

    # ---- stage 3: closure grid (extended + cluster) ----
    omega_n = float(res.get("omega_n", float("nan")))
    closure = list(case.closure_grid(omega_n if np.isfinite(omega_n) else None))
    print(f"[stage3] closure grid: {len(closure)} nodes "
          f"[{closure[0]}, {closure[-1]}]", flush=True)
    missing = [w for w in closure if round(float(w), 9) not in points]
    print(f"[stage3] {len(missing)} cluster nodes to solve: {missing}", flush=True)
    for w in missing:
        _solve_one(outdir, args.level, w)
    points = _load_points(outdir, args.level)

    # ---- stage 4: gates on full closure grid ----
    closure_sweep = case.sweep_from_points(points, closure)
    analysis = case.gates_from_sweep(closure_sweep, reference=reference)
    # narrow-band A/B control: the round-2 OMEGA_GRID subset of the same run
    narrow = [w for w in case.OMEGA_GRID if round(float(w), 9) in points]
    narrow_block = None
    if len(narrow) == len(case.OMEGA_GRID):
        narrow_sweep = case.sweep_from_points(points, narrow)
        narrow_block = _gates_block(narrow_sweep, reference, "narrow_round2")

    results = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": geometry,
        "rotation_center": list(case.ROTATION_CENTER),
        "level": args.level,
        "extended_grid": list(case.EXTENDED_GRID),
        "closure_grid": closure,
        "heave_resonance": res,
        "gates_definition": case.GATES,
        "reference": {
            "path": str(case.CAPYTAINE_REF_PATH),
            "metadata": reference["metadata"],
        },
        "closure_band": _gates_block(closure_sweep, reference, "closure"),
        "narrow_round2_control": narrow_block,
        "verdict": analysis["verdict"],
        "failed": analysis["failed"],
        "studies": {
            name: {
                "added_mass_inf": s["added_mass_inf"],
                "G2": s["G2"],
                "G3": s["G3"],
                "G3b": s["G3b"],
                "G5": s["G5"],
                "kernel_diagnostics": s["kernel_diagnostics"],
            }
            for name, s in analysis["studies"].items()
        },
        "free_decay": {k: v for k, v in analysis["free_decay"].items()
                       if k != "half_power"},
        "half_power": analysis["free_decay"]["half_power"],
        "cross_solver": {k: v for k, v in analysis["cross_solver"].items()
                         if k != "rows"},
    }
    out = outdir / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[stage4] -> {out}", flush=True)
    print(f"[stage4] verdict={analysis['verdict']} failed={analysis['failed']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
