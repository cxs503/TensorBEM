"""Driver for CASE-BEM-CUMMINS-WIGLEY (6-DOF memory kernel), mission m17209.

Two stages so the (minutes-long) BEM sweeps are computed once and reused:

    PYTHONPATH=src python scripts/m17209_cummins_wigley_kernel_6dof.py --stage sweep
    PYTHONPATH=src python scripts/m17209_cummins_wigley_kernel_6dof.py --stage gates
    PYTHONPATH=src python scripts/m17209_cummins_wigley_kernel_6dof.py --stage all   # both

Artifacts (``artifacts/cummins_wigley_kernel_6dof/``):
  ``sweep_<level>.json`` radiation coefficients per mesh level (BEM output)
  ``results.json``       gate table, kernels, free decay, cross-solver table
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kernel_6dof as case

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__import__("os").environ.get(
    "M17209_OUT", str(ROOT / "artifacts" / "cummins_wigley_kernel_6dof")))
OMEGA = tuple(float(x) for x in __import__("os").environ.get("M17209_OMEGA", "0.5,0.7,1.0,1.5,2.2,3.2,4.2,6.0").split(","))


def sweep_path(level: str) -> pathlib.Path:
    return OUT / f"sweep_{level}.json"


def _one(args):
    """One (level, omega) radiation solve -- module level so it can be pickled."""
    import numpy as np
    from tensorbem.free_surface_solver3d import solve_radiation_3d
    from tensorbem.wigley3_freq import (WIGLEY3_B, WIGLEY3_C, WIGLEY3_L, WIGLEY3_T,
                                        wigley3_wetted_hull_mesh)
    level, om, n_sections, n_levels = args
    mesh = wigley3_wetted_hull_mesh(L=WIGLEY3_L, B=WIGLEY3_B, T=WIGLEY3_T, c=WIGLEY3_C,
                                    n_sections=n_sections, n_levels=n_levels)
    rad = solve_radiation_3d(mesh, om * om / 9.80665, density=1025.0, gravity=9.80665,
                             chief_points="auto")
    return (float(om), np.asarray(rad.added_mass).tolist(), np.asarray(rad.damping).tolist(),
            float(getattr(rad, "relative_residual", float("nan"))))


def do_sweep(levels, omega=OMEGA, workers=1) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for level in levels:
        t0 = time.time()
        if workers > 1:
            import multiprocessing as mp
            n_sections, n_levels = case.MESH_LEVELS[level]
            jobs = [(level, om, n_sections, n_levels) for om in omega]
            with mp.Pool(workers) as pool:
                out = pool.map(_one, jobs)
            out.sort(key=lambda row: row[0])
            sweep = {
                "level": case.level_label(n_sections, n_levels),
                "panels": n_sections * n_levels * 2,
                "omega": [row[0] for row in out],
                "added_mass": [row[1] for row in out],
                "damping": [row[2] for row in out],
                "relative_residual": [row[3] for row in out],
                "density": 1025.0,
                "gravity": 9.80665,
            }
        else:
            sweep = case.radiation_sweep(omega, level=level)
        sweep["wall_seconds"] = round(time.time() - t0, 1)
        sweep_path(level).write_text(json.dumps(sweep, indent=1))
        print(f"[sweep] {level} {sweep['level']} panels={sweep['panels']} "
              f"wall={sweep['wall_seconds']}s -> {sweep_path(level).name}", flush=True)


def cached_sweep_fn(omega, *, level):
    payload = json.loads(sweep_path(level).read_text())
    if [round(float(x), 6) for x in payload["omega"]] != [round(float(x), 6) for x in omega]:
        raise SystemExit(f"{sweep_path(level)} was computed on a different omega grid; "
                         f"rerun --stage sweep")
    return payload


def do_gates(level: str, levels, dt: float, tau_max: float) -> int:
    t0 = time.time()
    results = case.run_case(level=level, levels=levels, omegas=OMEGA, dt=dt,
                            tau_max=tau_max, sweep_fn=cached_sweep_fn, progress=lambda *_: None)
    results["wall_seconds_gates"] = round(time.time() - t0, 1)
    results["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                              "platform": sys.platform}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("verdict:", results["verdict"], "wall_s:", results["wall_seconds_gates"], flush=True)
    for gate in results["gates"].values():
        print(f"  {gate['id']}: value={gate['value']:.4g} tol={gate['tol']:.3g} "
              f"passed={gate['passed']}", flush=True)
    return 0 if results["verdict"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("sweep", "gates", "all"))
    ap.add_argument("--levels", default="coarse,medium")
    ap.add_argument("--level", default="medium")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--tau-max", type=float, default=20.0)
    args = ap.parse_args(argv)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    if args.level not in levels:
        levels = levels + (args.level,)
    if args.stage in ("sweep", "all"):
        do_sweep(levels, workers=args.workers)
    if args.stage in ("gates", "all"):
        return do_gates(args.level, levels, args.dt, args.tau_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())