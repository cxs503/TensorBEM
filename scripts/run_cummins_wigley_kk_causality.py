"""Driver for the Cummins / Kramers-Kronig Wigley III case (case pool anchor).

Stages (each writes files under ``--outdir`` so a sharded cluster run can be
resumed and every number in the report is traceable to one job):

``point``   solve one ``(level, omega)``; JSON per frequency.
``sweep``   solve the shard ``--shard i --shards n`` of :data:`OMEGA_GRID`.
``gates``   assemble the cached points into sweeps and evaluate G1-G5, including
            the legacy-band A/B control; writes ``results.json``.

Examples::

    python scripts/run_cummins_wigley_kk_causality.py --mode swatch ...
    python scripts/run_cummins_wigley_kk_causality.py \\
        --mode sweep --level medium --shard 0 --shards 6 \\
        --outdir artifacts/case_bem_cummins_wigley
    python scripts/run_cummins_wigley_kk_causality.py \\
        --mode gates --levels coarse,medium \\
        --outdir artifacts/case_bem_cummins_wigley
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


def run_point(args) -> dict:
    outdir = pathlib.Path(args.outdir)
    (outdir / "points").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = case.solve_point(args.level, float(args.omega))
    path = _point_path(outdir, args.level, float(args.omega))
    path.write_text(json.dumps(payload, indent=1))
    print(f"[point] {args.level} omega={args.omega:.3f} wall={payload['wall_seconds']:.1f}s "
          f"residual={payload['relative_residual']:.3e} -> {path} "
          f"({time.perf_counter() - started:.1f}s)", flush=True)
    return payload


def run_sweep(args) -> list[dict]:
    grid = list(args.omega or case.OMEGA_GRID)
    shard, shards = int(args.shard), int(args.shards)
    if not 0 <= shard < shards:
        raise SystemExit("--shard must satisfy 0 <= shard < --shards")
    subset = [grid[i] for i in range(len(grid)) if i % shards == shard]
    print(f"[shard {shard}/{shards}] level={args.level} omegas={subset}", flush=True)
    out = []
    for omega in subset:
        out.append(run_point(argparse.Namespace(
            level=args.level, omega=omega, outdir=args.outdir)))
    return out


def run_gates(args) -> dict:
    outdir = pathlib.Path(args.outdir)
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    reference = case.load_capytaine_reference()
    geometry = case.geometry_gate()
    results = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": geometry,
        "rotation_center": list(case.ROTATION_CENTER),
        "omega_grid": list(case.OMEGA_GRID),
        "legacy_grid": list(case.LEGACY_GRID),
        "gates_definition": case.GATES,
        "reference": {
            "path": str(case.CAPYTAINE_REF_PATH),
            "metadata": reference["metadata"],
        },
        "levels": {},
    }
    for level in levels:
        points = case.load_points(outdir / "points", level)
        if not points:
            raise SystemExit(f"no cached points for level {level!r} in {outdir/'points'}")
        have = sorted(points)
        full = [w for w in case.OMEGA_GRID if round(w, 9) in points]
        missing = [w for w in case.OMEGA_GRID if round(w, 9) not in points]
        print(f"[gates] level={level} cached {len(have)} frequencies, "
              f"grid coverage {len(full)}/{len(case.OMEGA_GRID)}", flush=True)
        if missing:
            print(f"[gates] WARNING missing {missing}", flush=True)
        entry = {
            "signature": case.sweep_signature(level),
            "cached_omega": have,
            "missing_omega": missing,
            "sweep": None,
            "gates": None,
            "legacy_band": None,
        }
        if len(full) >= 8:
            sweep = case.sweep_from_points(points, full)
            entry["sweep"] = {
                "omega": sweep["omega"],
                "added_mass_diag": [
                    list(np.diag(np.asarray(sweep["added_mass"])[i]))
                    for i in range(len(sweep["omega"]))
                ],
                "damping_diag": [
                    list(np.diag(np.asarray(sweep["damping"])[i]))
                    for i in range(len(sweep["omega"]))
                ],
                "relative_residual": sweep["relative_residual"],
                "wall_seconds": sweep["wall_seconds"],
                "rotation_center": sweep["rotation_center"],
            }
            entry["heave_resonance"] = case.heave_resonance_band(sweep)
            analysis = case.gates_from_sweep(sweep, reference=reference)
            entry["gates"] = {
                "verdict": analysis["verdict"],
                "failed": analysis["failed"],
                "detail": analysis["gates"],
                "free_decay": {k: v for k, v in analysis["free_decay"].items()
                               if k != "half_power"},
                "half_power": analysis["free_decay"]["half_power"],
                "cross_solver": analysis["cross_solver"],
                "per_dof": {
                    name: {
                        "added_mass_inf": s["added_mass_inf"],
                        "G2": s["G2"],
                        "G3": s["G3"],
                        "G5": s["G5"],
                        "kernel_diagnostics": s["kernel_diagnostics"],
                        "kernel": s["kernel"],
                    }
                    for name, s in analysis["studies"].items()
                },
            }
            # A/B control: the round-1 band, same solver, same mesh.
            legacy = [w for w in case.LEGACY_GRID if round(w, 9) in points]
            if len(legacy) == len(case.LEGACY_GRID):
                legacy_sweep = case.sweep_from_points(points, legacy)
                try:
                    legacy_analysis = case.gates_from_sweep(
                        legacy_sweep, reference=reference)
                    entry["legacy_band"] = {
                        "verdict": legacy_analysis["verdict"],
                        "failed": legacy_analysis["failed"],
                        "detail": legacy_analysis["gates"],
                        "free_decay": {
                            k: v for k, v in legacy_analysis["free_decay"].items()
                            if k != "half_power"},
                        "half_power": legacy_analysis["free_decay"]["half_power"],
                        "omega": legacy,
                    }
                except Exception as exc:  # honest record of a configuration that cannot run
                    entry["legacy_band"] = {"error": f"{type(exc).__name__}: {exc}",
                                            "omega": legacy}
            else:
                entry["legacy_band"] = {"error": "legacy grid not cached",
                                        "have": legacy}
        results["levels"][level] = entry
        print(f"[gates] level={level} verdict="
              f"{entry['gates']['verdict'] if entry['gates'] else 'n/a'}", flush=True)

    out = outdir / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[gates] -> {out}", flush=True)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("point", "sweep", "gates", "sanity"))
    parser.add_argument("--level", default="medium")
    parser.add_argument("--levels", default="coarse,medium")
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--omega-list", default=None,
                        help="comma separated frequency list overriding OMEGA_GRID")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--outdir", default="artifacts/case_bem_cummins_wigley")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args(argv)
    if args.omega_list:
        args.omega = [float(v) for v in args.omega_list.split(",") if v.strip()]
    if args.mode == "point":
        if args.omega is None:
            parser.error("--mode point needs --omega")
        run_point(args)
    elif args.mode == "sweep":
        run_sweep(args)
    elif args.mode == "gates":
        run_gates(args)
    else:
        print(json.dumps({
            "geometry": case.geometry_gate(),
            "rotation_center": list(case.ROTATION_CENTER),
            "omega_grid": list(case.OMEGA_GRID),
            "signature_medium": case.sweep_signature("medium"),
        }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())