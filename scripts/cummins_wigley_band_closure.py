"""Evaluate the Cummins / Kramers-Kronig case on the *closure* band (round 4).

Round 2 failed G3/G4/G5 and explained the residual in prose ("band/scale
effect").  Round 3 measured that explanation away (`tensorbem.benchmarks.
kk_band_floor`) and found that the max-over-midpoints statistic of G3 migrates
to the band edge when the same datum is re-expressed on a finer grid -- but it
could not supply an extended-band number, because its sweep job never ran
(report ``docs/dev_report_m17773_cummins_band_budget.md`` §5: `band_attributable`
= null, jobs cj1658/cj1659 stuck in the queue).

This driver consumes the *solved* extended sweep and answers three questions
with numbers instead of prose:

``band``      what changes when the same estimator sees [0.10, 20] rad/s instead
              of [0.30, 12.5] rad/s (:func:`kk_band_floor.estimator_floor` with
              ``extended=``, and the same-run narrow subset as the control);
``edge``      how much of the G3 max is the outermost interval (interior max and
              rms are reported beside it);
``direction`` the damping direction of the Hilbert pair (``B`` from ``A``,
              normalised by ``max|B|``), which is the direction the Cummins
              kernel integrates.

Nothing here changes the predeclared gates: G1-G5 keep the definitions in
:mod:`tensorbem.benchmarks.cummins_wigley_kk_causality`.  G3b and the rms /
interior statistics are *supplementary*: a passing supplement does not turn a
failing gate into a pass.

Usage::

    PYTHONPATH=src python3 scripts/cummins_wigley_band_closure.py \\
        --points-dir artifacts/case_bem_cummins_wigley_ext/points --level medium \\
        --outdir artifacts/case_bem_cummins_wigley_ext
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.benchmarks import kk_band_floor


def _diag(sweep: dict, dof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega, added, damping = case._as_diag(sweep, dof)
    return omega, added, damping


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


def _floor_block(sweep: dict, extended: dict) -> dict:
    """Band attribution per DOF: narrow band vs extended band, same estimator."""
    out = {}
    for dof, label in enumerate(case.DOF_LABELS):
        omega_n, added_n, damping_n = _diag(sweep, dof)
        if extended is None:
            ext = None
        else:
            omega_e, added_e, damping_e = _diag(extended, dof)
            ext = (omega_e, added_e, damping_e)
        report = kk_band_floor.estimator_floor(
            omega_n, added_n, damping_n, extended=ext)
        profile_n = report.pop("refined_profile")
        report.pop("coarse_profile", None)
        if extended is not None:
            # the extended direction, refined: what the wider band leaves behind
            ext_grid, ext_a, ext_b = ext
            dense = kk_band_floor.refine_sweep(ext_grid, ext_a, ext_b, points=601)
            wide_refined = kk_band_floor.kk_residual(
                *dense, added_mass_inf=report["added_mass_inf"],
                scale=report["scale"])
            report["extended_refined_max"] = wide_refined["max"]
            report["extended_refined_rms"] = wide_refined["rms"]
            report["extended_refined_argmax_omega"] = wide_refined["argmax_omega"]
            report["extended_refined_profile"] = {
                "omega_mid": wide_refined["omega_mid"],
                "residual": wide_refined["residual"],
            }
        report["narrow_refined_profile"] = profile_n
        report["dof_label"] = label
        out[label] = report
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", default="artifacts/case_bem_cummins_wigley_ext/points")
    parser.add_argument("--level", default="medium")
    parser.add_argument("--outdir", default="artifacts/case_bem_cummins_wigley_ext")
    parser.add_argument("--out", default="closure.json")
    parser.add_argument("--resonance", type=float, default=7.378504598323515,
                        help="heave resonance the local cluster was placed around")
    args = parser.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    points = case.load_points(args.points_dir, args.level)
    if not points:
        raise SystemExit(f"no cached points in {args.points_dir} for level {args.level!r}")
    have = sorted(points)
    print(f"[closure] {len(have)} cached frequencies: {have[0]:.4f} .. {have[-1]:.4f}",
          flush=True)

    closure = [w for w in case.closure_grid(args.resonance) if round(w, 9) in points]
    extended = [w for w in case.EXTENDED_GRID if round(w, 9) in points]
    narrow = [w for w in case.OMEGA_GRID if round(w, 9) in points]
    legacy = [w for w in case.LEGACY_GRID if round(w, 9) in points]
    print(f"[closure] coverage: closure {len(closure)}/{len(case.closure_grid(args.resonance))}"
          f", extended {len(extended)}/{len(case.EXTENDED_GRID)}"
          f", narrow {len(narrow)}/{len(case.OMEGA_GRID)}"
          f", legacy {len(legacy)}/{len(case.LEGACY_GRID)}", flush=True)

    reference = case.load_capytaine_reference()
    results = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "round": 4,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": args.level,
        "geometry": case.geometry_gate(),
        "signature": case.sweep_signature(args.level),
        "rotation_center": list(case.ROTATION_CENTER),
        "grids": {
            "closure": list(closure),
            "extended": list(extended),
            "narrow": list(narrow),
            "legacy": list(legacy),
            "resonance_cluster_centre": args.resonance,
            "resonance_half_width": case.RESONANCE_HALF_WIDTH,
            "resonance_step": case.RESONANCE_STEP,
        },
        "bands": {},
        "band_attribution": None,
        "resonance": None,
    }

    sweeps = {
        "closure": (closure, "extended band + local resonance cluster"),
        "extended": (extended, "[0.10, 20] rad/s, 28 nodes"),
        "narrow": (narrow, "[0.30, 12.5] rad/s, 22 nodes (round 2)"),
        "legacy": (legacy, "[0.5, 6] rad/s, 8 nodes (round 1)"),
    }
    for name, (grid, description) in sweeps.items():
        if len(grid) < 8:
            results["bands"][name] = {"error": "grid not cached",
                                      "have": list(grid), "description": description}
            continue
        sweep = case.sweep_from_points(points, grid)
        results["bands"][name] = _gates_block(sweep, reference, f"{name}: {description}")
        results["bands"][name]["description"] = description
        print(f"[closure] band={name:9s} points={len(grid):3d} "
              f"verdict={results['bands'][name]['verdict']}", flush=True)
        if name == "closure":
            results["resonance"] = case.heave_resonance_band(sweep)

    if len(narrow) >= 8 and len(extended) >= 8:
        narrow_sweep = case.sweep_from_points(points, narrow)
        extended_sweep = case.sweep_from_points(points, extended)
        results["band_attribution"] = _floor_block(narrow_sweep, extended_sweep)
        for label, entry in results["band_attribution"].items():
            print(f"[closure] band_attributable {label:6s} = "
                  f"{entry['band_attributable']}", flush=True)

    out = outdir / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[closure] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())