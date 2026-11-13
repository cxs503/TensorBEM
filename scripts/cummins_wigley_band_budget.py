"""Band budget of the Cummins/Kramers-Kronig gates on the Wigley III hull.

Round 2 of ``CASE-BEM-CUMMINS-WIGLEY`` left G3 (Kramers-Kronig, 53-81 %),
G4 (free decay, ``IndexError``) and G5 (round trip, 22-53 %) failing and could
only *assert* that the residual was a band effect (the round-2 sweep realised 16
of its 22 declared nodes and started at 0.30 rad/s).  This driver answers the
question with measurements instead of prose:

1. it re-evaluates G1-G5 on
   * the case grid (:data:`tensorbem.benchmarks.cummins_wigley_kk_causality.OMEGA_GRID`,
     22 nodes on [0.30, 12.5]), and
   * an **extended band** (28 nodes on [0.10, 20.0], same mesh), which is the
     prescription the round-2 report wrote for itself ("diagnosing it needs a
     denser low-frequency sweep (0.1-0.3 rad/s)");
2. it measures, per DOF, the **causal floor** of the KK residual at each band
   (:func:`tensorbem.benchmarks.kk_band_floor.band_floor_report`): the residual an
   exactly-causal exponential-sum surrogate with the measured spectral shape
   still shows on the same grid, split into the band part (dense sampling) and
   the grid-discretisation part.

So each DOF ends up with three numbers -- measured, floor on the same grid, floor
on a dense grid -- and the differences say whether the panel sweep carries
non-causal content or whether the estimator/band is the whole story.

Every number written by this driver comes from cached per-frequency solves
(``points/``, one JSON per ``(level, omega)``, written by
``scripts/run_cummins_wigley_kk_causality.py --mode sweep``); nothing is
hand-typed and a missing frequency raises instead of being interpolated.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.benchmarks import kk_band_floor

#: The band the round-2 report asked for: four nodes below the old start plus two
#: above the old stop, on top of the unchanged case grid.
LOW_EXTENSION = (0.10, 0.15, 0.20, 0.25)
HIGH_EXTENSION = (15.00, 20.00)
EXTENDED_GRID = LOW_EXTENSION + tuple(case.OMEGA_GRID) + HIGH_EXTENSION


def _grids(points: dict[float, dict]) -> dict[str, list[float]]:
    """Band name -> frequencies, keeping only bands fully cached."""
    available = {round(float(w), 9) for w in points}
    grids = {}
    for name, grid, required in (
        ("case_band_22", case.OMEGA_GRID, case.OMEGA_GRID),
        ("extended_band_28", EXTENDED_GRID, EXTENDED_GRID),
        ("legacy_band_8", case.LEGACY_GRID, case.LEGACY_GRID),
    ):
        missing = [float(w) for w in required if round(float(w), 9) not in available]
        grids[name] = {
            "omega": [float(w) for w in grid],
            "complete": not missing,
            "missing": missing,
        }
    return grids


def _dof_block(sweep: dict, dof: int, dof_name: str,
               extended: dict | None = None) -> dict:
    """KK residual of one DOF next to the estimator's own floor."""
    omega, added, damping = case._as_diag(sweep, dof)
    ext = None
    if extended is not None:
        ext = tuple(case._as_diag(extended, dof))
    floor = kk_band_floor.estimator_floor(omega, added, damping, extended=ext)
    keep = ("band", "points", "residual_coarse", "residual_refined",
            "residual_coarse_rms", "residual_refined_rms",
            "argmax_omega_coarse", "argmax_omega_refined",
            "discretisation_share", "band_attributable", "extended_band",
            "added_mass_inf", "scale")
    return {
        "dof": dof_name,
        "estimator_floor": {key: floor[key] for key in keep},
        "profiles": {"coarse": floor["coarse_profile"],
                     "refined": floor["refined_profile"]},
    }


def load_npz_sweep(path: pathlib.Path, label: str) -> dict:
    """A cached ``(omega, A, B)`` sweep shipped as ``.npz``.

    ``benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz`` holds one
    such sweep (192-panel coarse mesh, 13 nodes on [0.15, 8.0] rad/s) written by
    the earlier case-pool mission; carrying it here lets the band budget run
    without re-solving a single frequency, and keeps the numbers traceable to the
    artifact that produced them.
    """
    data = np.load(path)
    omega = np.asarray(data["omega"], dtype=float)
    return {
        "level": label,
        "level_key": label,
        "rotation_center": list(case.ROTATION_CENTER),
        "omega": omega.tolist(),
        "added_mass": np.asarray(data["added_mass"], dtype=float).tolist(),
        "damping": np.asarray(data["damping"], dtype=float).tolist(),
        "relative_residual": [float("nan")] * omega.size,
        "wall_seconds": [float("nan")] * omega.size,
        "density": case.DENSITY,
        "gravity": case.GRAVITY,
        "source": str(path),
    }


def build_budget(points_dir: pathlib.Path, level: str) -> dict:
    points = case.load_points(points_dir, level)
    if not points:
        raise SystemExit(f"no cached points for level {level!r} in {points_dir}")
    grids = _grids(points)
    reference = case.load_capytaine_reference()
    out = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "stage": "band_budget",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "points_dir": str(points_dir),
        "cached_omega": sorted(float(w) for w in points),
        "grids": grids,
        "gates_definition": case.GATES,
        "rotation_center": list(case.ROTATION_CENTER),
        "bands": {},
    }
    for name, info in grids.items():
        if not info["complete"]:
            out["bands"][name] = {"skipped": "grid not fully cached",
                                  "missing": info["missing"]}
            continue
        sweep = case.sweep_from_points(points, info["omega"])
        extended_sweep = None
        ext_info = grids.get("extended_band_28", {})
        if ext_info.get("complete") and name != "extended_band_28":
            extended_sweep = case.sweep_from_points(points, ext_info["omega"])
        analysis = case.gates_from_sweep(sweep, reference=reference)
        decay = {k: v for k, v in analysis["free_decay"].items() if k != "half_power"}
        out["bands"][name] = {
            "omega": info["omega"],
            "gates": {gid: {k: v for k, v in gate.items()}
                      for gid, gate in analysis["gates"].items()},
            "failed": analysis["failed"],
            "verdict": analysis["verdict"],
            "free_decay": decay,
            "half_power": analysis["free_decay"]["half_power"],
            "heave_resonance": case.heave_resonance_band(sweep),
            "cross_solver": analysis["cross_solver"],
            "dof": {
                case.DOF_LABELS[dof]: _dof_block(
                    sweep, dof, case.DOF_LABELS[dof],
                    extended=extended_sweep)
                for dof in range(6)
            },
        }
        print(f"[budget] {name}: {analysis['verdict']}", flush=True)
    return out


def build_budget_from_sweep(sweep: dict, label: str, *,
                            with_cross: bool = False) -> dict:
    """Band budget on one already-assembled sweep (no solver call)."""
    reference = case.load_capytaine_reference() if with_cross else None
    analysis = case.gates_from_sweep(
        sweep, reference=reference, include_cross_solver=with_cross)
    band = {
        "omega": sweep["omega"],
        "source": sweep.get("source", "cached points"),
        "gates": dict(analysis["gates"]),
        "failed": analysis["failed"],
        "verdict": analysis["verdict"],
        "free_decay": {k: v for k, v in analysis["free_decay"].items()
                       if k != "half_power"},
        "half_power": analysis["free_decay"]["half_power"],
        "heave_resonance": case.heave_resonance_band(sweep),
        "cross_solver": analysis["cross_solver"],
        "dof": {
            case.DOF_LABELS[dof]: _dof_block(sweep, dof, case.DOF_LABELS[dof])
            for dof in range(6)
        },
    }
    print(f"[budget] {label}: {analysis['verdict']}", flush=True)
    return {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "stage": "band_budget",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": label,
        "rotation_center": list(case.ROTATION_CENTER),
        "gates_definition": case.GATES,
        "bands": {label: band},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", default="artifacts/case_bem_cummins_wigley_ext/points")
    parser.add_argument("--level", default="medium")
    parser.add_argument("--npz", default=None,
                        help="cached sweep .npz (omega/added_mass/damping) to analyse "
                             "instead of a points directory")
    parser.add_argument("--label", default=None, help="band label when --npz is used")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.npz:
        npz = pathlib.Path(args.npz)
        label = args.label or npz.stem
        payload = build_budget_from_sweep(load_npz_sweep(npz, label), label)
        out = pathlib.Path(args.out) if args.out else npz.parent / f"band_budget_{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1))
        print(f"[budget] -> {out}", flush=True)
        return 0
    points_dir = pathlib.Path(args.points_dir)
    payload = build_budget(points_dir, args.level)
    out = pathlib.Path(args.out) if args.out else points_dir.parent / f"band_budget_{args.level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"[budget] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())