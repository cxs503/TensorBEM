"""CASE-BEM-CUMMINS-WIGLEY: added mass at the two spectral limits.

Computes, with the image/doubled-body method of
:mod:`tensorbem.added_mass_limit`, the added-mass matrix of the Wigley III hull

* in the ``omega -> infinity`` limit (Dirichlet calm-water surface, the
  ``A_inf`` that the Kramers-Kronig damping prediction needs), and
* in the ``omega -> 0`` limit (rigid-wall calm-water surface),

on a mesh ladder, and evaluates the existing G3/G5 closure gates with the
physically computed ``A_inf`` instead of the ill-conditioned tail fit.

Run (coarse only, few seconds)::

    PYTHONPATH=src python3 scripts/cummins_wigley_added_mass_limit.py \
        --levels coarse --outdir docs/cummins_wigley_added_mass_limit
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

from tensorbem.wigley3_hull import LEGACY_WIGLEY_ALPHA
import numpy as np

from tensorbem.added_mass_limit import AddedMassLimitResult, added_mass_limit
from tensorbem.benchmarks.cummins_wigley_kk_causality import (
    DENSITY,
    DOF_LABELS,
    GRAVITY,
    MESH_LEVELS,
    geometry_gate,
    wetted_mesh,
)
from tensorbem.causality import fit_added_mass_asymptote, kkr_added_mass, kkr_damping
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh

CENTER = (0.0, 0.0, -0.09375)
SWEEP = pathlib.Path("docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz")
ROTATION_CENTER = (-0.0, -0.0, -0.09375)


def _diag(result: AddedMassLimitResult) -> dict:
    return {DOF_LABELS[i]: float(result.diagonal()[i]) for i in range(6)}


def _round(result: AddedMassLimitResult) -> dict:
    return {
        "limit": result.limit,
        "parity": result.parity,
        "diag": _diag(result),
        "matrix": np.asarray(result.added_mass_matrix, dtype=float).tolist(),
        "lower_panel_count": result.lower_panel_count,
        "doubled_panel_count": result.doubled_panel_count,
        "doubled_volume": result.doubled_volume,
        "hull_volume": result.hull_volume,
        "volume_ratio": result.doubled_volume / result.hull_volume,
        "condition_number": result.condition_number,
        "max_relative_residual_l2": result.max_relative_residual_l2,
        "symmetry_error": result.symmetry_error,
        "double_layer_row_sum_error": result.double_layer_row_sum_error,
    }


def _load_sweep() -> dict:
    payload = np.load(SWEEP)
    return {
        "omega": [float(w) for w in payload["omega"]],
        "added_mass": payload["added_mass"],
        "damping": payload["damping"],
        "level": "n_sections=10,n_levels=5",
        "level_key": "coarse",
        "rotation_center": list(ROTATION_CENTER),
        "density": DENSITY,
        "gravity": GRAVITY,
    }


def closure_residual(
    omega: np.ndarray,
    added: np.ndarray,
    damping: np.ndarray,
    added_mass_inf: float,
) -> dict:
    """G3/G5 statistics of one DOF for a *given* ``A_inf``.

    ``G3`` predicts the added mass from the damping sweep (KK), ``G5`` is the
    ``B -> K -> (B, A)`` round trip; both are normalised by the scale
    ``max |A - A_inf|`` exactly as the anchor's gates are.
    """
    mid = 0.5 * (omega[:-1] + omega[1:])
    measured = 0.5 * (added[:-1] + added[1:])
    predicted = np.array([
        kkr_added_mass(omega, damping, float(w), added_mass_inf=float(added_mass_inf))
        for w in mid
    ])
    abs_res = np.abs(predicted - measured)
    scale = max(float(np.max(np.abs(measured - added_mass_inf))), 1e-30)
    r3 = float(np.max(abs_res)) / scale
    # affine dependence of the G3 statistic on A_inf, measured by a finite step
    step = 1.0e-3 * max(abs(float(added_mass_inf)), 1.0)
    pred2 = np.array([
        kkr_added_mass(omega, damping, float(w),
                       added_mass_inf=float(added_mass_inf) + step)
        for w in mid
    ])
    scale2 = max(float(np.max(np.abs(measured - added_mass_inf - step))), 1e-30)
    r3_step = float(np.max(np.abs(pred2 - measured))) / scale2
    return {
        "G3": r3,
        "G3_added_mass_inf": float(added_mass_inf),
        "scale": scale,
        "G3_derivative": (r3_step - r3) / step,
        "argmax_omega": float(mid[int(np.argmax(abs_res))]),
        # the mirror KK direction: A -> B (depends on A_inf only through the
        # integrand A - A_inf, which is what the sweep and A_inf together give)
        "G3_reverse": float(
            np.max(np.abs(
                np.array([
                    kkr_damping(omega, added, float(w),
                                added_mass_inf=float(added_mass_inf))
                    for w in mid
                ]) - np.interp(mid, omega, damping)
            )) / max(float(np.max(np.abs(np.interp(mid, omega, damping)))), 1e-30)
        ),
    }


def profile_optimal_inf(omega, added, damping) -> float:
    """The ``A_inf`` that minimises the G3 statistic (m20594's closed form).

    ``A_pred(omega; a) = q(omega) + a`` is affine with slope ``+1`` in ``a``, so
    the max-abs residual is a convex piecewise-linear function of ``a`` and the
    minimiser is the mid-range of the error vector ``e_i = A_i - q_i``.
    """
    mid = 0.5 * (omega[:-1] + omega[1:])
    measured = 0.5 * (added[:-1] + added[1:])
    base = np.array([kkr_added_mass(omega, damping, float(w), added_mass_inf=0.0)
                     for w in mid])
    error = measured - base
    return float(0.5 * (np.max(error) + np.min(error)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["coarse"])
    parser.add_argument("--outdir", default="docs/cummins_wigley_added_mass_limit")
    arguments = parser.parse_args()

    outdir = pathlib.Path(arguments.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gate = geometry_gate()
    if not gate["passed"]:
        raise SystemExit(f"geometry gate failed: {gate}")

    report: dict = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "rotation_center": list(CENTER),
        "geometry_gate": gate,
        "density": DENSITY,
        "gravity": GRAVITY,
        "levels": {},
        "analytic_check": {},
    }

    # ---- analytic check: half-immersed sphere, A_inf(heave) = m_full / 4 ----
    analytic = []
    for nlat, nlon in ((4, 12), (8, 24), (12, 36)):
        mesh = _hemisphere(nlat, nlon)
        res = added_mass_limit(mesh, "infinite_frequency", density=DENSITY)
        exact = DENSITY * 4.0 / 3.0 * math.pi / 4.0
        analytic.append({
            "nlat": nlat, "nlon": nlon, "panels": mesh.panel_count,
            "hull_volume_exact": 2.0 / 3.0 * math.pi,
            "hull_volume": mesh.volume,
            "A_inf_heave": float(res.diagonal()[2]),
            "exact_heave": exact,
            "rel_err": float(abs(res.diagonal()[2] - exact) / exact),
        })
    report["analytic_check"] = {
        "case": "half-immersed sphere: image closes to a full sphere, heave "
                "becomes a rigid translation, A_inf(heave) = m_full/4 exactly",
        "rows": analytic,
        "worst_rel_err": max(row["rel_err"] for row in analytic),
        "rel_err_finest": analytic[-1]["rel_err"],
    }

    # ---- Wigley ladder ----
    for level in arguments.levels:
        nx, nz = MESH_LEVELS[level]
        mesh = wigley3_wetted_hull_mesh(nx=nx, nz=nz, alpha=LEGACY_WIGLEY_ALPHA)
        entry: dict = {"n_sections": nx, "n_levels": nz,
                       "panels": mesh.panel_count,
                       "hull_volume": float(mesh.volume)}
        for limit in ("infinite_frequency", "zero_frequency"):
            started = time.time()
            result = added_mass_limit(mesh, limit, center=CENTER)
            entry[limit] = _round(result)
            entry[limit]["wall_seconds"] = time.time() - started
        report["levels"][level] = entry

    # ---- closure gates with the physical A_inf ----
    sweep = _load_sweep()
    omega = np.asarray(sweep["omega"], dtype=float)
    added_matrix = np.asarray(sweep["added_mass"], dtype=float)
    damping_matrix = np.asarray(sweep["damping"], dtype=float)
    level_for_closure = "coarse"
    a_inf = report["levels"][level_for_closure]["infinite_frequency"]["diag"]
    a_zero = report["levels"][level_for_closure]["zero_frequency"]["diag"]
    closure = {}
    for dof, label in enumerate(DOF_LABELS):
        added = added_matrix[:, dof, dof]
        damping = damping_matrix[:, dof, dof]
        fit = fit_added_mass_asymptote(omega, added)
        optimal = profile_optimal_inf(omega, added, damping)
        row = {
            "tail_fit_inf": fit.added_mass_inf,
            "tail_fit_rel_residual": fit.relative_residual,
            "profile_optimal_inf": optimal,
            "physical_inf": a_inf[label],
            "zero_frequency_limit": a_zero[label],
            "sweep_A_at_lowest_omega": float(added[0]),
            "sweep_omega_lowest": float(omega[0]),
            "residuals": {
                "tail_fit": closure_residual(omega, added, damping, fit.added_mass_inf),
                "profile_optimal": closure_residual(omega, added, damping, optimal),
                "physical_inf": closure_residual(omega, added, damping, a_inf[label]),
            },
        }
        closure[label] = row
    report["closure"] = {
        "sweep": str(SWEEP),
        "sweep_band": [float(omega[0]), float(omega[-1])],
        "sweep_points": int(omega.size),
        "g3_tol": 0.10,
        "g5_tol": 0.05,
        "per_dof": closure,
        "worst": {
            key: max(closure[label]["residuals"][key]["G3"] for label in DOF_LABELS)
            for key in ("tail_fit", "profile_optimal", "physical_inf")
        },
    }

    path = outdir / "added_mass_limit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "levels": {k: {lim: v[lim]["diag"] for lim in
                       ("infinite_frequency", "zero_frequency")}
                   for k, v in report["levels"].items()},
        "analytic_worst_rel_err": report["analytic_check"]["worst_rel_err"],
        "closure_worst_G3": report["closure"]["worst"],
    }, indent=2, sort_keys=True))
    print(f"wrote {path}")
    return 0


def _hemisphere(nlat: int, nlon: int, radius: float = 1.0):
    """Lower hemisphere mesh (waterline at the equator) for the analytic check."""
    from tensorbem.geometry3d import WettedHullMesh

    verts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for i in range(nlat):
        theta = 0.5 * math.pi * i / nlat
        z = -radius * math.sin(theta)
        r = radius * math.cos(theta)
        row = []
        for j in range(nlon):
            phi = 2.0 * math.pi * j / nlon
            row.append(len(verts))
            verts.append((r * math.cos(phi), r * math.sin(phi), z))
        rings.append(row)
    pole = len(verts)
    verts.append((0.0, 0.0, -radius))
    faces: list[tuple[int, int, int]] = []
    for i in range(nlat - 1):
        for j in range(nlon):
            j2 = (j + 1) % nlon
            a, b = rings[i][j], rings[i][j2]
            c, d = rings[i + 1][j2], rings[i + 1][j]
            faces.append((a, b, c))
            faces.append((a, c, d))
    for j in range(nlon):
        j2 = (j + 1) % nlon
        faces.append((rings[-1][j], rings[-1][j2], pole))
    return WettedHullMesh(np.asarray(verts, float),
                          np.asarray(faces, np.int64)[:, ::-1])


if __name__ == "__main__":
    raise SystemExit(main())