"""CASE-BEM-MARUO-DRIFT -- Maruo far-field mean drift force against its closed form
in terms of the scattering amplitudes (fixed bottom-mounted vertical cylinder).

What is being decided
---------------------
Does TensorBEM's 3-D BEM route
(``solve_diffraction_3d`` -> ``kochin_function_3d`` -> reduced Maruo far-field
kernel) reproduce the *closed form* of the Maruo (1957) far-field integral,
written in terms of the scattering amplitudes of the exact MacCamy-Fuchs
scattering field, to better than 5 % in the mean horizontal drift force?

Reference (the closed form, not a typed table)
----------------------------------------------
Maruo's far-field formula for the mean drift force on a fixed body in regular
head seas, deep water, is

    F_x = (rho g A^2 / (2 pi k)) Int_0^{2 pi} |S(theta)|^2 (1 - cos theta) dtheta,

with ``S(theta)`` the far-field scattering amplitude of the disturbed potential
(``psi_s ~ sqrt(2/(pi k r)) exp(i(k r - pi/4)) S(theta)``).  This is the
matching instance of

    F_j = (rho omega^2 / 4k) Int |K(theta)|^2 (e_beta - e_theta)_j dtheta

(Maruo 1957; Maruo 1960; Newman 1977, Marine Hydrodynamics, Sec. 7.3).

For a circular cylinder the scattering field is the exact MacCamy-Fuchs series

    psi_s = sum_m A_m H_m^(1)(k r) e^{i m theta},   A_m = -i^m J_m'(ka)/H_m'^(1)(ka),

whose far field is a *finite* trigonometric polynomial
``S(theta) = sum_m c_m e^{i m theta}``, ``c_m = A_m exp(-i m pi/2)``.  Because
that spectrum is finite, the Maruo azimuthal integral can be evaluated in exact
closed form -- this is the whole point of the case: no quadrature at all,

    Int |S|^2 dtheta                = 2 pi sum_m |c_m|^2,
    Int |S|^2 cos(theta) dtheta     = 2 pi Re sum_m c_m conj(c_{m+1}),
    => Int |S|^2 (1 - cos theta) dtheta = 2 pi [ sum_m |c_m|^2
                                                 - Re sum_m c_m conj(c_{m+1}) ].

The ``(ka) -> 0`` limit of this same closed form is the classical long-wave law

    F_x = (5 pi^2 / 16) rho g A^2 a (ka)^3 = 3.0842513753... rho g A^2 a (ka)^3,

which is used here as an independent symbolic anchor (the constant is evaluated
from ``math.pi`` in code, never typed in).

Route under test (``got``)
--------------------------
``solve_diffraction_3d`` on a watertight wall-sided cylinder mesh (draft chosen
so the deep-water Green function does not see the bottom), its physical Kochin
function from ``drift_far_field3d.bem_kochin`` (amplitude normalisation: the
incident potential carries ``g A / omega``, see
``drift_far_field3d.AMPLITUDE_NORMALISATION``) and the radius-free reduced Maruo
kernel ``drift_far_field3d.maruo_far_field_drift``.

Evidence discipline
-------------------
Every number in the emitted JSON is computed by this module.  Nothing is typed
from a paper: the reference *is* the closed form above, evaluated in floating
point, and it is cross-checked against the repository's independent quadrature
route (``cylinder_drift.far_field_cylinder_drift``) and, at long waves, against
the symbolic constant ``5 pi^2 / 16``.

Usage
-----
    PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case \
        --ka 0.5 1.0 --panels 32 --levels 12 --grading 3.0 \
        --json artifacts/maruo_drift_far_field_case/cell.json

The study driver (``run_study``) sweeps the case grid and writes one JSON with
the full comparison table, the panel/quadrature sensitivities and the gates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import cylinder_drift as _cd
from ..diffraction_drift_maturity import vertical_cylinder_hull_mesh
from ..drift_far_field3d import bem_kochin, maruo_far_field_drift
from ..free_surface_solver3d import solve_diffraction_3d

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.81
DEFAULT_M_MAX = 14

#: Study grid of the case: the four wave numbers span the long-wave band, the
#: drift-force maximum (ka ~ 1) and the short-wave band (ka = 2).
DEFAULT_KA: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0)

#: Panel resolutions (name, n_theta panels, vertical rings, grading).  The
#: grading exponent clusters rings near the free surface; it was found to matter
#: far more than the ring count alone (see docs/maruo_drift_far_field_case.md).
RESOLUTIONS: tuple[tuple[str, int, int, float], ...] = (
    ("coarse", 24, 6, 2.5),
    ("medium", 32, 8, 2.5),
    ("fine", 32, 12, 3.0),
    ("finer", 48, 12, 3.0),
)

__all__ = [
    "DEFAULT_KA",
    "RESOLUTIONS",
    "bem_case",
    "closed_form_case",
    "long_wave_anchor",
    "maruo_integral_closed_form",
    "merge_studies",
    "quadrature_sensitivity",
    "run_study",
]


# ---------------------------------------------------------------------------
# 1. the reference: exact reduction of the Maruo integral
# ---------------------------------------------------------------------------
def maruo_integral_closed_form(ka: float, *, m_max: int = DEFAULT_M_MAX) -> dict[str, Any]:
    """Exact closed form of ``Int |S|^2 (1 - cos theta) dtheta`` for the cylinder.

    ``S(theta) = sum_m c_m e^{i m theta}`` with ``c_m = A_m exp(-i m pi/2)`` and
    ``A_m`` the MacCamy-Fuchs scattering coefficients.  The two Maruo moments are
    ``2 pi`` times a finite sum over the modal spectrum, so no azimuthal
    quadrature is involved.
    """
    x = float(ka)
    if not (x > 0.0):
        raise ValueError("ka must be positive")
    m_max = int(m_max)
    modes = _cd.maccamy_fuchs_modes(x, m_max=m_max)
    m = np.arange(-m_max, m_max + 1, dtype=float)
    c = modes * np.exp(-0.5j * m * math.pi)
    isotropic = float(np.sum(np.abs(c) ** 2))
    projection = float(np.real(np.sum(c[:-1] * np.conj(c[1:]))))
    integral = 2.0 * math.pi * (isotropic - projection)
    return {
        "ka": x,
        "m_max": m_max,
        "isotropic_moment": 2.0 * math.pi * isotropic,
        "projection_moment": 2.0 * math.pi * projection,
        "integral_weighted": integral,
        "modal_energy_sum": isotropic,
    }


def closed_form_case(
    ka: float,
    *,
    amplitude: float = 1.0,
    radius: float = 1.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    m_max: int = DEFAULT_M_MAX,
) -> dict[str, Any]:
    """Closed-form mean drift force ``F_x = rho g A^2 I / (2 pi k)``."""
    amp = float(amplitude)
    a = float(radius)
    rho = float(density)
    g = float(gravity)
    if amp <= 0.0 or a <= 0.0 or rho <= 0.0 or g <= 0.0:
        raise ValueError("amplitude, radius, density and gravity must be positive")
    red = maruo_integral_closed_form(ka, m_max=m_max)
    k = float(ka) / a
    force = rho * g * amp**2 * red["integral_weighted"] / (2.0 * math.pi * k)
    out = dict(red)
    out.update(
        {
            "wavenumber": k,
            "radius": a,
            "amplitude": amp,
            "density": rho,
            "gravity": g,
            "mean_drift_force_x": force,
            # dimensionless form: F / (rho g A^2 a)
            "force_coefficient": force / (rho * g * amp**2 * a),
        }
    )
    return out


def long_wave_anchor(
    *, ka: float = 0.02, radius: float = 1.0, amplitude: float = 1.0,
    density: float = DEFAULT_DENSITY, gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Long-wave law ``F = (5 pi^2/16) rho g A^2 a (ka)^3`` vs the closed form."""
    x = float(ka)
    ref = closed_form_case(x, amplitude=amplitude, radius=radius,
                           density=density, gravity=gravity)
    coefficient = math.pi**2 * 5.0 / 16.0
    law = coefficient * density * gravity * amplitude**2 * radius * x**3
    measured = ref["mean_drift_force_x"] / (
        density * gravity * amplitude**2 * radius * x**3
    )
    return {
        "ka": x,
        "long_wave_coefficient_closed_form": coefficient,
        "long_wave_coefficient_measured": measured,
        "long_wave_force": law,
        "closed_form_force": ref["mean_drift_force_x"],
        "rel_error": abs(law - ref["mean_drift_force_x"]) / ref["mean_drift_force_x"],
    }


# ---------------------------------------------------------------------------
# 2. the route under test: BEM panels -> Kochin -> reduced Maruo kernel
# ---------------------------------------------------------------------------
def bem_case(
    ka: float,
    *,
    panels: int = 32,
    levels: int = 12,
    grading: float = 3.0,
    radius: float = 1.0,
    amplitude: float = 1.0,
    depth_factor: float = 16.0,
    n_azimuth: int = 96,
    n_theta_quad: int = 2048,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """BEM solution, its Kochin function and the far-field Maruo drift force."""
    t0 = time.time()
    k = float(ka) / float(radius)
    mesh = vertical_cylinder_hull_mesh(
        float(radius), float(depth_factor) * float(radius),
        n_theta=int(panels), n_levels=int(levels), grading=float(grading),
    )
    diffraction = solve_diffraction_3d(
        mesh, k, amplitude=float(amplitude), density=density, gravity=gravity
    )
    kochin = bem_kochin(mesh, diffraction, n_azimuth=int(n_azimuth))
    far = maruo_far_field_drift(
        kochin.kochin, kochin.azimuth, k,
        amplitude=float(amplitude), density=density, gravity=gravity,
        n_theta=int(n_theta_quad),
    )
    elapsed = time.time() - t0
    ref = closed_form_case(
        ka, amplitude=amplitude, radius=radius, density=density, gravity=gravity
    )
    got = float(far["force_x"])
    return {
        "ka": float(ka),
        "wavenumber": k,
        "panels": int(panels),
        "levels": int(levels),
        "grading": float(grading),
        "panel_count": int(mesh.panel_count),
        "n_azimuth": int(n_azimuth),
        "n_theta_quad": int(n_theta_quad),
        "bem_force": got,
        "bem_force_coefficient": got / (density * gravity * amplitude**2 * radius),
        "ref_force": float(ref["mean_drift_force_x"]),
        "ref_force_coefficient": float(ref["force_coefficient"]),
        "ref_integral": float(ref["integral_weighted"]),
        "bem_integral_total": float(far["integral_total"]),
        "bem_integral_weighted": float(far["integral_weighted"]),
        "rel_error": abs(got - ref["mean_drift_force_x"]) / abs(ref["mean_drift_force_x"]),
        "solver_relative_residual": float(diffraction.relative_residual),
        "kochin_azimuth_rel_change": float(kochin.azimuth_rel_change),
        "elapsed_seconds": elapsed,
    }


def quadrature_sensitivity(
    ka: float,
    *,
    n_thetas: Sequence[int] = (128, 256, 512, 1024, 2048, 4096),
    m_max: int = DEFAULT_M_MAX,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Azimuthal-quadrature sensitivity of the Maruo kernel on the exact field.

    Uses the exact MacCamy-Fuchs Kochin table (repo convention) shifted into the
    physical normalisation so that the reduced kernel's numbers are comparable
    with the closed form; only the *spread* over ``n_theta`` is interpreted.
    """
    x = float(ka)
    k = x
    theta = (2.0 * math.pi / 1440) * np.arange(1440, dtype=float)
    kochin = _cd.kochin_3d(x, theta, m_max=m_max)
    rows = []
    for n_t in n_thetas:
        far = maruo_far_field_drift(kochin, theta, k, n_theta=int(n_t),
                                    density=density, gravity=gravity)
        rows.append({"n_theta": int(n_t), "integral_total": float(far["integral_total"]),
                     "integral_weighted": float(far["integral_weighted"]),
                     "force_x": float(far["force_x"])})
    values = np.array([row["integral_weighted"] for row in rows], dtype=float)
    reference_force = closed_form_case(x, density=density, gravity=gravity)["mean_drift_force_x"]
    return {
        "ka": x,
        "rows": rows,
        "spread_rel": float((values.max() - values.min()) / abs(values.mean())),
        "last_step_rel": float(abs(values[-1] - values[-2]) / abs(values[-1])),
        # amplitude-convention audit (drift_far_field3d.AMPLITUDE_NORMALISATION):
        # a unit-potential-amplitude K (as produced by cylinder_drift.kochin_3d)
        # understates the force by (g/omega)^2 = g/k, i.e. the measured ratio of
        # this unit-amplitude kernel to the closed form must be k/g.
        "unit_amplitude_force_ratio": float(rows[-1]["force_x"] / reference_force),
        "expected_unit_amplitude_ratio_k_over_g": float(x / gravity),
    }


# ---------------------------------------------------------------------------
# 3. study driver
# ---------------------------------------------------------------------------
def _long_wave_gate(anchors: Sequence[dict[str, Any]], tolerance: float = 1e-3) -> dict:
    """Gate on the long-wave law: measured where its own validity band holds.

    The departure of the closed form from ``(5 pi^2/16)(ka)^3`` is O((ka)^2)
    (1.5e-5 at ka = 0.01, 1.2e-3 at ka = 0.05), so only the anchors with
    ``ka <= 0.02`` are gated; larger ``ka`` is reported for the trend.
    """
    in_band = [row for row in anchors if row["ka"] <= 0.02]
    if not in_band:
        raise ValueError("no long-wave anchor inside ka <= 0.02")
    measured = max(row["rel_error"] for row in in_band)
    return _gate(
        "G2_long_wave_coefficient", tolerance, measured, measured < tolerance,
        "closed form vs symbolic 5 pi^2/16 at ka = "
        + " / ".join(f"{row['ka']:g}" for row in in_band)
        + "; the ka = 0.05 anchor is reported for the O(ka^2) trend, not gated",
    )


def _gate(name: str, threshold: float, measured: float, passed: bool, note: str = "") -> dict:
    return {
        "name": name,
        "threshold": threshold,
        "measured": measured,
        "passed": bool(passed),
        "note": note,
    }


def run_study(
    *,
    ka_values: Sequence[float] = DEFAULT_KA,
    resolutions: Sequence[tuple[str, int, int, float]] = RESOLUTIONS,
    production: str = "fine",
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_azimuth: int = 96,
    n_theta_quad: int = 2048,
    log: bool = True,
) -> dict[str, Any]:
    """Run the whole case: reference, BEM table, sensitivities, gates."""
    cells: list[dict[str, Any]] = []
    for ka in ka_values:
        for name, panels, levels, grading in resolutions:
            cell = bem_case(ka, panels=panels, levels=levels, grading=grading,
                            n_azimuth=n_azimuth, n_theta_quad=n_theta_quad,
                            density=density, gravity=gravity)
            cell["resolution"] = name
            cells.append(cell)
            if log:
                print(
                    f"[case] ka={ka:5.2f} {name:6s} panels={cell['panel_count']:4d} "
                    f"got={cell['bem_force']:12.4f} ref={cell['ref_force']:12.4f} "
                    f"rel={100.0 * cell['rel_error']:6.2f}% "
                    f"({cell['elapsed_seconds']:.0f}s)",
                    flush=True,
                )

    reference_rows = [closed_form_case(ka, density=density, gravity=gravity)
                      for ka in ka_values]
    # independent quadrature route of the same closed form (repo capability)
    cross_rows = []
    for ka in ka_values:
        quad = _cd.far_field_cylinder_drift(ka, density=density, gravity=gravity)
        closed = closed_form_case(ka, density=density, gravity=gravity)
        cross_rows.append({
            "ka": float(ka),
            "closed_form": closed["mean_drift_force_x"],
            "cylinder_drift_quadrature": quad["mean_drift_force_x"],
            "rel_error": abs(closed["mean_drift_force_x"] - quad["mean_drift_force_x"])
            / quad["mean_drift_force_x"],
        })

    anchors = [long_wave_anchor(ka=ka, density=density, gravity=gravity)
               for ka in (0.01, 0.02, 0.05)]
    quad_sens = {
        str(ka): quadrature_sensitivity(ka, density=density, gravity=gravity)
        for ka in ka_values
    }

    production_cells = [c for c in cells if c["resolution"] == production]
    worst_production = max(c["rel_error"] for c in production_cells)

    gates = [
        _gate("G1_closed_form_vs_quadrature", 1e-10,
              max(row["rel_error"] for row in cross_rows),
              max(row["rel_error"] for row in cross_rows) < 1e-10,
              "closed form (modal sum) vs repo quadrature route"),
        _long_wave_gate(anchors),
        _gate("G3_bem_vs_ref_production", 0.05, worst_production,
              worst_production < 0.05,
              f"worst cell at resolution '{production}' (task tolerance 5%)"),
        _gate("G4_quadrature_spread", 1e-2,
              max(q["spread_rel"] for q in quad_sens.values()),
              max(q["spread_rel"] for q in quad_sens.values()) < 1e-2,
              "Maruo kernel spread over n_theta = 128 ... 4096 on the exact field"),
    ]

    return {
        "case": "CASE-BEM-MARUO-DRIFT",
        "generated_by": "tensorbem.benchmarks.maruo_drift_far_field_case.run_study",
        "parameters": {
            "density": density,
            "gravity": gravity,
            "radius": 1.0,
            "amplitude": 1.0,
            "depth_factor": 16.0,
            "n_azimuth": n_azimuth,
            "n_theta_quad": n_theta_quad,
            "m_max": DEFAULT_M_MAX,
            "ka_values": [float(x) for x in ka_values],
            "resolutions": [list(r) for r in resolutions],
            "production_resolution": production,
        },
        "cells": cells,
        "reference": reference_rows,
        "cross_check_closed_form_vs_quadrature": cross_rows,
        "long_wave_anchors": anchors,
        "quadrature_sensitivity": quad_sens,
        "production_worst_rel_error": worst_production,
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
    }


# ---------------------------------------------------------------------------
# 4. shard merge (one JSON per wave number -> one study JSON)
# ---------------------------------------------------------------------------
def merge_studies(studies: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-shard study JSONs (same parameters, disjoint ``ka`` lists).

    The heavy BEM cells are the only part worth sharding; the closed-form and
    quadrature blocks are cheap and are recomputed here from the merged
    parameters so that the merged file is never a union of stale subsets.
    """
    if not studies:
        raise ValueError("no studies to merge")
    pars = studies[0]["parameters"]
    for study in studies[1:]:
        if study["parameters"]["resolutions"] != pars["resolutions"]:
            raise ValueError("resolutions differ between shards")
    ka_values = sorted({ka for study in studies for ka in study["parameters"]["ka_values"]})
    cells = [cell for study in studies for cell in study["cells"]]
    cells.sort(key=lambda c: (c["ka"], c["panel_count"]))
    production = pars["production_resolution"]
    production_errors = [c["rel_error"] for c in cells if c["resolution"] == production]
    if not production_errors:
        raise ValueError(f"no cells at production resolution {production!r}")
    missing = set(ka_values) - {c["ka"] for c in cells if c["resolution"] == production}
    if missing:
        raise ValueError(f"shards missing production cells for ka={sorted(missing)}")

    density = float(pars["density"])
    gravity = float(pars["gravity"])
    merged_pars = dict(pars)
    merged_pars["ka_values"] = [float(x) for x in ka_values]

    cross_rows = []
    for ka in ka_values:
        quad = _cd.far_field_cylinder_drift(ka, density=density, gravity=gravity)
        closed = closed_form_case(ka, density=density, gravity=gravity)
        cross_rows.append({
            "ka": float(ka),
            "closed_form": closed["mean_drift_force_x"],
            "cylinder_drift_quadrature": quad["mean_drift_force_x"],
            "rel_error": abs(closed["mean_drift_force_x"] - quad["mean_drift_force_x"])
            / quad["mean_drift_force_x"],
        })
    anchors = [long_wave_anchor(ka=ka, density=density, gravity=gravity)
               for ka in (0.01, 0.02, 0.05)]
    quad_sens = {str(ka): quadrature_sensitivity(ka, density=density, gravity=gravity)
                 for ka in ka_values}
    worst = max(production_errors)
    quad_spread = max(q["spread_rel"] for q in quad_sens.values())
    gates = [
        _gate("G1_closed_form_vs_quadrature", 1e-10,
              max(row["rel_error"] for row in cross_rows),
              max(row["rel_error"] for row in cross_rows) < 1e-10,
              "closed form (modal sum) vs repo quadrature route"),
        _long_wave_gate(anchors),
        _gate("G3_bem_vs_ref_production", 0.05, worst, worst < 0.05,
              f"worst cell at resolution {production!r} (task tolerance 5%)"),
        _gate("G4_quadrature_spread", 1e-2, quad_spread, quad_spread < 1e-2,
              "Maruo kernel spread over n_theta = 128 ... 4096 on the exact field"),
    ]
    return {
        "case": "CASE-BEM-MARUO-DRIFT",
        "generated_by": "tensorbem.benchmarks.maruo_drift_far_field_case.merge_studies",
        "parameters": merged_pars,
        "cells": cells,
        "reference": [closed_form_case(ka, density=density, gravity=gravity) for ka in ka_values],
        "cross_check_closed_form_vs_quadrature": cross_rows,
        "long_wave_anchors": anchors,
        "quadrature_sensitivity": quad_sens,
        "production_worst_rel_error": worst,
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ka", type=float, nargs="+", default=list(DEFAULT_KA))
    parser.add_argument("--resolutions", nargs="+", default=None,
                        help="subset of resolution names (default: all)")
    parser.add_argument("--production", default="fine")
    parser.add_argument("--n-azimuth", type=int, default=96)
    parser.add_argument("--n-theta-quad", type=int, default=2048)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--merge", type=Path, nargs="+", default=None,
                        help="merge per-shard study JSONs instead of running BEM")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.merge:
        study = merge_studies([json.loads(path.read_text()) for path in args.merge])
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(study, indent=2) + "\n")
            print(f"[case] wrote {args.json}")
        for gate in study["gates"]:
            print(f"[gate] {gate['name']:34s} measured={gate['measured']:.6g} "
                  f"passed={gate['passed']}")
        return 0 if study["all_passed"] else 1
    resolutions = RESOLUTIONS
    if args.resolutions:
        wanted = set(args.resolutions)
        resolutions = tuple(r for r in RESOLUTIONS if r[0] in wanted)
        missing = wanted - {r[0] for r in resolutions}
        if missing:
            raise SystemExit(f"unknown resolutions: {sorted(missing)}")
    study = run_study(
        ka_values=args.ka, resolutions=resolutions, production=args.production,
        n_azimuth=args.n_azimuth, n_theta_quad=args.n_theta_quad,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(study, indent=2) + "\n")
        print(f"[case] wrote {args.json}")
    for gate in study["gates"]:
        print(f"[gate] {gate['name']:34s} measured={gate['measured']:.6g} "
              f"threshold={gate['threshold']:g} passed={gate['passed']}")
    if args.figures:
        from .maruo_drift_far_field_case_figures import plot_study

        paths = plot_study(study, args.figures)
        print(f"[case] figures: {[str(p) for p in paths]}")
    return 0 if study["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())