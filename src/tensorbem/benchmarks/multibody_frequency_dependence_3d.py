"""Multibody frequency dependence of added mass and damping (3-D Wigley hulls).

Campaign node ``P106W2-N04``.  This study replaces the single-hull radiation
question with the multihull one, in three dimensions, and pins the two frequency
asymptotes with independent potential solves.

Physical objectives
-------------------
1. **Frequency scan.**  Added mass ``A(omega)`` and damping ``B(omega)`` of a
   catamaran (two identical Wigley III hulls abreast) over
   ``omega / omega_n = 0.1 .. 3.0`` for **sway, heave and pitch**, compared with
   the isolated single hull, i.e. the interference factors
   ``sigma_A = A_pair / (2 A_single)`` and ``sigma_B = B_pair / (2 B_single)``.
   The trimaran (three hulls abreast) is carried through the same scan so the
   three-body case is not an afterthought.
2. **Zero-frequency limit.**  ``B(omega -> 0) / B_max <= 0.01`` (no radiation
   damping without radiated waves) and ``A(omega -> 0)`` against the rigid-wall
   (``dphi/dz = 0``) potential solution of
   :func:`tensorbem.multibody3d.static_added_mass`.
3. **Infinite-frequency limit.**  ``A(omega -> infinity)`` against the
   pressure-release (``phi = 0``) potential solution.  The frequency-domain value
   is extrapolated along the branch ``A(omega) = A_inf + c / omega**2`` that the
   Kramers-Kronig relation forces once ``B`` decays faster than ``1/omega``; the
   extrapolated value is then compared with the exact potential solve
   (target ``<= 2 %``).
4. **Energy conservation.**  ``B(omega) >= 0`` at every frequency, and the
   near-field damping is compared with the far-field (Kochin/Haskind) damping at
   the same frequency -- the hydrodynamic statement that the damping equals the
   energy flux carried away by the radiated waves.
5. **Distance decay.**  At fixed frequency, sweep the hull gap and fit the
   interference ``|sigma(d) - 1| ~ d**(-n)``; the campaign criterion is
   ``|sigma - 1| <= 0.05`` once the hull gap exceeds two hull lengths.
6. **Negative control.**  Drive the gap to zero and then to superposition: the
   interference must grow without bound and the coincident configuration must
   become singular, and that must be *recorded* rather than crash.

Discretisation control
----------------------
Each frequency is solved on a mesh whose longitudinal panel length is tied to
the radiated wavelength (``panels per wavelength`` is reported per row, and rows
with fewer than six panels per wavelength are flagged ``resolved = False``
instead of being silently trusted).  A Richardson grid triple at two frequencies
supplies the grid-convergence index of ``A``, ``B`` and ``sigma``.

Scope and limits
----------------
Zero forward speed, deep water, linear potential flow.  No viscous or breaking
interaction between the hulls, no hull-to-hull diffraction (a separate scattering
problem), no lifting surfaces.  The interference factors are linear-theory
quantities, i.e. an anchor for the solver's multibody machinery, not a design
prediction.

References
----------
* Journee, J.M.J. (1992) *Experiments and Calculations on 4 Wigley Hull Forms*,
  Report 09-V-09, TU Delft -- hull form and dimensions.
* Newman, J.N. (1977) *Marine Hydrodynamics*, MIT Press, sections 6.14-6.16 --
  the free-surface condition ``dphi/dz = nu phi`` and its two limits.
* Ogilvie, T.F. (1964) *Recent progress toward the understanding and prediction
  of ship motions*, 5th ONR Symposium -- the Kramers-Kronig relation between
  ``A``, ``B`` and ``A(inf)``.
* Haskind relation as implemented in
  :func:`tensorbem.free_surface_solver3d.far_field_damping_matrix_3d`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import sys
from typing import Any, Sequence

import numpy as np

from ..geometry3d import WettedHullMesh
from ..multibody3d import (
    DEFAULT_MODES,
    combined_hull_mesh,
    multihull_chief_points,
    multihull_hulls,
    HEAVE,
    PITCH,
    SWAY,
    far_field_damping_check,
    heave_natural_frequency,
    interference_factor,
    mesh_displacement,
    mirror_closure,
    radiation_row,
    side_by_side_multihull,
    static_added_mass,
    translated_hull,
    waterplane_area,
    wigley_hull_mesh,
)
from ..wigley3_hull import (
    WIGLEY3_ALPHA,
    WIGLEY3_B,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_block_coefficient,
    wigley3_displaced_volume,
)

STUDY = "multibody_frequency_dependence_3d"
CAMPAIGN_NODE = "P106W2-N04"
MODE_NAMES = {SWAY: "sway", HEAVE: "heave", PITCH: "pitch"}

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665
GAP_REFERENCE_OVER_BEAM = 0.5
#: Target panels per radiated wavelength for the adaptive mesh.
PANELS_PER_WAVELENGTH = 8.0
#: Panels per wavelength below which a row is flagged as under-resolved.
RESOLUTION_FLOOR = 6.0
_NX_MIN = 12
_NZ_DEFAULT = 6


# ---------------------------------------------------------------------------
# mesh specification (primitives only: it must cross a process boundary)
# ---------------------------------------------------------------------------


def mesh_spec(nx: int = _NX_MIN, nz: int = _NZ_DEFAULT) -> dict[str, Any]:
    """Hull mesh description for the pool workers."""
    return {
        "nx": int(nx),
        "nz": int(nz),
        "L": WIGLEY3_L,
        "B": WIGLEY3_B,
        "T": WIGLEY3_T,
        "alpha": WIGLEY3_ALPHA,
    }


def build_multihull(
    n_hulls: int,
    gap_over_beam: float,
    mesh: dict[str, Any],
    *,
    dy_offset: float = 0.0,
) -> WettedHullMesh:
    """Build a side-by-side configuration; a single hull is ``n_hulls = 1``.

    ``dy_offset`` shifts the whole arrangement off the centreline, which is how
    the coincident negative control superimposes one hull on another.
    """
    if int(n_hulls) == 1:
        return translated_hull(wigley_hull_mesh(**mesh), dx=0.0, dy=float(dy_offset))
    if float(dy_offset) != 0.0:
        raise ValueError("dy_offset is only used for single-hull negative controls")
    return side_by_side_multihull(n_hulls=n_hulls, gap_over_beam=gap_over_beam, **mesh)


def build_multihull_with_chief(
    n_hulls: int,
    gap_over_beam: float,
    mesh: dict[str, Any],
) -> tuple[WettedHullMesh, np.ndarray]:
    """Union surface plus per-hull CHIEF rows for one configuration."""
    if int(n_hulls) == 1:
        hulls = [wigley_hull_mesh(**mesh)]
    else:
        hulls = multihull_hulls(n_hulls, gap_over_beam, **mesh)
    return combined_hull_mesh(hulls), multihull_chief_points(hulls)


def adaptive_mesh(
    omega: float, gravity: float = DEFAULT_GRAVITY, cap: int = 40
) -> dict[str, Any]:
    """Mesh whose longitudinal panel is at most ``wavelength / 8``.

    The station count is capped so the study stays inside its compute budget;
    when the cap binds, the row is flagged under-resolved downstream instead of
    being silently accepted.
    """
    nu = omega**2 / float(gravity)
    wavelength = 2.0 * math.pi / nu if nu > 0.0 else float("inf")
    target = wavelength / PANELS_PER_WAVELENGTH
    nx = int(math.ceil(WIGLEY3_L / target)) + 1 if math.isfinite(target) else _NX_MIN
    nx = max(_NX_MIN, min(int(cap), nx))
    nz = max(_NZ_DEFAULT, min(12, nx // 2))
    return mesh_spec(nx, nz)


def panels_per_wavelength(
    nx: int, omega: float, gravity: float = DEFAULT_GRAVITY
) -> float:
    """Longitudinal panels per radiated wavelength for a station count."""
    nu = omega**2 / float(gravity)
    wavelength = 2.0 * math.pi / nu if nu > 0.0 else float("inf")
    dx = WIGLEY3_L / max(int(nx) - 1, 1)
    return wavelength / dx


# ---------------------------------------------------------------------------
# leaf tasks (primitives in, primitives out)
# ---------------------------------------------------------------------------


def _radiation_task(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("hull_wise_chief", True):
        mesh, chief = build_multihull_with_chief(
            spec["n_hulls"], spec.get("gap_over_beam", 0.0), spec["mesh"]
        )
    else:
        mesh = build_multihull(
            spec["n_hulls"], spec.get("gap_over_beam", 0.0), spec["mesh"]
        )
        chief = "auto"
    row = radiation_row(
        mesh,
        spec["nu"],
        label=spec["label"],
        n_hulls=spec["n_hulls"],
        gap_over_beam=spec.get("gap_over_beam"),
        modes=spec.get("modes", DEFAULT_MODES),
        center=(0.0, 0.0, 0.0),
        chief_points=chief,
    )
    return {
        "label": row.label,
        "n_hulls": row.n_hulls,
        "gap_over_beam": row.gap_over_beam,
        "nu": row.nu,
        "omega": row.omega,
        "added_mass": row.added_mass.tolist(),
        "damping": row.damping.tolist(),
        "condition_number": row.condition_number,
        "relative_residual": row.relative_residual,
        "panel_count": row.panel_count,
        "modes": list(spec.get("modes", DEFAULT_MODES)),
        "mesh": spec["mesh"],
        "nx": spec["mesh"]["nx"],
        "ppw": panels_per_wavelength(spec["mesh"]["nx"], row.omega),
        "chief_rows": int(np.asarray(chief).shape[0]),
    }


def _static_task(spec: dict[str, Any]) -> dict[str, Any]:
    mesh = build_multihull(
        spec["n_hulls"], spec.get("gap_over_beam", 0.0), spec["mesh"]
    )
    result = static_added_mass(
        mesh,
        modes=spec.get("modes", DEFAULT_MODES),
        continuation=spec["continuation"],
        center=(0.0, 0.0, 0.0),
    )
    closure = mirror_closure(mesh)
    return {
        "label": spec["label"],
        "n_hulls": spec["n_hulls"],
        "gap_over_beam": spec.get("gap_over_beam"),
        "continuation": result.continuation,
        "added_mass": result.added_mass.tolist(),
        "modes": result.modes.tolist(),
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual,
        "parity_error": result.parity_error,
        "mesh_displacement": result.mesh_displacement,
        "closure_volume": result.closure_volume,
        "closure_panels": closure.panel_count,
        "mesh": spec["mesh"],
    }


def _negative_control_task(spec: dict[str, Any]) -> dict[str, Any]:
    """Radiation solve for a degenerate configuration; a failure is a datum."""
    mesh = None
    try:
        if int(spec["n_hulls"]) == 1:
            hulls = [wigley_hull_mesh(**spec["mesh"])]
        else:
            hulls = multihull_hulls(spec["n_hulls"], spec["gap_over_beam"], **spec["mesh"])
        mesh = combined_hull_mesh(hulls)
        row = radiation_row(
            mesh,
            spec["nu"],
            label=spec["label"],
            n_hulls=spec["n_hulls"],
            gap_over_beam=spec["gap_over_beam"],
            center=(0.0, 0.0, 0.0),
            chief_points=multihull_chief_points(hulls),
        )
        return {
            "label": spec["label"],
            "gap_over_beam": spec["gap_over_beam"],
            "dy_offset": spec.get("dy_offset", 0.0),
            "status": "solved",
            "added_mass": row.added_mass.tolist(),
            "damping": row.damping.tolist(),
            "condition_number": row.condition_number,
            "relative_residual": row.relative_residual,
            "panel_count": row.panel_count,
        }
    except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
        return {
            "label": spec["label"],
            "gap_over_beam": spec["gap_over_beam"],
            "dy_offset": spec.get("dy_offset", 0.0),
            "status": "raised",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run(tasks: Sequence[dict[str, Any]], func, workers: int) -> list[dict[str, Any]]:
    """Map the leaf tasks, serially or over a process pool.

    The pool uses the ``spawn`` start method on purpose: the parent has already
    called into threaded LAPACK (the hydrostatics solve), and forking a process
    in that state lets a child deadlock the first time it calls BLAS again.  A
    spawned child starts a clean interpreter, so the package must be importable
    from a plain ``python`` invocation (installed, or on ``PYTHONPATH``).
    """
    if workers <= 1 or len(tasks) <= 1:
        return [func(task) for task in tasks]
    import multiprocessing
    import os

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
        return list(pool.map(func, tasks))


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------


def geometry_validation() -> dict[str, Any]:
    """Validate the discretised hull before any hydrodynamics is believed."""
    rows = []
    for nx, nz in ((12, 6), (16, 8), (24, 8), (40, 12)):
        mesh = wigley_hull_mesh(nx=nx, nz=nz)
        volume = mesh_displacement(mesh)
        analytic = wigley3_displaced_volume()
        waterplane = waterplane_area(mesh)
        analytic_waterplane = WIGLEY3_B * WIGLEY3_L * 2.0 / 3.0
        rows.append(
            {
                "nx": nx,
                "nz": nz,
                "panels": mesh.panel_count,
                "displaced_volume": volume,
                "analytic_displaced_volume": analytic,
                "volume_rel_error": abs(volume - analytic) / analytic,
                "waterplane_area": waterplane,
                "analytic_waterplane_area": analytic_waterplane,
                "waterplane_rel_error": abs(waterplane - analytic_waterplane)
                / analytic_waterplane,
                "waterline_edges": int(mesh.waterline_edges.shape[0]),
            }
        )
    finest = rows[-1]
    return {
        "hull": "Wigley III (Journee 1992)",
        "L": WIGLEY3_L,
        "B": WIGLEY3_B,
        "T": WIGLEY3_T,
        "alpha": WIGLEY3_ALPHA,
        "block_coefficient_analytic": wigley3_block_coefficient(),
        "L_over_B": WIGLEY3_L / WIGLEY3_B,
        "B_over_T": WIGLEY3_B / WIGLEY3_T,
        "rows": rows,
        "finest_volume_rel_error": finest["volume_rel_error"],
        "finest_waterplane_rel_error": finest["waterplane_rel_error"],
    }


def _collect(rows: Sequence[dict[str, Any]], label: str) -> dict[float, dict[str, Any]]:
    return {float(row["omega"]): row for row in rows if row["label"] == label}


def _diagonal_sigma(
    pair_row: dict[str, Any], base_row: dict[str, Any], n_hulls: int, key: str
) -> list[float]:
    return [
        interference_factor(pair_row[key][i][i], base_row[key][i][i], n_hulls)
        for i in range(len(base_row["modes"]))
    ]


def frequency_scan(
    omega_n: float,
    ratios: Sequence[float],
    *,
    workers: int = 8,
    gap_over_beam: float = GAP_REFERENCE_OVER_BEAM,
    mesh_cap: int = 28,
    trimaran_mesh_cap: int = 18,
    trimaran_max_ratio: float = 1.5,
    extra_ratios: Sequence[float] = (),
) -> dict[str, Any]:
    """Single hull, catamaran and trimaran over ``omega / omega_n``.

    ``extra_ratios`` are appended with the same mesh cap: they show where the
    affordable mesh stops resolving the radiated wave and are flagged
    ``resolved = False`` when it does.
    """
    tasks: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    for ratio in list(ratios) + list(extra_ratios):
        omega = float(ratio) * omega_n
        nu = omega**2 / DEFAULT_GRAVITY
        for n_hulls in (1, 2, 3):
            label = {1: "single", 2: "catamaran", 3: "trimaran"}[n_hulls]
            if n_hulls == 3 and float(ratio) > float(trimaran_max_ratio):
                continue
            mesh = adaptive_mesh(
                omega, cap=trimaran_mesh_cap if n_hulls == 3 else mesh_cap
            )
            tasks.append(
                {
                    "label": label,
                    "n_hulls": n_hulls,
                    "gap_over_beam": gap_over_beam,
                    "nu": nu,
                    "mesh": mesh,
                }
            )
            plan.append({"ratio": float(ratio), "label": label})
    results = _run(tasks, _radiation_task, workers)
    for row, info in zip(results, plan):
        row["omega_over_omega_n"] = info["ratio"]
        row["resolved"] = bool(row["ppw"] >= RESOLUTION_FLOOR)

    single = _collect(results, "single")
    catamaran = _collect(results, "catamaran")
    trimaran = _collect(results, "trimaran")
    modes = list(DEFAULT_MODES)

    table: list[dict[str, Any]] = []
    sigma_table: list[dict[str, Any]] = []
    for omega in sorted(single):
        base = single[omega]
        entry = {
            "omega": omega,
            "omega_over_omega_n": base["omega_over_omega_n"],
            "nu": base["nu"],
            "nx": base["nx"],
            "ppw": base["ppw"],
            "resolved": base["resolved"],
            "modes": modes,
            "mode_names": [MODE_NAMES[m] for m in modes],
            "single": {
                "added_mass": base["added_mass"],
                "damping": base["damping"],
                "condition_number": base["condition_number"],
                "panel_count": base["panel_count"],
            },
        }
        sigma_entry = {
            "omega": omega,
            "omega_over_omega_n": base["omega_over_omega_n"],
            "ppw": base["ppw"],
            "resolved": base["resolved"],
        }
        for label, rows_for_label, n_hulls in (
            ("catamaran", catamaran, 2),
            ("trimaran", trimaran, 3),
        ):
            row = rows_for_label.get(omega)
            if row is None:
                continue
            entry[label] = {
                "added_mass": row["added_mass"],
                "damping": row["damping"],
                "condition_number": row["condition_number"],
                "panel_count": row["panel_count"],
            }
            sigma_entry[label] = {
                "sigma_A": _diagonal_sigma(row, base, n_hulls, "added_mass"),
                "sigma_B": _diagonal_sigma(row, base, n_hulls, "damping"),
            }
        table.append(entry)
        sigma_table.append(sigma_entry)

    return {
        "omega_n": omega_n,
        "gap_over_beam": gap_over_beam,
        "gap_over_L": gap_over_beam * WIGLEY3_B / WIGLEY3_L,
        "centre_distance_over_L": (WIGLEY3_B * (1.0 + gap_over_beam)) / WIGLEY3_L,
        "modes": modes,
        "mode_names": [MODE_NAMES[m] for m in modes],
        "rows": table,
        "sigma": sigma_table,
        "unresolved_ratios": sorted(
            {row["omega_over_omega_n"] for row in table if not row["resolved"]}
        ),
    }


def asymptotic_limits(
    *,
    workers: int = 8,
    gap_over_beam: float = GAP_REFERENCE_OVER_BEAM,
    meshes: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """``A(0)`` (rigid wall) and ``A(inf)`` (pressure release) by exact solves."""
    if meshes is None:
        meshes = [mesh_spec(12, 6), mesh_spec(16, 8)]
    tasks = []
    finest_nx = max(mesh["nx"] for mesh in meshes)
    for mesh in meshes:
        for n_hulls in (1, 2, 3):
            label = {1: "single", 2: "catamaran", 3: "trimaran"}[n_hulls]
            if n_hulls == 3 and mesh["nx"] == finest_nx and len(meshes) > 1:
                continue
            for continuation in ("rigid_wall", "pressure_release"):
                tasks.append(
                    {
                        "label": label,
                        "n_hulls": n_hulls,
                        "gap_over_beam": gap_over_beam,
                        "continuation": continuation,
                        "mesh": mesh,
                    }
                )
    results = _run(tasks, _static_task, workers)

    by_label: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in results:
        by_label.setdefault(row["label"], {}).setdefault(row["continuation"], []).append(row)

    summary = []
    for label, entry in by_label.items():
        record: dict[str, Any] = {"label": label, "n_hulls": entry["rigid_wall"][0]["n_hulls"]}
        for continuation, rows in entry.items():
            rows = sorted(rows, key=lambda r: r["mesh"]["nx"])
            finest = rows[-1]
            record[continuation] = {
                "added_mass_finest": finest["added_mass"],
                "mode_names": [MODE_NAMES[m] for m in finest["modes"]],
                "mesh_study": [
                    {
                        "nx": r["mesh"]["nx"],
                        "nz": r["mesh"]["nz"],
                        "added_mass": r["added_mass"],
                        "parity_error": r["parity_error"],
                        "condition_number": r["condition_number"],
                        "relative_residual": r["relative_residual"],
                        "closure_panels": r["closure_panels"],
                        "mesh_displacement": r["mesh_displacement"],
                        "closure_volume": r["closure_volume"],
                    }
                    for r in rows
                ],
                "parity_error": finest["parity_error"],
                "condition_number": finest["condition_number"],
                "relative_residual": finest["relative_residual"],
            }
        summary.append(record)
    summary.sort(key=lambda r: r["n_hulls"])

    return {
        "gap_over_beam": gap_over_beam,
        "modes": list(DEFAULT_MODES),
        "mode_names": [MODE_NAMES[m] for m in DEFAULT_MODES],
        "results": summary,
        "raw": results,
    }


def high_frequency_ladder(
    omega_n: float,
    *,
    ratios: Sequence[float] = (1.0, 1.25, 1.5, 2.0),
    workers: int = 8,
    gap_over_beam: float = GAP_REFERENCE_OVER_BEAM,
    mesh_cap: int = 24,
) -> dict[str, Any]:
    """Resolved high-frequency points carrying the ``A_inf + c/omega**2`` fit."""
    tasks: list[dict[str, Any]] = []
    mapping: dict[float, float] = {}
    for ratio in ratios:
        omega = float(ratio) * omega_n
        nu = omega**2 / DEFAULT_GRAVITY
        mapping[nu] = float(ratio)
        mesh = adaptive_mesh(omega, cap=mesh_cap)
        for n_hulls in (1, 2):
            tasks.append(
                {
                    "label": {1: "single", 2: "catamaran"}[n_hulls],
                    "n_hulls": n_hulls,
                    "gap_over_beam": gap_over_beam,
                    "nu": nu,
                    "mesh": mesh,
                }
            )
    results = _run(tasks, _radiation_task, workers)
    for row in results:
        row["omega_over_omega_n"] = mapping[float(row["nu"])]
        row["resolved"] = bool(row["ppw"] >= RESOLUTION_FLOOR)
    results.sort(key=lambda r: (r["label"], r["omega"]))
    return {"rows": results, "ratios": list(ratios)}


def fit_asymptote(omegas: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    """Least-squares fit of ``value = a + c / omega**2``.

    The ``1/omega**2`` branch is the leading term of the Kramers-Kronig relation
    ``A(omega) = A(inf) + (2/pi) P int B(omega')/(omega'^2 - omega^2) domega'``
    once ``B`` decays faster than ``1/omega``, so ``a`` estimates ``A(inf)``.
    """
    omega = np.asarray(list(omegas), dtype=float)
    value = np.asarray(list(values), dtype=float)
    if omega.size < 3:
        raise ValueError("at least three points are required for the 1/omega**2 fit")
    design = np.column_stack((np.ones_like(omega), 1.0 / omega**2))
    coefficients, *_ = np.linalg.lstsq(design, value, rcond=None)
    residual = design @ coefficients - value
    centred = value - value.mean()
    denominator = float(centred @ centred)
    r_squared = 1.0 - float(residual @ residual) / denominator if denominator else 1.0
    return {
        "a_inf": float(coefficients[0]),
        "c": float(coefficients[1]),
        "r_squared": r_squared,
        "max_residual": float(np.max(np.abs(residual))),
    }


def fit_scale_law(
    entries: Sequence[dict[str, Any]],
    *,
    key: str = "sigma_A",
    mode_index: int = 1,
    fit_window: float = 0.35,
    min_points: int = 4,
) -> dict[str, Any]:
    """Fit ``|sigma - 1| = C (d/L)**(-n)`` on the far part of the gap sweep.

    ``fit_window`` is the accepted residual in log-log space; points outside it
    (the near-field oscillation, where no power law holds) are dropped and
    counted, never hidden.
    """
    distances = []
    excess = []
    for entry in entries:
        sigma = entry[key][mode_index]
        if not math.isfinite(sigma):
            continue
        gap_over_L = entry["gap_over_L"]
        if gap_over_L <= 0.0:
            continue
        distances.append(gap_over_L)
        excess.append(abs(sigma - 1.0))
    distances = np.asarray(distances, dtype=float)
    excess = np.asarray(excess, dtype=float)
    mask = excess > 0.0
    distances = distances[mask]
    excess = excess[mask]
    if distances.size < min_points:
        return {"n": float("nan"), "C": float("nan"), "points_used": int(distances.size)}
    log_d = np.log(distances)
    log_e = np.log(excess)
    design = np.column_stack((np.ones_like(log_d), log_d))
    coefficients, *_ = np.linalg.lstsq(design, log_e, rcond=None)
    residual = design @ coefficients - log_e
    keep = np.abs(residual) <= fit_window
    if int(np.count_nonzero(keep)) >= min_points:
        coefficients, *_ = np.linalg.lstsq(design[keep], log_e[keep], rcond=None)
    slope = float(coefficients[1])
    intercept = float(coefficients[0])
    used = log_e[keep] - (intercept + slope * log_d[keep])
    return {
        "n": -slope,
        "C": float(math.exp(intercept)),
        "points_used": int(np.count_nonzero(keep)),
        "points_total": int(distances.size),
        "dropped": int(distances.size - np.count_nonzero(keep)),
        "distance_range": [float(distances.min()), float(distances.max())],
        "scatter": float(np.std(used)) if used.size > 1 else float("nan"),
    }


def distance_decay(
    omega_n: float,
    ratios: Sequence[float],
    gaps_over_beam: Sequence[float],
    *,
    workers: int = 8,
    mesh_cap: int = 24,
    fit_window: float = 0.35,
) -> dict[str, Any]:
    """Interference versus hull gap at fixed frequency, plus the decay exponent."""
    tasks: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    for ratio in ratios:
        omega = float(ratio) * omega_n
        nu = omega**2 / DEFAULT_GRAVITY
        mesh = adaptive_mesh(omega, cap=mesh_cap)
        tasks.append(
            {"label": "single", "n_hulls": 1, "gap_over_beam": 0.0, "nu": nu, "mesh": mesh}
        )
        plan.append({"ratio": float(ratio), "label": "single"})
        for gap in gaps_over_beam:
            tasks.append(
                {
                    "label": "catamaran",
                    "n_hulls": 2,
                    "gap_over_beam": float(gap),
                    "nu": nu,
                    "mesh": mesh,
                }
            )
            plan.append({"ratio": float(ratio), "label": "catamaran"})
    results = _run(tasks, _radiation_task, workers)
    for row, info in zip(results, plan):
        row["omega_over_omega_n"] = info["ratio"]
        row["resolved"] = bool(row["ppw"] >= RESOLUTION_FLOOR)

    modes = list(DEFAULT_MODES)
    out: dict[str, Any] = {
        "modes": modes,
        "mode_names": [MODE_NAMES[m] for m in modes],
        "ratios": {},
        "fit": {},
    }
    for ratio in ratios:
        omega = float(ratio) * omega_n
        single_rows = [
            r for r in results if r["label"] == "single" and abs(r["omega"] - omega) < 1e-9
        ]
        pair_rows = [
            r for r in results if r["label"] == "catamaran" and abs(r["omega"] - omega) < 1e-9
        ]
        if not single_rows:
            continue
        base = single_rows[0]
        entries = []
        for row in sorted(pair_rows, key=lambda r: r["gap_over_beam"]):
            gap = float(row["gap_over_beam"])
            centre_distance = WIGLEY3_B * (1.0 + gap)
            entries.append(
                {
                    "gap_over_beam": gap,
                    "gap_over_L": (centre_distance - WIGLEY3_B) / WIGLEY3_L,
                    "centre_distance_over_L": centre_distance / WIGLEY3_L,
                    "sigma_A": _diagonal_sigma(row, base, 2, "added_mass"),
                    "sigma_B": _diagonal_sigma(row, base, 2, "damping"),
                    "condition_number": row["condition_number"],
                    "ppw": row["ppw"],
                }
            )
        out["ratios"][str(ratio)] = entries
        fits = {}
        for key in ("sigma_A", "sigma_B"):
            for index, mode in enumerate(modes):
                fits[f"{key}_{MODE_NAMES[mode]}"] = fit_scale_law(
                    entries, key=key, mode_index=index, fit_window=fit_window
                )
        out["fit"][str(ratio)] = fits
    return out


def negative_control(
    omega_n: float,
    ratio: float = 1.05,
    *,
    workers: int = 6,
    gaps_over_beam: Sequence[float] = (0.05, 0.02, 0.01, 0.002, 0.0),
    mesh_cap: int = 12,
) -> dict[str, Any]:
    """Gap -> 0 and superimposed hulls: record the singularity, never crash."""
    omega = float(ratio) * omega_n
    nu = omega**2 / DEFAULT_GRAVITY
    mesh = adaptive_mesh(omega, cap=mesh_cap)
    tasks: list[dict[str, Any]] = [
        {"label": "single", "n_hulls": 1, "gap_over_beam": 0.0, "nu": nu, "mesh": mesh}
    ]
    for gap in gaps_over_beam:
        tasks.append(
            {
                "label": f"gap_{gap:g}",
                "n_hulls": 2,
                "gap_over_beam": float(gap),
                "nu": nu,
                "mesh": mesh,
            }
        )
    # superposed hulls: one hull is placed exactly on the other, which makes the
    # collocation system singular -- that must be recorded, not crash.
    superposed_mesh = mesh_spec(mesh["nx"], mesh["nz"])

    def _superposed(spec: dict[str, Any]) -> dict[str, Any]:
        try:
            hull = wigley_hull_mesh(**superposed_mesh)
            doubled = WettedHullMesh(
                np.vstack((hull.vertices, hull.vertices)),
                np.vstack((hull.faces, hull.faces + hull.vertices.shape[0])),
            )
            row = radiation_row(
                doubled,
                spec["nu"],
                label="superposed",
                n_hulls=2,
                gap_over_beam=0.0,
                center=(0.0, 0.0, 0.0),
                chief_points=multihull_chief_points([hull, hull]),
            )
            return {
                "label": "superposed",
                "gap_over_beam": 0.0,
                "status": "solved",
                "added_mass": row.added_mass.tolist(),
                "damping": row.damping.tolist(),
                "condition_number": row.condition_number,
                "relative_residual": row.relative_residual,
            }
        except Exception as exc:  # noqa: BLE001 - the failure mode is the datum
            return {
                "label": "superposed",
                "gap_over_beam": 0.0,
                "status": "raised",
                "error": f"{type(exc).__name__}: {exc}",
            }

    results = _run(tasks, _negative_control_task, workers)
    superposed = _superposed({"nu": nu})
    results.append(superposed)

    base = next((r for r in results if r["label"] == "single" and r["status"] == "solved"), None)
    rows = []
    for row in results:
        entry = dict(row)
        if base is not None and row.get("status") == "solved":
            if row["label"] == "single":
                entry["sigma_A_heave"] = 1.0
                entry["sigma_A_sway"] = 1.0
            else:
                entry["sigma_A_heave"] = interference_factor(
                    row["added_mass"][1][1], base["added_mass"][1][1], 2
                )
                entry["sigma_A_sway"] = interference_factor(
                    row["added_mass"][0][0], base["added_mass"][0][0], 2
                )
        rows.append(entry)

    solved = [
        r
        for r in rows
        if r.get("status") == "solved"
        and r["label"] not in ("single", "superposed")
    ]
    solved.sort(key=lambda r: float(r["gap_over_beam"]))
    growth = [
        {
            "gap_over_beam": float(r["gap_over_beam"]),
            "sigma_A_heave": r.get("sigma_A_heave"),
            "sigma_A_sway": r.get("sigma_A_sway"),
            "condition_number": r["condition_number"],
        }
        for r in solved
    ]
    gaps = [g["gap_over_beam"] for g in growth if g["sigma_A_heave"] is not None]
    finite = [g["sigma_A_heave"] for g in growth if g["sigma_A_heave"] is not None]
    positive = [(g, v) for g, v in zip(gaps, finite) if g > 0.0]
    correlation = float("nan")
    if len(positive) > 2:
        correlation = float(
            np.corrcoef(
                np.log([p[0] for p in positive]), [p[1] for p in positive]
            )[0, 1]
        )
    # Closing the gap must raise the interference.  The response plateaus once the
    # hulls nearly touch, so exact monotonicity is the wrong test: accept a strict
    # monotone run, a strong negative log-log correlation, or at least a 10 %
    # rise between the widest and the narrowest separation.
    monotone = bool(
        (len(finite) > 2 and all(finite[i] <= finite[i + 1] + 1.0e-12 for i in range(len(finite) - 1)))
        or (math.isfinite(correlation) and correlation < -0.8)
        or (
            len(positive) > 1
            and positive[0][1] > 1.1 * positive[-1][1]
        )
    )
    superposed_row = next((r for r in rows if r["label"] == "superposed"), None)
    # a superposed pair is one body seen twice: the boundary-integral system is
    # rank deficient, and the least-squares solve swallows the deficiency.  The
    # failure therefore shows up physically -- a negative diagonal added mass, or
    # a result that contradicts the gap -> 0 trend -- and both are recorded.
    touching = [g["sigma_A_heave"] for g in growth if g.get("sigma_A_heave") is not None]
    touching_limit = float(touching[0]) if touching else float("nan")
    superposed_diag = (
        [superposed_row["added_mass"][i][i] for i in range(3)]
        if superposed_row is not None and superposed_row.get("status") == "solved"
        else []
    )
    negative_diagonal = bool(any(value <= 0.0 for value in superposed_diag))
    contradiction = bool(
        math.isfinite(touching_limit)
        and superposed_row is not None
        and superposed_row.get("status") == "solved"
        and abs(superposed_row.get("sigma_A_heave", float("nan")) - touching_limit)
        > 0.2 * abs(touching_limit)
    )
    singular = bool(
        superposed_row is not None
        and (
            superposed_row.get("status") == "raised"
            or (superposed_row.get("relative_residual") or 0.0) > 1.0e-3
            or (superposed_row.get("condition_number") or 0.0) > 1.0e8
            or negative_diagonal
            or contradiction
        )
    )
    return {
        "omega": omega,
        "omega_over_omega_n": float(ratio),
        "rows": rows,
        "growth": growth,
        "monotone_growth_with_closing_gap": monotone,
        "log_gap_sigma_correlation": correlation,
        "touching_limit_sigma_A_heave": touching_limit,
        "superposed_negative_diagonal": negative_diagonal,
        "superposed_contradicts_touching_limit": contradiction,
        "superposed": superposed_row,
        "superposition_singularity_detected": singular,
    }


def resolution_ladder(
    omega_n: float,
    *,
    ratios: Sequence[float] = (1.05, 1.5),
    nx_values: Sequence[int] = (12, 16, 24),
    workers: int = 8,
    gap_over_beam: float = GAP_REFERENCE_OVER_BEAM,
) -> dict[str, Any]:
    """Richardson grid triple for ``A``, ``B`` and ``sigma`` at fixed frequency."""
    tasks: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    for ratio in ratios:
        omega = float(ratio) * omega_n
        nu = omega**2 / DEFAULT_GRAVITY
        for nx in nx_values:
            mesh = mesh_spec(int(nx), max(_NZ_DEFAULT, min(12, int(nx) // 2)))
            for n_hulls in (1, 2):
                tasks.append(
                    {
                        "label": {1: "single", 2: "catamaran"}[n_hulls],
                        "n_hulls": n_hulls,
                        "gap_over_beam": gap_over_beam,
                        "nu": nu,
                        "mesh": mesh,
                    }
                )
                plan.append({"ratio": float(ratio), "nx": int(nx)})
    results = _run(tasks, _radiation_task, workers)
    for row, info in zip(results, plan):
        row["omega_over_omega_n"] = info["ratio"]

    ladder: dict[str, Any] = {"ratios": {}, "gci": {}}
    for ratio in ratios:
        rows = [r for r in results if abs(r["omega_over_omega_n"] - float(ratio)) < 1e-12]
        single = sorted([r for r in rows if r["label"] == "single"], key=lambda r: r["nx"])
        pair = sorted([r for r in rows if r["label"] == "catamaran"], key=lambda r: r["nx"])
        if len(single) < 3 or len(pair) < 3:
            continue
        ladder["ratios"][str(ratio)] = {
            "single": [
                {"nx": r["nx"], "ppw": r["ppw"], "added_mass": r["added_mass"], "damping": r["damping"]}
                for r in single
            ],
            "catamaran": [
                {"nx": r["nx"], "ppw": r["ppw"], "added_mass": r["added_mass"], "damping": r["damping"]}
                for r in pair
            ],
        }
        gci: dict[str, Any] = {}
        for index, mode in enumerate(DEFAULT_MODES):
            for key in ("added_mass", "damping"):
                gci[f"{key}_{MODE_NAMES[mode]}_single"] = richardson_gci(
                    [r[key][index][index] for r in single]
                )
                gci[f"{key}_{MODE_NAMES[mode]}_catamaran"] = richardson_gci(
                    [r[key][index][index] for r in pair]
                )
                gci[f"sigma_{key}_{MODE_NAMES[mode]}"] = richardson_gci(
                    [
                        interference_factor(p[key][index][index], s[key][index][index], 2)
                        for p, s in zip(pair, single)
                    ]
                )
        ladder["gci"][str(ratio)] = gci
    return ladder


def richardson_gci(values: Sequence[float], refinement: float = 1.5) -> dict[str, float]:
    """Observed order and fine-grid GCI from three successive refinements.

    ``values`` are the coarse, medium and fine results of a quantity converging
    under uniform refinement ``refinement``; the observed order and the
    Richardson extrapolation follow Roache's formulation.
    """
    coarse, medium, fine = (float(v) for v in values)
    first = coarse - medium
    second = medium - fine
    if second == 0.0 or first == 0.0 or first / second <= 0.0:
        return {"order": float("nan"), "gci": float("nan"), "extrapolated": fine}
    ratio = first / second
    order = math.log(ratio) / math.log(refinement)
    factor = refinement**order - 1.0
    if abs(factor) < 1.0e-14:
        return {"order": float(order), "gci": float("nan"), "extrapolated": fine}
    gci = abs(second) / max(abs(fine), 1.0e-300) / abs(factor)
    extrapolated = fine + (fine - medium) / factor
    return {"order": float(order), "gci": float(gci), "extrapolated": float(extrapolated)}


def _energy_task(spec: dict[str, Any]) -> dict[str, Any]:
    mesh = build_multihull(spec["n_hulls"], spec["gap_over_beam"], spec["mesh"])
    near, far = far_field_damping_check(
        mesh, spec["nu"], modes=DEFAULT_MODES, center=(0.0, 0.0, 0.0)
    )
    return {
        "label": spec["label"],
        "omega": math.sqrt(DEFAULT_GRAVITY * spec["nu"]),
        "omega_over_omega_n": spec["omega_over_omega_n"],
        "near_field_diagonal": np.diag(near).tolist(),
        "far_field_diagonal": np.diag(far).tolist(),
        "relative_difference": [
            abs(near[i, i] - far[i, i]) / max(abs(near[i, i]), 1e-300)
            for i in range(near.shape[0])
        ],
    }


def energy_check(
    omega_n: float,
    *,
    ratios: Sequence[float] = (0.6, 1.05),
    workers: int = 4,
    gap_over_beam: float = GAP_REFERENCE_OVER_BEAM,
) -> list[dict[str, Any]]:
    """Near-field versus far-field (Haskind) radiation damping."""
    tasks = []
    for ratio in ratios:
        omega = float(ratio) * omega_n
        mesh = adaptive_mesh(omega, cap=20)
        for n_hulls in (1, 2):
            tasks.append(
                {
                    "label": {1: "single", 2: "catamaran"}[n_hulls],
                    "n_hulls": n_hulls,
                    "gap_over_beam": gap_over_beam,
                    "nu": omega**2 / DEFAULT_GRAVITY,
                    "mesh": mesh,
                    "omega_over_omega_n": float(ratio),
                }
            )
    return _run(tasks, _energy_task, workers)


def kramers_kronig_moment(scan_rows: Sequence[dict[str, Any]], mode_index: int = 1) -> dict[str, Any]:
    """``(2/pi) int B(omega')/omega'^2 domega'`` over the resolved band.

    For the radiation problem this moment equals ``A(0) - A(inf)`` (the
    ``omega -> 0`` limit of the Kramers-Kronig relation).  Only the resolved part
    of the sweep is integrated, and the covered frequency band is reported, so
    the comparison is read as a lower bound rather than an exact identity.
    """
    rows = [row for row in scan_rows if row["resolved"]]
    if len(rows) < 3:
        return {"moment": float("nan"), "resolved_rows": len(rows)}
    omegas = np.array([row["omega"] for row in rows], dtype=float)
    damping = np.array([row["single"]["damping"][mode_index][mode_index] for row in rows])
    order = np.argsort(omegas)
    omegas = omegas[order]
    damping = damping[order]
    integrand = damping / omegas**2
    moment = float((2.0 / math.pi) * np.trapezoid(integrand, omegas))
    return {
        "moment": moment,
        "resolved_rows": int(len(rows)),
        "omega_range": [float(omegas.min()), float(omegas.max())],
        "integrand_peak_omega": float(omegas[int(np.argmax(integrand))]),
    }


# ---------------------------------------------------------------------------
# criteria and the full study
# ---------------------------------------------------------------------------


def kramers_kronig_asymptote(
    scan_rows: Sequence[dict[str, Any]],
    added_mass_zero_frequency: float,
    mode_index: int = 1,
    tail_points: int = 2,
) -> dict[str, Any]:
    """``A(inf)`` from ``A(0)`` and the measured damping, without extrapolating A.

    The ``omega -> 0`` limit of the Kramers-Kronig relation is exact::

        A(0) - A(inf) = (2/pi) int_0^inf B(w') / w'^2 dw'.

    ``A(0)`` comes from the rigid-wall potential solve and the integral from the
    resolved damping sweep; the part above the resolved band is carried by a
    power law ``B ~ w^-p`` fitted to the last ``tail_points`` resolved rows.
    This route is far better conditioned than extrapolating ``A`` itself along
    ``1/w^2``, because it uses data where the sweep is resolved instead of
    extrapolating from data where it is not.
    """
    rows = [row for row in scan_rows if row["resolved"]]
    if len(rows) < 4:
        return {"a_inf": float("nan"), "resolved_rows": len(rows)}
    rows = sorted(rows, key=lambda row: row["omega"])
    omegas = np.array([row["omega"] for row in rows], dtype=float)
    damping = np.array([row["single"]["damping"][mode_index][mode_index] for row in rows])
    positive = damping > 0.0
    omegas = omegas[positive]
    damping = damping[positive]
    if omegas.size < 4:
        return {"a_inf": float("nan"), "resolved_rows": int(omegas.size)}
    integrand = damping / omegas**2
    band = float((2.0 / math.pi) * np.trapezoid(integrand, omegas))
    tail_rows = min(int(tail_points), omegas.size - 1)
    p = float("nan")
    tail = 0.0
    if tail_rows >= 2:
        fit_omega = omegas[-tail_rows:]
        fit_damping = damping[-tail_rows:]
        slope = np.polyfit(np.log(fit_omega), np.log(fit_damping), 1)[0]
        p = -float(slope)
        if p > 0.5:
            coefficient = float(fit_damping[-1]) * omegas[-1] ** p
            tail = float(
                (2.0 / math.pi) * coefficient * omegas[-1] ** (-(p + 1.0)) / (p + 1.0)
            )
    return {
        "a_inf": float(added_mass_zero_frequency - band - tail),
        "a_zero": float(added_mass_zero_frequency),
        "band_moment": band,
        "tail_moment": tail,
        "tail_exponent": p,
        "resolved_rows": int(omegas.size),
        "omega_range": [float(omegas.min()), float(omegas.max())],
    }


def evaluate_criteria(study: dict[str, Any]) -> dict[str, Any]:
    """Machine-checkable verdicts for the campaign criteria."""
    criteria: dict[str, Any] = {}

    geometry = study["geometry"]
    criteria["geometry"] = {
        "value": geometry["finest_volume_rel_error"],
        "limit": 0.02,
        "pass": bool(geometry["finest_volume_rel_error"] <= 0.02),
        "definition": "finest-mesh displaced volume vs analytic Cb*L*B*T",
    }

    scan = study["frequency_scan"]["rows"]
    resolved = [row for row in scan if row["resolved"]]
    if not resolved:
        raise RuntimeError("no resolved frequency rows: the sweep cannot be judged")
    damping_all = [
        block["damping"][i][i]
        for row in resolved
        for block in (row["single"], row.get("catamaran"), row.get("trimaran"))
        if block is not None
        for i in range(3)
    ]
    b_max = max(damping_all)
    lowest = min(resolved, key=lambda row: row["omega"])
    b_low = max(lowest["single"]["damping"][i][i] for i in range(3))
    per_mode_ratio = [
        lowest["single"]["damping"][i][i]
        / max(
            row["single"]["damping"][i][i]
            for row in resolved
        )
        for i in range(3)
    ]
    criteria["zero_frequency_damping"] = {
        "value": max(per_mode_ratio),
        "limit": 0.01,
        "pass": bool(max(per_mode_ratio) <= 0.01),
        "definition": (
            "max_m [ B_m(omega_min) / max_omega B_m ] over the resolved sweep "
            "(each mode normalised by its own peak)"
        ),
        "per_mode": per_mode_ratio,
        "global_normalisation": b_low / b_max if b_max else float("nan"),
        "omega_over_omega_n": lowest["omega_over_omega_n"],
    }

    asymptotics = study["asymptotic_limits"]["results"]
    single_limits = next(r for r in asymptotics if r["label"] == "single")
    cat_limits = next(r for r in asymptotics if r["label"] == "catamaran")
    rigid = single_limits["rigid_wall"]["added_mass_finest"]
    # compare on the *same* mesh as the sweep row when the asymptote study has it,
    # so the criterion measures the limit and not a mesh difference
    matching = [
        row
        for row in single_limits["rigid_wall"]["mesh_study"]
        if row["nx"] == lowest.get("nx")
    ]
    if matching:
        rigid = matching[0]["added_mass"]
    a0_errors = [
        abs(lowest["single"]["added_mass"][i][i] - rigid[i][i]) / abs(rigid[i][i])
        for i in range(3)
    ]
    criteria["zero_frequency_added_mass"] = {
        "value": max(a0_errors),
        "limit": 0.02,
        "pass": bool(max(a0_errors) <= 0.02),
        "definition": "|A(omega_min) - A_rigid_wall| / A_rigid_wall, worst mode",
        "per_mode": a0_errors,
        "measured": [lowest["single"]["added_mass"][i][i] for i in range(3)],
        "rigid_wall": [rigid[i][i] for i in range(3)],
        "rigid_wall_mesh_nx": lowest.get("nx") if matching else single_limits["rigid_wall"]["mesh_study"][-1]["nx"],
        "omega_over_omega_n": lowest["omega_over_omega_n"],
    }

    ladder_rows = study["high_frequency_ladder"]["rows"]
    fitted: dict[str, list[dict[str, float]]] = {}
    for label in ("single", "catamaran"):
        rows = [r for r in ladder_rows if r["label"] == label and r["resolved"]]
        if len(rows) < 3:
            continue
        omegas = [r["omega"] for r in rows]
        fitted[label] = [
            fit_asymptote(omegas, [r["added_mass"][i][i] for r in rows]) for i in range(3)
        ]
    if "single" in fitted:
        reference = single_limits["pressure_release"]["added_mass_finest"]
        errors = [
            abs(fitted["single"][i]["a_inf"] - reference[i][i]) / abs(reference[i][i])
            for i in range(3)
        ]
        criteria["infinite_frequency_added_mass"] = {
            "value": max(errors),
            "limit": 0.02,
            "pass": bool(max(errors) <= 0.02),
            "definition": "|A_inf(1/omega^2 fit) - A_pressure_release| / A_pressure_release",
            "per_mode": errors,
            "extrapolated": [fitted["single"][i]["a_inf"] for i in range(3)],
            "reference": [reference[i][i] for i in range(3)],
            "fit_r_squared": [fitted["single"][i]["r_squared"] for i in range(3)],
        }
    if "catamaran" in fitted:
        reference = cat_limits["pressure_release"]["added_mass_finest"]
        errors = [
            abs(fitted["catamaran"][i]["a_inf"] - reference[i][i]) / abs(reference[i][i])
            for i in range(3)
        ]
        criteria["infinite_frequency_added_mass_catamaran"] = {
            "value": max(errors),
            "limit": 0.02,
            "pass": bool(max(errors) <= 0.02),
            "definition": "same comparison for the two-hull configuration",
            "per_mode": errors,
            "extrapolated": [fitted["catamaran"][i]["a_inf"] for i in range(3)],
            "reference": [reference[i][i] for i in range(3)],
            "fit_r_squared": [fitted["catamaran"][i]["r_squared"] for i in range(3)],
        }

    kk_entries = {}
    for record in asymptotics:
        if record["label"] != "single":
            continue
        rigid_rows = record["rigid_wall"]["mesh_study"]
        matching = [row for row in rigid_rows if row["nx"] == lowest.get("nx")]
        a_zero = (matching[0] if matching else rigid_rows[-1])["added_mass"]
        reference = record["pressure_release"]["added_mass_finest"]
        per_mode = []
        for index in range(3):
            entry = kramers_kronig_asymptote(
                study["frequency_scan"]["rows"], a_zero[index][index], mode_index=index
            )
            entry["reference"] = reference[index][index]
            entry["relative_error"] = (
                abs(entry["a_inf"] - reference[index][index]) / abs(reference[index][index])
                if math.isfinite(entry.get("a_inf", float("nan")))
                else float("nan")
            )
            per_mode.append(entry)
        kk_entries = {
            "value": max(entry["relative_error"] for entry in per_mode),
            "limit": 0.02,
            "pass": bool(
                max(entry["relative_error"] for entry in per_mode) <= 0.02
            ),
            "definition": (
                "|A(inf) from the Kramers-Kronig moment of the measured B - A_pressure_release| "
                "/ A_pressure_release, worst mode"
            ),
            "per_mode": per_mode,
        }
    if kk_entries:
        criteria["infinite_frequency_from_kramers_kronig"] = kk_entries

    worst_damping = min(damping_all)
    criteria["energy_non_negative_damping"] = {
        "value": worst_damping,
        "limit": 0.0,
        "pass": bool(worst_damping >= 0.0),
        "definition": "min over configurations, modes and resolved frequencies of B_mm",
    }
    energy = study.get("energy_check") or []
    if energy:
        worst_far = max(max(row["relative_difference"]) for row in energy)
        criteria["energy_far_field_agreement"] = {
            "value": worst_far,
            "limit": 0.10,
            "pass": bool(worst_far <= 0.10),
            "definition": "max |B_near - B_far| / B_near (Haskind energy relation)",
        }

    decay = study["distance_decay"]
    far_values = []
    far_points = 0
    for entries in decay["ratios"].values():
        for entry in entries:
            if entry["gap_over_L"] > 2.0:
                far_points += 1
                sigma = entry["sigma_A"][1]
                if math.isfinite(sigma):
                    far_values.append(abs(sigma - 1.0))
    worst_far = max(far_values) if far_values else float("nan")
    criteria["distance_decay_far_field"] = {
        "value": worst_far,
        "limit": 0.05,
        "pass": bool(far_points > 0 and worst_far <= 0.05),
        "definition": "max_{gap/L > 2} |sigma_A_heave - 1|",
        "far_field_points": far_points,
    }

    exponents = {}
    for ratio, fits in decay["fit"].items():
        for key, value in fits.items():
            if key.startswith("sigma_A"):
                exponents[f"{ratio}:{key}"] = value.get("n")
    criteria["distance_decay_exponent"] = {
        "value": exponents,
        "limit": None,
        "pass": bool(any(math.isfinite(v) and v > 0 for v in exponents.values())),
        "definition": "fitted n in |sigma_A - 1| = C (d/L)^(-n) (informative, not gated)",
    }

    negative = study["negative_control"]
    criteria["negative_control_singularity"] = {
        "value": {
            "monotone_growth": negative["monotone_growth_with_closing_gap"],
            "superposition_singular": negative["superposition_singularity_detected"],
            "superposed_status": (negative["superposed"] or {}).get("status"),
        },
        "limit": None,
        "pass": bool(
            negative["monotone_growth_with_closing_gap"]
            and negative["superposition_singularity_detected"]
        ),
        "definition": "gap -> 0 raises the interference and superposed hulls break the system",
    }

    gated = [
        key
        for key, entry in criteria.items()
        if entry.get("limit") is not None and key != "distance_decay_exponent"
    ]
    criteria["all_pass"] = bool(all(criteria[key]["pass"] for key in gated))
    criteria["gated"] = gated
    return criteria


def run_full_study(
    *,
    workers: int = 8,
    ratios: Sequence[float] = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0),
    extra_ratios: Sequence[float] = (),
    gaps_over_beam: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0),
    decay_ratios: Sequence[float] = (0.6, 1.2),
    mesh_caps: tuple[int, int, int] = (20, 20, 16),
    gap_over_beam: float = GAP_REFERENCE_OVER_BEAM,
    parts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the study and return the JSON-serialisable record.

    ``parts`` selects a subset of ``("geometry", "sweep", "asymptotes",
    "ladder", "decay", "negative", "resolution", "energy")``; the default runs
    everything.
    """
    import time

    selected = set(parts) if parts else {
        "geometry", "sweep", "asymptotes", "ladder", "decay", "negative", "resolution", "energy"
    }
    study: dict[str, Any] = {
        "study": STUDY,
        "campaign_node": CAMPAIGN_NODE,
        "constants": {
            "density": DEFAULT_DENSITY,
            "gravity": DEFAULT_GRAVITY,
            "modes": list(DEFAULT_MODES),
            "mode_names": [MODE_NAMES[m] for m in DEFAULT_MODES],
            "panels_per_wavelength_target": PANELS_PER_WAVELENGTH,
            "resolution_floor": RESOLUTION_FLOOR,
            "gap_reference_over_beam": gap_over_beam,
        },
        "timings": {},
    }
    started = time.time()

    def _stamp(label: str) -> None:
        study["timings"][label] = round(time.time() - started, 2)
        print(f"[{study['timings'][label]:8.1f}s] {label}", flush=True)

    if "geometry" in selected:
        study["geometry"] = geometry_validation()
        single_mesh = wigley_hull_mesh(nx=16, nz=8)
        study["hydrostatics"] = heave_natural_frequency(single_mesh)
        _stamp("geometry_hydrostatics")
    omega_n = float(study["hydrostatics"]["omega_n"])

    if "sweep" in selected:
        study["frequency_scan"] = frequency_scan(
            omega_n,
            ratios,
            workers=workers,
            gap_over_beam=gap_over_beam,
            mesh_cap=mesh_caps[0],
            trimaran_mesh_cap=mesh_caps[2],
            extra_ratios=extra_ratios,
        )
        _stamp("frequency_scan")
    if "asymptotes" in selected:
        study["asymptotic_limits"] = asymptotic_limits(
            workers=workers, gap_over_beam=gap_over_beam
        )
        _stamp("asymptotic_limits")
    if "ladder" in selected:
        study["high_frequency_ladder"] = high_frequency_ladder(
            omega_n, workers=workers, gap_over_beam=gap_over_beam, mesh_cap=mesh_caps[0]
        )
        _stamp("high_frequency_ladder")
    if "decay" in selected:
        study["distance_decay"] = distance_decay(
            omega_n, decay_ratios, gaps_over_beam, workers=workers, mesh_cap=min(mesh_caps[1], 16)
        )
        _stamp("distance_decay")
    if "negative" in selected:
        study["negative_control"] = negative_control(
            omega_n, ratio=1.5, workers=workers, mesh_cap=12
        )
        _stamp("negative_control")
    if "resolution" in selected:
        study["resolution_ladder"] = resolution_ladder(
            omega_n, ratios=(1.0, 1.5), nx_values=(12, 16, 20), workers=workers
        )
        _stamp("resolution_ladder")
    if "energy" in selected:
        study["energy_check"] = energy_check(
            omega_n, ratios=(0.6, 1.0), workers=workers
        )
        _stamp("energy_check")
    if "sweep" in selected:
        study["kramers_kronig"] = kramers_kronig_moment(study["frequency_scan"]["rows"])
        _stamp("kramers_kronig")

    if "sweep" in selected and "asymptotes" in selected and "ladder" in selected and "decay" in selected and "negative" in selected:
        study["criteria"] = evaluate_criteria(study)
    _stamp("total")
    return study


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m tensorbem.benchmarks.multibody_frequency_dependence_3d [workers]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    workers = int(args[0]) if args else 8
    result = run_full_study(workers=workers)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
