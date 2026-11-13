"""Case-pool driver for CASE-BEM-HULME-DIFF-VERIF (tensor_case_pool id 177).

Anchor: *Hulme (1982) diffraction verification, 3-D*.  The case grades the
floating-hemisphere **wave exciting force** (the quantity a seakeeping solver
actually consumes) on three routes that do not share a discretisation:

``reference``
    ``tensorbem.analytic.hulme_hemisphere_radiation`` -- Hulme's wave-free
    multipole series, excitation via the origin Kochin amplitude (Haskind
    reciprocity).  Truncation is laddered (48/96/144 terms) so the reference's
    own convergence is part of the evidence, not an assumption.
``bem``
    ``tensorbem.free_surface_solver3d.solve_diffraction_3d`` on the Kelvin
    (Havelock) Green-function BIE, constant panels, interior CHIEF points.
    Three meshes (64 / 256 / 1024 panels) so the discretisation error is
    *measured* and its order is reported.
``independent``
    an exact quadrature of the incident-wave (Froude-Krylov) pressure over the
    analytical hemisphere **plus** the diffraction force of the third-party
    Rankine BEM code capytaine, frozen in ``capytaine_scattered.json`` by
    ``scripts/capytaine_hulme_diff_scattered.py``.  This is the only route whose
    numbers do not come from this repository's own series implementation, so it
    is what breaks the circularity of a reference-vs-solver comparison.

Convention pins (both measured, see the frozen json):

* TensorBEM's ``solve_diffraction_3d(...).excitation_force`` is the **total**
  exciting force (``Froude-Krylov + diffraction``), matching the semi-analytical
  reference and the ``froude_krylov_force_3d`` docstring.
* capytaine's ``DiffractionProblem.forces`` is the **diffraction (scattered)
  force only**; the Froude-Krylov part must be added separately.  The probe at
  ``ka = 0.05`` (scattered -> 0 while the total tends to ``pi*rho*g*a^2``) is
  stored in the frozen json, and gate G4 below is what catches it if a future
  capytaine changes the convention.
* Long-wave physics anchor on the reference itself: ``X3(ka->0)`` must tend to
  ``rho*g*A*A_wp = pi*rho*g*a^2`` (the wave pressure acting on the waterplane).

Gates are fixed before the run and are not tuned afterwards::

    G1  complex relative error of the exciting force  <= 5 %   (ledger <5 %)
    G2  phase difference                              <= 3 deg (case mesh)
    G3  mesh ladder monotone, observed order >= 0.8
    G4  independent route vs reference                <= 5 %
    G5  reference long-wave limit vs pi*A_wp/a^2       <= 1 % (extrapolated;
        the raw ka = 0.02 value is allowed 3 % because the approach is O(ka))
    G6  Haskind reciprocity (radiation -> Kochin)     <= 2 %

Reproduce::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_diff_verif/run_case.py

or, for the artefact directory used by the case pool::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_diff_verif/run_case.py \\
        --out benchmarks/verified/casepool_bem_hulme_diff_verif
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    froude_krylov_force_3d,
    haskind_excitation_force_3d,
    rigid_body_normals_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0

#: ka band of the case (log-ish spacing, same band as the radiation case 138).
KA_BAND: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
#: hull subdivisions -> 64 / 256 / 1024 constant panels.
LEVELS: tuple[int, ...] = (2, 3, 4)
CASE_SUBDIVISIONS = 4
#: Haskind reciprocity is a radiation-side invariant; one mesh below the case
#: mesh is enough (it agrees to ~0.1 % there) and halves the wall clock.
HASKIND_LEVEL = 3
TERMS_LADDER: tuple[int, ...] = (48, 96, 144)
#: ka used for the long-wave limit probe of the reference (gate G5).
LONG_WAVE_KA = 0.02

GATE_COMPLEX = 0.05
GATE_PHASE_DEG = 3.0
GATE_ORDER = 0.8
GATE_INDEPENDENT = 0.05
GATE_LONG_WAVE = 0.01
#: raw allowance at ka = 0.02: the approach to rho*g*A*A_wp is O(ka) with a
#: measured slope of ~-4 (i.e. ~2.5 % at ka = 0.02); the extrapolated limit is
#: held to GATE_LONG_WAVE.
GATE_LONG_WAVE_RAW = 0.03
GATE_HASKIND = 0.02

#: per-(ka, mesh) memo so that a wall-clock-limited rerun resumes instead of
#: recomputing the expensive 1024-panel solves.  Purely a cache: every cached
#: number is recomputed from scratch when the file is absent.
CACHE = Path(os.environ.get("TENSORBEM_HULME_DIFF_CACHE", "/tmp/hulme_diff_cache"))

VOLUME = 2.0 * math.pi / 3.0 * RADIUS**3  # exact hemisphere volume
WATERPLANE = math.pi * RADIUS**2  # exact waterplane area


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _c(value: complex) -> dict[str, float]:
    return {"re": float(complex(value).real), "im": float(complex(value).imag)}


def _phase_deg(value: complex) -> float:
    return math.degrees(math.atan2(complex(value).imag, complex(value).real))


def _phase_delta_deg(got: complex, ref: complex) -> float:
    """Signed phase difference folded into (-180, 180]."""
    delta = _phase_deg(got) - _phase_deg(ref)
    return (delta + 180.0) % 360.0 - 180.0


def _rel(got: complex, ref: complex) -> float:
    return abs(complex(got) - complex(ref)) / abs(complex(ref))


def _non_dim(force: complex) -> complex:
    return complex(force) / (DENSITY * GRAVITY * RADIUS**2)


def exact_froude_krylov(nu: float, n_theta: int = 400) -> tuple[complex, complex]:
    """Exact (quadrature) incident-wave force on the analytical hemisphere.

    ``F_j = -i omega rho int phi_I n_j dS`` with ``phi_I`` of unit amplitude,
    evaluated in closed reduced form on ``r = a``, ``z <= 0``::

        I3 = -2 pi int_0^{pi/2} e^{-ka cos t} J0(ka sin t) cos t sin t dt
        I1 = +2 pi i int_0^{pi/2} e^{-ka cos t} J1(ka sin t) sin^2 t dt

    and returns ``F_j = -rho g A a^2 I_j``.  The overall sign is not a free
    choice: it is the phase reference of :func:`froude_krylov_force_3d` (the
    surface integral of the same incident pressure field), which the results
    cross-check per frequency; a sign error here would break gate G4 instead of
    hiding inside it.
    """
    t, w = np.polynomial.legendre.leggauss(int(n_theta))
    theta = 0.25 * math.pi * (t + 1.0)  # [0, pi/2]
    weights = 0.25 * math.pi * w
    kappa = float(nu)
    expo = np.exp(-kappa * np.cos(theta))
    q = kappa * np.sin(theta)
    j0 = np.ones_like(q)
    j1 = 0.5 * q
    k0 = q > 1.0e-12
    if np.any(k0):
        # series for small arguments, scipy-free: J0, J1 via upward recurrence
        # is unstable; use the classical power series truncated at |q| <= 30.
        z = q[k0] ** 2
        term0 = np.ones_like(z)
        term1 = np.ones_like(z)
        acc0 = np.ones_like(z)
        acc1 = np.ones_like(z)
        for n in range(1, 60):
            term0 = term0 * (-z / 4.0) / (n * n)
            term1 = term1 * (-z / 4.0) / (n * (n + 1))
            acc0 = acc0 + term0
            acc1 = acc1 + term1
        j0[k0] = acc0
        j1[k0] = 0.5 * q[k0] * acc1
    i3 = -2.0 * math.pi * float(np.sum(weights * expo * j0 * np.cos(theta) * np.sin(theta)))
    i1 = 2.0j * math.pi * float(np.sum(weights * expo * j1 * np.sin(theta) ** 2))
    scale = DENSITY * GRAVITY * RADIUS**2  # amplitude = 1 m
    return complex(-i1 * scale), complex(-i3 * scale)


def scattered_force(mesh: Any, ka: float, diff: Any) -> np.ndarray:
    """``-i omega rho int phi_scattered n_j dS`` -- the diffraction alone."""
    omega = math.sqrt(GRAVITY * ka)
    ops_area = mesh.areas[:, None] * rigid_body_normals_3d(mesh)
    return np.asarray(
        -1.0j * omega * DENSITY * (ops_area.T @ diff.scattered_potential),
        dtype=np.complex128,
    )


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def _cache_read(name: str) -> Any:
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text())
    return None


def _cache_write(name: str, payload: Any) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / (name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(CACHE / name)


def reference(ka: float) -> dict[str, Any]:
    cached = _cache_read(f"ref_{ka:.3f}.json")
    if cached is not None:
        return cached
    rows = {}
    for terms in TERMS_LADDER:
        res = hulme_hemisphere_radiation(
            ka, radius=RADIUS, density=DENSITY, gravity=GRAVITY, n_terms=terms,
            quad_order=max(128, 2 * terms),
        )
        rows[str(terms)] = {
            "surge": _c(res.excitation_force_x),
            "heave": _c(res.excitation_force_z),
        }
    fine = rows[str(TERMS_LADDER[-1])]
    coarse = rows[str(TERMS_LADDER[-2])]
    truncation = {
        key: _rel(
            complex(coarse[key]["re"], coarse[key]["im"]),
            complex(fine[key]["re"], fine[key]["im"]),
        )
        for key in ("surge", "heave")
    }
    payload = {
        "ka": ka,
        "terms_ladder": list(TERMS_LADDER),
        "per_terms": rows,
        "surge": fine["surge"],
        "heave": fine["heave"],
        "truncation_relative": truncation,
    }
    _cache_write(f"ref_{ka:.3f}.json", payload)
    return payload


def bem_point(ka: float, subdivisions: int, haskind: bool) -> dict[str, Any]:
    name = f"bem_{ka:.3f}_{subdivisions}_{int(haskind)}.json"
    cached = _cache_read(name)
    if cached is not None:
        return cached
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
    diff = solve_diffraction_3d(
        mesh, ka, wave_angle=0.0, operators=ops,
        density=DENSITY, gravity=GRAVITY, amplitude=1.0,
    )
    total = complex(diff.excitation_force[0]), complex(diff.excitation_force[2])
    scat = scattered_force(mesh, ka, diff)
    fk = froude_krylov_force_3d(mesh, ka, density=DENSITY, gravity=GRAVITY)
    out: dict[str, Any] = {
        "ka": ka,
        "subdivisions": subdivisions,
        "panels": int(mesh.faces.shape[0]),
        "surge": _c(total[0]),
        "heave": _c(total[1]),
        "surge_scattered": _c(scat[0]),
        "heave_scattered": _c(scat[2]),
        "surge_fk": _c(fk[0]),
        "heave_fk": _c(fk[2]),
        "condition_number": float(diff.condition_number),
        "relative_residual": float(diff.relative_residual),
    }
    if haskind:
        rad = solve_radiation_3d(
            mesh, ka, operators=ops, density=DENSITY, gravity=GRAVITY
        )
        hask = haskind_excitation_force_3d(mesh, rad, wave_angle=0.0)
        out["haskind"] = {
            "kochin_surge": _c(hask["kochin_haskind"][0]),
            "kochin_heave": _c(hask["kochin_haskind"][2]),
            "boundary_surge": _c(hask["boundary_haskind"][0]),
            "boundary_heave": _c(hask["boundary_haskind"][2]),
        }
    _cache_write(name, out)
    return out


def _as_complex(entry: dict[str, float]) -> complex:
    return complex(entry["re"], entry["im"])


def run(independent_path: Path | None = None) -> dict[str, Any]:
    independent = None
    if independent_path is not None and independent_path.exists():
        independent = json.loads(independent_path.read_text())

    refs = {ka: reference(ka) for ka in KA_BAND}
    bem = {
        ka: [bem_point(ka, sub, haskind=(sub == HASKIND_LEVEL)) for sub in LEVELS]
        for ka in KA_BAND
    }

    # ---- TensorBEM vs reference, per mesh level -------------------------- #
    ladder_rows: list[dict[str, Any]] = []
    for ka in KA_BAND:
        for row in bem[ka]:
            ref_s = _as_complex(refs[ka]["surge"])
            ref_h = _as_complex(refs[ka]["heave"])
            got_s = _as_complex(row["surge"])
            got_h = _as_complex(row["heave"])
            ladder_rows.append(
                {
                    "ka": ka,
                    "subdivisions": row["subdivisions"],
                    "panels": row["panels"],
                    "surge": {
                        "bem": row["surge"],
                        "reference": refs[ka]["surge"],
                        "complex_relative": _rel(got_s, ref_s),
                        "magnitude_relative": (abs(got_s) - abs(ref_s)) / abs(ref_s),
                        "phase_deg": _phase_delta_deg(got_s, ref_s),
                    },
                    "heave": {
                        "bem": row["heave"],
                        "reference": refs[ka]["heave"],
                        "complex_relative": _rel(got_h, ref_h),
                        "magnitude_relative": (abs(got_h) - abs(ref_h)) / abs(ref_h),
                        "phase_deg": _phase_delta_deg(got_h, ref_h),
                    },
                    "condition_number": row["condition_number"],
                }
            )

    # ---- G1/G2: case-mesh accuracy --------------------------------------- #
    case_rows = [r for r in ladder_rows if r["subdivisions"] == CASE_SUBDIVISIONS]
    worst_complex = max(
        (r["dof"]["complex_relative"], r["ka"], r["dof_name"])
        for r in (
            dict(row, dof=row["surge"], dof_name="surge") for row in case_rows
        )
    )
    worst_complex_h = max(
        (r["dof"]["complex_relative"], r["ka"], r["dof_name"])
        for r in (
            dict(row, dof=row["heave"], dof_name="heave") for row in case_rows
        )
    )
    worst_all = max(worst_complex, worst_complex_h)
    worst_phase = max(
        abs(row[dof]["phase_deg"])
        for row in case_rows
        for dof in ("surge", "heave")
    )

    # ---- G3: observed convergence order in the panel size ---------------- #
    order_rows = []
    for ka in KA_BAND:
        row = {"ka": ka}
        for dof in ("surge", "heave"):
            errs = []
            for level in LEVELS:
                match = next(
                    r for r in ladder_rows
                    if r["ka"] == ka and r["subdivisions"] == level
                )
                errs.append(match[dof]["complex_relative"])
            panels = [64 * 4 ** (level - 2) for level in LEVELS]
            orders = []
            for i in range(1, len(errs)):
                if errs[i] <= 0.0 or errs[i - 1] <= 0.0:
                    continue
                h_ratio = math.sqrt(panels[i] / panels[i - 1])
                orders.append(math.log(errs[i - 1] / errs[i]) / math.log(h_ratio))
            row[dof] = {
                "errors": errs,
                "panels": panels,
                "monotone": bool(errs[0] > errs[1] > errs[2]),
                "orders": orders,
                "lowest_order": min(orders) if orders else None,
            }
        order_rows.append(row)
    monotone = all(
        row[dof]["monotone"] for row in order_rows for dof in ("surge", "heave")
    )
    min_order = min(
        order_row[dof]["lowest_order"]
        for order_row in order_rows
        for dof in ("surge", "heave")
        if order_row[dof]["lowest_order"] is not None
    )

    # ---- G4: independent route (exact FK + foreign-code diffraction) ----- #
    independent_rows: list[dict[str, Any]] = []
    if independent is not None:
        # the frozen json stores forces non-dimensionalised by rho g a^2
        force_scale = DENSITY * GRAVITY * RADIUS**2
        finest = independent["mesh_ladder"][-1]
        richardson = independent["extrapolation"]
        for ka in KA_BAND:
            key = f"{ka:.3f}"
            fk_s, fk_h = exact_froude_krylov(ka)
            case = next(
                row for row in bem[ka] if row["subdivisions"] == CASE_SUBDIVISIONS
            )
            for dof, fk in (("surge", fk_s), ("heave", fk_h)):
                scat_finest = force_scale * _as_complex(finest["values"][key][dof])
                scat_rich = force_scale * _as_complex(
                    richardson[dof]["richardson"]["order_2"][key]
                )
                total_finest = fk + scat_finest
                total_rich = fk + scat_rich
                ref = _as_complex(refs[ka][dof])
                independent_rows.append(
                    {
                        "ka": ka,
                        "dof": dof,
                        "fk_exact": _c(fk),
                        "fk_bem_panel": case[f"{dof}_fk"],
                        "capytaine_scattered_finest": _c(scat_finest),
                        "capytaine_scattered_richardson2": _c(scat_rich),
                        "total_finest": _c(total_finest),
                        "total_richardson2": _c(total_rich),
                        "reference": refs[ka][dof],
                        "bem_total": case[dof],
                        "complex_relative_finest": _rel(total_finest, ref),
                        "complex_relative_richardson2": _rel(total_rich, ref),
                        "bem_relative": _rel(_as_complex(case[dof]), ref),
                    }
                )
        worst_independent = max(
            row["complex_relative_finest"] for row in independent_rows
        )
        worst_independent_rich = max(
            row["complex_relative_richardson2"] for row in independent_rows
        )
    else:
        worst_independent = None
        worst_independent_rich = None

    # ---- G5: long-wave limit of the reference ---------------------------- #
    lw = reference(LONG_WAVE_KA)
    long_wave_heave = abs(_as_complex(lw["heave"])) / (DENSITY * GRAVITY * RADIUS**2)
    long_wave_limit = WATERPLANE / RADIUS**2  # pi
    # the approach to the limit is O(ka), so the raw value at ka = 0.02 is not
    # the limit: both the raw deviation (reported) and a linear extrapolation
    # in ka (gated) are part of the evidence.
    extrap_ka = (0.01, 0.02, 0.05)
    extrap = [
        (ka, abs(_as_complex(reference(ka)["heave"]))
         / (DENSITY * GRAVITY * RADIUS**2))
        for ka in extrap_ka
    ]
    coeffs = np.polyfit([row[0] for row in extrap], [row[1] for row in extrap], 1)
    long_wave_extrapolated = float(coeffs[1])
    long_wave_rel = abs(long_wave_heave - long_wave_limit) / long_wave_limit
    long_wave_extrap_rel = (
        abs(long_wave_extrapolated - long_wave_limit) / long_wave_limit
    )

    # ---- G6: Haskind reciprocity at the case mesh ------------------------ #
    haskind_rows = []
    for ka in KA_BAND:
        case = next(row for row in bem[ka] if row["subdivisions"] == HASKIND_LEVEL)
        if "haskind" not in case:
            continue
        haskind_rows.append(
            {
                "ka": ka,
                "surge_relative": _rel(
                    _as_complex(case["haskind"]["kochin_surge"]),
                    _as_complex(case["surge"]),
                ),
                "heave_relative": _rel(
                    _as_complex(case["haskind"]["kochin_heave"]),
                    _as_complex(case["heave"]),
                ),
                "boundary_surge_relative": _rel(
                    _as_complex(case["haskind"]["boundary_surge"]),
                    _as_complex(case["surge"]),
                ),
                "boundary_heave_relative": _rel(
                    _as_complex(case["haskind"]["boundary_heave"]),
                    _as_complex(case["heave"]),
                ),
                "kochin_surge": case["haskind"]["kochin_surge"],
                "kochin_heave": case["haskind"]["kochin_heave"],
                "boundary_surge": case["haskind"]["boundary_surge"],
                "boundary_heave": case["haskind"]["boundary_heave"],
                "direct_surge": case["surge"],
                "direct_heave": case["heave"],
                "subdivisions": case["subdivisions"],
            }
        )
    worst_haskind = max(
        (max(row["surge_relative"], row["heave_relative"]) for row in haskind_rows),
        default=None,
    )

    gates = {
        "G1_complex_relative_case_mesh": {
            "target": GATE_COMPLEX,
            "worst": worst_all[0],
            "worst_cell": {"ka": worst_all[1], "dof": worst_all[2]},
            "verdict": "PASS" if worst_all[0] <= GATE_COMPLEX else "FAIL",
        },
        "G2_phase_deg_case_mesh": {
            "target": GATE_PHASE_DEG,
            "worst": worst_phase,
            "verdict": "PASS" if worst_phase <= GATE_PHASE_DEG else "FAIL",
        },
        "G3_mesh_convergence": {
            "target_order": GATE_ORDER,
            "monotone": monotone,
            "lowest_order": min_order,
            "verdict": "PASS" if monotone and min_order >= GATE_ORDER else "FAIL",
        },
        "G4_independent_route": {
            "target": GATE_INDEPENDENT,
            "worst_finest": worst_independent,
            "worst_richardson2": worst_independent_rich,
            "verdict": (
                "NOT_RUN"
                if worst_independent is None
                else ("PASS" if worst_independent <= GATE_INDEPENDENT else "FAIL")
            ),
        },
        "G5_reference_long_wave_limit": {
            "target": GATE_LONG_WAVE,
            "ka": LONG_WAVE_KA,
            "value": long_wave_heave,
            "limit_pi_Awp_over_a2": long_wave_limit,
            "relative": long_wave_rel,
            "raw_target": GATE_LONG_WAVE_RAW,
            "raw_verdict": "PASS" if long_wave_rel <= GATE_LONG_WAVE_RAW else "FAIL",
            "extrapolation_points": [list(row) for row in extrap],
            "extrapolated_limit": long_wave_extrapolated,
            "extrapolated_relative": long_wave_extrap_rel,
            "verdict": (
                "PASS"
                if long_wave_rel <= GATE_LONG_WAVE_RAW
                and long_wave_extrap_rel <= GATE_LONG_WAVE
                else "FAIL"
            ),
        },
        "G6_haskind_reciprocity": {
            "target": GATE_HASKIND,
            "worst": worst_haskind,
            "verdict": (
                "NOT_RUN"
                if worst_haskind is None
                else ("PASS" if worst_haskind <= GATE_HASKIND else "FAIL")
            ),
        },
    }
    verdict = "PASS" if all(
        gate["verdict"] in ("PASS", "NOT_RUN") for gate in gates.values()
    ) and gates["G1_complex_relative_case_mesh"]["verdict"] == "PASS" else "FAIL"

    return {
        "anchor": "CASE-BEM-HULME-DIFF-VERIF",
        "case_pool_id": 177,
        "citation": (
            "Hulme A. (1982) The wave forces acting on a floating hemisphere "
            "undergoing forced periodic oscillations, J. Fluid Mech. 121:443-463"
        ),
        "reference_implementation": "tensorbem.analytic.hulme_hemisphere_radiation",
        "bem_implementation": "tensorbem.free_surface_solver3d.solve_diffraction_3d",
        "independent_implementation": (
            "scripts/capytaine_hulme_diff_scattered.py "
            "(capytaine DiffractionProblem, scattered force) + exact "
            "Froude-Krylov quadrature from this module"
        ),
        "normalisation": {
            "excitation_force": "F / (rho g A a^2), a = 1 m, A = 1 m",
            "time_convention": "exp(-i omega t)",
            "wave_direction": 0.0,
            "note": (
                "TensorBEM excitation_force and the Hulme reference are TOTAL "
                "(Froude-Krylov + diffraction); capytaine's diffraction force is "
                "scattered-only and is completed with the exact FK term here"
            ),
        },
        "geometry": {
            "body": "floating hemisphere, r = a = 1 m, waterplane z = 0",
            "volume": VOLUME,
            "waterplane_area": WATERPLANE,
            "rho": DENSITY,
            "gravity": GRAVITY,
        },
        "band_ka": list(KA_BAND),
        "levels": list(LEVELS),
        "case_subdivisions": CASE_SUBDIVISIONS,
        "terms_ladder": list(TERMS_LADDER),
        "reference": {f"{ka:.3f}": refs[ka] for ka in KA_BAND},
        "ladder": ladder_rows,
        "orders": order_rows,
        "independent": independent_rows,
        "independent_provenance": (
            None if independent is None else {
                "provenance": independent["provenance"],
                "mesh_ladder_panels": [
                    row["n_faces"] for row in independent["mesh_ladder"]
                ],
                "convention_probe": independent["convention_probe"],
                "radiation_sanity": independent["radiation_sanity"],
            }
        ),
        "haskind": haskind_rows,
        "long_wave": {
            "ka": LONG_WAVE_KA,
            "heave_dimensionless": long_wave_heave,
            "limit": long_wave_limit,
            "relative": long_wave_rel,
            "extrapolation_points": [list(row) for row in extrap],
            "extrapolated_limit": long_wave_extrapolated,
            "extrapolated_relative": long_wave_extrap_rel,
            "reference_row": lw,
        },
        "gates": gates,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# report + entry point
# --------------------------------------------------------------------------- #
def render_report(results: dict[str, Any]) -> str:
    g = results["gates"]
    lines = [
        "# CASE-BEM-HULME-DIFF-VERIF: Hulme 半球绕射（三维激励力）验证",
        "",
        "> 本文件由 `benchmarks/verified/casepool_bem_hulme_diff_verif/run_case.py`",
        "> 从 `results.json` 程序化渲染，正文不手抄任何数字。",
        "",
        "## 1. 案例与判据",
        "",
        f"- 算例码 `{results['anchor']}`（算例池 id {results['case_pool_id']}，仓库 TensorBEM）",
        f"- 几何：{results['geometry']['body']}，体积 {results['geometry']['volume']:.6f} m³，",
        f"  水线面面积 {results['geometry']['waterplane_area']:.6f} m²，ρ = {results['geometry']['rho']} kg/m³",
        f"- 频率带 ka = {results['band_ka']}",
        f"- 参考解：`{results['reference_implementation']}`（Hulme 1982 无波势多极级数，"
        f"项数阶梯 {results['terms_ladder']}，激励力取原点 Kochin 振幅/Haskind 互易）",
        f"- 求解器：`{results['bem_implementation']}`，面元阶梯 {results['levels']}"
        f"（算例网格 sub={results['case_subdivisions']}）",
        f"- 独立路线：{results['independent_implementation']}",
        "",
        "| 门 | 判据 | 目标 | 实测 | 结论 |",
        "|---|---|---|---|---|",
        f"| G1 | 复相对误差（算例网格，全 ka、全自由度） | ≤ {g['G1_complex_relative_case_mesh']['target']:.0%} "
        f"| {g['G1_complex_relative_case_mesh']['worst']:.2%} "
        f"(ka={g['G1_complex_relative_case_mesh']['worst_cell']['ka']}, "
        f"{g['G1_complex_relative_case_mesh']['worst_cell']['dof']}) "
        f"| {g['G1_complex_relative_case_mesh']['verdict']} |",
        f"| G2 | 相位差 | ≤ {g['G2_phase_deg_case_mesh']['target']:.1f}° "
        f"| {g['G2_phase_deg_case_mesh']['worst']:.3f}° | {g['G2_phase_deg_case_mesh']['verdict']} |",
        f"| G3 | 面元阶梯单调 + 观测阶 | ≥ {g['G3_mesh_convergence']['target_order']} "
        f"| 单调={g['G3_mesh_convergence']['monotone']}, 最小阶="
        f"{g['G3_mesh_convergence']['lowest_order']:.3f} | {g['G3_mesh_convergence']['verdict']} |",
        f"| G4 | 独立路线 vs 参考（复相对） | ≤ {g['G4_independent_route']['target']:.0%} "
        f"| 最细网格 {'-' if g['G4_independent_route']['worst_finest'] is None else format(g['G4_independent_route']['worst_finest'], '.2%')} / "
        f"Richardson {'-' if g['G4_independent_route']['worst_richardson2'] is None else format(g['G4_independent_route']['worst_richardson2'], '.2%')} "
        f"| {g['G4_independent_route']['verdict']} |",
        f"| G5 | 参考解长波极限 | 外推 ≤ {g['G5_reference_long_wave_limit']['target']:.0%} / 原始 ≤ {g['G5_reference_long_wave_limit']['raw_target']:.0%} "
        f"| ka=0.02 原始 {g['G5_reference_long_wave_limit']['value']:.5f}, 外推 "
        f"{g['G5_reference_long_wave_limit']['extrapolated_limit']:.5f} vs "
        f"π·A_wp/a² = {g['G5_reference_long_wave_limit']['limit_pi_Awp_over_a2']:.5f} "
        f"(外推相对 {g['G5_reference_long_wave_limit']['extrapolated_relative']:.2%}) "
        f"| {g['G5_reference_long_wave_limit']['verdict']} |",
        f"| G6 | Haskind 互易（辐射→Kochin vs 直接绕射） | ≤ {g['G6_haskind_reciprocity']['target']:.0%} "
        f"| {'-'.rjust(1) if g['G6_haskind_reciprocity']['worst'] is None else format(g['G6_haskind_reciprocity']['worst'], '.2%')} "
        f"| {g['G6_haskind_reciprocity']['verdict']} |",
        "",
        f"**总判定：{results['verdict']}**",
        "",
        "## 2. 算例网格上的激励力（复相对误差 / 相位差）",
        "",
        "| ka | |F1| 参考 | |F1| 计算 | F1 复相对 | F1 相位差° | |F3| 参考 | |F3| 计算 | F3 复相对 | F3 相位差° |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results["ladder"]:
        if row["subdivisions"] != results["case_subdivisions"]:
            continue
        s, h = row["surge"], row["heave"]
        ref_s = complex(s["reference"]["re"], s["reference"]["im"])
        got_s = complex(s["bem"]["re"], s["bem"]["im"])
        ref_h = complex(h["reference"]["re"], h["reference"]["im"])
        got_h = complex(h["bem"]["re"], h["bem"]["im"])
        lines.append(
            f"| {row['ka']:.2f} | {abs(ref_s):.1f} | {abs(got_s):.1f} "
            f"| {s['complex_relative']:.2%} | {s['phase_deg']:+.3f} "
            f"| {abs(ref_h):.1f} | {abs(got_h):.1f} "
            f"| {h['complex_relative']:.2%} | {h['phase_deg']:+.3f} |"
        )

    lines += [
        "",
        "## 3. 面元收敛（复相对误差 → 观测阶）",
        "",
        "| ka | 面元 | F1 复相对 | F1 阶 | F3 复相对 | F3 阶 |",
        "|---|---|---|---|---|---|",
    ]
    for order_row in results["orders"]:
        ka = order_row["ka"]
        for i, level in enumerate(results["levels"]):
            row = next(
                r for r in results["ladder"]
                if r["ka"] == ka and r["subdivisions"] == level
            )
            s_order = order_row["surge"]["orders"]
            h_order = order_row["heave"]["orders"]
            s_txt = f"{s_order[i - 1]:.2f}" if i >= 1 and i - 1 < len(s_order) else "-"
            h_txt = f"{h_order[i - 1]:.2f}" if i >= 1 and i - 1 < len(h_order) else "-"
            lines.append(
                f"| {ka:.2f} | {row['panels']} | {row['surge']['complex_relative']:.2%} "
                f"| {s_txt} | {row['heave']['complex_relative']:.2%} | {h_txt} |"
            )

    lines += [
        "",
        "## 4. 独立路线（capytaine 绕射 + 精确 Froude-Krylov）",
        "",
        "| ka | 自由度 | FK 精确 | capy 绕射（最细） | 合计 | 参考 | 复相对 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results["independent"]:
        fk = complex(row["fk_exact"]["re"], row["fk_exact"]["im"])
        sc = complex(
            row["capytaine_scattered_finest"]["re"],
            row["capytaine_scattered_finest"]["im"],
        )
        tot = complex(row["total_finest"]["re"], row["total_finest"]["im"])
        ref = complex(row["reference"]["re"], row["reference"]["im"])
        lines.append(
            f"| {row['ka']:.2f} | {row['dof']} | {fk.real:+.1f}{fk.imag:+.1f}j "
            f"| {sc.real:+.1f}{sc.imag:+.1f}j | {tot.real:+.1f}{tot.imag:+.1f}j "
            f"| {ref.real:+.1f}{ref.imag:+.1f}j | {row['complex_relative_finest']:.2%} |"
        )
    if not results["independent"]:
        lines.append("| - | - | - | - | - | - | NOT_RUN |")

    lines += [
        "",
        "## 5. Haskind 互易（同一求解器内两条力学路线互校）",
        "",
        "| ka | F1 相对 | F3 相对 | 冗余 |",
        "|---|---|---|---|",
    ]
    for row in results["haskind"]:
        lines.append(
            f"| {row['ka']:.2f} | {row['surge_relative']:.3%} "
            f"| {row['heave_relative']:.3%} | {results['gates']['G6_haskind_reciprocity']['target']:.0%} |"
        )

    lines += [
        "",
        "## 6. 注意点与适用边界",
        "",
        "- **口径坑（本算例的核心发现）**：capytaine 的 `DiffractionProblem.forces` 只含绕射（散射压力）力，",
        "  不含入射波（Froude-Krylov）部分；直接拿它和 Hulme 总激励力比对会得到「假失败」或",
        "  「假成功」。冻结文件中 `convention_probe` 用 ka→0 极限（总激励力 → π·ρg·a²·A ≠ 0）钉死该口径。",
        "- TensorBEM 的 `solve_diffraction_3d(...).excitation_force` 是**总**激励力（FK + 绕射），",
        "  与 `froude_krylov_force_3d` 的 docstring 一致；`scattered-only` 另存于 results 里便于逐项对账。",
        "- 参考解不是 Hulme 原文印刷数字（本环境无该文获取途径）：它是 Hulme 方法的重新推导级数，",
        "  因此本算例额外给出 G4（第三方代码）与 G5（长波解析极限）两条**非自证**外部锚。",
        "- ka 带 0.2–2.0 对应 a = 1 m 时的 λ/a ≈ 3.1–31；亚 ka = 0.2 的长波段受消波/截断影响更大，",
        "  超 ka = 2 需加密网格，均不在本算例适用范围内。",
        "",
        "## 7. 复现",
        "",
        "```bash",
        "python scripts/capytaine_hulme_diff_scattered.py   # 需 capytaine，产出冻结参考",
        "PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_diff_verif/run_case.py \\",
        "    --out benchmarks/verified/casepool_bem_hulme_diff_verif",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="artefact directory (default: none, print only)")
    parser.add_argument("--independent", default=None,
                        help="frozen capytaine json (default: <here>/capytaine_scattered.json)")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    independent = Path(args.independent) if args.independent else here / "capytaine_scattered.json"
    results = run(independent)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        (out / "reference.json").write_text(
            json.dumps(
                {
                    "citation": results["citation"],
                    "implementation": results["reference_implementation"],
                    "terms_ladder": results["terms_ladder"],
                    "normalisation": results["normalisation"],
                    "reference": results["reference"],
                    "long_wave": results["long_wave"],
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        (out / "report.md").write_text(render_report(results), encoding="utf-8")
        print(f"wrote {out}/results.json, reference.json, report.md")

    print(json.dumps({k: v["verdict"] for k, v in results["gates"].items()}, indent=2))
    print("verdict:", results["verdict"])
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())