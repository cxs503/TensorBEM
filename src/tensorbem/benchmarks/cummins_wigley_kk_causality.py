"""Cummins-equation / Kramers-Kronig causal-consistency case for the Wigley III hull.

What this case decides
----------------------
Case-pool anchor ``CASE-BEM-CUMMINS-WIGLEY``
(``ref_source = "Cummins equation + Kramers-Kronig causality"``, ``tol = <10 %``).
It asks one question with predeclared tolerances: *is the radiation impedance
that the TensorBEM 3-D panel solve returns for a ship-shaped, surface-piercing
hull a single causal linear system?* -- i.e. can one memory kernel

    K(t) = (2/pi) int_0^inf B(w) cos(w t) dw

reproduce the whole frequency sweep, through the Cummins equation

    (M + A_inf) x'' + int_0^t K(t - s) x'(s) ds + C x = F(t),

and through the Hilbert pair (Kramers-Kronig, Kotik & Mangulis 1962; Newman
1977, *Marine Hydrodynamics*, ch. 6)

    B(w)   = -(2/pi) PV int_0^inf s^2 (A(s) - A_inf) / (s^2 - w^2) ds
    A(w)   = A_inf + (2/pi) PV int_0^inf B(s) / (s^2 - w^2) ds .

Gates (predeclared, machine-decided)
------------------------------------
======================  ======  ==================================================
G1 cross_solver         10 %    diagonal A_ii, B_ii at w = 1 rad/s vs the frozen
                                capytaine 3.1 reference on the same wetted surface
G2 kernel_at_zero        2 %    K(0) sampled vs (2/pi) int B dw of the same sweep
G3 kramers_kronig       10 %    A(w) vs Hilbert(B) at the sweep midpoints
G4 free_decay           10 %    zeta of the Cummins free decay vs the half-power
                                zeta of the frequency-domain response
G5 round_trip            5 %    B -> K -> (B, A) inside the transform band
======================  ======  ==================================================

Round-2 corrections (both findings of the independent audit of the round-1
attempt, ``docs/board_audit/m17264/AUDIT_m17209.md``)
-----------------------------------------------------
**F1 rotation centre.**  The radiation sweep must pass
``center = ROTATION_CENTER = (0, 0, -T/2)`` -- the same centre that produced the
capytaine reference.  Without it roll/pitch moments are taken about the
waterline (the solver default), which is a *different body*, and G1 fails on
roll by ~50 % while surge/sway/heave/yaw are exactly unaffected (they have no
moment arm).  :func:`test_rotation_center_convention` in
``tests/test_cummins_wigley_kk_causality.py`` pins the convention.

**F2 frequency band.**  The round-1 grid was ``[0.5, 6] rad/s`` with 8 nodes.
Two of the gates need the band, not just the solver:

* heave resonates at ``w_n = sqrt(C33 / (m + A33(w)))`` which for this hull is
  **above 6 rad/s** (``A33`` falls from ~195 kg at w = 1 to ~47 kg at w = 6, so
  the ratio crosses 1 only past 6.5 rad/s).  With no resonance inside the band
  the half-power estimator cannot produce ``zeta`` at all -- G4 is *undefined*,
  not inaccurate.
* the two Kramers-Kronig integrals are principal values over ``s in
  (0, inf)``; a sweep that stops at ``0.6 * w_max`` for the round trip and 6
  rad/s for the tail does not see the ``A -> A_inf`` plateaus at either end, so
  the causality residual is dominated by band truncation.

:data:`OMEGA_GRID` therefore spans ``0.30 .. 12.5 rad/s`` with 22 nodes (about
2.7 nodes per decade-wide structure, resolution 12 % in ``w``), which brackets
the heave resonance found at run time (:func:`heave_resonance_band`).  The
round-1 grid is kept as :data:`LEGACY_GRID` and the case reports the same gates
on it as an A/B control: the difference between the two is what the band fix
buys, measured, not asserted.

Reference provenance
--------------------
No digitised Cummins-kernel or Wigley radiation table is shipped, and recalled
literature constants are recorded as debt by
:mod:`tensorbem.reference_provenance`.  G1 is therefore anchored on an
independent *code* (capytaine 3.1.dev0, frozen in
``benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`` with its full
environment and rotation centre), and G2-G5 are the exact identities the
Cummins/Kramers-Kronig formalism supplies.  The consequence is stated honestly:
G2-G5 are cross-route checks of one and the same sweep, so they detect
*inconsistency* (wrong Green function, irregular frequencies, band truncation,
a phase/sign error) but cannot detect a systematic error shared by both routes;
only G1 can, and only at the single frequency the reference file covers.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem import time_domain as td
from tensorbem.causality import fit_added_mass_asymptote, kkr_added_mass
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_C,
    WIGLEY3_CB,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_hydrostatic_reference,
    wigley3_wetted_hull_mesh,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "CAPYTAINE_REF_PATH",
    "DENSITY",
    "DOF_LABELS",
    "GATES",
    "GRAVITY",
    "LEGACY_GRID",
    "MESH_LEVELS",
    "OMEGA_GRID",
    "ROTATION_CENTER",
    "analysis_gates",
    "cross_solver_table",
    "geometry_gate",
    "heave_decay_gate",
    "heave_resonance_band",
    "load_capytaine_reference",
    "load_points",
    "solve_point",
    "sweep_from_points",
    "sweep_signature",
    "wetted_mesh",
]

DOF_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DOF_INDEX = {label: i for i, label in enumerate(DOF_LABELS)}

DENSITY = 1025.0
GRAVITY = 9.80665

#: Rotation centre shared with the capytaine reference (mid-draft, on the
#: centreline, amidships).  See module docstring, finding F1.
ROTATION_CENTER = (0.0, 0.0, -WIGLEY3_T / 2.0)

#: Mesh ladder; ``panels`` below counts the *quadrilateral* panels the builder
#: emits, ``triangles`` the count the solver assembles (2 per quad).  Both are
#: reported so the two counting conventions cannot be confused again (audit F5).
MESH_LEVELS = {"coarse": (10, 5), "medium": (20, 10), "fine": (40, 20)}

#: Round-2 frequency grid (rad/s): 22 nodes over [0.30, 12.5].  Must contain
#: 1.0 rad/s (the reference frequency) and bracket the heave resonance.
OMEGA_GRID = (
    0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.20, 1.50, 1.85, 2.30, 2.80,
    3.40, 4.00, 4.70, 5.40, 6.20, 7.00, 7.80, 8.60, 9.60, 11.00, 12.50,
)

#: Round-1 band, kept as the A/B control (audit F2).
LEGACY_GRID = (0.5, 0.7, 1.0, 1.5, 2.2, 3.2, 4.2, 6.0)

#: Predeclared gates.  ``tol`` is a relative tolerance; G1 carries the case-pool
#: tolerance (<10 %), G2/G5 are identities between two numerical routes and are
#: held tighter, G3 is the case-tolerance causality residual, G4 compares two
#: estimators of the same damping.
GATES = {
    "G1_cross_solver": {"tol": 0.10, "quantity": "diag A_ii, B_ii at omega=1.0"},
    "G2_kernel_at_zero": {"tol": 0.02, "quantity": "K(0) vs (2/pi) int B dw"},
    "G3_kramers_kronig": {"tol": 0.10, "quantity": "A(w) vs Hilbert(B)"},
    "G4_free_decay": {"tol": 0.10, "quantity": "zeta_decay vs zeta_half_power"},
    "G5_round_trip": {"tol": 0.05, "quantity": "band-limited B, A"},
}

CAPYTAINE_REF_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "verified"
    / "wigley3_recalled_ref"
    / "capytaine_ref.json"
)


# ---------------------------------------------------------------------------
# geometry and the per-frequency radiation solve
# ---------------------------------------------------------------------------


def geometry_gate(density: float = DENSITY, gravity: float = GRAVITY) -> dict:
    """Validate the hull before any number is quoted (Cb / L/B / B/T / volume)."""
    ref = wigley3_hydrostatic_reference(density=density, gravity=gravity)
    volume = float(ref["volume"])
    cb_analytic = WIGLEY3_CB
    cb_geometry = volume / (WIGLEY3_L * WIGLEY3_B * WIGLEY3_T)
    return {
        "L": WIGLEY3_L,
        "B": WIGLEY3_B,
        "T": WIGLEY3_T,
        "c": WIGLEY3_C,
        "L_over_B": WIGLEY3_L / WIGLEY3_B,
        "B_over_T": WIGLEY3_B / WIGLEY3_T,
        "volume": volume,
        "Cb_declared": cb_analytic,
        "Cb_from_volume": cb_geometry,
        "Cb_abs_error": abs(cb_geometry - cb_analytic),
        "waterplane_area": float(ref["waterplane_area"]),
        "C33": float(ref["C33"]),
        "displaced_mass": float(volume * density),
        "rotation_center": list(ROTATION_CENTER),
        "passed": bool(
            abs(cb_geometry - cb_analytic) <= 1e-9
            and abs(WIGLEY3_L / WIGLEY3_B - 10.0) <= 1e-9
            and abs(WIGLEY3_B / WIGLEY3_T - 1.6) <= 1e-9
        ),
    }


def wetted_mesh(level: str = "medium"):
    """Wetted-surface mesh for the Wigley III hull at the requested level."""
    n_sections, n_levels = MESH_LEVELS[level]
    return wigley3_wetted_hull_mesh(
        L=WIGLEY3_L, B=WIGLEY3_B, T=WIGLEY3_T, c=WIGLEY3_C,
        n_sections=n_sections, n_levels=n_levels,
    )


def sweep_signature(level: str = "medium") -> dict:
    """Mesh identity of a level, in both counting conventions."""
    n_sections, n_levels = MESH_LEVELS[level]
    mesh = wetted_mesh(level)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=float)
    tri = verts[faces]
    volume = float(
        np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
    )
    return {
        "level": f"n_sections={n_sections},n_levels={n_levels}",
        "n_sections": n_sections,
        "n_levels": n_levels,
        "quad_panels": n_sections * n_levels * 2,
        "triangles": int(len(faces)),
        "mesh_volume": volume,
        "mesh_volume_rel_error": volume / float(
            wigley3_hydrostatic_reference()["volume"]
        ) - 1.0,
    }


def solve_point(
    level: str,
    omega: float,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    center: Sequence[float] | None = None,
) -> dict:
    """One frequency of the radiation sweep, with the fixed rotation centre.

    ``center`` is exposed only so the convention can be *tested* (the test suite
    solves the same mesh with the solver default and shows which DOF move); the
    case itself always uses :data:`ROTATION_CENTER`.
    """
    centre = ROTATION_CENTER if center is None else tuple(float(v) for v in center)
    mesh = wetted_mesh(level)
    started = time.perf_counter()
    result = solve_radiation_3d(
        mesh,
        float(omega) ** 2 / gravity,
        center=centre,
        density=density,
        gravity=gravity,
        chief_points="auto",
    )
    wall = time.perf_counter() - started
    return {
        "level": level,
        "omega": float(omega),
        "added_mass": np.asarray(result.added_mass, dtype=float).tolist(),
        "damping": np.asarray(result.damping, dtype=float).tolist(),
        "relative_residual": float(getattr(result, "relative_residual", float("nan"))),
        "rotation_center": list(centre),
        "density": density,
        "gravity": gravity,
        "wall_seconds": wall,
    }


def load_points(directory: pathlib.Path | str, level: str) -> dict[float, dict]:
    """Read every cached single-frequency solve of ``level`` from ``directory``."""
    directory = pathlib.Path(directory)
    points: dict[float, dict] = {}
    for path in sorted(directory.glob(f"{level}_omega_*.json")):
        payload = json.loads(path.read_text())
        if payload.get("level") != level:
            continue
        points[round(float(payload["omega"]), 9)] = payload
    return points


def sweep_from_points(points: dict[float, dict], omegas: Sequence[float]) -> dict:
    """Assemble an ``(omega, A, B)`` sweep from cached points, in the given order.

    A missing frequency raises instead of being interpolated: a sweep is only
    evidence for the frequencies that were actually solved.
    """
    grid = np.asarray(omegas, dtype=float)
    missing = [float(w) for w in grid if round(float(w), 9) not in points]
    if missing:
        raise KeyError(f"no cached solve for omega = {missing} (level "
                       f"{points[next(iter(points))]['level'] if points else '?'})")
    first = points[round(float(grid[0]), 9)] if grid.size else None
    if first is None:
        raise ValueError("empty sweep")
    n6 = len(first["added_mass"])
    added = np.zeros((grid.size, n6, n6))
    damping = np.zeros((grid.size, n6, n6))
    residuals = []
    walls = []
    for i, w in enumerate(grid):
        payload = points[round(float(w), 9)]
        added[i] = np.asarray(payload["added_mass"], dtype=float)
        damping[i] = np.asarray(payload["damping"], dtype=float)
        residuals.append(float(payload.get("relative_residual", float("nan"))))
        walls.append(float(payload.get("wall_seconds", float("nan"))))
    n_sections, n_levels = MESH_LEVELS[first["level"]]
    return {
        "level": f"n_sections={n_sections},n_levels={n_levels}",
        "level_key": first["level"],
        "rotation_center": first["rotation_center"],
        "omega": grid.tolist(),
        "added_mass": added.tolist(),
        "damping": damping.tolist(),
        "relative_residual": residuals,
        "wall_seconds": walls,
        "density": first["density"],
        "gravity": first["gravity"],
    }


def _as_diag(sweep: dict, dof: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damping = np.asarray(sweep["damping"], dtype=float)
    return omega, added[:, dof, dof], damping[:, dof, dof]


# ---------------------------------------------------------------------------
# reference and gates
# ---------------------------------------------------------------------------


def load_capytaine_reference(
    path: pathlib.Path | str | None = None, builder: str = "wigley3_freq"
) -> dict:
    """Load the independent capytaine reference (G1), level-keyed, nothing retyped."""
    path = pathlib.Path(path) if path is not None else CAPYTAINE_REF_PATH
    payload = json.loads(path.read_text())
    entries = payload["builders"][builder]
    levels = {}
    for entry in entries:
        levels[entry["level"]] = {
            "added_mass": {k: float(v) for k, v in entry["added_mass"].items()},
            "damping": {k: float(v) for k, v in entry["damping"].items()},
            "panels": entry.get("panels"),
            "mesh_volume": entry.get("mesh_volume"),
        }
    return {
        "path": str(path),
        "builder": builder,
        "metadata": payload.get("metadata", {}),
        "levels": levels,
    }


def cross_solver_table(
    sweep: dict, reference: dict, *, omega_ref: float = 1.0
) -> dict:
    """G1: TensorBEM vs the independent capytaine reference at ``omega_ref``."""
    label = sweep["level"]
    if label not in reference["levels"]:
        raise KeyError(f"reference has no level {label!r}; have "
                       f"{sorted(reference['levels'])}")
    ref = reference["levels"][label]
    omega = np.asarray(sweep["omega"], dtype=float)
    index = int(np.argmin(np.abs(omega - omega_ref)))
    if abs(omega[index] - omega_ref) > 1e-9:
        raise ValueError(f"omega={omega_ref} not on the sweep grid")
    added = np.asarray(sweep["added_mass"], dtype=float)[index]
    damping = np.asarray(sweep["damping"], dtype=float)[index]
    rows = []
    for dof, name in enumerate(DOF_LABELS):
        ref_a = ref["added_mass"][name.capitalize()]
        ref_b = ref["damping"][name.capitalize()]
        got_a = float(added[dof, dof])
        got_b = float(damping[dof, dof])
        rows.append({
            "dof": dof,
            "dof_label": name,
            "omega": float(omega[index]),
            "added_mass_ref": ref_a,
            "added_mass_got": got_a,
            "added_mass_rel_err": (got_a - ref_a) / ref_a if ref_a else None,
            "damping_ref": ref_b,
            "damping_got": got_b,
            "damping_rel_err": (got_b - ref_b) / ref_b if ref_b else None,
        })
    finite = [abs(r["added_mass_rel_err"]) for r in rows
              if r["added_mass_rel_err"] is not None]
    finite_b = [abs(r["damping_rel_err"]) for r in rows
                if r["damping_rel_err"] is not None]
    return {
        "reference": {
            "path": reference["path"],
            "builder": reference["builder"],
            "level": label,
            "panels": ref["panels"],
            "metadata": reference["metadata"],
        },
        "rows": rows,
        "worst_added_mass_rel_err": max(finite),
        "worst_damping_rel_err": max(finite_b),
        "worst_rel_err": max(max(finite), max(finite_b)),
        "rotation_center_sweep": sweep["rotation_center"],
        "rotation_center_reference": reference["metadata"].get("rotation_center"),
    }


def analysis_gates(
    sweep: dict,
    dof: int,
    *,
    dt: float = 0.02,
    tau_max: float = 40.0,
) -> dict:
    """G2, G3, G5 for one DOF of one sweep (the frequency-domain half of the case)."""
    omega, added, damping = _as_diag(sweep, dof)
    fit = fit_added_mass_asymptote(omega, added)
    a_inf = float(fit.added_mass_inf)

    tau = np.arange(0.0, tau_max + 0.5 * dt, dt)
    kernel = td.radiation_memory_kernel(omega, damping, tau)
    diag = td.kernel_diagnostics(kernel, omega, damping)

    # G2 is the quadrature identity K(0) = (2/pi) int_0^inf B dw, so it is
    # evaluated on the *same* sweep with the below-band closure switched off:
    # with ``low_frequency_model="zero"`` the forward leg is nothing but the
    # exact piecewise-linear integral of B, and any residual is a defect of the
    # transform.  The closure that the production kernel adds below omega_min
    # (the long-wave model B ~ c w^3) is reported next to it as a diagnostic,
    # not folded into the gate -- it is a modelling choice, not an identity.
    kernel_zero = td.radiation_memory_kernel(
        omega, damping, tau, low_frequency_model="zero")
    k_zero_sampled = float(kernel_zero.values[0])
    k_zero_integral = float(np.sum(
        0.5 * (damping[1:] + damping[:-1]) * np.diff(omega)
    ) * 2.0 / math.pi)
    r2 = abs(k_zero_sampled - k_zero_integral) / max(abs(k_zero_integral), 1e-30)
    k_zero_production = float(kernel.values[0])
    closure_fraction = (abs(k_zero_production - k_zero_integral)
                        / max(abs(k_zero_integral), 1e-30))

    # G3: the principal value needs the sample *strictly inside* a grid interval,
    # so the check is made at the midpoints, where the measured A is linearly
    # interpolated (no new physics enters -- the sweep is the datum).
    kk_omega = 0.5 * (omega[:-1] + omega[1:])
    kk_measured = 0.5 * (added[:-1] + added[1:])
    kk_pred = np.array([
        kkr_added_mass(omega, damping, float(w), added_mass_inf=a_inf)
        for w in kk_omega
    ])
    kk_abs = np.abs(kk_pred - kk_measured)
    scale = max(float(np.max(np.abs(kk_measured - a_inf))), 1e-30)
    r3 = float(np.max(kk_abs)) / scale
    r3_pointwise = (kk_abs / np.maximum(np.abs(kk_measured), 1e-30)).tolist()

    # G5: band-limited round trip B -> K -> (B, A)
    trip = td.fourier_round_trip(omega, added, damping, a_inf, dt=dt, tau_max=tau_max)
    damping_exact = np.asarray(trip.damping_exact, dtype=float)
    added_exact = np.asarray(trip.added_mass_exact, dtype=float)
    damping_err = np.abs(np.asarray(trip.damping_round_trip, dtype=float) - damping_exact) \
        / max(float(np.max(np.abs(damping_exact))), 1e-30)
    added_err = np.abs(np.asarray(trip.added_mass_round_trip, dtype=float) - added_exact) \
        / max(float(np.max(np.abs(added_exact - a_inf))), 1e-30)
    band = tuple(getattr(trip, "band", ()) or ())
    query = np.asarray(trip.omega_query, dtype=float)
    if len(band) == 2:
        mask = (query >= band[0] - 1e-12) & (query <= band[1] + 1e-12)
    else:  # pragma: no cover - the routine always reports its band
        mask = np.ones_like(query, dtype=bool)
    r5 = float(np.nanmax(np.concatenate([damping_err[mask], added_err[mask]])))

    return {
        "dof": int(dof),
        "dof_label": DOF_LABELS[dof],
        "omega": omega.tolist(),
        "added_mass": added.tolist(),
        "damping": damping.tolist(),
        "added_mass_inf": a_inf,
        "added_mass_inf_tail_fraction": getattr(fit, "tail_fraction", None),
        "kernel": {
            "dt": dt,
            "tau_max": tau_max,
            "n_samples": int(tau.size),
            "tau": kernel.tau.tolist(),
            "values": kernel.values.tolist(),
        },
        "kernel_diagnostics": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in diag.items()
        },
        "G2": {
            "K_at_zero_sampled": k_zero_sampled,
            "K_at_zero_integral": k_zero_integral,
            "rel_err": r2,
            "K_at_zero_production": k_zero_production,
            "low_frequency_closure_fraction": closure_fraction,
        },
        "G3": {
            "rel_err": r3,
            "pointwise_rel_err": r3_pointwise,
            "omega": kk_omega.tolist(),
        },
        "G5": {
            "rel_err": r5,
            "damping_rel_err": damping_err.tolist(),
            "added_mass_rel_err": added_err.tolist(),
            "band": list(band),
        },
    }


def heave_resonance_band(
    sweep: dict, *, density: float = DENSITY, gravity: float = GRAVITY
) -> dict:
    """Locate the heave resonance of the tabulated sweep (used to choose the band)."""
    omega, added, damping = _as_diag(sweep, DOF_INDEX["heave"])
    ref = wigley3_hydrostatic_reference(density=density, gravity=gravity)
    mass = float(ref["volume"] * density)
    restoring = float(ref["C33"])
    try:
        props = td.resonance_properties(omega, added, damping, mass=mass,
                                        restoring=restoring)
    except ValueError as exc:  # honest record: no crossing inside the band
        return {"mass": mass, "restoring_C33": restoring,
                "error": str(exc), "inside_band": False}
    omega_n = float(props["omega_n"])
    return {
        "mass": mass,
        "restoring_C33": restoring,
        "omega_n": omega_n,
        "period": 2.0 * math.pi / omega_n,
        "inside_band": bool(omega[0] < omega_n < omega[-1]),
        "band": [float(omega[0]), float(omega[-1])],
        "properties": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in props.items()},
        "added_mass_inf_heave": float(fit_added_mass_asymptote(omega, added).added_mass_inf),
    }


def heave_decay_gate(
    study: dict,
    sweep: dict,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    dt: float = 0.02,
    duration: float = 120.0,
) -> dict:
    """G4: Cummins-equation free decay in heave vs the half-power damping ratio.

    Heave is the DOF whose mass and restoring follow from the geometry alone
    (``rho * volume`` and ``C33 = rho g A_wp``), so no empirical radius of
    gyration enters.  The two numbers compared come from *different routes*: a
    time-domain integration of the memory integral vs a frequency-domain
    half-power scan of ``1/|kappa|``.
    """
    omega, added, damping = _as_diag(sweep, DOF_INDEX["heave"])
    ref = wigley3_hydrostatic_reference(density=density, gravity=gravity)
    mass = float(ref["volume"] * density)
    restoring = float(ref["C33"])
    a_inf = float(study["added_mass_inf"])

    tau = np.asarray(study["kernel"]["tau"], dtype=float)
    kernel = td.radiation_memory_kernel(omega, damping, tau)
    decay = td.cummins_free_decay(
        kernel, mass=mass, restoring=restoring, added_mass_inf=a_inf,
        dt=dt, duration=duration,
    )
    metrics = td.decay_metrics(decay.time, decay.displacement)
    rate = float(metrics.get("decay_rate", float("nan")))
    period = metrics.get("period")
    omega_d = float(2.0 * math.pi / period) if period else float("nan")
    omega_n = (math.hypot(omega_d, rate)
               if np.isfinite(omega_d) and np.isfinite(rate) else float("nan"))
    zeta_decay = rate / omega_n if omega_n and np.isfinite(omega_n) else float("nan")

    dense = np.linspace(float(omega[0]), float(omega[-1]), 801)
    try:
        half_power = td.half_power_damping(
            dense, np.interp(dense, omega, added), np.interp(dense, omega, damping),
            mass=mass, restoring=restoring,
        )
    except (ValueError, IndexError) as exc:
        # honest record instead of a fabricated number: no resonance inside the
        # tabulated band (ValueError) or a half-power crossing that falls outside
        # the scan window (IndexError) both mean "this band cannot answer G4".
        half_power = {"error": f"{type(exc).__name__}: {exc}"}
    zeta_fd = float(half_power.get("zeta", float("nan")))
    rel = (abs(zeta_decay - zeta_fd) / abs(zeta_fd)
           if np.isfinite(zeta_fd) and zeta_fd != 0.0 else float("nan"))

    return {
        "dof": 2,
        "dof_label": "heave",
        "mass": mass,
        "restoring_C33": restoring,
        "waterplane_area": float(ref["waterplane_area"]),
        "added_mass_inf": a_inf,
        "kernel_dt": dt,
        "kernel_tau_max": float(tau[-1]),
        "duration": duration,
        "decay_rate": rate,
        "decay_period": period,
        "omega_d": omega_d,
        "omega_n_decay": omega_n,
        "zeta_decay": zeta_decay,
        "zeta_half_power": zeta_fd,
        "half_power": half_power,
        "rel_err": rel,
        "decay_envelope_r2": metrics.get("r_squared"),
    }


def _verdict(identifier: str, value: float | None, tol: float,
             extra: dict | None = None) -> dict:
    passed = value is not None and np.isfinite(value) and abs(value) <= tol
    out = {"id": identifier, "value": value, "tol": tol, "passed": bool(passed)}
    if extra:
        out.update(extra)
    return out


def gates_from_sweep(
    sweep: dict,
    *,
    reference: dict | None = None,
    dt: float = 0.02,
    tau_max: float = 40.0,
    decay_duration: float = 120.0,
    include_cross_solver: bool = True,
) -> dict:
    """Evaluate all five gates on one sweep (all DOF for G1/G2/G3/G5, heave G4).

    ``include_cross_solver=False`` marks G1 as *skipped* instead of evaluating it,
    for sweeps whose mesh level the frozen capytaine reference does not cover
    (a cached sweep from another artifact, an off-ladder mesh).  A skipped gate is
    neither passed nor failed; it is reported as ``passed: None`` and left out of
    the failure list, so a band study cannot borrow G1's authority by accident.
    """
    studies = {}
    for dof in range(6):
        studies[DOF_LABELS[dof]] = analysis_gates(sweep, dof, dt=dt, tau_max=tau_max)
    decay = heave_decay_gate(studies["heave"], sweep, dt=dt,
                             duration=decay_duration)
    if include_cross_solver:
        cross = cross_solver_table(sweep, reference or load_capytaine_reference())
        g1 = _verdict("G1_cross_solver", cross["worst_rel_err"],
                      GATES["G1_cross_solver"]["tol"],
                      {"worst_added_mass": cross["worst_added_mass_rel_err"],
                       "worst_damping": cross["worst_damping_rel_err"],
                       "omega": 1.0, "level": sweep["level"]})
    else:
        cross = {"skipped": "no independent reference for this mesh level",
                 "level": sweep["level"]}
        g1 = {"id": "G1_cross_solver", "value": None, "tol": GATES["G1_cross_solver"]["tol"],
              "passed": None, "skipped": cross["skipped"], "level": sweep["level"]}
    g2 = _verdict("G2_kernel_at_zero",
                  max(s["G2"]["rel_err"] for s in studies.values()),
                  GATES["G2_kernel_at_zero"]["tol"],
                  {"per_dof": {k: v["G2"]["rel_err"] for k, v in studies.items()}})
    worst_r3 = max(studies.values(), key=lambda s: s["G3"]["rel_err"])
    g3 = _verdict("G3_kramers_kronig", worst_r3["G3"]["rel_err"],
                  GATES["G3_kramers_kronig"]["tol"],
                  {"worst_dof": worst_r3["dof_label"],
                   "per_dof": {k: v["G3"]["rel_err"] for k, v in studies.items()}})
    g4 = _verdict("G4_free_decay", decay["rel_err"], GATES["G4_free_decay"]["tol"],
                  {"zeta_decay": decay["zeta_decay"],
                   "zeta_half_power": decay["zeta_half_power"],
                   "half_power_error": decay["half_power"].get("error")})
    worst_r5 = max(studies.values(), key=lambda s: s["G5"]["rel_err"])
    g5 = _verdict("G5_round_trip", worst_r5["G5"]["rel_err"],
                  GATES["G5_round_trip"]["tol"],
                  {"worst_dof": worst_r5["dof_label"],
                   "per_dof": {k: v["G5"]["rel_err"] for k, v in studies.items()}})
    gates = {g["id"]: g for g in (g1, g2, g3, g4, g5)}
    failed = [key for key, value in gates.items() if value["passed"] is False]
    return {
        "level": sweep["level"],
        "omega": sweep["omega"],
        "rotation_center": sweep["rotation_center"],
        "gates": gates,
        "failed": failed,
        "verdict": "PASS" if not failed else "FAIL:" + ",".join(failed),
        "studies": studies,
        "free_decay": decay,
        "cross_solver": cross,
    }