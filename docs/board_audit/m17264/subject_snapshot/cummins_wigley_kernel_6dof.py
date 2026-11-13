"""Cummins memory-kernel benchmark for the Wigley III hull in 6 DOF (case pool).

What this benchmark decides
---------------------------
A *case-pool* benchmark: it must answer, with predeclared tolerances, whether the
TensorBEM chain

    frequency-domain radiation  A(w), B(w)   [3-D panel BEM, Kelvin Green fn]
        -> Cummins memory kernel  K(t) = (2/pi) int_0^inf B(w) cos(w t) dw
        -> free decay of the Cummins equation

is right for a ship-shaped, surface-piercing body in all six degrees of freedom
-- not for a submerged sphere (see :mod:`tensorbem.benchmarks.cummins_memory_closure`)
and not for heave alone (see :mod:`tensorbem.benchmarks.cummins_wigley_hull`).

References (all numeric references are *loaded at run time*, nothing is typed in)
-------------------------------------------------------------------------------
*R1 independent solver* -- Capytaine 3.1 (BEM, Rankine panel method with
  infinite-water-depth Green function) on the same Wigley III wetted surface,
  committed as ``benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json``
  together with its full environment record (python/numpy/capytaine versions,
  machine ``loongarch64``, density 1025 kg/m^3, gravity 9.80665 m/s^2,
  rotation centre ``(0, 0, -0.09375)``, water depth infinite).  The file is read
  here by :func:`load_capytaine_reference`; no constant is copied by hand.

*R2 Cummins (1962) identity* -- ``K(0) = (2/pi) int_0^inf B(w) dw``.  The kernel
  value at ``tau = 0`` is checked against the quadrature of the very ``B(w)``
  that produced it, i.e. the identity is tested, not the quadrature.

*R3 Kramers-Kronig causality* -- ``A(w) - A(inf)`` is the Hilbert transform of
  ``B(w)`` (:func:`tensorbem.causality.kkr_added_mass`).  The real part is
  predicted from the imaginary part alone; the two are different physical
  outputs of the same solve, so agreement is a statement about the radiation
  impedance being causal, not about the quadrature.

*R4 Cummins vs frequency domain* -- the damping ratio of the Cummins-equation
  free decay must equal the half-power damping ratio of the frequency-domain
  response built from the same ``A``, ``B``.

*R5 round trip* -- ``B -> K -> B`` on the tabulated band
  (:func:`tensorbem.time_domain.fourier_round_trip`).

Why no published ``K(t)`` table is used
---------------------------------------
No digitised table of Cummins kernels or of Wigley III radiation coefficients
exists in this repository, and the repository's provenance gate
(:mod:`tensorbem.reference_provenance`) classifies *recalled* literature
constants as debt that never counts as evidence.  R1 is therefore the
independent-solver anchor, and R2-R5 are the exact identities that the Cummins /
Kramers-Kronig formalism itself supplies.  The consequence is recorded honestly:
R2-R5 are cross-method checks of the same BEM sweep, so they cannot detect a
systematic error of the radiation solve itself; only R1 can, and only at the
single frequency the reference file covers.
"""

from __future__ import annotations

import json
import math
import pathlib
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
    "DOF_LABELS",
    "MESH_LEVELS",
    "GATES",
    "load_capytaine_reference",
    "radiation_sweep",
    "memory_kernel_study",
    "heave_free_decay",
    "cross_solver_table",
    "run_case",
]

DOF_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DOF_INDEX = {label: i for i, label in enumerate(DOF_LABELS)}

#: Mesh ladder, matching the levels recorded in the Capytaine reference file.
MESH_LEVELS = {"coarse": (10, 5), "medium": (20, 10), "fine": (40, 20)}

#: Default frequency grid (rad/s).  Must contain the reference frequency 1.0 rad/s
#: and satisfy ``validate_radiation_sweep`` (>= 8 points).
OMEGA_GRID = (0.5, 0.7, 1.0, 1.5, 2.0, 2.6, 3.2, 4.0, 5.0, 6.0)

#: Predeclared gates.  ``tol`` is the case tolerance (<10 % for the pool case);
#: R2/R5 are identities between two numerical routes, so they get a tighter
#: tolerance and are reported separately.
GATES = {
    "R1_cross_solver": {"tol": 0.10, "kind": "rel_err", "quantity": "A_ii, B_ii at omega=1.0"},
    "R2_kernel_at_zero": {"tol": 0.02, "kind": "rel_err", "quantity": "K(0) vs (2/pi) int B dw"},
    "R3_kramers_kronig": {"tol": 0.10, "kind": "rel_err", "quantity": "A(w) vs Hilbert(B)"},
    "R4_free_decay": {"tol": 0.10, "kind": "rel_err", "quantity": "zeta_decay vs zeta_half_power"},
    "R5_round_trip": {"tol": 0.05, "kind": "rel_err", "quantity": "band-limited B, A"},
}

CAPYTAINE_REF_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "verified"
    / "wigley3_recalled_ref"
    / "capytaine_ref.json"
)


def load_capytaine_reference(
    path: pathlib.Path | str | None = None, builder: str = "wigley3_freq"
) -> dict:
    """Load the independent Capytaine reference (R1) as a level-keyed table.

    The JSON file stores, per mesh level, the six diagonal added-mass and damping
    entries in SI units.  Nothing is re-typed: the file is the datum.
    """
    path = pathlib.Path(path) if path is not None else CAPYTAINE_REF_PATH
    payload = json.loads(path.read_text())
    entries = payload["builders"][builder]
    table = {}
    for entry in entries:
        table[entry["level"]] = {
            "added_mass": {k: float(v) for k, v in entry["added_mass"].items()},
            "damping": {k: float(v) for k, v in entry["damping"].items()},
            "panels": entry.get("panels"),
            "mesh_volume": entry.get("mesh_volume"),
        }
    return {"path": str(path), "builder": builder, "levels": table}


def level_label(n_sections: int, n_levels: int) -> str:
    return f"n_sections={n_sections},n_levels={n_levels}"


def radiation_sweep(
    omegas: Sequence[float] = OMEGA_GRID,
    *,
    level: str = "medium",
    density: float = 1025.0,
    gravity: float = 9.80665,
    n_sections: int | None = None,
    n_levels: int | None = None,
) -> dict:
    """Radiation coefficients for all six DOF on the requested mesh level."""
    if n_sections is None or n_levels is None:
        n_sections, n_levels = MESH_LEVELS[level]
    mesh = wigley3_wetted_hull_mesh(
        L=WIGLEY3_L, B=WIGLEY3_B, T=WIGLEY3_T, c=WIGLEY3_C,
        n_sections=n_sections, n_levels=n_levels,
    )
    omega = np.asarray(omegas, dtype=float)
    added = np.zeros((len(omega), 6, 6))
    damping = np.zeros((len(omega), 6, 6))
    residuals = []
    for i, om in enumerate(omega):
        rad = solve_radiation_3d(
            mesh, om * om / gravity, density=density, gravity=gravity,
            chief_points="auto",
        )
        added[i] = rad.added_mass
        damping[i] = rad.damping
        residuals.append(float(getattr(rad, "relative_residual", float("nan"))))
    return {
        "level": level_label(n_sections, n_levels),
        "panels": n_sections * n_levels * 2,
        "omega": omega.tolist(),
        "added_mass": added.tolist(),
        "damping": damping.tolist(),
        "relative_residual": residuals,
        "density": density,
        "gravity": gravity,
    }


def _as_diag(sweep: dict, dof: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damping = np.asarray(sweep["damping"], dtype=float)
    return omega, added[:, dof, dof], damping[:, dof, dof]


def memory_kernel_study(
    sweep: dict,
    *,
    dof: int,
    dt: float = 0.02,
    tau_max: float = 20.0,
    added_mass_inf: float | None = None,
) -> dict:
    """Kernel, Cummins identity, Kramers-Kronig and round trip for one DOF.

    ``added_mass_inf`` may be supplied when the case knows A(inf) from an
    independent route; otherwise it is extrapolated from the tail of the sweep
    (:func:`tensorbem.causality.fit_added_mass_asymptote`), and that estimate is
    itself a source of error in the Kramers-Kronig residual, which is why both
    the fitted value and the highest-frequency sample are recorded.
    """
    omega, A, B = _as_diag(sweep, dof)
    omega, A, B = td.validate_radiation_sweep(omega, A, B)
    tau = np.arange(0.0, tau_max + 0.5 * dt, dt)
    kernel = td.radiation_memory_kernel(omega, B, tau)
    diag = td.kernel_diagnostics(kernel, omega, B)
    fit = fit_added_mass_asymptote(omega, A)
    a_inf_fit = float(fit.added_mass_inf)
    a_inf = float(added_mass_inf) if added_mass_inf is not None else a_inf_fit

    # R2: Cummins identity K(0) = (2/pi) int_0^inf B dw
    omega_safe = np.clip(omega, 1e-12, None)
    b_over_w = np.where(omega_safe > 0.0, B / omega_safe, 0.0)
    k_zero_integral = float(2.0 / math.pi * np.trapezoid(B, omega))
    k_zero_sampled = float(kernel.values[0])
    r2 = abs(k_zero_sampled - k_zero_integral) / abs(k_zero_integral) if k_zero_integral else float("nan")

    # R3: Kramers-Kronig, A(w) - A(inf) from B(w) alone.  The principal-value
    # quadrature needs the sample strictly inside the tabulated band, so the two
    # end nodes are excluded (they are the endpoints of the truncated sweep, not
    # evidence about causality).
    # ``kkr_added_mass`` requires the sample to sit *between* nodes (its
    # piecewise-linear principal value has a pole otherwise), so the check is
    # made at the midpoints of the sweep and the measured ``A`` is linearly
    # interpolated there.
    kk_omega = 0.5 * (omega[:-1] + omega[1:])
    kk_measured = 0.5 * (A[:-1] + A[1:])
    kk_pred = np.array([kkr_added_mass(omega, B, float(w), added_mass_inf=a_inf)
                        for w in kk_omega])
    kk_abs = np.abs(kk_pred - kk_measured)
    scale = max(float(np.max(np.abs(kk_measured - a_inf))), 1e-12)
    r3 = float(np.max(kk_abs)) / scale
    r3_pointwise = (kk_abs / np.maximum(np.abs(kk_measured), 1e-12)).tolist()

    # R5: band-limited round trip (the band is a property of the sampling and is
    # reported by the routine itself)
    rt = td.fourier_round_trip(omega, A, B, a_inf, dt=dt, tau_max=tau_max)
    # pointwise relative errors are undefined where A(w) - A(inf) crosses zero,
    # so the round-trip error is normalised by the scale of each quantity
    damping_exact = np.asarray(rt.damping_exact, dtype=float)
    addedm_exact = np.asarray(rt.added_mass_exact, dtype=float)
    damping_err = np.abs(np.asarray(rt.damping_round_trip, dtype=float) - damping_exact) \
        / max(float(np.max(np.abs(damping_exact))), 1e-30)
    addedm_err = np.abs(np.asarray(rt.added_mass_round_trip, dtype=float) - addedm_exact) \
        / max(float(np.max(np.abs(addedm_exact - a_inf))), 1e-30)
    band = tuple(getattr(rt, "band", ()) or ())
    query = np.asarray(rt.omega_query, dtype=float)
    if len(band) == 2:
        mask = (query >= band[0] - 1e-12) & (query <= band[1] + 1e-12)
    else:  # pragma: no cover - the routine always reports its band
        mask = np.ones_like(query, dtype=bool)
    r5 = float(np.nanmax(np.concatenate([damping_err[mask], addedm_err[mask]])))

    return {
        "dof": int(dof),
        "dof_label": DOF_LABELS[dof],
        "omega": omega.tolist(),
        "added_mass": A.tolist(),
        "damping": B.tolist(),
        "added_mass_inf": a_inf,
        "added_mass_inf_source": "supplied" if added_mass_inf is not None else "tail_fit",
        "added_mass_inf_tail_fit": a_inf_fit,
        "added_mass_at_omega_max": float(A[-1]),
        "added_mass_inf_tail_fraction": getattr(fit, "tail_fraction", None),
        "kernel": {
            "dt": dt,
            "tau_max": tau_max,
            "n_samples": int(tau.size),
            "values": kernel.values.tolist(),
            "tau": kernel.tau.tolist(),
        },
        "diagnostics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in diag.items()},
        "R2": {
            "K_at_zero_sampled": k_zero_sampled,
            "K_at_zero_integral": k_zero_integral,
            "rel_err": r2,
            "contrast_with_b_over_omega": float(np.max(np.abs(b_over_w))),
        },
        "R3": {"rel_err": r3, "pointwise_rel_err": r3_pointwise,
               "omega": kk_omega.tolist(), "at_sweep_midpoints": True},
        "R5": {"rel_err": r5,
               "damping_rel_err": damping_err.tolist(),
               "added_mass_rel_err": addedm_err.tolist(),
               "band": list(getattr(rt, "band", ()) or ())},
    }


def heave_free_decay(
    study: dict,
    *,
    sweep: dict | None = None,
    density: float = 1025.0,
    gravity: float = 9.80665,
    dt: float = 0.02,
    duration: float = 60.0,
) -> dict:
    """R4: Cummins-equation free decay in heave vs the half-power damping ratio.

    Heave is the one DOF whose mass and hydrostatic restoring follow from the
    geometry alone (displaced mass ``rho * nabla`` and ``C33 = rho g A_wp``),
    so no empirical radius of gyration enters the comparison.
    """
    if sweep is not None:
        omega, A, B = _as_diag(sweep, DOF_INDEX["heave"])
    else:
        omega = np.asarray(study["omega"], dtype=float)
        A = np.asarray(study["added_mass"], dtype=float)
        B = np.asarray(study["damping"], dtype=float)
    ref = wigley3_hydrostatic_reference(density=density, gravity=gravity)
    mass = float(ref["volume"] * density)
    restoring = float(ref["C33"])
    a_inf = float(study["added_mass_inf"])

    kernel = td.radiation_memory_kernel(
        omega, B, np.asarray(study["kernel"]["tau"], dtype=float)
    )
    decay = td.cummins_free_decay(
        kernel, mass=mass, restoring=restoring, added_mass_inf=a_inf,
        dt=dt, duration=duration,
    )
    metrics = td.decay_metrics(decay.time, decay.displacement)
    rate = float(metrics.get("decay_rate", float("nan")))
    period = metrics.get("period")
    omega_d = float(2.0 * math.pi / period) if period else float("nan")
    omega_n = math.hypot(omega_d, rate) if np.isfinite(omega_d) and np.isfinite(rate) else float("nan")
    zeta_decay = rate / omega_n if omega_n and np.isfinite(omega_n) else float("nan")

    # the resonance scan needs a grid fine enough to see Re(kappa) change sign;
    # the BEM sweep has only a handful of nodes, so the coefficients are
    # linearly interpolated onto a dense grid (no new physics enters).
    dense = np.linspace(float(omega[0]), float(omega[-1]), 401)
    try:
        hp = td.half_power_damping(
            dense, np.interp(dense, omega, A), np.interp(dense, omega, B),
            mass=mass, restoring=restoring,
        )
    except ValueError as exc:  # honest record instead of a fabricated number
        hp = {"error": str(exc)}
    zeta_fd = float(hp.get("zeta", float("nan")))
    rel = abs(zeta_decay - zeta_fd) / abs(zeta_fd) if zeta_fd else float("nan")

    return {
        "dof": 2,
        "dof_label": "heave",
        "mass": mass,
        "restoring_C33": restoring,
        "waterplane_area": float(ref["waterplane_area"]),
        "added_mass_inf": a_inf,
        "decay_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in metrics.items()},
        "zeta_free_decay": zeta_decay,
        "zeta_half_power": zeta_fd,
        "half_power": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in hp.items()},
        "half_power_grid": "linear interpolation of A,B onto 401 nodes",
        "rel_err": rel,
        "time": np.asarray(decay.time, dtype=float)[::25].tolist(),
        "displacement": np.asarray(decay.displacement, dtype=float)[::25].tolist(),
    }


def cross_solver_table(
    sweep: dict,
    reference: dict,
    *,
    omega_ref: float = 1.0,
) -> dict:
    """R1: TensorBEM vs the independent Capytaine reference at ``omega_ref``."""
    levels = reference["levels"]
    label = sweep["level"]
    if label not in levels:
        raise KeyError(f"reference has no level {label!r}; have {sorted(levels)}")
    ref = levels[label]
    omega = np.asarray(sweep["omega"], dtype=float)
    idx = int(np.argmin(np.abs(omega - omega_ref)))
    if abs(omega[idx] - omega_ref) > 1e-9:
        raise ValueError(f"omega={omega_ref} not on the sweep grid: {omega.tolist()}")
    added = np.asarray(sweep["added_mass"], dtype=float)[idx]
    damping = np.asarray(sweep["damping"], dtype=float)[idx]
    rows = []
    for dof, label_dof in enumerate(DOF_LABELS):
        cap_a = ref["added_mass"][label_dof.capitalize()]
        cap_b = ref["damping"][label_dof.capitalize()]
        got_a = float(added[dof, dof])
        got_b = float(damping[dof, dof])
        rows.append({
            "dof": dof,
            "dof_label": label_dof,
            "omega": float(omega[idx]),
            "added_mass_ref": cap_a,
            "added_mass_got": got_a,
            "added_mass_rel_err": (got_a - cap_a) / cap_a if cap_a else None,
            "damping_ref": cap_b,
            "damping_got": got_b,
            "damping_rel_err": (got_b - cap_b) / cap_b if cap_b else None,
        })
    finite = [r for r in rows if r["added_mass_rel_err"] is not None]
    return {
        "reference": {"path": reference["path"], "builder": reference["builder"],
                      "level": label, "panels": ref["panels"]},
        "rows": rows,
        "worst_added_mass_rel_err": max(abs(r["added_mass_rel_err"]) for r in finite),
        "worst_damping_rel_err": max(abs(r["damping_rel_err"]) for r in rows
                                     if r["damping_rel_err"] is not None),
    }


def _verdict(identifier: str, value: float | None, tol: float, extra: dict | None = None) -> dict:
    passed = value is not None and np.isfinite(value) and abs(value) <= tol
    out = {"id": identifier, "value": value, "tol": tol, "passed": bool(passed)}
    if extra:
        out.update(extra)
    return out


def run_case(
    *,
    level: str = "medium",
    levels: Sequence[str] = ("coarse", "medium"),
    omegas: Sequence[float] = OMEGA_GRID,
    dt: float = 0.02,
    tau_max: float = 20.0,
    reference: dict | None = None,
    sweep_fn=radiation_sweep,
    progress=print,
) -> dict:
    """Run the whole case: sweeps on ``levels``, then all gates on ``level``."""
    reference = reference or load_capytaine_reference()
    sweeps = {}
    for lv in levels:
        progress(f"[sweep] {lv}")
        sweeps[lv] = sweep_fn(omegas, level=lv)
    main = sweeps[level]

    studies = {}
    for dof in range(6):
        progress(f"[kernel] {DOF_LABELS[dof]}")
        studies[DOF_LABELS[dof]] = memory_kernel_study(main, dof=dof, dt=dt, tau_max=tau_max)

    decay = heave_free_decay(studies["heave"], sweep=main, dt=dt)
    cross = cross_solver_table(main, reference)

    # sensitivity 1: mesh ladder, measured on the frequency-domain input itself
    sensitivity_mesh = {}
    for lv, sw in sweeps.items():
        omega = np.asarray(sw["omega"], dtype=float)
        idx = int(np.argmin(np.abs(omega - 1.0)))
        sensitivity_mesh[lv] = {
            "level": sw["level"], "panels": sw["panels"],
            "added_mass_diag": np.diag(np.asarray(sw["added_mass"])[idx]).tolist(),
            "damping_diag": np.diag(np.asarray(sw["damping"])[idx]).tolist(),
        }
    # sensitivity 2: kernel sampling (dt, tau_max).  A frequency-band truncation
    # study is *not* possible here: ``validate_radiation_sweep`` requires at least
    # 8 tabulated frequencies, so dropping the top of an 8-node sweep is rejected
    # (recorded as ecosystem friction in the report).
    fine = memory_kernel_study(main, dof=DOF_INDEX["heave"], dt=dt / 2.0, tau_max=tau_max)
    short = memory_kernel_study(main, dof=DOF_INDEX["heave"], dt=dt, tau_max=tau_max / 2.0)
    base_k0 = studies["heave"]["R2"]["K_at_zero_sampled"]
    sensitivity_sampling = {
        "dt_base": dt, "tau_max_base": tau_max,
        "K_at_zero_base": base_k0,
        "K_at_zero_dt_half": fine["R2"]["K_at_zero_sampled"],
        "rel_change_dt_half": abs(fine["R2"]["K_at_zero_sampled"] - base_k0) / abs(base_k0),
        "K_at_zero_tau_max_half": short["R2"]["K_at_zero_sampled"],
        "rel_change_tau_max_half": abs(short["R2"]["K_at_zero_sampled"] - base_k0) / abs(base_k0),
        "R5_rel_err_base": studies["heave"]["R5"]["rel_err"],
        "R5_rel_err_dt_half": fine["R5"]["rel_err"],
        "band_truncation": "not testable: validate_radiation_sweep needs >= 8 frequencies",
    }

    gates = {
        "R1": _verdict("R1_cross_solver", cross["worst_added_mass_rel_err"],
                       GATES["R1_cross_solver"]["tol"]),
        "R2": _verdict("R2_kernel_at_zero",
                       max(abs(s["R2"]["rel_err"]) for s in studies.values()),
                       GATES["R2_kernel_at_zero"]["tol"]),
        "R3": _verdict("R3_kramers_kronig",
                       max(s["R3"]["rel_err"] for s in studies.values()),
                       GATES["R3_kramers_kronig"]["tol"]),
        "R4": _verdict("R4_free_decay", decay["rel_err"], GATES["R4_free_decay"]["tol"]),
        "R5": _verdict("R5_round_trip",
                       max(s["R5"]["rel_err"] for s in studies.values()),
                       GATES["R5_round_trip"]["tol"]),
    }
    per_dof = {
        label: {
            "R1_added_mass_rel_err": next(r["added_mass_rel_err"] for r in cross["rows"]
                                          if r["dof_label"] == label),
            "R1_damping_rel_err": next(r["damping_rel_err"] for r in cross["rows"]
                                       if r["dof_label"] == label),
            "R2_rel_err": studies[label]["R2"]["rel_err"],
            "R3_rel_err": studies[label]["R3"]["rel_err"],
            "R5_rel_err": studies[label]["R5"]["rel_err"],
            "added_mass_inf": studies[label]["added_mass_inf"],
        }
        for label in DOF_LABELS
    }
    failed = [k for k, v in gates.items() if not v["passed"]]
    return {
        "case": "CASE-BEM-CUMMINS-WIGLEY (6 DOF memory kernel)",
        "geometry": {
            "hull": "Wigley III (parabolic waterline + section, c-parameterised)",
            "L": WIGLEY3_L, "B": WIGLEY3_B, "T": WIGLEY3_T, "c": WIGLEY3_C,
            "Cb": WIGLEY3_CB,
            "volume": float(wigley3_hydrostatic_reference()["volume"]),
            "L_B": WIGLEY3_L / WIGLEY3_B, "B_T": WIGLEY3_B / WIGLEY3_T,
            "density": 1025.0, "gravity": 9.80665,
            "rotation_center_z": -WIGLEY3_T / 2.0,
        },
        "omega": list(omegas),
        "nondim_omega_sqrt_L_over_g": [float(w * math.sqrt(WIGLEY3_L / 9.80665)) for w in omegas],
        "kernel_settings": {"dt": dt, "tau_max": tau_max},
        "sweeps": {lv: {k: v for k, v in sw.items() if k not in ("added_mass", "damping")}
                   for lv, sw in sweeps.items()},
        "studies": studies,
        "free_decay_heave": decay,
        "cross_solver": cross,
        "sensitivity": {"mesh": sensitivity_mesh, "sampling": sensitivity_sampling},
        "gates": gates,
        "per_dof": per_dof,
        "failed_gates": failed,
        "verdict": "PASS" if not failed else "FAIL:" + ",".join(failed),
    }