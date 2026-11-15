"""Decompose the G5 (round-trip) failure of ``CASE-BEM-CUMMINS-WIGLEY``.

Question
--------
Gate G5 asks whether ``B -> K -> (B, A)`` closes on the tabulated Wigley III
radiation sweep; it fails at 20-48 % on every DOF.  Three sources are possible
and they demand different responses:

``transform``
    the inverse legs (Simpson quadratures of the Kramers-Kronig pair) or the
    forward leg (the memory kernel) are wrong;
``band``
    the sweep window ``[omega_min, omega_max]`` and the closure of the bands
    outside it;
``a_inf``
    the gate normalises the added-mass residual by ``max|A - A_inf|`` and takes
    ``A_inf`` from :func:`tensorbem.causality.fit_added_mass_asymptote`, a
    two-term tail fit that is known (round 5, task m20594) to be able to leave
    the physical range of the sweep.

What separates them is that each leaves a *different signature*:

* a transform error makes the residual **grow with** ``omega`` (the sine/cosine
  quadratures degrade as the period shrinks toward ``2 dt``);
* a band error lives at the **edges** of the query band;
* an ``A_inf`` error is **flat in** ``omega``: ``A_rt(omega) - A(omega) =
  A_inf_fit - A_inf_true`` for every ``omega``, because the kernel carries
  ``A - A_inf`` and the constant is added back separately.  The residual is then
  exactly ``|A_inf_fit - A_inf_true| / max|A - A_inf|``, a constant.

The script measures all three on the same datum and validates the decomposition
procedure on an arm where the truth is known.

Arms
----
``synthetic_exact_kernel``  the inverse legs alone, on ``K = c exp(-a tau)``
                            sampled on the case's own tau grid (both ``(A, B)``
                            exact by construction) -> the transform error.
``synthetic_biased_a_inf``  the same kernel with ``A_inf`` shifted by a known
                            ``delta`` -> must reproduce ``delta / scale`` to
                            machine precision, which is what makes the panel
                            attribution a measurement rather than a guess.
``panel``                   the shipped statistic, the flat-offset component,
                            the bias-free shape residual, and the same residual
                            with the Kramers-Kronig-consistent ``A_inf*`` of
                            :func:`tensorbem.causality.profile_added_mass_inf`.

Two side findings are measured as well, both reported by the doc they write:
G2's kernel-at-zero gate (``rel_err`` ~3e-16 on every DOF, i.e. the gate cannot
fail because the forward leg *is* the trapezoid rule it is compared against) and
the high-frequency closure of the forward leg (closing the band above
``omega_max`` with the algebraic tail ``B(omega_max)(omega_max/omega)^2`` buys
3-6 % on G5, not the 20-48 % that is missing).

Usage::

    PYTHONPATH=src python3 scripts/cummins_g5_ainf_attribution.py
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from tensorbem import time_domain as td
from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.causality import profile_added_mass_inf
from cummins_kernel_high_frequency_audit import (
    kernel_tail_closed,
    round_trip_errors,
    synthetic_arm,
)

DOF_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
BAND = (0.314, 7.2)
SWEEP_NPZ = pathlib.Path("docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz")


def exact_kernel(c: float, a: float, tau: np.ndarray) -> td.MemoryKernel:
    return td.MemoryKernel(
        tau=tau, values=c * np.exp(-a * tau), omega_max=float(tau.size and 12.0),
        damping_scale=0.0, low_frequency_model="zero",
        high_frequency_bound=0.0, low_frequency_bound=0.0, metadata={},
    )


def synthetic_biased_a_inf(omega: np.ndarray, tau: np.ndarray, *,
                           c: float = 300.0, a: float = 8.0,
                           added_mass_inf: float = 50.0,
                           delta: float = 7.5) -> dict:
    """Does a known ``A_inf`` shift reproduce the G5 residual exactly?"""
    damping = c * a / (a * a + omega * omega)
    added = added_mass_inf - c / (a * a + omega * omega)
    kernel = exact_kernel(c, a, tau)
    band = BAND
    base = round_trip_errors(kernel, omega, added, damping, added_mass_inf, band)
    shifted = round_trip_errors(kernel, omega, added, damping,
                               added_mass_inf + delta, band)
    query = np.asarray(base["query"], dtype=float)
    exact_a = np.interp(query, omega, added)
    scale = float(np.max(np.abs(exact_a - added_mass_inf)))
    # G5 normalises by max|A - A_inf|, so the normaliser moves with the shift too:
    # the residual the gate reports is |delta| / max|A - (A_inf + delta)|, which
    # understates a large bias instead of overstating it.
    scale_shifted = float(np.max(np.abs(exact_a - (added_mass_inf + delta))))
    profile = np.asarray(shifted["added_mass_profile"], dtype=float)
    return {
        "delta": delta,
        "scale": scale,
        "scale_shifted": scale_shifted,
        "predicted_residual": delta / scale_shifted,
        "measured_residual": float(shifted["added_mass_rel_err"]),
        "flatness": float(np.ptp(profile) / max(float(np.max(profile)), 1e-30)),
        "damping_error_unshifted": base["damping_rel_err"],
        "identity_error": abs(shifted["added_mass_rel_err"] - delta / scale_shifted),
    }


def panel_rows(tau: np.ndarray) -> dict:
    data = np.load(SWEEP_NPZ)
    omega = np.asarray(data["omega"], dtype=float)
    rows = {}
    for dof, label in enumerate(DOF_LABELS):
        added = np.asarray(data["added_mass"], dtype=float)[:, dof, dof]
        damping = np.asarray(data["damping"], dtype=float)[:, dof, dof]
        a_inf = float(td.fit_added_mass_asymptote(omega, added).added_mass_inf)
        kernel = td.radiation_memory_kernel(omega, damping, tau)
        shipped = round_trip_errors(kernel, omega, added, damping, a_inf, BAND)
        query = np.asarray(shipped["query"], dtype=float)
        exact_a = np.interp(query, omega, added)
        round_a = np.interp(query, omega, added) - np.asarray(
            shipped["added_mass_profile"], dtype=float) * np.max(
            np.abs(exact_a - a_inf)) * np.sign(1.0)
        residual = np.asarray(shipped["added_mass_profile"], dtype=float)
        offset = float(np.mean(residual))
        scale = float(np.max(np.abs(exact_a - a_inf)))
        prof = profile_added_mass_inf(omega, added, damping)
        calibrated = round_trip_errors(kernel, omega, added, damping,
                                       prof.best_added_mass_inf, BAND)
        closed = round_trip_errors(kernel_tail_closed(omega, damping, tau),
                                   omega, added, damping, a_inf, BAND)
        rows[label] = {
            "added_mass_inf_fit": a_inf,
            "added_mass_inf_kk": prof.best_added_mass_inf,
            "scale": scale,
            "g5_shipped": shipped["worst"],
            "added_mass_residual_flat_offset": offset,
            "bias_over_scale": abs(offset),
            "bias_explained_fraction": abs(offset) / max(shipped["added_mass_rel_err"], 1e-30),
            "g5_shape_residual": float(np.max(np.abs(residual - offset))),
            "g5_with_kk_a_inf": calibrated["added_mass_rel_err"],
            "g5_tail_closed": closed["worst"],
            "damping_leg_error": shipped["damping_rel_err"],
            "residual_profile": residual.tolist(),
            "query": query.tolist(),
        }
    return rows


def g2_vacuity(tau: np.ndarray) -> dict:
    """G2 compares K(0) with the trapezoid rule of the very table K(0) came from."""
    sweep = case.sweep_signature("medium") if hasattr(case, "sweep_signature") else None
    data = np.load(SWEEP_NPZ)
    omega = np.asarray(data["omega"], dtype=float)
    out = {}
    for dof, label in enumerate(DOF_LABELS):
        damping = np.asarray(data["damping"], dtype=float)[:, dof, dof]
        study = case.analysis_gates({"omega": omega.tolist(),
                                     "added_mass": np.asarray(
                                         data["added_mass"], dtype=float).tolist(),
                                     "damping": np.asarray(data["damping"], dtype=float).tolist(),
                                     "level": "cached", "rotation_center": list(case.ROTATION_CENTER),
                                     "density": case.DENSITY,
                                     "gravity": case.GRAVITY}, dof)
        trap = float(np.sum(0.5 * (damping[1:] + damping[:-1]) * np.diff(omega))) * 2.0 / math.pi
        out[label] = {
            "g2_rel_err": study["G2"]["rel_err"],
            "K_at_zero": study["G2"]["K_at_zero_sampled"],
            "trapezoid_integral": trap,
            "closure_fraction": study["G2"]["low_frequency_closure_fraction"],
        }
    out["note"] = ("K(0) is built by exact piecewise-linear quadrature, so the "
                   "trapezoid rule of the same nodes reproduces it to machine "
                   "precision: G2 tests the quadrature against itself and cannot fail.")
    return out


def figure(payload: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["panel"]
    labels = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, [rows[k]["g5_shipped"] for k in labels], 0.4,
                label="G5 shipped (tol 5 %)", color="#c0392b")
    axes[0].bar(x + 0.2, [rows[k]["bias_over_scale"] for k in labels], 0.4,
                label=r"flat $A_\infty$ offset / scale", color="#e67e22")
    axes[0].plot(x, [rows[k]["g5_shape_residual"] for k in labels], "o-",
                 color="#16a085", label="bias-free shape residual")
    axes[0].plot(x, [rows[k]["g5_with_kk_a_inf"] for k in labels], "s--",
                 color="#2c3e50", label=r"with KK-consistent $A_\infty^*$")
    axes[0].axhline(0.05, color="k", lw=0.8, ls=":")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("relative residual")
    axes[0].set_title(r"G5: the failure is an $A_\infty$ offset, not the transform")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for label, color in (("heave", "#2980b9"), ("roll", "#8e44ad"), ("yaw", "#27ae60")):
        q = np.asarray(rows[label]["query"], dtype=float)
        axes[1].plot(q, rows[label]["residual_profile"], color=color, label=label)
    axes[1].axhline(0.05, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(r"$\omega$ [rad/s]")
    axes[1].set_ylabel(r"$|A_{rt}-A|/\max|A-A_\infty|$")
    axes[1].set_title("residual is flat in $\\omega$ (offset signature)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    tau = np.arange(0, 40.0001, 0.02)
    data = np.load(SWEEP_NPZ)
    omega = np.asarray(data["omega"], dtype=float)
    payload = {
        "mission": 21140,
        "anchor": case.__name__.split(".")[-1],
        "band": list(BAND),
        "dt": 0.02,
        "tau_max": float(tau[-1]),
        "sweep": {"npz": str(SWEEP_NPZ), "n_omega": int(omega.size),
                  "omega_min": float(omega[0]), "omega_max": float(omega[-1])},
        "synthetic_transform": synthetic_arm(omega, tau),
        "synthetic_biased_a_inf": synthetic_biased_a_inf(omega, tau),
        "panel": panel_rows(tau),
        "g2": g2_vacuity(tau),
    }
    out = pathlib.Path("docs/case_bem_cummins_wigley_g5_attribution.json")
    out.write_text(json.dumps(payload, indent=1))
    figure(payload, pathlib.Path("docs/fig_m21140_g5_ainf_attribution.png"))

    syn = payload["synthetic_transform"]
    bias = payload["synthetic_biased_a_inf"]
    print("synthetic exact-kernel arm (transform only): "
          f"worst {syn['exact_kernel']['worst']:.4%}")
    print(f"known A_inf shift {bias['delta']}: predicted {bias['predicted_residual']:.4f} "
          f"measured {bias['measured_residual']:.4f} identity error {bias['identity_error']:.2e} "
          f"flatness {bias['flatness']:.2e}")
    for label, row in payload["panel"].items():
        print(f"{label:6s} G5 {row['g5_shipped']:.4%} = bias {row['bias_over_scale']:.4f}"
              f" ({row['bias_explained_fraction']:.2%}) + shape {row['g5_shape_residual']:.4%}"
              f" | with A_inf* {row['g5_with_kk_a_inf']:.4%} | tail-closed {row['g5_tail_closed']:.4%}")
    print("G2 rel_err:", {k: f"{v['g2_rel_err']:.2e}" for k, v in payload["g2"].items()
                          if k != "note"})
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())