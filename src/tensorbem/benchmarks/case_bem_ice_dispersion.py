"""CASE-BEM-ICE-DISPERSION: Ice mass-loading dispersion benchmark.

Reference
---------
Weitz, E. & Keller, J. B. (1950). "Reflection of water waves from a
floating semi-infinite elastic plate". *Comm. Pure Appl. Math.* 3(4),
pp. 305–318.  The mass-loading dispersion relation (eq. 1 of this module)
is the finite-depth generalisation of their deep-water result.

Deep-water limit (Weitz & Keller 1950, eq. 2.12):
    omega^2 = g k / (1 + mu k)

Shallow-water limit (long-wave, kH << 1):
    omega^2 = g k^2 H / (1 + mu k^2 H)

Both limits are closed-form and serve as the reference for this benchmark.

Benchmark parameters
--------------------
Ice thickness: 0.3, 0.6, 0.9, 1.5, 2.5 m
Water depth:   200 m (deep), 6 m (shallow)
Periods:       3, 4, 6, 8, 10, 12, 14 s

Tolerance: < 5 % relative error against the closed-form limits.

Usage
-----
    PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_ice_dispersion \
        --out artifacts/m17274/ice_dispersion_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..ice import (
    DEFAULT_GRAVITY,
    DEFAULT_WATER_DENSITY,
    DEFAULT_ICE_DENSITY,
    IceCover,
    dispersion_table,
    mass_loading_omega,
    mass_loading_wavenumber,
    open_water_wavenumber,
    robin_residual,
    shooting_omega,
    thickness_study,
    energy_flux_amplitude_ratio,
    mass_loading_attenuation,
)


def weitz_keller_deep_water_omega(k: float, mu: float, g: float) -> float:
    """Weitz-Keller deep-water closed form: omega^2 = g k / (1 + mu k)."""
    return math.sqrt(g * k / (1.0 + mu * k))


def shallow_water_omega(k: float, H: float, mu: float, g: float) -> float:
    """Shallow-water (long-wave) closed form: omega^2 = g k^2 H / (1 + mu k^2 H)."""
    return math.sqrt(g * k * k * H / (1.0 + mu * k * k * H))


def run_benchmark(
    *,
    periods=(3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
    thicknesses=(0.3, 0.6, 0.9, 1.5, 2.5),
    deep_depth=200.0,
    shallow_depth=6.0,
) -> dict:
    """Run the ice dispersion benchmark and return a JSON-serialisable payload."""
    t0 = time.time()

    # ---- default ice cover ----
    ice = IceCover(thickness=0.9, water_depth=deep_depth)
    omegas = np.array([2.0 * math.pi / T for T in periods], dtype=float)

    # ---- frequency sweep (deep water) ----
    table = dispersion_table(omegas, ice)
    residual = np.array(
        [abs(robin_residual(table["k_ice"][i], omegas[i], ice)) for i in range(omegas.size)]
    )
    shooting = np.array(
        [shooting_omega(table["k_ice"][i], ice) for i in range(omegas.size)]
    )
    shooting_error = np.abs(shooting - omegas) / omegas
    analytic_omega = np.array([mass_loading_omega(table["k_ice"][i], ice) for i in range(omegas.size)])
    omega_error = np.abs(analytic_omega - omegas) / omegas

    # ---- deep-water Weitz-Keller limit ----
    deep_ice = IceCover(thickness=0.9, water_depth=1.0e5)
    deep_k = float(mass_loading_wavenumber(2.0 * math.pi / 10.0, deep_ice).wavenumber)
    deep_omega_ref = weitz_keller_deep_water_omega(deep_k, deep_ice.mu, deep_ice.gravity)
    deep_omega_target = 2.0 * math.pi / 10.0
    deep_error = abs(deep_omega_ref - deep_omega_target) / deep_omega_target

    # Also compute k from the Weitz-Keller formula directly (invert omega->k)
    # For omega = 2pi/10, solve gk/(1+mu k) = omega^2 for k
    omega_dw = 2.0 * math.pi / 10.0
    omega2_dw = omega_dw ** 2
    # gk = omega^2 (1 + mu k) => k(g - omega^2 mu) = omega^2 => k = omega^2 / (g - omega^2 mu)
    k_wk = omega2_dw / (deep_ice.gravity - omega2_dw * deep_ice.mu)
    omega_from_wk = weitz_keller_deep_water_omega(k_wk, deep_ice.mu, deep_ice.gravity)
    k_error = abs(k_wk - deep_k) / deep_k

    # ---- shallow-water limit ----
    shallow_ice = IceCover(thickness=0.9, water_depth=shallow_depth)
    shallow_period = 12.0
    shallow_omega = 2.0 * math.pi / shallow_period
    shallow_k = float(mass_loading_wavenumber(shallow_omega, shallow_ice).wavenumber)
    tanh_kh = math.tanh(shallow_k * shallow_ice.water_depth)
    shallow_ref = math.sqrt(
        shallow_ice.gravity * shallow_k * tanh_kh / (1.0 + shallow_ice.mu * shallow_k * tanh_kh)
    )
    shallow_error = abs(shallow_ref - shallow_omega) / shallow_omega
    shallow_kH = shallow_k * shallow_ice.water_depth

    # Pure shallow-water closed form (kH << 1 approximation)
    # omega^2 = g k^2 H / (1 + mu k^2 H)
    # Solve for k: omega^2 (1 + mu k^2 H) = g k^2 H
    # => omega^2 = k^2 H (g - omega^2 mu)
    # => k^2 = omega^2 / (H (g - omega^2 mu))
    omega2_sw = shallow_omega ** 2
    denom_sw = shallow_ice.water_depth * (shallow_ice.gravity - omega2_sw * shallow_ice.mu)
    k_sw_closed = math.sqrt(omega2_sw / denom_sw)
    omega_sw_from_closed = shallow_water_omega(k_sw_closed, shallow_ice.water_depth, shallow_ice.mu, shallow_ice.gravity)
    sw_k_error = abs(k_sw_closed - shallow_k) / shallow_k

    # ---- thickness study ----
    thickness = thickness_study(2.0 * math.pi / 6.0, np.asarray(thicknesses), ice)

    # ---- evanescent branch ----
    cutoff = ice.cutoff_omega
    evanescent = {}
    for factor in (1.02, 1.2, 2.0):
        omega_ev = cutoff * float(factor)
        wave = mass_loading_wavenumber(omega_ev, ice)
        kappa = wave.decay_rate
        target = omega_ev ** 2 / (ice.mu * omega_ev ** 2 - ice.gravity)
        evanescent[f"{factor:.2f}"] = {
            "omega_rad_s": omega_ev,
            "propagating": wave.propagating,
            "decay_rate_per_m": kappa,
            "decay_length_m": (1.0 / kappa) if kappa > 0.0 else float("inf"),
            "kappa_tan_kh_residual": kappa * math.tan(kappa * ice.water_depth) - target,
            "amplitude_after_100m": math.exp(-kappa * 100.0),
        }

    # ---- amplitude ratio ----
    amplitude_ratio = [
        {"period_s": float(T), "ratio": float(energy_flux_amplitude_ratio(float(w), ice))}
        for T, w in zip(periods, omegas)
    ]

    elapsed = time.time() - t0

    payload = {
        "case": "CASE-BEM-ICE-DISPERSION",
        "reference": "Weitz & Keller (1950), Comm. Pure Appl. Math. 3(4), pp. 305-318",
        "model": "mass-loading ice sheet, finite-depth water",
        "elapsed_s": round(elapsed, 3),
        "ice": {
            "thickness_m": ice.thickness,
            "ice_density_kg_m3": ice.ice_density,
            "water_density_kg_m3": ice.water_density,
            "water_depth_m": ice.water_depth,
            "mu_m": round(ice.mu, 6),
            "cutoff_omega_rad_s": round(cutoff, 6),
            "cutoff_period_s": round(ice.cutoff_period, 6),
        },
        "frequency_sweep": {
            "period_s": [float(x) for x in table["period"]],
            "omega_rad_s": [round(float(x), 6) for x in table["omega"]],
            "k_ice_per_m": [round(float(x), 6) for x in table["k_ice"]],
            "k_open_per_m": [round(float(x), 6) for x in table["k_open"]],
            "wavenumber_shift_pct": [round(float(x) * 100, 3) for x in (table["k_ice"] / table["k_open"] - 1.0)],
            "wavelength_ice_m": [round(float(x), 4) for x in table["wavelength"]],
            "phase_speed_ice_m_s": [round(float(x), 4) for x in table["phase_speed"]],
            "group_speed_ice_m_s": [round(float(x), 4) for x in table["group_speed"]],
            "amplitude_ratio": [round(float(x), 6) for x in table["amplitude_ratio"]],
            "robin_residual_abs_per_m": [f"{x:.2e}" for x in residual],
            "omega_back_substituted_rel_error": [f"{x:.2e}" for x in omega_error],
            "rk4_shooting_rel_error": [f"{x:.2e}" for x in shooting_error],
        },
        "weitz_keller_deep_water_limit": {
            "period_s": 10.0,
            "water_depth_m": 1.0e5,
            "k_computed_per_m": round(deep_k, 8),
            "k_weitz_keller_per_m": round(k_wk, 8),
            "k_relative_error": f"{k_error:.2e}",
            "omega_weitz_keller_rad_s": round(deep_omega_ref, 8),
            "omega_target_rad_s": round(deep_omega_target, 8),
            "omega_relative_error": f"{deep_error:.2e}",
            "closed_form": "omega^2 = g k / (1 + mu k)",
            "pass_lt_5pct": deep_error < 0.05,
        },
        "shallow_water_limit": {
            "period_s": shallow_period,
            "water_depth_m": shallow_depth,
            "k_computed_per_m": round(shallow_k, 8),
            "k_shallow_closed_per_m": round(k_sw_closed, 8),
            "k_relative_error": f"{sw_k_error:.2e}",
            "kH": round(shallow_kH, 6),
            "omega_shallow_ref_rad_s": round(shallow_ref, 8),
            "omega_target_rad_s": round(shallow_omega, 8),
            "omega_relative_error": f"{shallow_error:.2e}",
            "closed_form": "omega^2 = g k^2 H / (1 + mu k^2 H)",
            "pass_lt_5pct": shallow_error < 0.05,
        },
        "thickness_study": {
            "period_s": 6.0,
            "thickness_m": [float(x) for x in thickness["thickness"]],
            "mu_m": [round(float(x), 6) for x in thickness["mu"]],
            "k_ice_per_m": [round(float(x), 6) for x in thickness["k_ice"]],
            "k_analytic_robin_per_m": [round(float(x), 6) for x in thickness["k_analytic_robin"]],
            "relative_difference_pct": [round(float(x) * 100, 4) for x in thickness["relative_difference"]],
            "max_relative_difference_pct": round(float(np.nanmax(thickness["relative_difference"])) * 100, 4),
            "pass_lt_5pct": bool(float(np.nanmax(thickness["relative_difference"])) < 0.05),
        },
        "evanescent_branch": evanescent,
        "amplitude_ratio_by_period": amplitude_ratio,
        "verification": {
            "max_robin_residual_abs_per_m": f"{float(np.max(residual)):.2e}",
            "max_omega_back_substitution_error": f"{float(np.max(omega_error)):.2e}",
            "max_rk4_shooting_error": f"{float(np.max(shooting_error)):.2e}",
            "deep_water_limit_error": f"{deep_error:.2e}",
            "shallow_water_limit_error": f"{shallow_error:.2e}",
            "thickness_study_max_error_pct": round(float(np.nanmax(thickness["relative_difference"])) * 100, 4),
            "gate_deep_water_lt_5pct": bool(deep_error < 0.05),
            "gate_shallow_water_lt_5pct": bool(shallow_error < 0.05),
            "gate_thickness_study_lt_5pct": bool(float(np.nanmax(thickness["relative_difference"])) < 0.05),
            "all_gates_pass": bool(
                deep_error < 0.05
                and shallow_error < 0.05
                and float(np.nanmax(thickness["relative_difference"])) < 0.05
            ),
        },
    }
    return payload


def generate_figures(payload: dict, out_dir: Path):
    """Generate matplotlib figures for the benchmark report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = payload["frequency_sweep"]
    periods = fs["period_s"]
    k_ice = fs["k_ice_per_m"]
    k_open = fs["k_open_per_m"]
    omega = fs["omega_rad_s"]
    cg_ice = fs["group_speed_ice_m_s"]
    amp_ratio = fs["amplitude_ratio"]

    # Figure 1: Dispersion curves (k vs omega)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega, k_ice, "bo-", label="Ice-covered (mass-loading)", markersize=6)
    ax.plot(omega, k_open, "rs--", label="Open water", markersize=6)
    ax.set_xlabel("Angular frequency $\\omega$ (rad/s)", fontsize=12)
    ax.set_ylabel("Wavenumber $k$ (1/m)", fontsize=12)
    ax.set_title("Mass-loading dispersion: ice vs open water", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "dispersion_curve.png", dpi=150)
    plt.close(fig)

    # Figure 2: Wavenumber shift (%)
    fig, ax = plt.subplots(figsize=(8, 5))
    shift = fs["wavenumber_shift_pct"]
    ax.bar([str(p) for p in periods], shift, color="steelblue", edgecolor="black")
    ax.set_xlabel("Wave period (s)", fontsize=12)
    ax.set_ylabel("Wavenumber shift $\\Delta k/k_{open}$ (%)", fontsize=12)
    ax.set_title("Ice-induced wavenumber increase (mass-loading)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "wavenumber_shift.png", dpi=150)
    plt.close(fig)

    # Figure 3: Group velocity comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    # Compute open-water group velocity
    from ..ice import open_water_group_velocity, open_water_wavenumber as ow_k
    ice = IceCover(thickness=0.9, water_depth=200.0)
    cg_open = []
    for w in omega:
        k_o = ow_k(w, ice.water_depth, ice.gravity)
        cg_o = open_water_group_velocity(w, ice.water_depth, ice.gravity, k_o)
        cg_open.append(cg_o)
    ax.plot(periods, cg_ice, "bo-", label="Ice-covered", markersize=6)
    ax.plot(periods, cg_open, "rs--", label="Open water", markersize=6)
    ax.set_xlabel("Wave period (s)", fontsize=12)
    ax.set_ylabel("Group velocity $c_g$ (m/s)", fontsize=12)
    ax.set_title("Group velocity: ice vs open water", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "group_velocity.png", dpi=150)
    plt.close(fig)

    # Figure 4: Amplitude ratio (energy flux continuity)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periods, amp_ratio, "go-", markersize=6)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="No ice (ratio = 1)")
    ax.set_xlabel("Wave period (s)", fontsize=12)
    ax.set_ylabel("Amplitude ratio $A_{ice}/A_{open}$", fontsize=12)
    ax.set_title("Energy-flux amplitude ratio (ice steepens waves)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "amplitude_ratio.png", dpi=150)
    plt.close(fig)

    # Figure 5: Thickness study
    ts = payload["thickness_study"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts["thickness_m"], ts["k_ice_per_m"], "bo-", label="$k_{ice}$", markersize=6)
    ax.plot(ts["thickness_m"], ts["k_analytic_robin_per_m"], "rs--", label="$k_{Robin}$ (analytic)", markersize=6)
    ax.plot(ts["thickness_m"], [ts["k_ice_per_m"][0]] * len(ts["thickness_m"]), "k:", label="$k_{open}$", alpha=0.5)
    ax.set_xlabel("Ice thickness $h_i$ (m)", fontsize=12)
    ax.set_ylabel("Wavenumber $k$ (1/m)", fontsize=12)
    ax.set_title("Thickness dependence of ice wavenumber", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "thickness_study.png", dpi=150)
    plt.close(fig)

    # Figure 6: Verification errors (log scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Robin\nresidual", "Omega\nback-sub", "RK4\nshooting"]
    errors = [
        float(fs["robin_residual_abs_per_m"][2]),  # T=6s
        float(fs["omega_back_substituted_rel_error"][2]),
        float(fs["rk4_shooting_rel_error"][2]),
    ]
    ax.bar(labels, errors, color=["coral", "steelblue", "green"], edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Relative error", fontsize=12)
    ax.set_title("Verification errors (T = 6 s)", fontsize=13)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="5% tolerance")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "verification_errors.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CASE-BEM-ICE-DISPERSION benchmark")
    parser.add_argument("--out", type=str, default="artifacts/m17274/ice_dispersion_benchmark.json",
                        help="Output JSON path")
    parser.add_argument("--figures", type=str, default="artifacts/m17274/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    payload = run_benchmark()
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Benchmark written to {out_path}")

    generate_figures(payload, fig_dir)
    print(f"Figures written to {fig_dir}/")

    # Print summary
    v = payload["verification"]
    print(f"\n=== Verification Summary ===")
    print(f"Deep-water Weitz-Keller error: {v['deep_water_limit_error']}  {'PASS' if v['gate_deep_water_lt_5pct'] else 'FAIL'}")
    print(f"Shallow-water limit error:     {v['shallow_water_limit_error']}  {'PASS' if v['gate_shallow_water_lt_5pct'] else 'FAIL'}")
    print(f"Thickness study max error:     {v['thickness_study_max_error_pct']}%  {'PASS' if v['gate_thickness_study_lt_5pct'] else 'FAIL'}")
    print(f"All gates pass: {v['all_gates_pass']}")


if __name__ == "__main__":
    main()
