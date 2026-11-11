#!/usr/bin/env python3
"""Verification of the unified flexural-gravity + mass-loading dispersion.

Generates (all numbers machine-written, no hand transcription):

    docs/flexural_gravity_mass_loading/verification_report.json   criteria + tables
    docs/flexural_gravity_mass_loading/results.md                 numeric tables
    docs/flexural_gravity_mass_loading/fig_*.png                  figures

Acceptance criteria (V1..V10) are listed in ``CRITERIA`` below; every entry
records the measured quantity, the reference it is compared to, and pass/fail.

Run:  PYTHONPATH=src python scripts/verify_flexural_gravity_mass_loading.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.flexural_gravity import (  # noqa: E402
    IcePlate,
    crossover_wavenumber,
    deep_water_crossover_wavenumber,
    flexural_gravity_group_velocity_mass_loading,
    flexural_gravity_omega,
    flexural_gravity_omega_mass_loading,
    flexural_gravity_omega2_mass_loading,
    flexural_gravity_wavenumber,
    flexural_gravity_wavenumber_mass_loading,
    ice_edge_coefficients,
    mass_loading_parameter,
    open_water_group_velocity,
    open_water_wavenumber,
    wavelength_ratio_mass_loading,
)
from tensorbem.ice import IceCover, mass_loading_omega  # noqa: E402

OUT = ROOT / "docs" / "flexural_gravity_mass_loading"
TWO_PI = 2.0 * math.pi
ICE_THICKNESSES = (0.1, 0.5, 1.0, 2.0)
FREQUENCIES = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rel_err(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def numeric_crossover_by_bisection(plate: IcePlate) -> float:
    """Independent bisection of c(k) - c_open(k), no module root-solver used."""
    def phase_diff(k: float) -> float:
        omega = flexural_gravity_omega_mass_loading(k, plate)
        c_ice = omega / k
        c_open = math.sqrt(plate.gravity * math.tanh(k * plate.water_depth) / k)
        return c_ice - c_open

    lo = 1.0e-6 * deep_water_crossover_wavenumber(plate)
    hi = 100.0 * deep_water_crossover_wavenumber(plate)
    if phase_diff(lo) >= 0.0 or phase_diff(hi) <= 0.0:  # pragma: no cover
        raise RuntimeError("crossover not bracketed")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if phase_diff(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1.0e-15 * hi:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# V1  inertia -> 0 recovers the rigidity-only relation (2)
# ---------------------------------------------------------------------------
def criterion_v1() -> dict:
    plate = IcePlate(thickness=1.0, water_depth=200.0)
    light_plate = IcePlate(thickness=1.0, water_depth=200.0, ice_density=1.0e-9)
    worst = 0.0
    for k in np.logspace(-4, 2, 61):
        a = flexural_gravity_omega_mass_loading(float(k), light_plate)
        b = flexural_gravity_omega(float(k), plate)
        worst = max(worst, rel_err(a, b))
    return {
        "description": "mu -> 0 (rho_i -> 0) recovers omega^2 = (gk + Dk^5/rho)tanh(kH)",
        "max_relative_error": worst,
        "threshold": 1.0e-9,
        "pass": worst < 1.0e-9,
    }


# ---------------------------------------------------------------------------
# V2  short-wave deep-water limit -> classical plate-in-water dispersion
# ---------------------------------------------------------------------------
def criterion_v2() -> dict:
    plate = IcePlate(thickness=1.0, water_depth=200.0)
    mu = mass_loading_parameter(plate)
    rows = []
    for k in (5.0, 10.0, 20.0):
        measured = flexural_gravity_omega2_mass_loading(k, plate)
        # omega^2 = D k^5 / (rho_w + rho_i h k)   (classical, gravity neglected)
        reference = plate.flexural_rigidity * k ** 5 / (
            plate.water_density * (1.0 + mu * k)
        )
        rows.append({"k": k, "measured": measured, "reference": reference,
                     "relative_error": rel_err(measured, reference)})
    worst = max(r["relative_error"] for r in rows)
    return {
        "description": "kH>>1, Dk^4>>gk: omega^2 -> D k^5/(rho_w + rho_i h k)",
        "rows": rows,
        "max_relative_error": worst,
        "threshold": 1.0e-6,
        "pass": worst < 1.0e-6,
    }


# ---------------------------------------------------------------------------
# V3  long-wave shallow-water limit -> gH k^2 / (1 + mu H k^2)
# ---------------------------------------------------------------------------
def criterion_v3() -> dict:
    rows = []
    for h_ice in ICE_THICKNESSES:
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        k = 1.0e-3 / plate.water_depth
        mu = mass_loading_parameter(plate)
        measured = flexural_gravity_omega2_mass_loading(k, plate)
        reference = plate.gravity * k ** 2 * plate.water_depth / (
            1.0 + mu * plate.water_depth * k ** 2
        )
        rows.append({"ice_thickness": h_ice, "k_times_H": k * plate.water_depth,
                     "measured": measured, "reference": reference,
                     "relative_error": rel_err(measured, reference)})
    worst = max(r["relative_error"] for r in rows)
    return {
        "description": "kH -> 0: omega^2 -> g H k^2/(1 + mu H k^2) (shallow mass-loading)",
        "rows": rows,
        "max_relative_error": worst,
        "threshold": 1.0e-5,
        "pass": worst < 1.0e-5,
    }


# ---------------------------------------------------------------------------
# V4  cross-module consistency: D -> 0 reproduces tensorbem.ice mass loading
# ---------------------------------------------------------------------------
def criterion_v4() -> dict:
    rows = []
    for h_ice in ICE_THICKNESSES:
        stiff = IcePlate(thickness=h_ice, water_depth=200.0)
        soft = IcePlate(thickness=h_ice, water_depth=200.0, young_modulus=1.0e-6)
        cover = IceCover(thickness=h_ice, water_depth=200.0)
        worst = 0.0
        for k in (1.0e-4, 1.0e-3, 1.0e-2, 0.1):
            a = flexural_gravity_omega_mass_loading(k, soft)
            b = mass_loading_omega(k, cover)
            worst = max(worst, rel_err(a, b))
        rows.append({
            "ice_thickness": h_ice,
            "mu_flexural_gravity_module": mass_loading_parameter(stiff),
            "mu_ice_module": cover.mu,
            "mu_agreement": rel_err(mass_loading_parameter(stiff), cover.mu),
            "max_omega_relative_error_D_to_zero": worst,
        })
    worst_omega = max(r["max_omega_relative_error_D_to_zero"] for r in rows)
    worst_mu = max(r["mu_agreement"] for r in rows)
    return {
        "description": "eq.(3) with E->0 equals tensorbem.ice mass-loading omega on k in [1e-4,0.1]",
        "rows": rows,
        "max_relative_error": worst_omega,
        "max_mu_relative_error": worst_mu,
        "threshold": 1.0e-9,
        "pass": worst_omega < 1.0e-9 and worst_mu < 1.0e-15,
    }


# ---------------------------------------------------------------------------
# V5  closed-form crossover wavenumber vs independent bisection
# ---------------------------------------------------------------------------
def criterion_v5() -> dict:
    rows = []
    for h_ice in ICE_THICKNESSES:
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        closed = deep_water_crossover_wavenumber(plate)
        finite = crossover_wavenumber(plate, deep_water=False)
        independent = numeric_crossover_by_bisection(plate)
        rows.append({
            "ice_thickness": h_ice,
            "k_c_closed_form": closed,
            "lambda_c_closed_form": TWO_PI / closed,
            "k_c_finite_depth_solver": finite,
            "k_c_independent_bisection": independent,
            "relative_error_vs_independent": rel_err(finite, independent),
            "deep_vs_finite_relative_difference": rel_err(closed, finite),
        })
    worst = max(r["relative_error_vs_independent"] for r in rows)
    return {
        "description": "k_c = (g mu/a)^(1/3) vs independent bisection of c(k)=c_open(k)",
        "rows": rows,
        "max_relative_error": worst,
        "threshold": 1.0e-10,
        "pass": worst < 1.0e-10,
    }


# ---------------------------------------------------------------------------
# V6  analytic group velocity vs central difference of omega(k)
# ---------------------------------------------------------------------------
def criterion_v6() -> dict:
    rows = []
    for h_ice in (0.5, 1.0):
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        worst = 0.0
        for omega in (0.2, 0.6, 2.0, 6.0):
            k = flexural_gravity_wavenumber_mass_loading(omega, plate)
            analytic = flexural_gravity_group_velocity_mass_loading(omega, plate, k)
            h = 1.0e-6 * k
            plus = flexural_gravity_omega_mass_loading(k + h, plate)
            minus = flexural_gravity_omega_mass_loading(k - h, plate)
            numeric = (plus - minus) / (2.0 * h)
            worst = max(worst, rel_err(analytic, numeric))
        rows.append({"ice_thickness": h_ice, "max_relative_error": worst})
    worst = max(r["max_relative_error"] for r in rows)
    return {
        "description": "cg = (N'D - ND')/D^2 /(2 omega) vs 4th-order-free central difference",
        "rows": rows,
        "max_relative_error": worst,
        "threshold": 1.0e-6,
        "pass": worst < 1.0e-6,
    }


# ---------------------------------------------------------------------------
# V7  wavenumber solver round trip omega -> k -> omega
# ---------------------------------------------------------------------------
def criterion_v7() -> dict:
    rows = []
    for h_ice in ICE_THICKNESSES:
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        worst = 0.0
        for omega in np.logspace(-2, 1.5, 25):
            k = flexural_gravity_wavenumber_mass_loading(float(omega), plate)
            back = flexural_gravity_omega_mass_loading(k, plate)
            worst = max(worst, rel_err(back, float(omega)))
        rows.append({"ice_thickness": h_ice, "max_relative_error": worst})
    worst = max(r["max_relative_error"] for r in rows)
    return {
        "description": "omega -> k -> omega round trip, omega in [1e-2, 31.6] rad/s",
        "rows": rows,
        "max_relative_error": worst,
        "threshold": 1.0e-9,
        "pass": worst < 1.0e-9,
    }


# ---------------------------------------------------------------------------
# V8  ice-edge energy conservation with plate inertia
# ---------------------------------------------------------------------------
def criterion_v8() -> dict:
    rows = []
    for h_ice in (0.5, 1.0):
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        worst = 0.0
        for frequency in (0.05, 0.1, 0.2, 0.5, 1.0):
            res = ice_edge_coefficients(
                TWO_PI * frequency, plate, include_plate_inertia=True
            )
            worst = max(worst, abs(res.energy_residual))
            rows.append({
                "ice_thickness": h_ice,
                "frequency": frequency,
                "R_energy": res.R_energy,
                "T_energy": res.T_energy,
                "energy_residual": res.energy_residual,
            })
    worst = max(abs(r["energy_residual"]) for r in rows)
    return {
        "description": "|R|^2 + |T|^2 = 1 at the ice edge with eq.(3) wavenumbers",
        "rows": rows,
        "max_abs_residual": worst,
        "threshold": 1.0e-12,
        "pass": worst < 1.0e-12,
    }


# ---------------------------------------------------------------------------
# V9  magnitude of the plate-inertia correction in the ocean-wave band
# ---------------------------------------------------------------------------
def inertia_effect_table() -> list:
    rows = []
    for h_ice in ICE_THICKNESSES:
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        k_c = deep_water_crossover_wavenumber(plate)
        for frequency in FREQUENCIES:
            omega = TWO_PI * frequency
            ratio_e3 = wavelength_ratio_mass_loading(omega, plate)
            k_e2 = flexural_gravity_wavenumber(omega, plate)
            k_open = open_water_wavenumber(omega, plate.water_depth, plate.gravity)
            ratio_e2 = k_open / k_e2
            rows.append({
                "ice_thickness": h_ice,
                "mu": mass_loading_parameter(plate),
                "crossover_wavelength": TWO_PI / k_c,
                "frequency": frequency,
                "lambda_ice_over_lambda_open_eq3": ratio_e3,
                "lambda_ice_over_lambda_open_eq2": ratio_e2,
                "inertia_correction_percent": 100.0 * (ratio_e3 / ratio_e2 - 1.0),
            })
    return rows


# ---------------------------------------------------------------------------
# V9b  material sensitivity of the plate-inertia correction
# ---------------------------------------------------------------------------
def material_sensitivity_table() -> list:
    """Inertia correction [%] over Young's modulus and ice thickness."""
    rows = []
    for young in (1.0e9, 5.0e9, 10.0e9):
        for h_ice in (0.5, 1.0, 2.0):
            plate = IcePlate(thickness=h_ice, water_depth=200.0, young_modulus=young)
            k_c = deep_water_crossover_wavenumber(plate)
            for frequency in FREQUENCIES:
                omega = TWO_PI * frequency
                ratio_e3 = wavelength_ratio_mass_loading(omega, plate)
                k_e2 = flexural_gravity_wavenumber(omega, plate)
                k_open = open_water_wavenumber(omega, plate.water_depth, plate.gravity)
                ratio_e2 = k_open / k_e2
                rows.append({
                    "young_modulus": young,
                    "ice_thickness": h_ice,
                    "crossover_wavelength": TWO_PI / k_c,
                    "frequency": frequency,
                    "lambda_ratio_eq3": ratio_e3,
                    "lambda_ratio_eq2": ratio_e2,
                    "inertia_correction_percent": 100.0 * (ratio_e3 / ratio_e2 - 1.0),
                })
    return rows


def criterion_v9(sensitivity: list) -> dict:
    worst = max(abs(r["inertia_correction_percent"]) for r in sensitivity)
    rows = []
    for young in (1.0e9, 5.0e9, 10.0e9):
        sub = [r for r in sensitivity if r["young_modulus"] == young]
        rows.append({
            "young_modulus": young,
            "max_abs_inertia_correction_percent": max(
                abs(r["inertia_correction_percent"]) for r in sub
            ),
        })
    return {
        "description": (
            "max |inertia correction| over E in {1,5,10} GPa, h in {0.5,1,2} m, "
            "f in 0.02..0.30 Hz stays below 10%"
        ),
        "rows": rows,
        "max_abs_inertia_correction_percent": worst,
        "measured": f"max |correction| = {worst:.2f}%",
        "threshold_percent": 10.0,
        "pass": worst < 10.0,
    }


# ---------------------------------------------------------------------------
# V10  arbitration of the "E5" expectation (lambda_trans < lambda_inc)
# ---------------------------------------------------------------------------
def criterion_v10(table: list) -> dict:
    """Sign of (lambda_ice - lambda_open) must flip exactly at the crossover.

    For each case the predicted crossover frequency is
    ``f_c = omega_eq3(k_c) / 2 pi``; the criterion is that
    ``sign(lambda_ratio - 1) == sign(f - f_c)`` for every case, i.e. waves
    longer than ``lambda_c`` are shortened and shorter waves lengthened.
    """
    rows = []
    for h_ice in ICE_THICKNESSES:
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        k_c = deep_water_crossover_wavenumber(plate)
        f_c = flexural_gravity_omega_mass_loading(k_c, plate) / TWO_PI
        sub = [r for r in table if r["ice_thickness"] == h_ice]
        agree = 0
        for row in sub:
            predicted = 1.0 if row["frequency"] > f_c else -1.0
            observed = 1.0 if row["lambda_ice_over_lambda_open_eq3"] > 1.0 else -1.0
            row["crossover_frequency_hz"] = f_c
            row["sign_agrees"] = predicted == observed
            agree += int(predicted == observed)
        rows.append({
            "ice_thickness": h_ice,
            "crossover_wavelength": TWO_PI / k_c,
            "crossover_frequency_hz": f_c,
            "n_cases": len(sub),
            "n_sign_agree": agree,
            "max_abs_inertia_correction_percent": max(
                abs(r["inertia_correction_percent"]) for r in sub
            ),
            "min_lambda_ratio_eq3": min(
                r["lambda_ice_over_lambda_open_eq3"] for r in sub
            ),
            "max_lambda_ratio_eq3": max(
                r["lambda_ice_over_lambda_open_eq3"] for r in sub
            ),
        })
    total = sum(r["n_cases"] for r in rows)
    agree_total = sum(r["n_sign_agree"] for r in rows)
    return {
        "description": (
            "sign(lambda_ice/lambda_open - 1) == sign(f - f_c) for every case; "
            "f_c = omega_eq3(k_c)/2pi"
        ),
        "rows": rows,
        "n_cases_total": total,
        "n_sign_agree": agree_total,
        "agreement_fraction": agree_total / total,
        "measured": f"{agree_total}/{total} cases sign-consistent",
        "max_abs_inertia_correction_percent": max(
            abs(r["inertia_correction_percent"]) for r in table
        ),
        "threshold": 1.0,
        "pass": agree_total == total,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_dispersion_models(plate: IcePlate, path: Path) -> None:
    k = np.logspace(-4, 1.5, 400)
    omega_eq2 = np.array([flexural_gravity_omega(float(x), plate) for x in k])
    omega_eq3 = np.array(
        [flexural_gravity_omega_mass_loading(float(x), plate) for x in k]
    )
    omega_open = np.array(
        [math.sqrt(plate.gravity * float(x) * math.tanh(float(x) * plate.water_depth))
         for x in k]
    )
    omega_bend = np.array(
        [math.sqrt(plate.D_over_rho * float(x) ** 5) for x in k]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.loglog(k, omega_open, "k--", lw=1.4, label="open water: $gk\\tanh(kH)$")
    ax.loglog(k, omega_eq2, "C0-", lw=1.6, label="eq.(2) bending only")
    ax.loglog(k, omega_eq3, "C3-", lw=1.6, label="eq.(3) bending + inertia")
    ax.loglog(k, omega_bend, "C2:", lw=1.4, label="pure flexural $Dk^5/\\rho_w$")
    ax.set_xlabel("wavenumber $k$ [1/m]")
    ax.set_ylabel("frequency $\\omega$ [rad/s]")
    ax.set_title(f"dispersion relations, $h$={plate.thickness} m, $H$={plate.water_depth} m")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ratio = omega_eq3 / omega_eq2
    ax.semilogx(k, 100.0 * (ratio - 1.0), "C3-", lw=1.6)
    k_c = deep_water_crossover_wavenumber(plate)
    ax.axvline(k_c, color="C1", ls="--", lw=1.2,
               label=f"$k_c$={k_c:.4f} 1/m ($\\lambda_c$={TWO_PI / k_c:.1f} m)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("wavenumber $k$ [1/m]")
    ax.set_ylabel("$(\\omega_3/\\omega_2 - 1)$ [%]")
    ax.set_title("plate-inertia correction relative to eq.(2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_phase_speed(table_like, path: Path) -> None:
    k = np.logspace(-3, 0.5, 300)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for idx, h_ice in enumerate(ICE_THICKNESSES):
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        c_ratio = []
        for x in k:
            omega = flexural_gravity_omega_mass_loading(float(x), plate)
            c_ice = omega / float(x)
            c_open = math.sqrt(
                plate.gravity * math.tanh(float(x) * plate.water_depth) / float(x)
            )
            c_ratio.append(c_ice / c_open)
        ax.semilogx(k, c_ratio, lw=1.6, color=f"C{idx}",
                    label=f"$h$={h_ice} m (eq.3)")
        k_c = deep_water_crossover_wavenumber(plate)
        ax.plot([k_c], [1.0], "o", color=f"C{idx}", ms=6)
        ax.axvline(k_c, color=f"C{idx}", ls=":", lw=1.0)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("wavenumber $k$ [1/m]")
    ax.set_ylabel("$c_{ice}/c_{open}$")
    ax.set_title("phase-speed ratio: slow (inertia) left of $k_c$, fast (bending) right")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_wavelength_ratio_vs_frequency(table: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for idx, h_ice in enumerate(ICE_THICKNESSES):
        sub = [r for r in table if r["ice_thickness"] == h_ice]
        freqs = [r["frequency"] for r in sub]
        ax.semilogy(freqs, [r["lambda_ice_over_lambda_open_eq3"] for r in sub],
                    "o-", color=f"C{idx}", lw=1.6, label=f"$h$={h_ice} m (eq.3)")
        ax.semilogy(freqs, [r["lambda_ice_over_lambda_open_eq2"] for r in sub],
                    "x--", color=f"C{idx}", lw=1.0, alpha=0.7,
                    label=f"$h$={h_ice} m (eq.2)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("frequency $f$ [Hz]")
    ax.set_ylabel("$\\lambda_{ice}/\\lambda_{open}$")
    ax.set_title("wavelength ratio in the ocean-wave band (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_edge_coefficients(path: Path) -> None:
    plate = IcePlate(thickness=1.0, water_depth=200.0)
    frequencies = np.linspace(0.02, 1.0, 60)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for flag, color, dash, label in (
        (False, "C0", "-", "eq.(2) bending only"),
        (True, "C3", "--", "eq.(3) bending + inertia"),
    ):
        R = []
        T = []
        for frequency in frequencies:
            res = ice_edge_coefficients(
                TWO_PI * frequency, plate, include_plate_inertia=flag
            )
            R.append(res.R_energy)
            T.append(res.T_energy)
        ax.plot(frequencies, R, color=color, ls=dash, lw=1.5, label=f"R ({label})")
        ax.plot(frequencies, T, color=color, ls=":", lw=1.3, label=f"T ({label})")
    ax.set_xlabel("frequency $f$ [Hz]")
    ax.set_ylabel("energy coefficient")
    ax.set_title("ice-edge reflection / transmission, $h$=1 m")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    resid = []
    for frequency in frequencies:
        res = ice_edge_coefficients(TWO_PI * frequency, plate, include_plate_inertia=True)
        resid.append(abs(res.energy_residual) + 1e-18)
    ax.semilogy(frequencies, resid, "C3-", lw=1.4)
    ax.set_xlabel("frequency $f$ [Hz]")
    ax.set_ylabel("$|R^2+T^2-1|$")
    ax.set_title("energy-conservation residual with plate inertia")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# markdown tables + main
# ---------------------------------------------------------------------------
def fmt(value) -> str:
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.6e}" if abs(value) < 1.0e-3 or abs(value) >= 1.0e4 else f"{value:.6f}"
    return str(value)


def write_results_markdown(report: dict, table: list, sensitivity: list, path: Path) -> None:
    lines = [
        "# 数值结果（程序化生成，零手抄）",
        "",
        "由 `scripts/verify_flexural_gravity_mass_loading.py` 自动写出；"
        "与本目录 `verification_report.json` 同源。",
        "",
        "## 判据汇总",
        "",
        "| 判据 | 内容 | 实测 | 阈值 | 判定 |",
        "|---|---|---|---|---|",
    ]
    for name, crit in report["criteria"].items():
        measured = crit.get("measured")
        if measured is None:
            measured = crit.get("max_relative_error")
        if measured is None:
            measured = crit.get("max_abs_residual")
        if measured is None:
            measured = crit.get("agreement_fraction")
        lines.append(
            f"| {name} | {crit['description']} | {fmt(measured)} | "
            f"{fmt(crit.get('threshold', crit.get('threshold_percent')))} | "
            f"{'PASS' if crit['pass'] else 'FAIL'} |"
        )
    lines += ["", "## 平板惯性对波长比的影响（海洋波频段）", "",
              "| h_ice [m] | mu [m] | lambda_c [m] | f [Hz] | "
              "lambda ratio eq.3 | lambda ratio eq.2 | 惯性修正 [%] |",
              "|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append(
            f"| {row['ice_thickness']} | {fmt(row['mu'])} | "
            f"{fmt(row['crossover_wavelength'])} | {row['frequency']} | "
            f"{fmt(row['lambda_ice_over_lambda_open_eq3'])} | "
            f"{fmt(row['lambda_ice_over_lambda_open_eq2'])} | "
            f"{row['inertia_correction_percent']:.3f} |"
        )
    lines += ["", "## 杨氏模量敏感性（惯性修正 [%]）", "",
              "| E [GPa] | h_ice [m] | " +
              " | ".join(f"f={f}" for f in FREQUENCIES) + " |",
              "|---|---|" + "---|" * len(FREQUENCIES)]
    for young in (1.0e9, 5.0e9, 10.0e9):
        for h_ice in (0.5, 1.0, 2.0):
            sub = [r for r in sensitivity
                   if r["young_modulus"] == young and r["ice_thickness"] == h_ice]
            cells = " | ".join(
                f"{next(r['inertia_correction_percent'] for r in sub if r['frequency'] == f):.2f}"
                for f in FREQUENCIES
            )
            lines.append(f"| {young / 1e9:.0f} | {h_ice} | {cells} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plate = IcePlate(thickness=1.0, water_depth=200.0)

    criteria = {
        "V1_inertia_zero_limit": criterion_v1(),
        "V2_short_wave_limit": criterion_v2(),
        "V3_long_wave_shallow_limit": criterion_v3(),
        "V4_ice_module_consistency": criterion_v4(),
        "V5_crossover_wavenumber": criterion_v5(),
        "V6_group_velocity": criterion_v6(),
        "V7_wavenumber_round_trip": criterion_v7(),
        "V8_ice_edge_energy": criterion_v8(),
    }
    table = inertia_effect_table()
    sensitivity = material_sensitivity_table()
    criteria["V9_inertia_correction_bound"] = criterion_v9(sensitivity)
    criteria["V10_e5_arbitration"] = criterion_v10(table)

    figures = []
    figure_dispersion_models(plate, OUT / "fig_dispersion_models.png")
    figures.append("fig_dispersion_models.png")
    figure_phase_speed(table, OUT / "fig_phase_speed_ratio.png")
    figures.append("fig_phase_speed_ratio.png")
    figure_wavelength_ratio_vs_frequency(table, OUT / "fig_wavelength_ratio.png")
    figures.append("fig_wavelength_ratio.png")
    figure_edge_coefficients(OUT / "fig_edge_coefficients.png")
    figures.append("fig_edge_coefficients.png")

    all_pass = all(c["pass"] for c in criteria.values())
    report = {
        "title": "TensorBEM unified flexural-gravity + mass-loading dispersion",
        "module": "src/tensorbem/flexural_gravity.py (eq. 3)",
        "model": "omega^2 = (g k + D k^5/rho_w) tanh(kH) / (1 + mu k tanh(kH))",
        "parameters": {
            "ice_thickness_values": list(ICE_THICKNESSES),
            "young_modulus": plate.young_modulus,
            "poisson_ratio": plate.poisson_ratio,
            "water_density": plate.water_density,
            "ice_density": plate.ice_density,
            "water_depth": plate.water_depth,
            "gravity": plate.gravity,
            "frequency_band": list(FREQUENCIES),
        },
        "criteria": criteria,
        "inertia_effect_table": table,
        "material_sensitivity_table": sensitivity,
        "figures": figures,
        "all_criteria_pass": all_pass,
    }
    (OUT / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_results_markdown(report, table, sensitivity, OUT / "results.md")

    for name, crit in criteria.items():
        measured = crit.get("max_relative_error", crit.get("max_abs_residual"))
        print(f"{name:32s} {'PASS' if crit['pass'] else 'FAIL'}  {measured!r}")
    print(f"all_criteria_pass = {all_pass}")
    print(f"artifacts -> {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())