#!/usr/bin/env python3
"""Generate the numeric tables of the wave-ice-ship report from the artifacts.

Reads ``artifacts/ice_dispersion.json``, ``artifacts/ice_classical_reference.json``
and (when present) ``artifacts/ice_ship_rao.json`` and writes
``docs/ice_water_ship_numbers.md``.  No number in the report is typed by hand:
every table is a direct rendering of the JSON evidence, so the report and the
machine-readable artifacts cannot drift apart.

Usage
-----
    PYTHONPATH=src python tools/ice_ship_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "docs" / "ice_water_ship_numbers.md"


def _fmt(value, digits=4) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "n/a"
    if value != 0.0 and (abs(value) >= 1.0e5 or abs(value) < 1.0e-3):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def dispersion_section(payload: dict) -> list[str]:
    ice = payload["ice"]
    sweep = payload["frequency_sweep"]
    ver = payload["verification"]
    lines = [
        "## A. Mass-loading ice-cover dispersion (no BEM)",
        "",
        f"Ice cover: h_i = {_fmt(ice['thickness_m'], 2)} m, rho_i = {_fmt(ice['ice_density'], 0)} kg/m^3, "
        f"H = {_fmt(ice['water_depth_m'], 0)} m, mu = {_fmt(ice['mu_m'], 4)} m, "
        f"cut-off omega = {_fmt(ice['cutoff_omega_rad_s'], 3)} rad/s "
        f"(T = {_fmt(ice['cutoff_period_s'], 3)} s).",
        "",
        "| T (s) | k_ice (1/m) | k_open (1/m) | k_ice/k_open - 1 | lambda_ice (m) | c_ice (m/s) | cg_ice (m/s) | A_ice/A_open | Robin residual (1/m) | RK4 shooting error |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, period in enumerate(sweep["period_s"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(period, 2),
                    _fmt(sweep["k_ice_per_m"][i], 5),
                    _fmt(sweep["k_open_per_m"][i], 5),
                    _fmt(sweep["wavenumber_shift"][i], 4),
                    _fmt(sweep["wavelength_ice_m"][i], 1),
                    _fmt(sweep["phase_speed_ice_m_s"][i], 2),
                    _fmt(sweep["group_speed_ice_m_s"][i], 2),
                    _fmt(sweep["amplitude_ratio_energy_flux"][i], 4),
                    _fmt(sweep["robin_residual_abs_per_m"][i], 2),
                    _fmt(sweep["rk4_shooting_relative_error"][i], 2),
                ]
            )
            + " |"
        )
    limits = payload["limits"]
    study = payload["thickness_study"]
    lines += [
        "",
        "Independent references (the verification gate of this leg):",
        "",
        "| reference | regime | error |",
        "|---|---|---|",
        f"| deep-water Weitz-Keller omega^2 = g k/(1+mu k) | T = {_fmt(limits['deep_water_weitz_keller']['period_s'], 1)} s | {_fmt(limits['deep_water_weitz_keller']['relative_error'], 2)} |",
        f"| shallow-water omega^2 = g k^2 H/(1+mu k^2 H) | T = {_fmt(limits['shallow_water']['period_s'], 1)} s, kH = {_fmt(limits['shallow_water']['kH'], 3)} | {_fmt(limits['shallow_water']['relative_error'], 2)} |",
        f"| RK4 shooting of W'' = k^2 W (no cosh) | T = 4-14 s | {_fmt(ver['max_rk4_shooting_error'], 2)} (worst) |",
        f"| Robin residual k tanh(kH) - omega^2/(g - mu omega^2) | T = 3-14 s | {_fmt(ver['max_robin_residual_abs_per_m'], 2)} 1/m |",
        f"| independent Robin root for k(h_i) | h_i = 0.3-2.5 m, T = {_fmt(study['period_s'], 1)} s | {_fmt(study['max_relative_difference'], 2)} (worst) |",
        "",
        "Ice-thickness study, T = "
        f"{_fmt(study['period_s'], 1)} s (gate: < 5 %):",
        "",
        "| h_i (m) | mu (m) | k_ice (1/m) | k analytic (1/m) | relative difference |",
        "|---|---|---|---|---|",
    ]
    for i, thickness in enumerate(study["thickness_m"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(thickness, 2),
                    _fmt(study["mu_m"][i], 4),
                    _fmt(study["k_ice_per_m"][i], 5),
                    _fmt(study["k_analytic_robin_per_m"][i], 5),
                    _fmt(study["relative_difference"][i], 2),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Evanescent branch above the cut-off (no propagating root exists; k = i kappa):",
        "",
        "| omega/cut-off | kappa (1/m) | decay length (m) | amplitude after 100 m | residual of kappa tan(kappa H) |",
        "|---|---|---|---|---|",
    ]
    for key, row in payload["evanescent_branch"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    _fmt(row["decay_rate_per_m"], 5),
                    _fmt(row["decay_length_m"], 1),
                    _fmt(row["amplitude_after_100m"], 4),
                    _fmt(row["kappa_tan_kh_residual"], 2),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def classical_section(payload: dict) -> list[str]:
    lines = [
        "## B. Open-water chain against a classical reference (Lamb sphere)",
        "",
        f"Submerged sphere r = {_fmt(payload['radius_m'], 1)} m at {_fmt(payload['submergence_m'], 1)} m depth; "
        f"reference: {payload['reference']}.",
        "",
        "| mesh | panels | omega (rad/s) | nu (1/m) | A33 (kg) | B33 (kg/s) | A33/A33_ref - 1 | A33/(rho V) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["mesh"],
                    _fmt(row["panels"]),
                    _fmt(row["omega"], 2),
                    _fmt(row["nu"], 5),
                    _fmt(row["A33"], 4),
                    _fmt(row["B33"], 3),
                    _fmt(row["A33_relative_error"], 4),
                    _fmt(row["displaced_mass_ratio"], 4),
                ]
            )
            + " |"
        )
    checks = payload["no_ice_limit_check"]
    worst_limit = max(
        max(check["added_mass_relative_difference"], check["damping_relative_difference"])
        for check in checks.values()
    )
    lines += [
        "",
        "Ice chain in the mu -> 0 limit versus the open-water solve (consistency gate):",
        "",
        "| mesh | wavenumber difference | added-mass difference | damping difference |",
        "|---|---|---|---|",
    ]
    for label, check in checks.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt(check["wavenumber_relative_difference"], 2),
                    _fmt(check["added_mass_relative_difference"], 2),
                    _fmt(check["damping_relative_difference"], 2),
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"Worst fine-mesh A33 error: {_fmt(payload['worst_fine_mesh_A33_relative_error'], 4)} "
        f"(gate < 15 %: {_fmt(payload['verification']['gate_added_mass_lt_15_percent'])}). "
        f"Worst mu -> 0 difference: {_fmt(worst_limit, 2)}.",
        "",
    ]
    return lines


def hull_section(payload: dict) -> list[str]:
    hull = payload["hull"]
    geometry = payload["geometry"]
    lines = [
        "## C. Idealised PC6-like hull: geometry gate and 6-DOF RAO",
        "",
        f"Hull: L = {_fmt(hull['length_m'], 1)} m, B = {_fmt(hull['breadth_m'], 1)} m, "
        f"T = {_fmt(hull['draft_m'], 1)} m, waterline exponent n = {_fmt(hull['waterline_exponent'], 1)}, "
        f"section exponent q = {_fmt(hull['section_exponent'], 1)}, keel ratio = {_fmt(hull['keel_ratio'], 2)}, "
        f"end ratio = {_fmt(hull['end_ratio'], 2)}.",
        "",
        "Geometry gate (mesh versus independent parametric quadrature):",
        "",
        "| mesh | panels | Cb (mesh) | Cb (reference) | Cb error | L error | B error | T error | Awp error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, check in geometry.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt(check["panels"]),
                    _fmt(check["block_coefficient"], 5),
                    _fmt(check["block_coefficient_reference"], 5),
                    _fmt(check["cb_relative_error"], 4),
                    _fmt(check["length_relative_error"], 1),
                    _fmt(check["breadth_relative_error"], 1),
                    _fmt(check["draft_relative_error"], 1),
                    _fmt(check["waterplane_area_relative_error"], 4),
                ]
            )
            + " |"
        )
    mass = payload["mass_properties"]
    ice = payload["ice"]
    lines += [
        "",
        f"Mass = {_fmt(mass['mass_kg'], 4)} kg at the design draft, CG = "
        f"({', '.join(_fmt(x, 2) for x in mass['center_of_gravity'])}) m; assumptions: {mass['assumptions']}.",
        f"Ice cover: h_i = {_fmt(ice['thickness_m'], 2)} m, H = {_fmt(ice['water_depth_m'], 1)} m, "
        f"mu = {_fmt(ice['mu_m'], 4)} m; head seas (wave angle "
        f"{_fmt(payload['wave_angle_deg'], 0)} deg).",
        "",
        "| T (s) | propagating | k_ice (1/m) | k_open (1/m) | k_i H | heave RAO open | heave RAO ice | pitch RAO open (rad/m) | pitch RAO ice (rad/m) | A33 open (kg) | A33 ice (kg) | B33 open (kg/s) | B33 ice (kg/s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["frequencies"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["period"], 1),
                    _fmt(row["propagating"]),
                    _fmt(row["k_ice"], 5) if math.isfinite(row["k_ice"]) else "n/a",
                    _fmt(row["k_open"], 5) if math.isfinite(row["k_open"]) else "n/a",
                    _fmt(row["kH"], 1) if math.isfinite(row["kH"]) else "n/a",
                    _fmt(row["heave_rao_open"], 4),
                    _fmt(row["heave_rao_ice"], 4),
                    _fmt(row["pitch_rao_open"], 4),
                    _fmt(row["pitch_rao_ice"], 4),
                    _fmt(row["a33_open"], 4),
                    _fmt(row["a33_ice"], 4),
                    _fmt(row["b33_open"], 4),
                    _fmt(row["b33_ice"], 4),
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"Mesh convergence at T = 8 s (heave RAO, head seas):",
        "",
        "| mesh | panels | A33 ice (kg) | A33 open (kg) | heave RAO ice | heave RAO open |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["mesh_convergence"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["mesh"],
                    _fmt(row["panels"]),
                    _fmt(row["a33_heave"], 4),
                    _fmt(row["a33_heave_open"], 4),
                    _fmt(row["heave_rao_ice"], 4),
                    _fmt(row["heave_rao_open"], 4),
                ]
            )
            + " |"
        )
    added = payload["added_resistance"]
    lines += [
        "",
        f"Added resistance in ice: **{added['status']}** -- {added['reason']}",
        "",
        f"Runtime of this stage: {_fmt(payload['runtime_s'], 0)} s.",
        "",
    ]
    return lines


def main() -> int:
    parts = ["# Wave-ice-ship numbers (generated)", ""]
    parts.append(
        "Every table below is generated from the JSON evidence written by "
        "`python -m tensorbem.benchmarks.ice_ship_verification` and rendered by "
        "`tools/ice_ship_report.py`; no value is transcribed by hand, so the report "
        "and the machine-readable evidence cannot drift apart."
    )
    parts.append("")
    dispersion = ARTIFACTS / "ice_dispersion.json"
    if dispersion.exists():
        parts += dispersion_section(json.loads(dispersion.read_text()))
    classical = ARTIFACTS / "ice_classical_reference.json"
    if classical.exists():
        parts += classical_section(json.loads(classical.read_text()))
    hull = ARTIFACTS / "ice_ship_rao.json"
    if hull.exists():
        parts += hull_section(json.loads(hull.read_text()))
    else:
        parts += [
            "## C. Hull stage",
            "",
            "`artifacts/ice_ship_rao.json` is not available in this checkout, so the hull "
            "tables are withheld rather than transcribed from memory.",
            "",
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT} ({len(parts)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())