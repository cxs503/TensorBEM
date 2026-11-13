#!/usr/bin/env python3
"""Render the ``BEM-ICE-SPHERE-HEAVE`` case evidence into tables and figures.

Reads ``artifacts/case_bem_ice_sphere_heave.json`` (written by
``python -m tensorbem.benchmarks.ice_sphere_heave``) and produces

* ``docs/case_bem_ice_sphere_heave_numbers.md`` -- every table, rendered from the
  JSON so no number in the report is transcribed by hand, and
* ``artifacts/figures/case_ice_sphere_*.png`` -- mesh-convergence, submergence,
  ice-thickness/frequency sensitivity and the heave surface potential.

Usage
-----
    PYTHONPATH=src python tools/ice_sphere_heave_report.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "case_bem_ice_sphere_heave.json"
NUMBERS = ROOT / "docs" / "case_bem_ice_sphere_heave_numbers.md"
FIGDIR = ROOT / "artifacts" / "figures"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    if number != 0.0 and (abs(number) >= 1.0e5 or abs(number) < 1.0e-3):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * float(value):+.{digits}f} %"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _rows_by(payload: dict, prefix: str) -> list[dict]:
    return [row for row in payload["rows"] if row["label"].startswith(prefix)]


def parameter_section(payload: dict) -> list[str]:
    ctx = payload["context"]
    rows = []
    for row in payload["rows"]:
        rows.append(
            [
                row["label"],
                _fmt(row["panels"]),
                _fmt(row["submergence_ratio"], 2),
                _fmt(row["omega_rad_s"], 3),
                _fmt(row["ice_thickness_m"], 4),
                _fmt(row["mu_over_R"], 5),
                _fmt(row["kR_ice"], 5),
                _fmt(row["kR_open"], 5),
                _fmt(row["omega_over_cutoff"], 4),
                _fmt(row["condition_number"], 3),
                _fmt(row["relative_residual"], 3),
            ]
        )
    return (
        [
            "## 1. Parameter table (all points, dimensionless groups included)",
            "",
            f"Radius `R = {_fmt(ctx['radius_m'], 1)} m`, density "
            f"`rho = {_fmt(ctx['density_kg_m3'], 1)} kg/m^3`, "
            f"`g = {_fmt(ctx['gravity_m_s2'], 5)} m/s^2`, water depth "
            f"`H = {_fmt(ctx['water_depth_m'], 1)} m` (ice validity `k_i H >= 3`).",
            "",
            "`mu = rho_i h_i / rho_w` is the mass-loading parameter, "
            "`k_i = omega^2/(g - mu omega^2)` the ice wavenumber, "
            "`omega_cut = sqrt(g/mu)` the mass-loading cut-off "
            "(`omega/omega_cut = 0` marks the open-water control).",
            "",
        ]
        + _table(
            [
                "point",
                "panels",
                "d/R",
                "omega [rad/s]",
                "h_i [m]",
                "mu/R",
                "k_i R",
                "k_0 R",
                "omega/omega_cut",
                "cond(A)",
                "relres",
            ],
            rows,
        )
        + [""]
    )


def comparison_section(payload: dict) -> list[str]:
    rows = []
    for row in payload["rows"]:
        rows.append(
            [
                row["label"],
                _fmt(row["panels"]),
                _fmt(row["A33_kg"], 2),
                _fmt(row["A33_dimensionless"], 5),
                _fmt(row["A33_lamb_kg"], 2),
                _pct(row["A33_relative_error_vs_lamb"]),
                "yes" if row["A33_relative_error_vs_lamb"] < payload["tolerance"] else "**NO**",
                _fmt(row["A33_exact_multipole_kg"], 2),
                _pct(row["A33_relative_error_vs_exact"]),
                _pct(row["exact_offset_vs_lamb"]),
                _fmt(row["B33_kg_per_s"], 3),
            ]
        )
    return (
        [
            "## 2. Reference vs computed, every point",
            "",
            "Reference `A33^Lamb = (2/3) pi rho R^3 = "
            f"{_fmt(payload['reference']['value_kg'], 4)} kg`; "
            "`A33^exact` is the multipole solution at the same `k_i R` and `d/R` "
            "(no discretisation), and `offset` is `(A33^exact - A33^Lamb)/A33^Lamb`, "
            "i.e. the *physics* offset of the finite-submergence ice/free surface.",
            "",
        ]
        + _table(
            [
                "point",
                "panels",
                "A33 [kg]",
                "A33/(rho R^3)",
                "A33^Lamb [kg]",
                "err vs Lamb",
                "<5 %?",
                "A33^exact [kg]",
                "err vs exact",
                "exact offset vs Lamb",
                "B33 [kg/s]",
            ],
            rows,
        )
        + [""]
    )


def gate_section(payload: dict) -> list[str]:
    verification = payload["verification"]
    keys = [key for key in verification if "gate" in key.lower() or key.endswith("(P1)")]
    rows = [[key, _fmt(verification[key])] for key in verification if isinstance(verification[key], bool)]
    extra = [
        ["primary_error_vs_lamb", _pct(verification["primary_error_vs_lamb"])],
        ["primary_error_vs_exact", _pct(verification["primary_error_vs_exact"])],
        ["ladder_observed_order", _fmt(verification.get("ladder_observed_order"), 3)],
        ["ladder_error_vs_exact", ", ".join(_pct(e) for e in verification["ladder_error_vs_exact"])],
        ["deep_error_vs_lamb", _pct(verification.get("deep_error_vs_lamb", float("nan")))],
        ["deep_exact_offset_vs_lamb", _pct(verification.get("deep_exact_offset_vs_lamb", float("nan")))],
    ]
    _ = keys
    return (
        ["## 3. Predeclared gates", ""]
        + _table(["gate", "result"], rows + extra)
        + ["", f"Deviations recorded: {payload.get('deviations') or 'none'}", ""]
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_convergence(payload: dict) -> Path | None:
    ladder = sorted(_rows_by(payload, "ladder__"), key=lambda row: row["panels"])
    if len(ladder) < 2:
        return None
    panels = np.array([row["panels"] for row in ladder], dtype=float)
    err_lamb = np.array([row["A33_relative_error_vs_lamb"] for row in ladder])
    err_exact = np.array([row["A33_relative_error_vs_exact"] for row in ladder])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(panels, 100.0 * err_lamb, "o-", label="BEM vs Lamb $(2/3)\\pi\\rho R^3$")
    ax.loglog(panels, 100.0 * err_exact, "s-", label="BEM vs exact multipole (same $k_iR$)")
    ax.axhline(5.0, color="crimson", ls="--", lw=1.0, label="case tolerance 5 %")
    ax.set_xlabel("panels $N$")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Submerged sphere heave under ice, $d/R=3$, $h_i=0.9$ m, $\\omega=0.5$ rad/s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = FIGDIR / "case_ice_sphere_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_submergence(payload: dict) -> Path | None:
    if not _rows_by(payload, "depth__"):
        return None
    med = sorted(_rows_by(payload, "depth__"), key=lambda row: row["submergence_ratio"])
    med = [row for row in med if row["label"].endswith("_med")]
    fine = [row for row in _rows_by(payload, "depth__") if row["label"].endswith("_fine")]
    ratios = np.array([row["submergence_ratio"] for row in med])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(
        2.0 * math.pi / 3.0, color="crimson", ls="--", lw=1.2,
        label="Lamb limit $(2/3)\\pi\\rho R^3$",
    )
    ax.plot(ratios, [row["A33_exact_multipole_kg"] / (payload["context"]["density_kg_m3"] * payload["context"]["radius_m"] ** 3) for row in med],
            "k-", lw=1.0, marker="", label="exact multipole")
    ax.plot(ratios, [row["A33_dimensionless"] for row in med], "o", label="BEM 352 panels")
    if fine:
        ratios_f = [row["submergence_ratio"] for row in fine]
        ax.plot(ratios_f, [row["A33_dimensionless"] for row in fine], "s", label="BEM 816 panels")
    ax.set_xlabel("submergence $d/R$")
    ax.set_ylabel("$A_{33}/(\\rho R^3)$")
    ax.set_title("Ice-covered heave added mass tends to the Lamb limit with submergence")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = FIGDIR / "case_ice_sphere_submergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_sensitivity(payload: dict) -> Path | None:
    ice = sorted(_rows_by(payload, "ice__"), key=lambda row: row["ice_thickness_m"])
    if not ice:
        return None
    freq = sorted(_rows_by(payload, "freq__"), key=lambda row: (row["omega_rad_s"], row["ice_thickness_m"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    axes[0].axhline(2.0 * math.pi / 3.0, color="crimson", ls="--", lw=1.0, label="Lamb limit")
    axes[0].axhline(1.0, color="grey", ls=":", lw=1.0, label="$\\rho R^3$ (displaced mass)")
    axes[0].plot(
        [max(row["ice_thickness_m"], 1.0e-3) for row in ice],
        [row["A33_dimensionless"] for row in ice],
        "o-",
        label="BEM 816 panels",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ice thickness $h_i$ [m] (1 mm = open-water control)")
    axes[0].set_ylabel("$A_{33}/(\\rho R^3)$")
    axes[0].set_title("$d/R=3$, $\\omega=0.5$ rad/s")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(2.0 * math.pi / 3.0, color="crimson", ls="--", lw=1.0, label="Lamb limit")
    for thickness, marker in ((0.9, "o"), (2.5, "s")):
        subset = [row for row in freq if row["ice_thickness_m"] == thickness]
        axes[1].plot(
            [row["omega_rad_s"] for row in subset],
            [row["A33_dimensionless"] for row in subset],
            marker + "-",
            label=f"BEM 816 panels, $h_i={thickness}$ m",
        )
    axes[1].set_xlabel("$\\omega$ [rad/s]")
    axes[1].set_ylabel("$A_{33}/(\\rho R^3)$")
    axes[1].set_title("frequency sweep at $d/R=3$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    path = FIGDIR / "case_ice_sphere_sensitivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_surface_field(payload: dict) -> Path | None:
    row = next((r for r in payload["rows"] if "surface_potential" in r), None)
    if row is None:
        return None
    centroids = np.asarray(row["surface_potential"]["centroids_m"], dtype=float)
    phi = np.asarray(row["surface_potential"]["phi_heave_real"], dtype=float)
    if phi.size != centroids.shape[0]:
        return None
    x, y, z = centroids.T
    radius = float(row["radius_m"])
    depth = float(row["submergence_m"])
    theta = np.arccos(np.clip((z - (-depth)) / radius, -1.0, 1.0))  # 0 = bottom pole
    phi_ang = np.mod(np.arctan2(y, x), 2.0 * math.pi)

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    scale = float(np.max(np.abs(phi))) or 1.0
    scatter = ax.scatter(
        np.degrees(phi_ang), np.degrees(theta), c=phi / scale, s=16,
        cmap="RdBu_r", vmin=-1.0, vmax=1.0,
    )
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel("polar angle from lower pole [deg]")
    ax.set_title(
        f"Unrolled heave radiation potential Re$\\phi_3$ on the sphere surface "
        f"({row['panels']} panels, $d/R={row['submergence_ratio']:g}$, "
        f"$h_i={row['ice_thickness_m']}$ m)\nnormalised by "
        f"{scale:.3e} m$^2$/s; $|\\phi|$ max/sum ratio shown by colour"
    )
    fig.colorbar(scatter, ax=ax, label="$\\mathrm{Re}\\,\\phi_3 / \\max|\\phi_3|$")
    path = FIGDIR / "case_ice_sphere_surface_potential.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--numbers", type=Path, default=NUMBERS)
    parser.add_argument("--figdir", type=Path, default=None)
    args = parser.parse_args(argv)

    global FIGDIR
    FIGDIR = args.figdir if args.figdir is not None else Path(FIGDIR)
    payload = json.loads(args.artifact.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        figure_convergence(payload),
        figure_submergence(payload),
        figure_sensitivity(payload),
        figure_surface_field(payload),
    ]
    figures = [path for path in candidates if path is not None]

    lines = [
        "# BEM-ICE-SPHERE-HEAVE — machine-rendered numbers",
        "",
        "Generated by `tools/ice_sphere_heave_report.py` from "
        "`artifacts/case_bem_ice_sphere_heave.json`; no value is typed by hand.",
        "",
        f"Case code `{payload['case_code']}`, case-pool entry "
        f"`{payload['case_pool_entry']}`, tolerance `{payload['tolerance']}`.",
        "",
        f"Reference: {payload['reference']['name']} — "
        f"`{payload['reference']['formula']}`; provenance: "
        f"{payload['reference']['provenance']}.",
        "",
        f"Model: {payload['model']['solver']}; interface condition "
        f"`{payload['model']['interface_condition']}`; exact-reference route "
        f"{payload['model']['exact_reference_route']}.",
        "",
    ]
    lines += parameter_section(payload)
    lines += comparison_section(payload)
    lines += gate_section(payload)
    lines += ["## 4. Figures", ""]
    lines += [f"* `{path.relative_to(ROOT)}`" for path in figures]
    args.numbers.parent.mkdir(parents=True, exist_ok=True)
    args.numbers.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.numbers}")
    for path in figures:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())