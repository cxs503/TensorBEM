#!/usr/bin/env python3
"""Generate comparison figures for CASE-BEM-HULME-HEMISPHERE.

Produces publication-quality PNGs comparing TensorBEM BEM results against
Hulme (1982) semi-analytical reference for the floating hemisphere:

1. ``hulme_radiation_coefficients.png`` -- A11, B11, A33, B33 vs ka
2. ``hulme_excitation_forces.png`` -- |Fx|, |Fz| and phases vs ka
3. ``hulme_error_bars.png`` -- relative error for all quantities vs ka

Usage:
    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/make_figures.py \\
        --out benchmarks/verified/casepool_bem_hulme_hemisphere
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from tensorbem.benchmarks import hulme_hemisphere_case as case


def _load_results(out_dir: Path) -> dict[str, Any]:
    """Load the committed results.json."""
    path = out_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"results.json not found at {path}")
    with open(path) as f:
        return json.load(f)


def _extract_band(rows: list[dict], kind: str, subdivisions: int) -> list[dict]:
    """Extract band rows of a given kind and subdivision level."""
    return sorted(
        [r for r in rows if r["kind"] == kind and r["subdivisions"] == subdivisions],
        key=lambda r: r["ka"],
    )


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)


def make_radiation_figure(band_case: list[dict], band_coarse: list[dict],
                          out_path: Path) -> str:
    """Figure 1: Radiation coefficients A11, B11, A33, B33 vs ka."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ka_case = [r["ka"] for r in band_case]
    ka_coarse = [r["ka"] for r in band_coarse]

    quantities = [
        ("A11_dimless", "A₁₁ / (ρ∇)", "Surge added mass"),
        ("B11_dimless", "B₁₁ / (ρω∇)", "Surge damping"),
        ("A33_dimless", "A₃₃ / (ρ∇)", "Heave added mass"),
        ("B33_dimless", "B₃₃ / (ρω∇)", "Heave damping"),
    ]

    for idx, (key, ylabel, title) in enumerate(quantities):
        ax = axes[idx // 2, idx % 2]
        ref_vals = [r["reference"][key] for r in band_case]

        ax.plot(ka_case, ref_vals, "k-", linewidth=2, label="Hulme (1982) ref", marker="o", markersize=6)
        ax.plot(ka_case, [r[key] for r in band_case], "b--", linewidth=1.5,
                label="BEM 1024 panels", marker="s", markersize=5)
        ax.plot(ka_coarse, [r[key] for r in band_coarse], "r:", linewidth=1.5,
                label="BEM 256 panels", marker="^", markersize=5)

        _style_ax(ax, "ka", ylabel, title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def make_excitation_figure(band_case: list[dict], band_coarse: list[dict],
                           out_path: Path) -> str:
    """Figure 2: Excitation forces magnitude and phase vs ka."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ka_case = [r["ka"] for r in band_case]
    ka_coarse = [r["ka"] for r in band_coarse]

    # |Fx|
    ax = axes[0, 0]
    ref_fx = [r["reference"]["Fx_dimless"] for r in band_case]
    ax.plot(ka_case, ref_fx, "k-", linewidth=2, label="Hulme (1982) ref", marker="o", markersize=6)
    ax.plot(ka_case, [r["Fx_dimless"] for r in band_case], "b--", linewidth=1.5,
            label="BEM 1024 panels", marker="s", markersize=5)
    ax.plot(ka_coarse, [r["Fx_dimless"] for r in band_coarse], "r:", linewidth=1.5,
            label="BEM 256 panels", marker="^", markersize=5)
    _style_ax(ax, "ka", "|Fx| / (ρgAa²)", "Surge excitation force magnitude")

    # |Fz|
    ax = axes[0, 1]
    ref_fz = [r["reference"]["Fz_dimless"] for r in band_case]
    ax.plot(ka_case, ref_fz, "k-", linewidth=2, label="Hulme (1982) ref", marker="o", markersize=6)
    ax.plot(ka_case, [r["Fz_dimless"] for r in band_case], "b--", linewidth=1.5,
            label="BEM 1024 panels", marker="s", markersize=5)
    ax.plot(ka_coarse, [r["Fz_dimless"] for r in band_coarse], "r:", linewidth=1.5,
            label="BEM 256 panels", marker="^", markersize=5)
    _style_ax(ax, "ka", "|Fz| / (ρgAa²)", "Heave excitation force magnitude")

    # Phase Fx
    ax = axes[1, 0]
    ref_pfx = [r["reference"]["phase_Fx_deg"] for r in band_case]
    ax.plot(ka_case, ref_pfx, "k-", linewidth=2, label="Hulme (1982) ref", marker="o", markersize=6)
    ax.plot(ka_case, [r["phase_Fx_deg"] for r in band_case], "b--", linewidth=1.5,
            label="BEM 1024 panels", marker="s", markersize=5)
    ax.plot(ka_coarse, [r["phase_Fx_deg"] for r in band_coarse], "r:", linewidth=1.5,
            label="BEM 256 panels", marker="^", markersize=5)
    _style_ax(ax, "ka", "Phase Fx (deg)", "Surge excitation force phase")

    # Phase Fz
    ax = axes[1, 1]
    ref_pfz = [r["reference"]["phase_Fz_deg"] for r in band_case]
    ax.plot(ka_case, ref_pfz, "k-", linewidth=2, label="Hulme (1982) ref", marker="o", markersize=6)
    ax.plot(ka_case, [r["phase_Fz_deg"] for r in band_case], "b--", linewidth=1.5,
            label="BEM 1024 panels", marker="s", markersize=5)
    ax.plot(ka_coarse, [r["phase_Fz_deg"] for r in band_coarse], "r:", linewidth=1.5,
            label="BEM 256 panels", marker="^", markersize=5)
    _style_ax(ax, "ka", "Phase Fz (deg)", "Heave excitation force phase")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def make_error_figure(band_case: list[dict], out_path: Path) -> str:
    """Figure 3: Relative error for all quantities vs ka."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ka = [r["ka"] for r in band_case]
    quantities = [
        ("rel_error_A11_dimless", "A₁₁"),
        ("rel_error_B11_dimless", "B₁₁"),
        ("rel_error_A33_dimless", "A₃₃"),
        ("rel_error_B33_dimless", "B₃₃"),
        ("rel_error_Fx_dimless", "|Fx|"),
        ("rel_error_Fz_dimless", "|Fz|"),
    ]

    markers = ["o", "s", "^", "v", "D", "x"]
    for idx, (key, label) in enumerate(quantities):
        vals = [r[key] * 100 for r in band_case]  # convert to %
        ax.plot(ka, vals, linewidth=1.5, label=label, marker=markers[idx], markersize=6)

    ax.axhline(y=5, color="r", linestyle="--", linewidth=1, alpha=0.7, label="5% gate")
    ax.axhline(y=-5, color="r", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    _style_ax(ax, "ka", "Relative error (%)", "BEM vs Hulme (1982) — all quantities")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/verified/casepool_bem_hulme_hemisphere")
    args = parser.parse_args(argv)

    if not HAS_MPL:
        print("matplotlib not available; cannot generate figures", flush=True)
        return 1

    out_dir = Path(args.out)
    results = _load_results(out_dir)
    rows = results["rows"]

    band_case = _extract_band(rows, "band", case.CASE_SUBDIVISIONS)
    band_coarse = _extract_band(rows, "band", case.COARSE_SUBDIVISIONS)

    if not band_case:
        print("No case-mesh band rows found in results.json", flush=True)
        return 1

    fig_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    paths.append(make_radiation_figure(band_case, band_coarse, fig_dir / "hulme_radiation_coefficients.png"))
    paths.append(make_excitation_figure(band_case, band_coarse, fig_dir / "hulme_excitation_forces.png"))
    paths.append(make_error_figure(band_case, fig_dir / "hulme_error_bars.png"))

    for p in paths:
        print(f"  wrote {p}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
