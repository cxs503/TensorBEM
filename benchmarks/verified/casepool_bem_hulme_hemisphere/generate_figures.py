#!/usr/bin/env python3
"""Generate the two figure artifacts for CASE-BEM-HULME-HEMISPHERE.

The case-pool contract tests (``test_casepool_hulme_hemisphere_results.py``
and ``test_casepool_hulme_independent_axis.py``) require two PNG figures to
exist next to the committed ``results.json``:

1. ``figure.png`` -- comparison curves (TensorBEM BEM vs Hulme (1982)
   reference) for all six quantities across the ka band, from the committed
   ``results.json``.
2. ``independent_axis_figure.png`` -- three-way comparison (TensorBEM solver
   vs Hulme reference vs Capytaine) from ``results.json`` +
   ``capytaine_excitation.json``.

Both are *derived* artefacts: every number is injected from the committed
run results, never typed by hand, so regenerating them is byte-reproducible
from the same inputs.  This script is the single entry point for that
regeneration.

Usage:
    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/generate_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

CASEPOOL_DIR = Path(__file__).resolve().parent
RESULTS = CASEPOOL_DIR / "results.json"
CAPYTAINE = CASEPOOL_DIR / "capytaine_excitation.json"
FIGURE = CASEPOOL_DIR / "figure.png"
IND_FIGURE = CASEPOOL_DIR / "independent_axis_figure.png"

QUANTITIES = [
    ("A11_dimless", r"$A_{11}/(\rho\nabla)$", "surge added mass"),
    ("B11_dimless", r"$B_{11}/(\rho\omega\nabla)$", "surge damping"),
    ("A33_dimless", r"$A_{33}/(\rho\nabla)$", "heave added mass"),
    ("B33_dimless", r"$B_{33}/(\rho\omega\nabla)$", "heave damping"),
    ("Fx_dimless", r"$|F_x|/(\rho g A a^2)$", "surge excitation"),
    ("Fz_dimless", r"$|F_z|/(\rho g A a^2)$", "heave excitation"),
]


def _band_rows(results: dict, subdivisions: int) -> list[dict]:
    """Band rows of a given subdivision level, sorted by ka."""
    return sorted(
        [r for r in results["rows"] if r["kind"] == "band" and r["subdivisions"] == subdivisions],
        key=lambda r: r["ka"],
    )


def generate_comparison_figure(out: Path | None = None) -> Path:
    """figure.png: BEM vs Hulme reference comparison curves.

    ``out`` lets a caller rebuild the figure somewhere else (the test suite
    rebuilds it under ``tmp_path``, because the delivery gate strips binary
    artefacts from commits); it is byte-for-byte the same plot either way.
    """
    destination = Path(out) if out is not None else FIGURE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(RESULTS.read_text())
    band_rows = _band_rows(results, 4)
    coarse_rows = _band_rows(results, 3)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (key, label, desc) in enumerate(QUANTITIES):
        ax = axes[idx]
        ka_vals = [r["ka"] for r in band_rows]
        bem_vals = [r[key] for r in band_rows]
        ref_vals = [r["reference"][key] for r in band_rows]
        coarse_vals = [r[key] for r in coarse_rows]

        ax.plot(ka_vals, ref_vals, "k-", linewidth=2, label="Hulme (1982) ref",
                marker="o", markersize=4)
        ax.plot(ka_vals, bem_vals, "b--", linewidth=1.5, label="BEM (1024 panels)",
                marker="s", markersize=4)
        ax.plot(ka_vals, coarse_vals, "r:", linewidth=1.2, label="BEM (256 panels)",
                marker="^", markersize=4)

        ax.set_xlabel(r"$ka$", fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(desc, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CASE-BEM-HULME-HEMISPHERE: Radiation & Diffraction Coefficients\n"
        "BEM vs Hulme (1982) Analytical Reference",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {destination} ({destination.stat().st_size} bytes)")
    return destination


def generate_independent_axis_figure() -> Path:
    """independent_axis_figure.png: three-way comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(RESULTS.read_text())
    capytaine = json.loads(CAPYTAINE.read_text())

    band_rows = _band_rows(results, 4)

    capy_key_map = {
        "A11_dimless": "A11", "B11_dimless": "B11",
        "A33_dimless": "A33", "B33_dimless": "B33",
        "Fx_dimless": "Fx_dimless", "Fz_dimless": "Fz_dimless",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (key, label, desc) in enumerate(QUANTITIES):
        ax = axes[idx]
        ka_vals = [r["ka"] for r in band_rows]
        bem_vals = [r[key] for r in band_rows]
        ref_vals = [r["reference"][key] for r in band_rows]

        capy_vals = []
        for r in band_rows:
            ka_str = f"{r['ka']:g}"
            capy_row = capytaine["reference"].get(ka_str, {})
            capy_vals.append(capy_row.get(capy_key_map.get(key, key), float("nan")))

        ax.plot(ka_vals, ref_vals, "k-", linewidth=2, label="Hulme (1982) ref",
                marker="o", markersize=4)
        ax.plot(ka_vals, bem_vals, "b--", linewidth=1.5, label="TensorBEM (1024 panels)",
                marker="s", markersize=4)
        ax.plot(ka_vals, capy_vals, "g-.", linewidth=1.2, label="Capytaine",
                marker="^", markersize=4)

        ax.set_xlabel(r"$ka$", fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(desc, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CASE-BEM-HULME-HEMISPHERE: Independent-Code Verification\n"
        "TensorBEM vs Hulme (1982) vs Capytaine",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IND_FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {IND_FIGURE} ({IND_FIGURE.stat().st_size} bytes)")
    return IND_FIGURE


if __name__ == "__main__":
    generate_comparison_figure()
    generate_independent_axis_figure()
    print("Done.")
