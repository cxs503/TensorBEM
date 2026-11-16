"""Figure for m23272: the R2 refinement ladder that the audit document argues from.

Reads ``benchmarks/verified/capytaine_crosscheck_r2/results.json`` only -- no
solve is re-run -- and plots, per body, the worst *scored* diagonal deviation
against the TensorBEM panel count, together with the fixed capytaine reference
panel count and the pre-registered bands (internal 5% / coupling 10%, owner 15%).

    python benchmarks/verified/make_m23272_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/verified/capytaine_crosscheck_r2/results.json"
OUT = ROOT / "docs/fig_xval_r2_degeneracy_ladder_m23272.png"


def main() -> int:
    study = json.loads(ARTIFACT.read_text())
    comparison = study["comparison"]
    convergence = study["convergence"]
    scoring = study["invariant_scoring"]["cases"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    colors = {
        "hemisphere_r1": "C0",
        "hemisphere_r1_cz-0.5": "C1",
        "sphere_r1_z-0.5": "C2",
        "sphere_r1_z-0.5_c0": "C3",
    }
    for case, rows in convergence.items():
        faces = [int(row["n_faces"]) for row in rows]
        diag = [abs(100.0 * float(row["worst_diagonal"])) for row in rows]
        coupl = [abs(100.0 * float(row["worst_coupling"])) for row in rows]
        colour = colors.get(case, "k")
        ax.plot(faces, diag, "o-", color=colour, label=f"{case} (diag)")
        ax.plot(faces, coupl, "s--", color=colour, alpha=0.55, lw=1.0)
        ref = int(comparison["cases"][case]["capytaine_faces"])
        ax.axvline(ref, color=colour, lw=0.8, ls=":", alpha=0.6)
        ax.annotate(
            f"capytaine {ref}",
            xy=(ref, ax.get_ylim()[1]),
            xytext=(2, -10),
            textcoords="offset points",
            fontsize=7,
            color=colour,
        )
    ax.axhline(5.0, color="k", lw=1.2, ls="-")
    ax.text(70, 5.4, "internal diagonal band 5%", fontsize=8)
    ax.axhline(10.0, color="k", lw=1.0, ls="--")
    ax.text(70, 10.4, "internal coupling band 10%", fontsize=8)
    ax.axhline(15.0, color="r", lw=1.0, ls="-.")
    ax.text(70, 15.4, "owner acceptance 15%", fontsize=8, color="r")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TensorBEM panels (log)")
    ax.set_ylabel("worst scored relative deviation [%] (log)")
    ax.set_title("R2 refinement ladder: solid = diagonal, dashed = coupling")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    names = list(scoring)
    values = [100.0 * float(scoring[name]["worst_physics_relative"]) for name in names]
    colours = [colors.get(name, "k") for name in names]
    bars = ax2.bar(range(len(names)), values, color=colours, alpha=0.85)
    for bar, value in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f}%",
            ha="center",
            fontsize=8,
        )
    ax2.axhline(3.0, color="k", lw=1.0, ls=":")
    ax2.text(-0.45, 3.15, "3% (precision goal)", fontsize=8)
    ax2.axhline(5.0, color="k", lw=1.0, ls="-")
    ax2.text(-0.45, 5.15, "5% internal band", fontsize=8)
    ax2.axhline(15.0, color="r", lw=1.0, ls="-.")
    ax2.text(-0.45, 15.3, "15% owner acceptance", fontsize=8, color="r")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("worst scored deviation [%]")
    ax2.set_ylim(0.0, 18.0)
    ax2.set_title("final mesh: scored physics deviation vs bands")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "XVAL-BEM-R2: exact-zero rotational block made decidable (audit m23272)", fontsize=11
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())