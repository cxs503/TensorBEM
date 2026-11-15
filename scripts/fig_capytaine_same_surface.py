#!/usr/bin/env python3
"""Figure for the same-surface capytaine cross-check.

Two panels: worst diagonal deviation and worst coupling deviation against the
number of panels, for TensorBEM on capytaine's surface.  The own-mesh curve of
the round-2 study (``capytaine_crosscheck_r2/results.json``) is drawn for the
same cases so the two comparisons are visibly different quantities.  Both axes
are logarithmic; a straight line indicates a constant observed order.

Usage::

    PYTHONPATH=src python scripts/fig_capytaine_same_surface.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAME = ROOT / "benchmarks/verified/capytaine_same_surface_crossval/results.json"
OWN = ROOT / "benchmarks/verified/capytaine_crosscheck_r2/results.json"
OUT = ROOT / "docs/fig_capytaine_same_surface.png"


def main() -> int:
    same = json.loads(SAME.read_text())
    own = json.loads(OWN.read_text()) if OWN.exists() else None
    cases = list(same["summary"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colours = plt.cm.tab10.colors
    for idx, case in enumerate(cases):
        rows = same["summary"][case]["rows"]
        faces = [r["triangles"] for r in rows]
        axes[0].plot(
            faces, [r["worst_diagonal"] * 100 for r in rows],
            "o-", color=colours[idx], label=case,
        )
        axes[1].plot(
            faces, [max(r["worst_coupling"], 1e-12) * 100 for r in rows],
            "o-", color=colours[idx], label=case,
        )
        if own is not None and case in own.get("convergence", {}):
            own_rows = own["convergence"][case]
            axes[0].plot(
                [r["n_faces"] for r in own_rows],
                [r["worst_diagonal"] * 100 for r in own_rows],
                "s--", color=colours[idx], alpha=0.55,
            )
            axes[1].plot(
                [r["n_faces"] for r in own_rows],
                [max(r["worst_coupling"], 1e-12) * 100 for r in own_rows],
                "s--", color=colours[idx], alpha=0.55,
            )
    axes[0].set_title("worst diagonal deviation (added mass, damping)")
    axes[1].set_title("worst active coupling deviation")
    for ax in axes:
        ax.set_xlabel("panels / triangles on the wetted surface")
        ax.set_ylabel("relative deviation vs capytaine [%]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(5.0, color="k", ls=":", lw=1.0)
        ax.axhline(15.0, color="k", ls="-", lw=1.0)
    axes[0].legend(fontsize=7, loc="best")
    axes[1].annotate("5 % diagonal band", (0.02, 0.06), xycoords="axes fraction",
                     fontsize=7)
    axes[1].annotate("15 % ledger band", (0.02, 0.02), xycoords="axes fraction",
                     fontsize=7)
    fig.suptitle(
        "TensorBEM on capytaine's own panel surface (solid) vs own-mesh round 2 (dashed)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())