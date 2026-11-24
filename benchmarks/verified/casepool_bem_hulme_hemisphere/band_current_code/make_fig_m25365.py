#!/usr/bin/env python
"""Render fig_m25365_band_cells.png for the PR#1195 reopen report (m25365).

Pure reader: every number comes from the committed artifacts
(``field_data.json`` for the surface field, ``cells/*.json`` for the band
errors and the CHIEF control pair).  No solve is run.

Panels
------
(a) |surge velocity potential| on the hemisphere surface (ka = 1.0, 256
    panels, committed field data) -- the field-quantity cloud map;
(b) per-quantity relative error vs ka on the 1024-panel extension band,
    with the 5 % case-pool gate and the 15 % hard limit; the ka = 4 A11
    mesh-limited failure is marked;
(c) CHIEF control at the irregular frequency: band-max error and condition
    number, CHIEF on vs off, ka = 2.58 and 3.0.

Run:  python benchmarks/verified/casepool_bem_hulme_hemisphere/band_current_code/make_fig_m25365.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless",
              "Fx_dimless", "Fz_dimless")
GATE, HARD = 0.05, 0.15


def band_max(cell: dict) -> float:
    return max(abs(cell[f"rel_error_{q}"]) for q in QUANTITIES)


def main() -> None:
    cells = {
        f.stem: json.loads(f.read_text())
        for f in sorted((HERE / "cells").glob("*.json"))
    }
    field = json.loads((HERE.parent / "field_data.json").read_text())
    f1 = next(f for f in field["fields"] if f["ka"] == 1.0)

    fig = plt.figure(figsize=(13.5, 4.6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.0], wspace=0.30)

    # --- (a) field cloud map: |surge potential| on the hemisphere ---------
    ax = fig.add_subplot(gs[0], projection="3d")
    ctr = np.asarray(f1["centres"])
    val = np.asarray(f1["surge_abs"])
    sc = ax.scatter(ctr[:, 0], ctr[:, 1], ctr[:, 2], c=val, cmap="viridis",
                    s=26, vmin=0.0, vmax=float(val.max()), edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
    cb.set_label(r"$|\phi_{surge}|$ (panel centres)")
    ax.set_title(f"(a) surge potential field, ka = {f1['ka']}, "
                 f"{f1['panels']} panels (committed field data)", fontsize=9)
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-60)

    # --- (b) per-quantity error vs ka, 1024-panel extension band -----------
    ax = fig.add_subplot(gs[1])
    ext = sorted((c for c in cells.values() if c["kind"] == "extension"),
                  key=lambda c: c["ka"])
    kas = [c["ka"] for c in ext]
    styles = {"A11_dimless": ("tab:blue", "o"), "B11_dimless": ("tab:cyan", "s"),
              "A33_dimless": ("tab:green", "^"), "B33_dimless": ("tab:orange", "D"),
              "Fx_dimless": ("tab:red", "v"), "Fz_dimless": ("tab:purple", "P")}
    for q in QUANTITIES:
        col, mk = styles[q]
        errs = [abs(c[f"rel_error_{q}"]) * 100 for c in ext]
        ax.plot(kas, errs, color=col, marker=mk, ms=5, lw=1.2, label=q.replace("_dimless", ""))
    ax.axhline(GATE * 100, color="k", ls="--", lw=1.0)
    ax.text(2.22, GATE * 100 + 0.35, "5 % case-pool gate", fontsize=8)
    ax.axhline(HARD * 100, color="r", ls=":", lw=1.0)
    ax.text(2.22, HARD * 100 + 0.35, "15 % hard limit", fontsize=8, color="r")
    c4 = next(c for c in ext if c["ka"] == 4.0)
    ax.annotate("ka = 4, A11\n(mesh-limited)",
                xy=(4.0, abs(c4["rel_error_A11_dimless"]) * 100),
                xytext=(3.05, 12.5), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("ka")
    ax.set_ylabel("|rel. error| vs re-derived reference (%)")
    ax.set_title("(b) extension band, 1024 panels, six scored quantities", fontsize=9)
    ax.set_xticks([2.2, 2.58, 3.0, 3.5, 4.0])
    ax.set_ylim(0, 16)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # --- (c) CHIEF control at / near the irregular frequency ---------------
    ax = fig.add_subplot(gs[2])
    pairs = []
    for ka in (2.58, 3.0):
        on = next(c for c in cells.values()
                  if c["kind"] == "extension" and c["ka"] == ka)
        off = next(c for c in cells.values()
                  if c["kind"] == "chief_off" and c["ka"] == ka)
        pairs.append((ka, on, off))
    x = np.arange(len(pairs))
    w = 0.36
    err_on = [band_max(on) * 100 for _, on, _ in pairs]
    err_off = [band_max(off) * 100 for _, _, off in pairs]
    ax.bar(x - w / 2, err_on, w, label="CHIEF on", color="tab:green")
    ax.bar(x + w / 2, err_off, w, label="CHIEF off (control)", color="tab:gray")
    ax.axhline(GATE * 100, color="k", ls="--", lw=1.0)
    for i, (ka, on, off) in enumerate(pairs):
        ax.text(i - w / 2, err_on[i] + 0.15, f"{err_on[i]:.2f}%\ncond {on['condition_number']:.1f}",
                ha="center", fontsize=7)
        ax.text(i + w / 2, err_off[i] + 0.15, f"{err_off[i]:.2f}%\ncond {off['condition_number']:.1f}",
                ha="center", fontsize=7)
    ax.axhline(GATE * 100, color="k", ls="--", lw=1.0)
    ax.set_xticks(x, [f"ka = {ka}" for ka, _, _ in pairs])
    ax.set_ylabel("band-max |rel. error| (%)")
    ax.set_title("(c) CHIEF control, band-max error + condition number", fontsize=9)
    ax.set_ylim(0, 7.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("CASE-BEM-HULME-HEMISPHERE -- band_current_code cells (m21845 deliverables, pinned by m25365)",
                 fontsize=10)
    out = HERE / "fig_m25365_band_cells.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
