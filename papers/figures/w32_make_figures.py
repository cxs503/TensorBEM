#!/usr/bin/env python3
"""W32 (mission 28172) -- figures for the RAO peak & m0 truncation convergence.

Reads the W32 artifact (``papers/data/w32-2026-11-16/seakeeping/
truncation_convergence.json``) and the W25 RAO table, and writes three PNGs:

    fig1_rao_curves.png      -- heave / pitch RAO with truncation levels
    fig2_m0_convergence.png  -- m0 truncation error vs upper band edge
    fig3_peak_convergence.png-- RAO peak truncation error vs upper band edge
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
W32 = ROOT / "papers/data/w32-2026-11-16/seakeeping/truncation_convergence.json"
W25 = ROOT / "papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json"
OUT = ROOT / "papers/figures/w32"

SEA_LABEL = {
    "ss_T1_1p00": r"$T_1=1.0$ s ($\omega_p=4.85$)",
    "ss_T1_1p40": r"$T_1=1.4$ s ($\omega_p=3.46$)",
    "ss_T1_1p80": r"$T_1=1.8$ s ($\omega_p=2.69$)",
}


def main() -> int:
    w32 = json.loads(W32.read_text())
    w25 = json.loads(W25.read_text())
    rt = w25["rao_table"]
    om = np.array([r[0] for r in rt["rows"]])
    heave = np.array([r[2] for r in rt["rows"]])
    pitch = np.array([r[3] for r in rt["rows"]])
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- fig 1: RAO curves with truncation levels ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(om, heave, "-", lw=1.6, label="heave RAO [m/m]")
    ax.plot(om, pitch, "--", lw=1.6, label="pitch RAO [rad/m]")
    for w in (3.0, 6.0, 8.0, 12.0):
        ax.axvline(w, color="0.6", lw=0.8, ls=":")
    ax.axvline(4.2369, color="crimson", lw=1.0, ls="--",
               label=r"pitch peak $\omega=4.24$")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("RAO amplitude")
    ax.set_title("Wigley III head-seas RAO (U=0, production mesh)\n"
                 "with truncation-frequency levels (dotted)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_rao_curves.png", dpi=150)
    plt.close(fig)

    # ---- fig 2: m0 truncation error vs upper band edge ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ss in ("ss_T1_1p00", "ss_T1_1p40", "ss_T1_1p80"):
        rows = [
            r for r in w32["axis_A_truncation_frequency"]["rows"]
            if r["sea_state"] == ss
        ]
        rows.sort(key=lambda r: r["omega_max"])
        w = [r["omega_max"] for r in rows]
        eh = [r["m0_heave_trunc_error_pct"] for r in rows]
        ep = [r["m0_pitch_trunc_error_pct"] for r in rows]
        ax.plot(w, eh, "-o", ms=4, label=f"{SEA_LABEL[ss]} heave m0")
        ax.plot(w, ep, "--s", ms=4, label=f"{SEA_LABEL[ss]} pitch m0")
    ax.axhline(0.5, color="crimson", lw=1.2, ls="--", label="gate 0.5%")
    ax.set_xlabel(r"$\omega_{\max}$ [rad/s] (upper band edge)")
    ax.set_ylabel("m0 truncation error vs full band [%]")
    ax.set_yscale("log")
    ax.set_title("Response m0 truncation error (axis A)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_m0_convergence.png", dpi=150)
    plt.close(fig)

    # ---- fig 3: RAO peak truncation error vs upper band edge ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ss in ("ss_T1_1p00", "ss_T1_1p40", "ss_T1_1p80"):
        rows = [
            r for r in w32["axis_A_truncation_frequency"]["rows"]
            if r["sea_state"] == ss
        ]
        rows.sort(key=lambda r: r["omega_max"])
        w = [r["omega_max"] for r in rows]
        eh = [r["peak_heave_trunc_error_pct"] for r in rows]
        ep = [r["peak_pitch_trunc_error_pct"] for r in rows]
        ax.plot(w, eh, "-o", ms=4, label=f"{SEA_LABEL[ss]} heave peak")
        ax.plot(w, ep, "--s", ms=4, label=f"{SEA_LABEL[ss]} pitch peak")
    ax.axhline(2.0, color="crimson", lw=1.2, ls="--", label="gate 2%")
    ax.set_xlabel(r"$\omega_{\max}$ [rad/s] (upper band edge)")
    ax.set_ylabel("RAO peak truncation error vs full band [%]")
    ax.set_yscale("log")
    ax.set_title("RAO peak truncation error (axis A)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_peak_convergence.png", dpi=150)
    plt.close(fig)

    print("figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name, p.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
