#!/usr/bin/env python3
"""Figures + numbers for the m20091 CASE-BEM-CRACK-SIF cross-audit report.

All numbers are computed here (nothing hand-copied into the report), and the
figure PNGs are written next to this script.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, "/root/tp-agent/work/m20091/src")
from tensorbem.crack_sif import (  # noqa: E402
    analytical_cod,
    analytical_sif,
    solve_crack_sif,
)

SIGMA, A, E, NU = 100.0e6, 0.01, 2.0e11, 0.3
EP = E / (1.0 - NU ** 2)
K_REF = analytical_sif(SIGMA, A)
NS = (10, 20, 40, 80, 160, 320, 640)


def collect() -> list[dict]:
    rows = []
    for n in NS:
        r = solve_crack_sif(sigma=SIGMA, a=A, young=E, poisson=NU, n_elements=n, method="constant")
        x, dd = r.cod_profile_x, r.element_dd
        basis = np.sqrt(np.clip(A ** 2 - x ** 2, 0.0, None))
        c = float(np.dot(basis, dd) / np.dot(basis, basis))       # Griffith-shape LS fit
        k_fit = EP * c * math.sqrt(math.pi * A) / 4.0
        ref_cod = analytical_cod(x, SIGMA, A, EP)
        rows.append({
            "n_elements": n,
            "K_delivered_MPa": r.sif_got / 1e6,
            "err_delivered_pct": r.error_pct,
            "K_shapefit_MPa": k_fit / 1e6,
            "err_shapefit_pct": abs(k_fit - K_REF) / K_REF * 100.0,
            "cod_rel_err_pct": 100.0 * float(np.linalg.norm(dd - ref_cod) / np.linalg.norm(ref_cod)),
            "cond": r.condition_number,
        })
    return rows


def figs(rows: list[dict]) -> dict:
    n = np.array([r["n_elements"] for r in rows], float)
    e_del = np.array([r["err_delivered_pct"] for r in rows])
    e_fit = np.array([r["err_shapefit_pct"] for r in rows])
    cod = np.array([r["cod_rel_err_pct"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax[0].loglog(n, e_del, "o-", color="#c0392b", label="delivered extract_sif (5-tip mean)")
    ax[0].loglog(n, e_fit, "s-", color="#1f77b4", label="Griffith-shape LS on same DD field")
    ax[0].axhline(3.0, ls="--", c="k", lw=1, label="platform 3%")
    ax[0].axhline(5.0, ls=":", c="grey", lw=1, label="ledger 5%")
    ax[0].set_xlabel("constant-DD elements (n)")
    ax[0].set_ylabel("|K_I - K_ref| / K_ref  [%]")
    ax[0].set_title("SIF error vs mesh: solve converges, extraction diverges")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].loglog(n, cod, "d-", color="#2e7d32", label="COD field vs analytic [%]")
    ax[1].loglog(n, e_del, "o--", color="#c0392b", alpha=0.6, label="K error, delivered [%]")
    ax[1].set_xlabel("constant-DD elements (n)")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("DD field is first-order accurate; K is not")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p1 = HERE / "fig1_error_vs_mesh.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["K_ref\n(closed form)", "K delivered\nn=640", "K shape-fit\nn=640", "K delivered\nn=20"]
    vals = [K_REF / 1e6, rows[-1]["K_delivered_MPa"], rows[-1]["K_shapefit_MPa"], rows[1]["K_delivered_MPa"]]
    cols = ["#34495e", "#c0392b", "#1f77b4", "#e67e22"]
    bars = ax.bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.4f}", ha="center", fontsize=9)
    ax.axhline(K_REF / 1e6, ls="--", c="k", lw=1)
    ax.set_ylim(16.0, 20.0)
    ax.set_ylabel("$K_I$ [MPa$\\sqrt{m}$]")
    ax.set_title("CASE-BEM-CRACK-SIF: delivered vs re-extracted SIF")
    fig.tight_layout()
    p2 = HERE / "fig2_sif_bars.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return {"p1": p1, "p2": p2}


if __name__ == "__main__":
    rows = collect()
    paths = figs(rows)
    payload = {
        "params": {"sigma_MPa": SIGMA / 1e6, "a_mm": A * 1e3, "E_GPa": E / 1e9, "nu": NU},
        "K_ref_MPa_sqrt_m": K_REF / 1e6,
        "rows": rows,
        "figures": {k: str(v) for k, v in paths.items()},
    }
    (HERE / "audit_numbers.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=1))