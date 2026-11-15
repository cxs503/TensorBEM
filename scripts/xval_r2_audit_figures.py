#!/usr/bin/env python3
"""Figures for the round-2 cross-validation re-adjudication.

(a) 6x6 deviation heatmaps: TensorBEM vs the re-solved capytaine reference, and
    the reference's own reproduction residual (stored vs re-solved), on the same
    colour scale family, so "the referee is exact" and "the codes differ by a
    few percent" can be compared directly.
(b) convergence of the worst physically-scored entry with mesh refinement, with
    the pre-registered 5 % band and the owner 15 % band drawn in, and the decay
    of the exact-zero (invariant) residual against an h^2 guide line.

Output: ``docs/fig_xval_r2_audit_heatmaps.png``, ``docs/fig_xval_r2_audit_convergence.png``.
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
sys.path.insert(0, str(ROOT / "scripts"))

from tensorbem.benchmarks.capytaine_crosscheck_r2 import (  # noqa: E402
    ARTIFACT_ROOT,
    DOF_ORDER,
    invariant_null_dofs,
)

BASE = ROOT / ARTIFACT_ROOT
CASE = "sphere_r1_z-0.5"
NU = 2.0
QUANTITY = "damping"


def _solve_deviation() -> tuple[np.ndarray, str]:
    """Re-solve one case/nu with capytaine and return the per-entry reproduction
    deviation of the stored reference (relative to the matrix max)."""
    import capytaine as cpt

    import verify_capytaine_reference_r2 as ver

    reference = json.loads((BASE / "capytaine_ref_r2.json").read_text())
    block = reference["cases"][CASE]
    meta = reference["metadata"]
    mesh = ver._mesh(cpt, CASE, block["reference_level"])
    body = ver._body(cpt, mesh, block["rotation_center"])
    solver = cpt.BEMSolver()
    stored = block["reference"]["{:.3f}".format(NU)][QUANTITY]
    live = np.zeros((6, 6))
    for dof in DOF_ORDER:
        j = DOF_ORDER.index(dof)
        res = solver.solve(
            cpt.RadiationProblem(
                body=body,
                omega=math.sqrt(NU * float(meta["gravity"])),
                radiating_dof=ver._CAPYTAINE_NAMES[dof],
                water_depth=np.inf,
                rho=float(meta["density"]),
                g=float(meta["gravity"]),
            )
        )
        key = "radiation_damping"
        vals = {k.lower(): float(v) for k, v in getattr(res, key).items()}
        for i, row in enumerate(DOF_ORDER):
            if row in vals:
                live[i, j] = vals[row]
    ref = np.asarray(stored, dtype=float)
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    return np.abs(live - ref) / scale, block["case"] if "case" in block else CASE


def _codes_deviation() -> np.ndarray:
    tb = json.loads((BASE / "solves" / (CASE + ".json")).read_text())
    cap = json.loads((BASE / "capytaine_ref_r2.json").read_text())["cases"][CASE]
    key = "{:.3f}".format(NU)
    a_tb = np.asarray(tb["reference"][key][QUANTITY], dtype=float)
    a_cap = np.asarray(cap["reference"][key][QUANTITY], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(a_tb - a_cap) / np.maximum(np.abs(a_cap), 1e-300)
    return dev


def figure_heatmaps() -> Path:
    repro, _ = _solve_deviation()
    codes = _codes_deviation()
    nulls = {DOF_ORDER.index(d) for d in invariant_null_dofs(CASE)}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    panels = (
        (axes[0], repro, "reference reproduction |stored - re-solved| / max|B|",
         "%.6e"),
        (axes[1], codes, "TensorBEM vs capytaine  |B_tb - B_cap| / |B_cap|", "%.6e"),
    )
    for ax, mat, title, fmt in panels:
        floor = 1e-19
        shown = np.where(mat > 0.0, mat, floor)
        im = ax.imshow(np.log10(shown), cmap="viridis", vmin=-18, vmax=0)
        ax.set_xticks(range(6), DOF_ORDER, rotation=45)
        ax.set_yticks(range(6), DOF_ORDER)
        ax.set_xlabel("radiating dof (column)")
        ax.set_ylabel("row dof")
        ax.set_title(title, fontsize=10)
        for i in range(6):
            for j in range(6):
                txt = ("%.1e" % mat[i, j]) if mat[i, j] > 0 else "0"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.2,
                        color="w" if np.log10(shown[i, j]) < -9 else "k")
                if i in nulls or j in nulls:
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                      edgecolor="red", lw=1.4)
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="log10(relative deviation)")
    fig.suptitle(
        "Round-2 radiation damping, {} @ nu={:g}: referee is exact, codes differ "
        "by percent (red = exact-zero/invariant entry)".format(CASE, NU),
        fontsize=10,
    )
    out = ROOT / "docs" / "fig_xval_r2_audit_heatmaps.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_convergence() -> Path:
    study = json.loads((BASE / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    for case, rows in study["convergence"].items():
        xs = [r["n_faces"] for r in rows]
        ys = [100.0 * max(r["worst_diagonal"], r["worst_coupling"]) for r in rows]
        ax.semilogx(xs, ys, "o-", label=case)
    ax.axhline(5.0, color="tab:orange", ls="--", lw=1.2, label="module self band 5 %")
    ax.axhline(15.0, color="tab:red", ls=":", lw=1.4, label="owner/ledger bound 15 %")
    ax.set_xlabel("panels (TensorBEM)")
    ax.set_ylabel("worst physically-scored deviation vs capytaine [%]")
    ax.set_title("(a) discretisation error decays, everything under 15 %", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax = axes[1]
    for entry in study["degeneracy"]["cases"]:
        xs = [lvl["n_faces"] for lvl in entry["levels"]]
        ys = [max(lvl["worst_ratio"], 1e-12) for lvl in entry["levels"]]
        ax.loglog(xs, ys, "o-", label=entry["case"])
        if len(xs) >= 2:
            h2 = ys[0] * (np.asarray(xs[0]) / np.asarray(xs, dtype=float)) ** 2 * 1.0
            ax.loglog(xs, h2, "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("panels")
    ax.set_ylabel("exact-zero residual (fraction of rho L**k)")
    ax.set_title("(b) invariant residual vs refinement (dashed: h^2 guide)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    out = ROOT / "docs" / "fig_xval_r2_audit_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    a = figure_heatmaps()
    b = figure_convergence()
    print(a)
    print(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())