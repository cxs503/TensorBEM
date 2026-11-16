"""Figure for m22876: what the C0 precondition separates, on the real artifacts.

Two panels, one line per committed sweep:

* left  -- the per-frequency relative residual of the linear solve (the number
           C0 compares against 1e-2);
* right -- the smallest diagonal radiation damping at that frequency (C0
           requires it to stay >= 0, because a negative radiation damping would
           be a solver artifact that *creates* energy).

The committed production sweep passes both; the two retained negative controls
(the n315 evidence sweep and the m20021 medium sweep) fail both, at the same
frequencies where their residuals blow up.

Usage::

    PYTHONPATH=src python tools/make_m22876_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
DOCS_DIR = ROOT / "docs" / "casepool_bem_cummins_wigley"

SOLVER_RESIDUAL_BAR = 0.01


def _load(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    damping = data["damping"]
    diagonal_min = np.array([np.diag(damping[i]).min() for i in range(damping.shape[0])])
    return {
        "omega": data["omega"].astype(float),
        "residual": data["solver_relative_residual"].astype(float),
        "damping_min": diagonal_min,
    }


def main() -> int:
    curves = {
        "committed medium (production)": (ARTIFACT_DIR / "sweep_medium.npz", "C0", "-", "#1f77b4"),
        "committed coarse": (ARTIFACT_DIR / "sweep_coarse.npz", "C0_coarse", "--", "#2ca02c"),
        "n315 evidence (coarse, FAIL)": (DOCS_DIR / "evidence_cluster_n315_sweep_coarse.npz", "negative control", ":", "#d62728"),
        "m20021 medium (FAIL)": (ROOT / "docs" / "m20021" / "sweep_medium.npz", "negative control", "-.", "#9467bd"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for label, (path, _role, style, colour) in curves.items():
        sweep = _load(path)
        axes[0].semilogy(sweep["omega"], sweep["residual"], style, color=colour, label=label, lw=1.6)
        axes[1].plot(sweep["omega"], sweep["damping_min"], style, color=colour, label=label, lw=1.6)

    axes[0].axhline(SOLVER_RESIDUAL_BAR, color="k", lw=1.0, ls=(0, (6, 3)))
    axes[0].annotate(
        "C0 bar = $10^{-2}$",
        xy=(0.02, SOLVER_RESIDUAL_BAR),
        xycoords=("axes fraction", "data"),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8,
    )
    axes[0].set_xlabel(r"$\omega$ [rad/s]")
    axes[0].set_ylabel(r"relative residual of the linear solve")
    axes[0].set_title("(a) C0 precondition: solver residual")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].axhline(0.0, color="k", lw=1.0, ls=(0, (6, 3)))
    axes[1].set_xlabel(r"$\omega$ [rad/s]")
    axes[1].set_ylabel(r"min$_{\mathrm{DOF}}$ diag $B(\omega)$ [non-dim.]")
    axes[1].set_title("(b) C0 precondition: radiation damping floor")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].set_ylim(-4.0, 1.0)

    fig.suptitle("CASE-BEM-CUMMINS-WIGLEY: C0 separates admissible sweeps from the retained negative controls", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = DOCS_DIR / "fig_c0_negative_controls.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())