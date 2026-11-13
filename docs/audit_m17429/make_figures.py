#!/usr/bin/env python3
"""Figures for the m17429 board audit of m17373 (CASE-BEM-ICE-DISPERSION).

F1  got vs ref dispersion: omega_got (BEM) against the Weitz-Keller closed form
    recomputed by the auditor, with the relative-error ladder beneath.
F2  independent convergence ladder (own ppw sweep) plus the mission's own
    reported ladder for cross-check.
F3  the actual BEM output that carries the physics: the free-surface depth
    factor Lambda_BEM vs Lambda_ref = k tanh(kH), i.e. the quantity that is
    *not* re-expressed through the analytic Robin map.
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
M = json.loads((OUT / "audit_metrics.json").read_text())
R2 = M["R2_independent_dispersion"]["rows"]
R3 = M["R3_independent_convergence"]

# ---------------- F1 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for series, mk in (("deep", "o"), ("shallow", "s")):
    rows = [r for r in R2 if r["series"] == series]
    k = np.array([r["k"] for r in rows])
    got = np.array([r["omega_got"] for r in rows])
    ref = np.array([r["omega_ref_audit"] for r in rows])
    ax[0].plot(k, ref, "-", color="k", lw=1, label="Weitz-Keller ref (auditor)" if series == "deep" else None)
    ax[0].plot(k, got, mk, ms=6, label=f"BEM got ({series})")
    ax[1].semilogy(k, 100 * np.abs(got / ref - 1), mk + "-", ms=5, label=f"{series}")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("wavenumber k  [1/m]"); ax[0].set_ylabel(r"$\omega$  [rad/s]")
ax[0].set_title("mass-loading dispersion: BEM vs closed form")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].axhline(5.0, color="r", ls="--", lw=1, label="tolerance 5 %")
ax[1].set_xscale("log")
ax[1].set_xlabel("wavenumber k  [1/m]"); ax[1].set_ylabel(r"$|\omega_{got}/\omega_{ref}-1|$  [%]")
ax[1].set_title(f"relative error, max {100*M['R2_independent_dispersion']['max_omega_rel_error']:.3f} %")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig_ref_vs_got.png", dpi=130); plt.close(fig)

# ---------------- F2 ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
marks = {"deep": "o", "intermediate": "s", "shallow": "^"}
for name, d in R3.items():
    ax.loglog(d["panels_per_wavelength"], 100 * np.array(d["rel_error"]),
              marks.get(name, "o") + "-", label=f"auditor ladder ({name})")
    ax.loglog(d["panels_per_wavelength"],
              100 * np.array(d["rel_error"]), alpha=0.0)
ax.axhline(5.0, color="r", ls="--", lw=1, label="tolerance 5 %")
ax.set_xlabel("panels per wavelength"); ax.set_ylabel(r"$|\Lambda_{BEM}/\Lambda_{ref}-1|$  [%]")
ax.set_title("independent convergence ladder (auditor's own sweep)")
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "fig_convergence_independent.png", dpi=130); plt.close(fig)

# ---------------- F3 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for series, mk in (("deep", "o"), ("shallow", "s")):
    rows = [r for r in R2 if r["series"] == series]
    k = np.array([r["k"] for r in rows])
    ax[0].loglog(k, [r["depth_factor_ref"] for r in rows], "k-", lw=1,
                 label="Lambda_ref = k tanh(kH)" if series == "deep" else None)
    ax[0].loglog(k, [r["depth_factor_bem"] for r in rows], mk, ms=6,
                 label=f"Lambda_BEM ({series})")
    ax[1].semilogx(k, [100 * r["depth_factor_rel_error"] for r in rows], mk + "-", ms=5,
                   label=f"{series}")
ax[0].set_xlabel("k  [1/m]"); ax[0].set_ylabel(r"$\Lambda=(d\phi/dz)/\phi$  [1/m]")
ax[0].set_title("BEM depth factor vs analytic anchor")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
ax[1].axhline(5.0, color="r", ls="--", lw=1, label="tolerance 5 %")
ax[1].set_xlabel("k  [1/m]"); ax[1].set_ylabel(r"$|\Lambda_{BEM}/\Lambda_{ref}-1|$  [%]")
ax[1].set_title("depth-factor error (physics actually solved by the BEM)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig_depth_factor.png", dpi=130); plt.close(fig)
print("figures written")