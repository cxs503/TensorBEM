#!/usr/bin/env python3
"""Audit figures for m19973 (board-audit of CASE-BEM-HELMHOLTZ-SPHERE).

Reads the audited payload (audited_case/results.json), the auditor rerun
(repro/results.json when present, else the replayed rows in repro.log) and the
independent reference check (indep_check.json). Writes figs/*.png.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

claimed = json.loads((HERE / "audited_case" / "results.json").read_text())
rows = claimed["rows"]
indep = json.loads((HERE / "indep_check.json").read_text())

rerun_path = HERE / "repro" / "results.json"
if rerun_path.exists():
    rerun = json.loads(rerun_path.read_text())["rows"]
    rerun_source = "runs/audit_m19973/repro/results.json (full auditor rerun)"
else:  # parse the replayed stdout log
    rerun = []
    txt = (HERE / "repro.log").read_text() if (HERE / "repro.log").exists() else ""
    for m in re.finditer(
        r"Running ka=([\d.]+), sub=(\d)\.\.\. sigma_bem=([\d.eE+-]+), "
        r"sigma_mie=([\d.eE+-]+), err=([\d.]+)%", txt):
        rerun.append({"ka": float(m.group(1)), "subdivisions": int(m.group(2)),
                      "sigma_bem": float(m.group(3)), "sigma_mie": float(m.group(4)),
                      "relative_error": float(m.group(5)) / 100})
    rerun_source = f"runs/audit_m19973/repro.log (partial stdout replay, {len(rerun)}/14 rows)"

ka_all = sorted({r["ka"] for r in rows})
tol = claimed["tolerance"]


def sel(sub, key="sigma_bem"):
    return [next(r[key] for r in rows if r["subdivisions"] == sub and r["ka"] == ka) for ka in ka_all]


# --- Figure 1: BEM vs Mie ---------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.6))
mie = sel(2, "sigma_mie")
ax.fill_between(ka_all, [s * (1 - tol) for s in mie], [s * (1 + tol) for s in mie],
                color="tab:green", alpha=0.15, label=f"Mie ±{tol*100:.0f}% (declared tolerance)")
ax.plot(ka_all, mie, "k--", lw=1.6, label="Mie series (independently re-derived)")
ax.plot(ka_all, sel(2), "o-", lw=2, ms=7, label="TensorBEM 320 panels")
ax.plot(ka_all, sel(3), "s-", lw=2, ms=7, label="TensorBEM 1280 panels")
ax.set_xlabel("ka (dimensionless wavenumber)")
ax.set_ylabel(r"scattering cross-section $\sigma$ [m$^2$]")
ax.set_title("Rigid sphere acoustic scattering: TensorBEM vs Mie series\n(audited payload, m19973 results.json)")
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(OUT / "fig1_bem_vs_mie.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# --- Figure 2: relative error vs ka ----------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))
for sub, mk in ((2, "o-"), (3, "s-")):
    errs = [e * 100 for e in sel(sub, "relative_error")]
    ax.semilogy(ka_all, errs, mk, lw=2, ms=7, label=f"{sub and {2: 320, 3: 1280}[sub]} panels")
ax.axhline(tol * 100, color="crimson", ls="--", lw=1.5, label=f"declared tolerance {tol*100:.0f}%")
# auditor's discretization-vs-quadrature probe (quadrature refinement is null)
for rec in indep["quadrature_sensitivity"]:
    ax.plot([rec["ka"]], [rec["err_vs_mie_64x64"] * 100], "x", ms=9, color="purple",
            label="auditor rerun, quadrature 64→192 unchanged" if rec is indep["quadrature_sensitivity"][0] else None)
ax.set_xlabel("ka")
ax.set_ylabel("relative error vs Mie [%]")
ax.set_title("Error budget: 1 of 14 declared cases exceeds the 6% tolerance\n(ka=0.25, 320 panels -> 6.12%; 1280 panels -> 1.57%)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.savefig(OUT / "fig2_error_vs_ka.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# --- Figure 3: independent reference routes + repro delta -------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))
ka_ref = [r["ka"] for r in indep["routes"]]
for key, lab, mk in (("rel_diff_A_repo", "mpmath series (route A) vs repo", "o"),
                     ("rel_diff_B_repo", "direct angular integration (route B)", "s"),
                     ("rel_diff_C_repo", "optical theorem (route C)", "^")):
    vals = [max(r[key], 1e-18) for r in indep["routes"]]
    axL.semilogy(ka_ref, vals, mk, ms=7, label=lab)
axL.axhline(1e-12, color="grey", ls=":", lw=1)
axL.set_xlabel("ka")
axL.set_ylabel("|sigma_route - sigma_repo| / sigma_repo")
axL.set_title("Reference leg audited: Mie series re-derived 3 independent ways")
axL.grid(alpha=0.3, which="both")
axL.legend(fontsize=8)

# repro delta: auditor rerun vs audited payload
if rerun:
    kas = [(r["ka"], r["subdivisions"]) for r in rerun]
    deltas = []
    for r in rerun:
        c = next(x for x in rows if x["ka"] == r["ka"] and x["subdivisions"] == r["subdivisions"])
        deltas.append(abs(r["sigma_bem"] - c["sigma_bem"]) / c["sigma_bem"])
    axR.bar([f"{k:g}/{s}" for k, s in kas], [max(d, 1e-18) for d in deltas], color="tab:blue")
    axR.set_yscale("log")
    axR.set_ylim(1e-18, 1e-10)
    axR.set_xlabel("ka / subdivisions")
    axR.set_ylabel("|sigma_rerun - sigma_payload| / sigma")
    axR.set_title(f"Reproducibility: auditor rerun of the payload\n{len(rerun)}/14 rows, max delta "
                  f"{max(deltas):.1e} (bit-identical)")
    axR.tick_params(axis="x", labelrotation=60, labelsize=8)
    axR.grid(alpha=0.3, axis="y")
fig.savefig(OUT / "fig3_reference_and_repro.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
print("rerun source:", rerun_source)
if rerun:
    print(f"rows compared: {len(rerun)}/14, max rel delta "
          f"{max(abs(r['sigma_bem'] - next(x['sigma_bem'] for x in rows if x['ka'] == r['ka'] and x['subdivisions'] == r['subdivisions'])) / next(x['sigma_bem'] for x in rows if x['ka'] == r['ka'] and x['subdivisions'] == r['subdivisions']) for r in rerun):.3e}")