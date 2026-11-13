#!/usr/bin/env python3
"""Figures for the m17395 audit of CASE-BEM-RIGID-LID-HEMISPHERE.

Reads only committed/produced artifacts (no hand-typed numbers):
  * benchmarks/verified/case_bem_rigid_lid_hemisphere/results.json  (case artifacts)
  * artifacts/m17395/independent_reference_check.json               (this audit)
  * artifacts/m17395/row_headline_1024.json                         (1024-panel rerun)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/m17395/figures")
OUT.mkdir(parents=True, exist_ok=True)
CASE = Path("/root/tp-agent/work/m17338/benchmarks/verified/case_bem_rigid_lid_hemisphere/results.json")
MINE = Path("artifacts/m17395/independent_reference_check.json")
HEAD = Path("artifacts/m17395/row_headline_1024.json")

case = json.loads(CASE.read_text())
mine = json.loads(MINE.read_text())
ref = case["reference"]["value_dimless"]
naive = case["reference"]["naive_translating_image_dimless"]
release = case["reference"]["pressure_release_dimless"]
tol = case["acceptance"]["tolerance"]

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
colors = {64: "tab:blue", 256: "tab:orange"}
for lvl in case["levels"]:
    rows = sorted(lvl["rows"], key=lambda r: r["ka"])
    ka = np.array([r["ka"] for r in rows])
    val = np.array([r["added_mass_dimless"] for r in rows])
    p = rows[0]["panels"]
    ax[0].plot(ka, val, "o-", color=colors.get(p, "tab:gray"), label="TensorBEM solver, %d panels" % p)
    ax[0].plot(ka, [r["hulme_dimless"] for r in rows], "x--", color=colors.get(p, "tab:gray"), alpha=0.6)
band = case["out_of_band"][0]["rows"]
ka_out = np.array([r["ka"] for r in band])
val_out = np.array([r["added_mass_dimless"] for r in band])
ax[0].plot(ka_out, val_out, "s-", color="tab:gray", label="solver, 256 panels (out of band)")
ax[0].axhline(ref, color="k", lw=1.4, label="rigid-lid reference %.7f" % ref)
ax[0].axhspan(ref * (1 - tol), ref * (1 + tol), color="tab:green", alpha=0.12, label="+/-5 % (pool)")
ax[0].axhline(naive, color="tab:red", ls=":", label="translating image 1.0 (refuted)")
ax[0].axhline(release, color="tab:purple", ls=":", label="pressure release 0.5 (exact)")
ax[0].set_xscale("log")
ax[0].set_xlabel("ka")
ax[0].set_ylabel(r"$A_{33}/(\rho\nabla)$")
ax[0].set_title("heave added mass, rigid-lid endpoint band")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=0.3)

# right: error vs panel count at ka = 1e-3, case solver + audit's independent BEM
series = {}
for lvl in case["levels"]:
    for r in lvl["rows"]:
        if abs(r["ka"] - 1e-3) < 1e-12:
            series.setdefault("case", []).append((r["panels"], 100 * r["relative_error"]))
if HEAD.exists():
    h = json.loads(HEAD.read_text())
    series.setdefault("case", []).append((h["row"]["panels"], 100 * h["row"]["relative_error"]))
bem = [(r["panels"], 100 * (r["dimless"] / ref - 1.0)) for r in mine["route_R2_image_bem"]["hemisphere"]]
for key, label, style in (
    ("case", "TensorBEM free-surface solver (ka=1e-3)", "o-"),
    ("bem", "independent image-BEM (this audit, ka=0)", "s-"),
):
    pts = sorted(series["case"] if key == "case" else bem)
    ax[1].plot([p[0] for p in pts], [p[1] for p in pts], style, label=label)
ax[1].axhspan(-5, 5, color="tab:green", alpha=0.12, label="pool tolerance +/-5 %")
ax[1].axhspan(-3, 3, color="tab:blue", alpha=0.08, label="ecosystem 3 %")
ax[1].axhline(0.0, color="k", lw=1.0)
ax[1].set_xscale("log")
ax[1].set_xlabel("panels")
ax[1].set_ylabel("relative error vs rigid-lid reference [%]")
ax[1].set_title("mesh convergence and independent route")
ax[1].legend(fontsize=7)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "audit_m17395_reference_and_convergence.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9))
ladder = mine["route_R1_series"]["ladder"]
ax[0].plot([r["degree_max"] for r in ladder], [r["dimless"] for r in ladder], "o-")
ax[0].axhline(ref, color="k", ls="--", lw=1.0, label="case reference %.9f" % ref)
ax[0].set_xlabel("even-mode truncation degree_max")
ax[0].set_ylabel(r"$A_{33}/(\rho\nabla)$")
ax[0].set_title("audit route R1: series convergence (my own derivation)")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

ctrl = mine["route_R2_image_bem"]
lev = [r["level"] for r in ctrl["sphere_selftest"]]
ax[1].plot(lev, [r["dimless"] for r in ctrl["sphere_selftest"]], "o-", label="full sphere (exact 1.0)")
ax[1].plot(lev, [r["pressure_release_dimless"] for r in ctrl["hemisphere"]], "s-",
           label="pressure-release hemisphere (exact 0.5, x2)")
ax[1].plot(lev, [r["dimless"] for r in ctrl["hemisphere"]], "^-", label="rigid-lid hemisphere (ref 0.83095)")
ax[1].axhline(1.0, color="k", ls=":", lw=0.8)
ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
ax[1].axhline(ref, color="k", ls="--", lw=0.8)
ax[1].set_xlabel("icosphere subdivision level")
ax[1].set_ylabel(r"$A_{33}/(\rho\nabla)$")
ax[1].set_title("audit route R2: image-BEM controls before reading the audited value")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "audit_m17395_independent_controls.png", dpi=140)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))