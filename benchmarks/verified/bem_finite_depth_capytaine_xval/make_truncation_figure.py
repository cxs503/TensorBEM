#!/usr/bin/env python3
"""Figures for the mode-cap truncation study (no cluster needed).

Figure 1: per-row mode requirement vs mesh refinement (sub 0..3) with the caps
          pinned by the cross-validation study drawn as horizontal lines.
Figure 2: heave added mass / damping of the cached sub=3 h/R=5 ka=1 cases vs
          mode cap (the recorded 9786 / 463427 / -2175 triple) against the
          capytaine reference, showing the cap - not the kernel - is the driver.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.finite_depth_solver3d import _row_mode_requirements  # noqa: E402
from tensorbem.geometry3d import submerged_sphere_mesh  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

requirements = {}
for sub in (0, 1, 2, 3):
    mesh = submerged_sphere_mesh(sub, radius=1.0, submergence=2.0)
    need = _row_mode_requirements(mesh, 5.0)
    requirements[sub] = {"panels": int(mesh.panel_count),
                         "min": int(need.min()), "median": float(np.median(need)),
                         "max": int(need.max())}

# ---- cached cap ladder at sub=3, h/R=5, ka=1 ------------------------------
cached = {}
for cap in (8, 16, 24, 32, 48, 64, 96):
    path = HERE / "cases" / f"tbem-modeM{cap}-sub3-h5-ka1.json"
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("status", "ok") == "ok" and "added_mass_heave" in d:
            cached[cap] = d
cpt_path = HERE / "cases" / "cpt-24x48-h5-ka1.json"
cpt = json.loads(cpt_path.read_text()) if cpt_path.exists() else {}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
subs = sorted(requirements)
ax.bar([f"sub{s}\n{requirements[s]['panels']} panels" for s in subs],
       [requirements[s]["max"] for s in subs], color="#4477aa", label="max row requirement")
ax.plot(range(len(subs)), [requirements[s]["median"] for s in subs], "o-",
        color="#ee6677", label="median row requirement")
for cap, style in ((48, "--"), (96, ":")):
    ax.axhline(cap, color="k", linestyle=style, linewidth=1,
               label=f"pinned cap {cap} (study)")
ax.set_yscale("log")
ax.set_ylabel("evanescent modes required (h/R = 5)")
ax.set_title("Cap vs requirement\n(requirement grows as the mesh refines)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
if cached:
    caps = sorted(cached)
    am = [cached[c]["added_mass_heave"] for c in caps]
    bd = [cached[c]["damping_heave"] for c in caps]
    ax.semilogy(caps, am, "o-", color="#4477aa", label="A33 TensorBEM")
    ax.semilogy(caps, bd, "s-", color="#ee6677", label="B33 TensorBEM")
    if cpt:
        ax.axhline(abs(cpt.get("added_mass_heave", np.nan)), color="#4477aa",
                   linestyle="--", label="A33 capytaine 24x48")
        ax.axhline(abs(cpt.get("damping_heave", np.nan)), color="#ee6677",
                   linestyle="--", label="B33 capytaine 24x48")
    ax.set_xlabel("pinned mode cap")
    ax.set_ylabel("heave coefficient")
    ax.set_title("sub=3 (1280 panels), h/R=5, ka=1\n"
                 "capped ladder is non-monotone: truncation, not the kernel")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
else:
    ax.text(0.5, 0.5, "no cached cap ladder", ha="center")
fig.tight_layout()
out = FIG / "finite_depth_mode_cap_truncation.png"
fig.savefig(out, dpi=160)
print("wrote", out)
print(json.dumps({"requirements": requirements,
                  "cached_caps": sorted(cached),
                  "cpt": {k: cpt.get(k) for k in ("added_mass_heave", "damping_heave")}},
                 indent=1))