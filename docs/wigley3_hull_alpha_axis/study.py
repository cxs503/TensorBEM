"""m16935: cross-builder radiation closure after the Wigley III geometry fix."""
import json, time, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorbem.benchmarks.wigley3_recalled_ref import (
    DENSITY, GRAVITY, OMEGA_STATED, build_mesh, mesh_volume)
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem import wigley3_hull as wh
from tensorbem import wigley3_freq as wf

L, B, T, A = wh.WIGLEY3_L, wh.WIGLEY3_B, wh.WIGLEY3_T, wh.WIGLEY3_ALPHA
scale = DENSITY * wh.wigley3_displaced_volume()
out = {"mission": "m16935", "omega": OMEGA_STATED, "density": DENSITY,
       "gravity": GRAVITY, "alpha": A, "analytic_volume": wh.wigley3_displaced_volume(),
       "panel_level": 10, "cells": {}}
for builder in ("wigley3_hull", "wigley3_freq"):
    mesh = build_mesh(builder, 10)
    t0 = time.time()
    res = solve_radiation_3d(mesh, OMEGA_STATED**2 / GRAVITY, modes=(2,),
                             density=DENSITY, gravity=GRAVITY,
                             center=(0.0, 0.0, -T / 2.0))
    out["cells"][builder] = {
        "panels": int(len(mesh.faces)),
        "A33": float(res.added_mass[0, 0]),
        "A33_over_rhoV": float(res.added_mass[0, 0]) / scale,
        "B33_over_rhoV_omega": float(res.damping[0, 0]) / (scale * OMEGA_STATED),
        "mesh_volume_rel_error": mesh_volume(mesh) / wh.wigley3_displaced_volume() - 1.0,
        "wall_s": round(time.time() - t0, 2)}
    print(builder, out["cells"][builder], flush=True)
h, f = out["cells"]["wigley3_hull"], out["cells"]["wigley3_freq"]
out["cross_builder"] = {
    "A33_rel_split": f["A33_over_rhoV"] / h["A33_over_rhoV"] - 1.0,
    "B33_rel_split": f["B33_over_rhoV_omega"] / h["B33_over_rhoV_omega"] - 1.0}
out["pre_fix_reference"] = {
    "source": "benchmarks/verified/wigley3_recalled_ref/results.json (m16892, level 20 meshes)",
    "A33_over_rhoV_wigley3_hull": 1.611257137008352,
    "A33_over_rhoV_wigley3_freq": 1.4556533984300866}
out["pre_fix_reference"]["A33_rel_split"] = (
    out["pre_fix_reference"]["A33_over_rhoV_wigley3_freq"]
    / out["pre_fix_reference"]["A33_over_rhoV_wigley3_hull"] - 1.0)

x = np.linspace(-0.5 * L, 0.5 * L, 2001)
y_now = np.asarray(wh.wigley3_half_breadth(x, np.zeros_like(x)), dtype=float)
y_ret = 0.5 * B * (1 - (2 * x / L) ** 2)          # retired zeta-placed recipe at z = 0
props = wh.compute_wigley3_properties()
out["geometry"] = {
    "Cwp_now": float(2 * np.trapezoid(y_now, x) / (L * B)),
    "Cwp_retired_recipe": float(2 * np.trapezoid(y_ret, x) / (L * B)),
    "Cwp_analytic": 2.0 / 3.0 + 2.0 * A / 15.0,
    "Cb": props.Cb, "Cm": 2.0 / 3.0, "KB_over_T": props.KB / T,
    "A_wp": props.waterplane_area,
    "max_abs_y_difference_vs_wigley3_freq": float(np.max(np.abs(
        y_now - wf.wigley3_halfbreadth(x, np.zeros_like(x), L, B, T))))}
pathlib.Path("docs/wigley3_hull_alpha_axis/study.json").write_text(json.dumps(out, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(x / L, y_now / (B / 2), "k-", lw=2, label=f"fixed: Cwp = {out['geometry']['Cwp_now']:.4f}")
ax[0].plot(x / L, y_ret / (B / 2), "r--", lw=1.6, label=f"retired recipe: Cwp = {out['geometry']['Cwp_retired_recipe']:.4f}")
ax[0].set_xlabel("x / L"); ax[0].set_ylabel("y(x, 0) / (B/2)")
ax[0].set_title("Wigley III calm waterline"); ax[0].legend(); ax[0].grid(alpha=.3)
names = ["wigley3_hull", "wigley3_freq"]
vals = [out["cells"][n]["A33_over_rhoV"] for n in names]
ax[1].bar(names, vals, color=["#2a6fdb", "#e08a1e"])
ax[1].axhline(np.mean(vals), ls=":", color="k", lw=1)
ax[1].set_ylim(1.9, 2.1); ax[1].set_ylabel(r"$A_{33}/(\rho\nabla)$ at $\omega=1.0$")
ax[1].set_title(f"cross-builder split = {100*out['cross_builder']['A33_rel_split']:+.2f} % (was "
                f"{100*out['pre_fix_reference']['A33_rel_split']:+.1f} %)")
ax[1].grid(alpha=.3, axis="y")
fig.tight_layout()
fig.savefig("docs/wigley3_hull_alpha_axis/fig_alpha_axis.png", dpi=140)
print("WROTE", json.dumps(out["cross_builder"]), json.dumps(out["geometry"]))
