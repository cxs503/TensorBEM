"""Figures for CASE-BEM-RIGID-BODY-MASS (Wigley hull) from run.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "run.json").read_text())
levels = data["levels"]
panels = np.array([lv["panels"] for lv in levels], dtype=float)
entry_err = np.array([lv["max_entry_rel_error"] for lv in levels])
vol_err = np.array([lv["volume_rel_error"] for lv in levels])
inertia_err = np.array([lv["inertia_rel_error"] for lv in levels])

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.loglog(panels, entry_err, "o-", label="max per-entry error (6x6)")
ax.loglog(panels, inertia_err, "s-", label="inertia tensor error")
ax.loglog(panels, vol_err, "^-", label="volume error")
ref_panels = np.array([panels[0], panels[-1]])
ax.loglog(ref_panels, entry_err[0] * (ref_panels / panels[0]) ** -1.0, "k--",
          lw=0.8, label=r"$O(h^2)$ reference slope")
ax.axhline(data["parameters"]["tolerance"], color="crimson", ls=":", label="3 % criterion")
ax.set_xlabel("panels on the closed mesh")
ax.set_ylabel("relative error")
ax.set_title("Wigley hull 6x6 rigid-body mass matrix: convergence")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_convergence.png", dpi=150)

ref = np.array(data["reference"]["matrix"])
worst = min(levels, key=lambda lv: lv["max_entry_rel_error"])
# rebuild the mesh-route matrix for the finest level
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from tensorbem.benchmarks.case_rigid_body_mass_wigley import HULL, mesh_route

route = mesh_route(HULL, worst["n_sections"], worst["n_levels"])
got = route["matrix"]
scale = np.max(np.abs(ref))
err = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-6 * scale)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
for ax, mat, title, fmt in (
    (axes[0], ref, "closed form (ref)", "%.3g"),
    (axes[1], got, f"BEM mesh route ({worst['panels']} panels)", "%.3g"),
    (axes[2], err, "per-entry relative error", "%.2g"),
):
    im = ax.imshow(mat, cmap="RdBu_r" if title != "per-entry relative error" else "viridis")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    labels = data["parameters"]["hull"]["mode_order"]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, "0" if mat[i, j] == 0 else fmt % mat[i, j],
                    ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(HERE / "fig_block_matrix.png", dpi=150)
print("wrote figures")
