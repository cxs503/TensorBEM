"""Regenerate the figure of docs/finite_depth_diffraction.md from results.json.

Nothing in the report is transcribed by hand: the curves come from the JSON that
``python -m tensorbem.benchmarks.finite_depth_diffraction`` wrote.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "results.json").read_text())

sweep = DATA["sweep"]
kas = sorted({row["ka"] for row in sweep})
depths = sorted({row["depth"] for row in sweep})

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

# (a) exciting force versus ka for the two depths
for depth in depths:
    rows = [row for row in sweep if row["depth"] == depth]
    rows.sort(key=lambda row: row["ka"])
    ka = [row["ka"] for row in rows]
    force = [row["excitation_direct_abs"] for row in rows]
    ref = [row["excitation_mfs_real"] ** 2 + row["excitation_mfs_imag"] ** 2 for row in rows]
    axes[0].plot(ka, force, "o-", label=f"panel h/a={depth:.0f}")
    axes[0].plot(ka, np.sqrt(ref), "x--", label=f"MFS h/a={depth:.0f}")
axes[0].set_xlabel("ka")
axes[0].set_ylabel("|X| [N/m] (|z|eta = 1 m)")
axes[0].set_title("heave exciting force")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)

# (b) reciprocity and independent-reference residuals
labels = [f"ka={row['ka']:.1f}\nh/a={row['depth']:.0f}" for row in sweep]
reciprocity = [row["haskind_reciprocity_rel"] for row in sweep]
mfs = [row["mfs_reference_rel"] for row in sweep]
x = np.arange(len(sweep))
axes[1].bar(x - 0.2, reciprocity, 0.4, label="Haskind vs diffraction")
axes[1].bar(x + 0.2, mfs, 0.4, label="MFS reference")
axes[1].set_yscale("log")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=6)
axes[1].set_ylabel("relative residual")
axes[1].set_title("independent checks")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3, axis="y")

# (c) panel convergence of the reciprocity and reference residuals
convergence = DATA["panel_convergence"]
panels = [row["panels"] for row in convergence]
axes[2].loglog(panels, [row["haskind_reciprocity_rel"] for row in convergence], "o-", label="Haskind reciprocity")
axes[2].loglog(panels, [row["mfs_reference_rel"] for row in convergence], "s-", label="MFS reference")
axes[2].set_xlabel("panels")
axes[2].set_ylabel("relative residual")
axes[2].set_title("ka=0.6, h/a=4: mesh refinement")
axes[2].legend(fontsize=7)
axes[2].grid(alpha=0.3, which="both")

fig.tight_layout()
out = HERE / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "finite_depth_diffraction.png", dpi=160)
print("wrote", out / "finite_depth_diffraction.png")