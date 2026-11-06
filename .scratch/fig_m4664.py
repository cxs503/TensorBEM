"""Figures for the m4664 report. Temporary helper (deleted before commit)."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path("/root/tp-agent/work/m4664")
OUT = REPO / ".scratch" / "fig_m4664"
OUT.mkdir(parents=True, exist_ok=True)

study = json.loads((REPO / "artifacts/analytical-study.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

styles = {
    "disk_cubic": ("o", "#1f77b4"),
    "ellipse_quadratic": ("s", "#d62728"),
    "l_shape_singular": ("^", "#2ca02c"),
    "square_separable": ("D", "#9467bd"),
}

h_all = []
for name, case in study["cases"].items():
    res = case["results"]
    h = np.array([r["h_max"] for r in res])
    fr = np.array([r["field_rmse"] for r in res])
    nd = np.array([r["normal_derivative_relative_l2_error"] for r in res])
    marker, color = styles[name]
    axes[0].loglog(h, fr, marker=marker, color=color, label=name)
    axes[1].loglog(h, nd, marker=marker, color=color, label=name)
    h_all.extend(h)

h_ref = np.array([min(h_all) * 0.9, max(h_all) * 1.1])
axes[0].loglog(h_ref, 0.024 * h_ref**2, "k--", lw=1, label=r"$O(h^2)$ reference")
axes[1].loglog(h_ref, 0.18 * h_ref, "k:", lw=1, label=r"$O(h)$ reference")

for ax, title, ylab in (
    (axes[0], "(a) interior field solution, constant panels", "field RMSE vs analytic"),
    (axes[1], "(b) boundary normal derivative (flux)", r"relative $L_2$ error"),
):
    ax.set_xlabel("mesh size $h_{\\max}$ [-]")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

fig.suptitle(
    "TensorBEM main a261e23 (+m4664): 2-D Laplace verification, 16/32/64/128 panels",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(OUT / "fig1_analytical_convergence.png", dpi=100)
print("wrote", OUT / "fig1_analytical_convergence.png")