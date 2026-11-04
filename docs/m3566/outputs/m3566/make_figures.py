"""Figures for m3566: rigid-body Neumann verification debt closure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/m3566"
bench = json.loads((OUT / "rigid_body.json").read_text(encoding="utf-8"))
report = json.loads((OUT / "standard_tier.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- figure 1
e2 = bench["ellipse_2d"]
levels2 = e2["levels"]
panels = np.array([lv["panels"] for lv in levels2], dtype=float)
kappa2 = np.array([lv["condition_number"] for lv in levels2])
modes2 = ("surge_x", "heave_y", "roll_xy")
labels2 = {"surge_x": r"$A_{11}'$ (surge)", "heave_y": r"$A_{22}'$ (heave)",
           "roll_xy": r"$A_{66}'$ (roll)"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
for mode in modes2:
    err = np.array([lv["diagonal_relative_errors"][mode] for lv in levels2])
    ax.loglog(panels, err, "o-", label=labels2[mode])
    ax.annotate(f"{err[-1]:.1e}", (panels[-1], err[-1]), textcoords="offset points",
                xytext=(-30, -12), fontsize=8)
slope = panels ** -2.0
ax.loglog(panels, 2.2 * slope / slope[0] * 3.2e-2, "k--", lw=1, label=r"$O(h^{2})$ slope")
ax.set_xlabel("panels $N$ (2-D elliptic cylinder)")
ax.set_ylabel("relative error")
ax.set_title("2-D rigid-body Neumann tensor vs Lamb (1932)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax2.semilogx(panels, kappa2, "s-", color="tab:red")
for x, y in zip(panels, kappa2):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)
ax2.set_xlabel("panels $N$")
ax2.set_ylabel(r"cond$(\frac{1}{2}I-H)$")
ax2.set_ylim(2.98, 3.09)
ax2.set_title(r"second-kind operator stays $O(1)$ conditioned")
ax2.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_rigid_2d.png", dpi=82)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
e3 = bench["ellipsoid_3d"]
levels3 = e3["levels"]
panels3 = np.array([lv["panels"] for lv in levels3], dtype=float)
modes3 = ("surge_x", "sway_y", "heave_z", "roll_x", "pitch_y", "yaw_z")
mode_labels = {"surge_x": "surge", "sway_y": "sway", "heave_z": "heave",
               "roll_x": "roll", "pitch_y": "pitch", "yaw_z": "yaw"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
for mode in modes3:
    err = np.array([lv["diagonal_relative_errors"][mode] for lv in levels3])
    style = "-o" if mode in ("surge_x", "sway_y", "heave_z") else "--s"
    ax.loglog(panels3, err, style, label=mode_labels[mode], markersize=5)
ref = panels3 ** -2.0
ax.loglog(panels3, 3.2e-1 * ref / ref[0], "k:", lw=1.2, label=r"$O(h^{2})$")
ax.set_xlabel("panels $N$ (triaxial ellipsoid)")
ax.set_ylabel("relative error")
ax.set_title("3-D $6\\times6$ Neumann tensor vs Lamb shape factors")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2)

# error map of the 6x6 tensor at the finest level (a "field" over the matrix)
diag_rel = levels3[-1]["diagonal_relative_errors"]
grid = np.full((6, 6), np.nan)
order = ("surge_x", "sway_y", "heave_z", "roll_x", "pitch_y", "yaw_z")
for i, mode in enumerate(order):
    grid[i, i] = diag_rel[mode]
im = ax2.imshow(np.log10(grid), cmap="viridis_r")
for i, mode in enumerate(order):
    ax2.annotate(f"{diag_rel[mode]:.1e}", (i, i), color="w", ha="center",
                 va="center", fontsize=7)
ax2.set_xticks(range(6), [mode_labels[m] for m in order], rotation=45, fontsize=8)
ax2.set_yticks(range(6), [mode_labels[m] for m in order], fontsize=8)
ax2.set_title(f"$\\log_{{10}}$ diagonal relative error, "
              f"$N={int(panels3[-1])}$")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT / "fig_rigid_3d.png", dpi=82)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
rows = [row for row in report["checks"] if row["group"] == "rigid_body"]
value_rows = [row for row in rows if row["comparison"] != "upper"]
keys = [row["key"].replace("rigid_body_", "") for row in value_rows]
dev = np.array([abs(row["relative_deviation"]) for row in value_rows])
tol = value_rows[0]["tolerance"]

fig, ax = plt.subplots(figsize=(9.6, 3.6))
ax.bar(range(len(dev)), dev, color="tab:blue", width=0.7)
ax.axhline(tol, color="tab:red", ls="--", label=f"drift tolerance {tol:g}")
ax.set_yscale("log")
ax.set_xticks(range(len(dev)), keys, rotation=90, fontsize=6)
ax.set_ylabel("|relative deviation| doc vs recomputed")
ax.set_title(f"44 documented numbers of §4.7 recomputed: "
             f"{sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)} PASS, "
             f"max {dev.max():.1e}")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_registry_deviations.png", dpi=82)
plt.close(fig)

for name in ("fig_rigid_2d.png", "fig_rigid_3d.png", "fig_registry_deviations.png"):
    print(name, (OUT / name).stat().st_size // 1024, "KB")