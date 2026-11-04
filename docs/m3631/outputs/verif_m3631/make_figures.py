"""Render the two report figures from the standard-tier artifact.

Both figures are generated from outputs/verif_m3631/standard_tier.json
(the registry check report) -- no number is hand-typed.

    python outputs/verif_m3631/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / "standard_tier.json").read_text())
rows = [r for r in report["checks"] if r["group"] in
        ("hulme_hemisphere", "hulme_chief", "submerged_sphere")]

# --------------------------------------------------------------------------
# Figure 1: convergence ladders, measured vs declared
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
hulme = [r for r in rows if r["group"] == "hulme_hemisphere"]
series = {
    "a33": ("hulme_a33_err_", "A33' heave added mass"),
    "b33": ("hulme_b33_err_", "B33' heave damping"),
    "a11": ("hulme_a11_err_", "A11' surge added mass"),
    "b11": ("hulme_b11_err_", "B11' surge damping"),
    "fz": ("hulme_fz_err_", "Fz excitation force"),
}
for name, (prefix, label) in series.items():
    pts = sorted(
        (r for r in hulme if r["key"].startswith(prefix)),
        key=lambda r: int(r["key"].rsplit("_", 1)[1]),
    )
    n = [int(r["key"].rsplit("_", 1)[1]) for r in pts]
    m = [r["measured"] for r in pts]
    d = [r["declared"] for r in pts]
    ax.plot(n, m, "o-", label=label)
    ax.plot(n, d, "--", alpha=0.55)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("panels N")
ax.set_ylabel("relative error vs Hulme (1982)")
ax.set_title("Hulme floating hemisphere, Ka=1.0\n(solid: measured, dashed: documented)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
sphere = [r for r in rows if r["group"] == "submerged_sphere"]
series = {
    "a33": ("sphere_a33_err_", "A33' heave added mass"),
    "a11": ("sphere_a11_err_", "A11' surge added mass"),
    "fx": ("sphere_fx_err_", "Fx excitation force"),
    "fz": ("sphere_fz_err_", "Fz excitation force"),
}
for name, (prefix, label) in series.items():
    pts = sorted(
        (r for r in sphere if r["key"].startswith(prefix)),
        key=lambda r: int(r["key"].rsplit("_", 1)[1]),
    )
    n = [int(r["key"].rsplit("_", 1)[1]) for r in pts]
    m = [r["measured"] for r in pts]
    d = [r["declared"] for r in pts]
    ax.plot(n, m, "s-", label=label)
    ax.plot(n, d, "--", alpha=0.55)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("panels N")
ax.set_ylabel("relative error vs Thorne multipole")
ax.set_title("Wang/Linton submerged sphere, f/a=1.5, Ka=0.8\n(solid: measured, dashed: documented)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_3d_free_surface_convergence.png", dpi=140)
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2: per-entry deviation vs tolerance
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.6))
order = {"hulme_hemisphere": 0, "hulme_chief": 1, "submerged_sphere": 2}
rows = sorted(rows, key=lambda r: (order[r["group"]], r["key"]))
x = range(len(rows))
dev = [abs(r["relative_deviation"]) for r in rows]
tol = [r["tolerance"] for r in rows]
colors = ["#1f77b4", "#d62728", "#2ca02c"]
ax.bar(x, dev, color=[colors[order[r["group"]]] for r in rows], width=1.0)
ax.plot(x, tol, "k.", ms=4, label="DRIFT tolerance")
ax.set_yscale("log")
ax.set_xticks(list(x)[::4])
ax.set_xticklabels([r["key"] for r in rows][::4], rotation=75, fontsize=6)
ax.set_ylabel("|measured - declared| / declared")
ax.set_title(
    "66 registered 3-D free-surface numbers: deviation from the documented value "
    f"(worst {max(dev):.1e})"
)
ax.grid(True, which="both", alpha=0.3, axis="y")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_3d_free_surface_deviation.png", dpi=140)
plt.close(fig)
print("figures written")
