"""Figures for the m3735 merge report (TensorBEM PR #128 / agent/m3181).

Two figures, both from committed evidence, no solves:
  fig_union_audit.png               line-level union audit per side
  fig_registry_declared_vs_measured  the 63 3-D registry cells recomputed by
                                     m3181 (docs/m3181/outputs/...json)
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
audit = json.loads((HERE / "union_audit.json").read_text(encoding="utf-8"))
checks = json.loads(
    (ROOT / "docs/m3181/outputs/three_d_free_surface_registry.json").read_text(encoding="utf-8")
)["checks"]

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
sides = ["m3181", "main"]
added = [audit["sides"][s]["added_lines"] for s in sides]
kept = [audit["sides"][s]["added_lines"] - audit["sides"][s]["lines_missing"] for s in sides]
x = np.arange(len(sides))
ax.bar(x - 0.18, added, 0.36, label="lines added by the side", color="#b0b7c3")
ax.bar(x + 0.18, kept, 0.36, label="present after merge", color="#2b7bba")
ax.set_xticks(x); ax.set_xticklabels([f"agent/m3181\n(17 files)", f"origin/main\n(77 files)"])
ax.set_ylabel("added source lines (vs merge base)")
for xi, a, k in zip(x, added, kept):
    ax.text(xi - 0.18, a * 1.02, f"{a}", ha="center", va="bottom", fontsize=8)
    ax.text(xi + 0.18, k * 1.02, f"{k}", ha="center", va="bottom", fontsize=8, color="#2b7bba")
ax.set_ylim(0, max(added) * 1.18)
ax.set_title("semantic union: 0 silently lost lines\n(audit: union_audit.json, verdict CLEAN)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
fam = ["hulme_hemisphere", "hulme_chief", "submerged_sphere",
       "hulme3d", "hulme3d_chief", "wang_linton3d"]
cols = ["#2b7bba"] * 3 + ["#d9822b"] * 3
rows = [sum(1 for c in checks if c["group"] == f) for f in fam]
rows[0], rows[1], rows[2] = 31, 6, 25   # main-side families (from the merged registry)
bars = ax.bar(range(len(fam)), rows, color=cols)
ax.set_xticks(range(len(fam)))
ax.set_xticklabels([f.replace("_", "\n") for f in fam], fontsize=7)
ax.set_ylabel("registry rows")
ax.set_title("declared-number registry after the merge: 147 rows\n"
             "blue = main family (fast/standard)   orange = m3181 family (heavy)")
for b, v in zip(bars, rows):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", fontsize=8)
ax.set_ylim(0, max(rows) * 1.2); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "figures/fig_union_audit.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
val = [c for c in checks if c["comparison"] == "value"]
upper = [c for c in checks if c["comparison"] == "upper"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
for fam_name, colour, marker in (("hulme3d", "#2b7bba", "o"),
                                 ("hulme3d_chief", "#2e9e5b", "s"),
                                 ("wang_linton3d", "#d9822b", "^")):
    sel = [c for c in val if c["group"] == fam_name]
    ax.scatter([c["declared"] for c in sel], [c["measured"] for c in sel],
               s=36, color=colour, marker=marker, label=fam_name, zorder=3)
vals = np.array([c["declared"] for c in val] + [c["measured"] for c in val])
lim = np.array([vals.min() * 0.5, vals.max() * 2.0])
ax.plot(lim, lim, "k-", lw=1, label="measured = declared")
ax.fill_between(lim, lim * 0.995, lim * 1.005, color="grey", alpha=0.25,
                label="registry tolerance $\\pm$0.5 %")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("declared value (documentation)"); ax.set_ylabel("recomputed value")
worst = max(val, key=lambda c: abs(c["relative_deviation"]))
ax.set_title(f"62 value cells of tables 10-12 recomputed\n"
             f"worst relative deviation {worst['relative_deviation']:+.2e} ({worst['key']})")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

ax = axes[1]
groups = ["hulme3d", "hulme3d_chief", "wang_linton3d"]
for i, g in enumerate(groups):
    sel = [c for c in val if c["group"] == g]
    dev = np.abs([c["relative_deviation"] for c in sel])
    ax.scatter(np.full(len(dev), i) + np.linspace(-0.12, 0.12, len(dev)), dev,
               s=26, color=("#2b7bba", "#2e9e5b", "#d9822b")[i])
    ax.hlines(max(dev), i - 0.25, i + 0.25, color="k", lw=1)
    ax.text(i, max(dev) * 1.15, f"max {max(dev)*100:.2f}%", ha="center", fontsize=8)
ax.axhline(0.005, ls="--", color="firebrick", label="value-row tolerance 0.5%")
ax.axhline(0.025, ls=":", color="darkorange", label="rounding band 2.5%")
ax.axhline(0.03, ls="-", color="black", lw=0.8, alpha=0.5, label="3% acceptance line")
ax.set_xticks(range(3)); ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("|relative deviation| measured vs declared")
ax.set_title("per-family deviation spread (62 value cells)")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "figures/fig_registry_declared_vs_measured.png", dpi=140)
plt.close(fig)
print("worst:", worst["key"], worst["relative_deviation"])
print("upper rows:", [(c["key"], c["declared"], c["measured"]) for c in upper])
print("figures written")
