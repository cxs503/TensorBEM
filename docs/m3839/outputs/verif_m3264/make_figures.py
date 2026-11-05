"""Regenerate the 3-D free-surface registry report and its figures."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check

OUT = "outputs/verif_m3264"
GROUPS = ("hulme_hemisphere", "hulme_chief", "submerged_sphere")

# ---------------------------------------------------------------- registry
report = check(
    tiers=("standard",), entries=[e for e in DECLARED_NUMBERS if e.group in GROUPS]
)
with open(f"{OUT}/registry_3d.json", "w") as handle:
    json.dump(report, handle, indent=1, sort_keys=True, allow_nan=False)

measured = json.load(open(f"{OUT}/measured.json"))
hulme = {row["subdivisions"]: row for row in measured["hulme_refinement"]["levels"]}
sphere = {row["subdivisions"]: row for row in measured["submerged_sphere_refinement"]["levels"]}

# ---------------------------------------------------------------- figure 1
CELLS = [
    ("heave_added_mass_rel_error", "$A'_{33}$", "o"),
    ("heave_damping_rel_error", "$B'_{33}$", "s"),
    ("surge_added_mass_rel_error", "$A'_{11}$", "^"),
    ("surge_damping_rel_error", "$B'_{11}$", "v"),
    ("excitation_force_z_rel_error", "$|F_z|$", "D"),
]
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
for ax, levels, title, group in (
    (axes[0], hulme, "Hulme floating hemisphere, $Ka=1$", "hulme_hemisphere"),
    (axes[1], sphere, r"Wang/Linton submerged sphere, $f/a=1.5$, $Ka=0.8$", "submerged_sphere"),
):
    declared = {
        (e.parameters["subdivisions"][0], e.quantity): e.declared
        for e in DECLARED_NUMBERS
        if e.group == group
    }
    h = [levels[k]["h_mean"] for k in sorted(levels)]
    for quantity, label, marker in CELLS:
        if quantity not in next(iter(levels.values())):
            continue
        err = [levels[k][quantity] for k in sorted(levels)]
        ax.plot(h, err, marker=marker, ls="-", label=f"{label} BEM", ms=6)
        doc = [declared.get((k, quantity)) for k in sorted(levels)]
        ax.plot(h, doc, marker=marker, ls="none", mfc="none", mec="k", ms=11, mew=1.0)
    guide = np.array(sorted(h))
    ref = levels[max(levels)]["heave_added_mass_rel_error"]
    ax.plot(guide, ref * (guide / guide.max()) ** 2, "k--", lw=1.0, label="$O(h^2)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean panel size $h$  [m]")
    ax.set_ylabel("relative error")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Table 10/12 cells: filled = recomputed, open = the number printed in the docs")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_3d_convergence.png", dpi=150)

# ---------------------------------------------------------------- figure 2
sweeps = {
    name: json.load(open(f"{OUT}/sweep_{name}.json"))["rows"]
    for name in ("hulme", "sphere")
}
fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6))
for col, (name, title) in enumerate(
    (("hulme", "Hulme hemisphere (256 panels)"), ("sphere", "Submerged sphere (320 panels)"))
):
    rows = sweeps[name]
    ka = [r["ka"] for r in rows]
    for row_idx, (key, label) in enumerate(
        (("a33", "heave added mass $A'_{33}$"), ("b33", "heave damping $B'_{33}$"))
    ):
        ax = axes[row_idx][col]
        ax.plot(ka, [r[f"exact_{key}"] for r in rows], "k-", lw=1.4, label="exact (multipole)")
        ax.plot(
            ka,
            [r[f"bem_{key}"] for r in rows],
            "o",
            ms=6,
            mfc="none",
            mec="C3",
            mew=1.4,
            label="BEM",
        )
        ax.set_xlabel("$Ka$")
        ax.set_ylabel(label)
        ax.grid(True, ls=":", lw=0.5)
        ax.legend(fontsize=8)
        ax.set_title(title if row_idx == 0 else "")
fig.suptitle("Frequency dependence, BEM vs semi-analytical reference")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_3d_sweep.png", dpi=150)

# ---------------------------------------------------------------- figure 3
rows = report["checks"]
labels = [r["key"] for r in rows]
dev = np.array([abs(r["relative_deviation"]) for r in rows])
order = np.argsort(dev)
fig, ax = plt.subplots(figsize=(7.4, 11.0))
colors = ["C0" if rows[i]["comparison"] == "value" else "C1" for i in order]
ax.barh(np.arange(len(rows)), dev[order], color=colors, height=0.75)
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels([labels[i] for i in order], fontsize=4.6)
ax.set_xscale("log")
ax.set_xlabel("|measured - declared| / |declared|")
ax.set_title(
    f"{len(rows)} 3-D free-surface rows: all PASS\n"
    f"worst equality row {max(d for d, r in zip(dev, rows) if r['comparison'] == 'value'):.2e}"
)
ax.grid(True, axis="x", which="both", ls=":", lw=0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_registry_deviation.png", dpi=150)

summary = report["summary"]
print(json.dumps({k: summary[k] for k in ("checked", "passed", "drifted", "verdict")}, indent=1))
print("worst value-row deviation:", summary["worst_value_relative_deviation"])
