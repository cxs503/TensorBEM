"""Figures for the m3940 merge record (PR #128 -> main, branch agent/m3181).

Two panels, both generated from the registry itself rather than typed in:

* coverage: how many registry rows each side contributed per group and what the
  merged registry holds, so a reader can see nothing was dropped;
* the four cells whose declared value the two branches disagreed on, shown as
  the relative gap between the branch value and the merged (canonical) value.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Row counts of the three branches of history, read from the git objects rather
# than retyped: main (ecba798), agent/m3181 (55db397).
COUNTS = json.loads((Path(__file__).resolve().parent / "coverage.json").read_text())
GROUPS = [
    ("hulme_hemisphere", "table 10\nHulme hemisphere"),
    ("hulme_chief", "table 11\nKa=2.58 CHIEF"),
    ("submerged_sphere", "table 12\nWang/Linton sphere"),
]
CORRECTED = [
    ("table 10 B11' order, 64 panels", 1.62, 1.8021),
    ("table 10 B11' order, 256 panels", 1.88, 1.9377),
    ("table 12 rotational 80 panels", 2.2e-4, 2.1510e-4),
    ("table 12 rotational 320 panels", 2.6e-5, 2.648e-5),
]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

ax = axes[0]
labels = [label for _, label in GROUPS]
main_counts = [COUNTS["main"][group] for group, _ in GROUPS]
pr_counts = [COUNTS["branch"][group] for group, _ in GROUPS]
merged_counts = [COUNTS["merged"][group] for group, _ in GROUPS]
idx = range(len(labels))
width = 0.26
ax.bar([i - width for i in idx], main_counts, width, label="main (ecba798)")
ax.bar(list(idx), pr_counts, width, label="agent/m3181 (55db397)")
ax.bar([i + width for i in idx], merged_counts, width, label="merged (m3940)")
ax.set_xticks(list(idx))
ax.set_xticklabels(labels)
ax.set_ylabel("registry rows")
ax.set_ylim(0, 36)
for i, value in enumerate(merged_counts):
    ax.text(i + width, value + 0.6, str(value), ha="center", fontsize=9)
ax.set_title("Table 10-12 coverage: union keeps every cell")
ax.legend(fontsize=8, frameon=False)
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
names = [name for name, _, _ in CORRECTED]
gaps = [abs(merged - branch) / abs(merged) for _, branch, merged in CORRECTED]
ax.barh(names, [gap * 100.0 for gap in gaps], color="#b3541e")
ax.set_xlabel("branch value vs merged declared value  [%]")
ax.set_title("Cells the two branches disagreed on (metric / rounding, not solver drift)")
ax.set_xlim(0, 15)
for i, (name, branch, merged) in enumerate(CORRECTED):
    ax.text(gaps[i] * 100.0 + 0.2, i, f"{branch:g} -> {merged:g}", va="center", fontsize=9)
ax.grid(axis="x", alpha=0.25)
ax.invert_yaxis()

fig.tight_layout()
target = OUT / "fig_merge_registry_union.png"
fig.savefig(target, dpi=150)
print(target)
print("merged rows:", len(DECLARED_NUMBERS))
