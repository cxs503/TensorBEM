"""Figures for the PR #139 merge report.

fig_pr139_registry_union.png  -- what the merge actually carries (entry sets)
fig_pr139_order_adjudication.png -- log-log B11' convergence with the two
                                    mesh-size conventions that disagree on the
                                    table-10 row-2 observed order
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
AUDIT = json.loads((HERE / "union_audit.json").read_text())
ADJ = json.loads((HERE / "table10_row2_order_adjudication.json").read_text())

# --------------------------------------------------------------- figure 1 ----
reg = AUDIT["registry"]
labels = ["merge base\n4f5b73e", "main\n(ecba798)", "agent/m3306\n(36ea849)", "merged\n(union)"]
values = [reg["keys_base"], reg["keys_main"], reg["keys_m3306"], reg["keys_merged"]]
colors = ["#9e9e9e", "#1f77b4", "#ff7f0e", "#2ca02c"]
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=170)
bars = ax.bar(labels, values, color=colors, width=0.6)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", fontsize=10)
ax.set_ylabel("declared numbers in the registry")
ax.set_title("PR #139 merge: registry entry sets (117 = 84 ∪ 55, 22 shared with base)")
ax.set_ylim(0, max(values) * 1.18)
ax.grid(axis="y", alpha=0.3)
ax.annotate(
    "no entry dropped:\naudit set-equality holds",
    xy=(3, values[3]),
    xytext=(1.35, values[3] * 1.02),
    fontsize=9,
    arrowprops={"arrowstyle": "->", "alpha": 0.6},
)
fig.tight_layout()
fig.savefig(HERE / "fig_pr139_registry_union.png")
plt.close(fig)

# --------------------------------------------------------------- figure 2 ----
levels = ADJ["levels"]
h = [row["h_mean"] for row in levels]
e = [row["surge_damping_rel_error"] for row in levels]
panels = [row["panel_count"] for row in levels]

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=170)
ax.loglog(h, e, "o-", color="#1f77b4", label="B11' relative error (Hulme 1982, Ka=1)")
for i, n in enumerate(panels):
    ax.annotate(f"{n} panels", (h[i], e[i]), textcoords="offset points", xytext=(8, 6), fontsize=8)

# slope guides through the 16-panel point
h0, e0 = h[0], e[0]
for slope, colour, label in (
    (1.8020558782321292, "#2ca02c", "slope 1.80 (measured h_mean ratio: table-10 cell, main)"),
    (1.6247006308715708, "#d62728", "slope 1.62 (nominal refinement factor 2: m3306's kept cell)"),
):
    xs = [h0, h[-1]]
    ys = [e0, e0 * (h[-1] / h0) ** (-slope)]
    ax.loglog(xs, ys, "--", color=colour, lw=1.3, label=label)

ax.set_xlabel("panel mean edge length $h_{mean}$")
ax.set_ylabel("relative error $|B'_{11,panel}-B'_{11,Hulme}|/B'_{11,Hulme}$")
ax.set_title("The one contested cell, adjudicated by recomputation: 1.80 at $h_{mean}$, 1.62 at $\\sqrt{N}$")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(HERE / "fig_pr139_order_adjudication.png")
plt.close(fig)

print("wrote", (HERE / "fig_pr139_registry_union.png").name, (HERE / "fig_pr139_order_adjudication.png").name)