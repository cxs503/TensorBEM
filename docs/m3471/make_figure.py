"""Figure for docs/m3471/mutation_audit_3d.md (pure post-processing of JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
BASE = json.loads((HERE / "outputs/mutation_audit_3d/mutation_report.json").read_text())
AFTER = json.loads((HERE / "outputs/mutation_audit_3d_after/mutation_report.json").read_text())
# 2-D reference sweep from the 2026-10-24 audit (docs/m2582), for context only.
BEFORE_2D = {"tensorbem.kernels": 0.750, "tensorbem.free_surface": 0.625,
             "tensorbem.hydrostatic": 0.500, "tensorbem.solver": 0.875,
             "tensorbem.finite_depth": 0.500}


def score(payload: dict) -> tuple[list[str], list[float]]:
    modules, values = [], []
    for name, entry in payload["modules"].items():
        modules.append(name.replace("tensorbem.", ""))
        values.append(entry["mutation_score"])
    return modules, values


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))

mods, before = score(BASE)
_, after = score(AFTER)
x = range(len(mods))
width = 0.38
ax1.bar([i - width / 2 for i in x], before, width, label="3-D suite as merged", color="#c0504d")
ax1.bar([i + width / 2 for i in x], after, width, label="+ mutation_gaps_3d guards", color="#4f81bd")
for i, (b, a) in enumerate(zip(before, after)):
    ax1.text(i - width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    ax1.text(i + width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
ax1.set_xticks(list(x))
ax1.set_xticklabels(mods, rotation=18)
ax1.set_ylim(0.0, 1.15)
ax1.set_ylabel("mutation score (killed / killed+survived)")
ax1.set_title("3-D BEM core: guard strength before/after m3471")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(axis="y", alpha=0.3)

all_before = dict(BEFORE_2D)
all_before.update({f"3d:{m}": v for m, v in zip(mods, before)})
all_after = dict(BEFORE_2D)
all_after.update({f"3d:{m}": v for m, v in zip(mods, after)})
labels = list(all_before)
vals = [all_before[k] for k in labels]
cols = ["#9dbfdf" if not k.startswith("3d:") else "#4f81bd" for k in labels]
ax2.barh(range(len(labels)), vals, color=cols)
for i, (k, v) in enumerate(zip(labels, vals)):
    ax2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    if k.startswith("3d:"):
        ax2.text(0.02, i, f"-> {all_after[k]:.2f}", va="center", fontsize=7, color="white")
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlim(0.0, 1.1)
ax2.set_xlabel("mutation score")
ax2.set_title("Audited surface: 2-D (m2582, light) vs 3-D (m3471, dark)")
ax2.grid(axis="x", alpha=0.3)

fig.tight_layout()
out = HERE / "figures/mutation_score_3d.png"
fig.savefig(out, dpi=160)
print(out, out.stat().st_size)
