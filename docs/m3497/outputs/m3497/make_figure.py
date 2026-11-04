"""Figure for the 2-D free-surface mutation audit (m3497)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sweep = json.loads((ROOT / "outputs/m3497/mutation_sweep.json").read_text())
screen = json.loads((ROOT / "outputs/m3497/mutation_screen.json").read_text())
equiv = {r["mutant"]: r["equivalent"] for r in screen["rows"]}
dev = {r["mutant"]: r["probe_max_rel_deviation"] for r in screen["rows"]}

names = [r["mutant"].split("_", 1)[0] for r in sweep["before"]]
short = {
    "M1": "PV cos regulariser sign",
    "M2": "PV sin regulariser dropped",
    "M3": "kernel scaled by 1.001",
    "M4": "gradient components swapped",
    "M5": "validity guard disabled",
    "M6": "finite depth -> deep-water root",
    "M7": "mode norm factor",
    "M8": "c_g = c_p/2 everywhere",
    "M9": "evanescent roots +1%",
    "M10": "wave factor scaled by 1.01",
}
before = [r["killed"] for r in sweep["before"]]
after = [r["killed"] for r in sweep["after"]]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
y = range(len(names))
for i, (b, a) in enumerate(zip(before, after)):
    ax.barh(i + 0.19, 1.0, height=0.36, color="#2a9d8f" if a else "#e76f51", alpha=0.95)
    ax.barh(i - 0.19, 1.0, height=0.36, color="#8ab17d" if b else "#f4a261", alpha=0.55)
ax.set_yticks(list(y))
ax.set_yticklabels([f"{n}  {short[n]}" for n in names], fontsize=9)
ax.set_xticks([])
ax.set_xlim(0, 1.28)
ax.invert_yaxis()
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(1.02, i - 0.19, "killed" if b else ("equivalent" if equiv[sweep["before"][i]["mutant"]] else "ESCAPED"),
            va="center", fontsize=8, color="#333")
    ax.text(1.02, i + 0.19, "killed" if a else ("equivalent" if equiv[sweep["after"][i]["mutant"]] else "ESCAPED"),
            va="center", fontsize=8, color="#333")
ax.set_title("upper bar = guard set + new quantitative gates\nlower bar = original guard set", fontsize=10)
ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color="#8ab17d", alpha=0.55, label="before: killed"),
    plt.Rectangle((0, 0), 1, 1, color="#f4a261", label="before: escaped"),
    plt.Rectangle((0, 0), 1, 1, color="#2a9d8f", label="after: killed"),
    plt.Rectangle((0, 0), 1, 1, color="#e76f51", label="after: escaped"),
], loc="lower right", fontsize=8, frameon=False)
for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)

xs = [dev[r["mutant"]] for r in screen["rows"]]
cols = ["#e76f51" if equiv[r["mutant"]] else "#2a9d8f" for r in screen["rows"]]
ax2.bar(range(len(xs)), [max(v, 1e-17) for v in xs], color=cols)
ax2.set_yscale("log")
ax2.axhline(1e-12, color="k", ls="--", lw=1)
ax2.text(0.02, 1.3e-12, "equivalence threshold 1e-12", fontsize=8)
ax2.set_xticks(range(len(xs)))
ax2.set_xticklabels(names, fontsize=8)
ax2.set_ylabel("max relative deviation of kernel probes")
ax2.set_title("equivalence screening: only M1/M2 are no-ops\n(symmetric Legendre nodes => sum w/t = 0)", fontsize=10)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)
fig.tight_layout()
out = ROOT / "outputs/m3497/fig_mutation_fs2d.png"
fig.savefig(out, dpi=105)
print("wrote", out)
