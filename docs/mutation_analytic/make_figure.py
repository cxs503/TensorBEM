"""Figure for the analytic-oracle mutation audit (data read from the JSON runs)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
before = json.loads((here / "before" / "results.json").read_text())
detail = json.loads((here / "rejudge_detail.json").read_text())
n = before["score"]["selected"]
b_kill = before["score"]["killed"]
b_to = before["score"]["timeout"]
b_surv = before["score"]["survived"]
after_kill = b_kill + sum(1 for v in detail.values() if v["verdict"] == "killed")
equivalent = sum(1 for v in detail.values() if v["verdict"] == "survived")
denom = n - equivalent

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
labels = ["killed\n(assertion failed)", "killed by timeout\n(no clean assertion)", "survived"]
before_vals = [b_kill, b_to, b_surv]
after_vals = [after_kill, 0, equivalent]
x = range(len(labels))
ax.bar([i - 0.19 for i in x], before_vals, width=0.38, label="before (mapped suite only)", color="#9aa0a6")
ax.bar([i + 0.19 for i in x], after_vals, width=0.38, label="after (guards back-filled)", color="#1f6feb")
for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
    ax.text(i - 0.19, bv + 0.15, str(bv), ha="center", fontsize=9)
    ax.text(i + 0.19, av + 0.15, str(av), ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(f"mutants (n={n})"); ax.set_ylim(0, max(before_vals) + 3)
ax.set_title("tensorbem.analytic: verdict of 24 seeded defects")
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
scores = [b_kill / n, after_kill / n, after_kill / denom]
names = ["before", "after\n(equivalent counted in)", "after\n(equivalent excluded)"]
bars = ax.bar(names, scores, color=["#9aa0a6", "#1f6feb", "#0b7a3b"], width=0.55)
for bar, value in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.12); ax.set_ylabel("mutation score (killed / scored)")
ax.set_title("Discriminating power of the analytic oracle's guard")
ax.axhline(1.0, color="k", lw=0.6, ls=":")
fig.tight_layout()
out = here / "fig_analytic_mutation_score.png"
fig.savefig(out, dpi=160)
print("wrote", out, out.stat().st_size)
