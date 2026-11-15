import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = "benchmarks/verified/casepool_bem_cummins_wigley/case_result.json"
d = json.load(open(A))
c5c = d["gates"]["C5c"]["values"]
mw = c5c["multi_window"]
bar = mw["bar"]
zf = c5c["zeta_frequency_domain"]
zc = c5c["zeta_decay_consensus"]
rel = c5c["relative_error"]
rows = sorted(mw["per_floor"], key=lambda r: -r["floor"])   # 1% -> 0.01%
floors = [r["floor"] * 100 for r in rows]                   # % of peak
zetas = [r["zeta"] for r in rows]
peaks = [r["n_peaks"] for r in rows]
spread = mw["spread"]

# ---- Fig 1: monotone window drift vs the frequency-domain reference
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axhspan(zf * (1 - bar), zf * (1 + bar), color="tab:green", alpha=0.15,
           label=f"±{bar:.0%} band around $\\zeta_{{fd}}$")
ax.axhline(zf, color="tab:green", lw=1.2, ls="--",
           label=f"$\\zeta_{{fd}}$ = {zf:.4f} (impedance route)")
ax.plot(floors, zetas, "o-", color="tab:blue", lw=1.8, ms=7,
        label="per-window $\\zeta$ (log decrement)")
ax.axhline(zc, color="tab:red", lw=1.2, ls=":",
           label=f"consensus (median) = {zc:.4f}")
for x, y, n in zip(floors, zetas, peaks):
    ax.annotate(f"{n} peaks", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color="tab:blue")
ax.set_xscale("log")
ax.set_xlabel("amplitude floor (% of peak)  —  wider window →")
ax.set_ylabel("damping ratio $\\zeta$")
ax.set_title(f"C5c roll: windows do NOT converge (spread {spread:.1%}) "
             f"yet consensus meets $\\zeta_{{fd}}$ within {rel:.2%}")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("outputs/m22091/fig1_window_drift.png", dpi=140); plt.close(fig)

# ---- Fig 2: peak count vs the >=5 peak qualification rule
fig, ax = plt.subplots(figsize=(7.2, 3.6))
bars = ax.bar([f"{x:g}%" for x in floors], peaks, color="tab:purple", alpha=0.8)
ax.axhline(5, color="tab:red", ls="--", lw=1.3, label="qualification: ≥5 peaks")
for b, n in zip(bars, peaks):
    ax.text(b.get_x() + b.get_width() / 2, n + 0.4, str(n), ha="center", fontsize=9)
ax.set_ylabel("peaks captured")
ax.set_title("C5c: all three windows qualify — so the gate has ≥2 windows to form a 'consensus'")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("outputs/m22091/fig2_peaks.png", dpi=140); plt.close(fig)

# ---- Fig 3: injected failure branches vs the decision boundary
from tensorbem.benchmarks.case_bem_cummins_wigley import c5c_decision
cases = [("nominal", True, 0.0, 3), ("at bar", True, bar, 3),
         ("just over", True, bar * (1 + 1e-9), 3), ("30% off", True, 0.30, 3),
         ("out of band", False, 0.0, 3), ("1 window", True, 0.0, 1),
         ("inf", True, math.inf, 3)]
fig, ax = plt.subplots(figsize=(7.6, 3.9))
xs = np.arange(len(cases))
for i, (name, ib, r, n) in enumerate(cases):
    status, passed = c5c_decision(in_band=ib, relative=r, n_qualifying=n, bar=bar)
    ax.bar(i, 0.5, bottom=0, color="tab:green" if passed else "tab:red", alpha=0.75)
    ax.text(i, 0.25, "PASS" if passed else "FAIL", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")
ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cases], fontsize=8, rotation=15)
ax.set_yticks([])
ax.set_title("C5c failure injection: all seven branches falsified once (not a rubber stamp)")
fig.tight_layout(); fig.savefig("outputs/m22091/fig3_branches.png", dpi=140); plt.close(fig)
print("figs ok", spread, zf, zc, rel, floors, zetas, peaks)
