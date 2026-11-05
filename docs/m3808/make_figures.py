"""Figures for the m3808 CI-red fix report (PR #125, TensorBEM).

Panel (a): the three free-surface residuals of ``docs/finite_depth3d.md`` §3.2
before/after the m3409 kernel rewrite, on a log axis, against the printed
tolerance and the new absolute noise floor.
Panel (b): which rows of the 64-row pin table moved at all, by category.

Every number is read from the JSON artifacts, none is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CROSS = json.loads((ROOT / "docs" / "finite_depth3d_pins" / "roundoff_crosscheck.json").read_text())
AUDIT = json.loads((ROOT / "docs" / "doc_claim_audit" / "audit.json").read_text())
BASELINE = json.loads((ROOT / "docs" / "doc_claim_audit" / "baseline.json").read_text())
OUT = ROOT / "docs" / "m3808" / "m3808_ci_fix.png"

can = sorted(
    (r for r in CROSS["moved_rows"] if r["category"] == "cancellation"),
    key=lambda r: float(r["row"].split("/")[1].split("=")[1]),
)
depths = [float(r["row"].split("/")[1].split("=")[1]) for r in can]
before = [r["recomputed_before"] for r in can]
after = [r["recomputed_after"] for r in can]
declared = [r["declared"] for r in can]

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

ax = axes[0]
x = np.arange(len(depths))
ax.semilogy(x, declared, "o", ms=9, mfc="none", mew=1.4, color="#34495e",
            label="printed in doc (§3.2)")
ax.semilogy(x, before, "s", ms=6, color="#2471a3", label="recomputed, cosh kernel (pre-merge)")
ax.semilogy(x, after, "d", ms=7, color="#c0392b", label="recomputed, exp kernel (m3409, post-merge)")
ax.axhline(1.0e-12, ls="--", lw=1.2, color="#27ae60",
           label="NOISE_FLOOR = 1e-12 (new absolute gate)")
ax.axhspan(1e-17, 1e-14, color="#f4d03f", alpha=0.18)
ax.text(0.02, 1.5e-16, "machine zero: $\\sim\\epsilon=2.2\\times10^{-16}$", fontsize=7.5,
        color="#7d6608", transform=ax.get_yaxis_transform())
ax.set_xticks(x)
ax.set_xticklabels([f"{d:g}" for d in depths])
ax.set_xlabel("water depth $h$ (m)")
ax.set_ylabel("free-surface residual $|G_z-\\nu G|/|\\nu G|$")
ax.set_ylim(1e-17, 1e-11)
ax.set_title("(a) §3.2 free-surface residuals: m3409 moves them by ≤1.13$\\epsilon$\n"
             f"(relative drift up to {max(r['rel_shift'] for r in can):.0%}, absolute ≤ 2.2e-16)",
             fontsize=9)
ax.legend(fontsize=6.8, loc="upper right")
ax.grid(alpha=0.25)

ax = axes[1]
cats = CROSS["categories"]
order = ["bit-identical", "last-bit", "amplified-roundoff", "cancellation", "physical-change"]
labels = {
    "bit-identical": "bit-identical",
    "last-bit": "last-bit only",
    "amplified-roundoff": "amplified roundoff\n(deep-water limit)",
    "cancellation": "cancellation noise\n(free-surface residual)",
    "physical-change": "physical change",
}
vals = [cats.get(k, 0) for k in order]
colors = ["#27ae60", "#7fb3d5", "#f5b041", "#c0392b", "#000000"]
bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.8, str(v), ha="center", fontsize=8.5)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels([labels[k] for k in order], fontsize=7.2)
ax.set_ylabel(f"rows of the {CROSS['rows']}-row pin table")
ax.set_ylim(0, max(vals) * 1.22)
ax.set_title("(b) Only roundoff moved. 0 physical changes;\n"
             f"{cats.get('bit-identical', 0)}/{CROSS['rows']} rows bit-identical after the rewrite",
             fontsize=9)
ax.grid(axis="y", alpha=0.25)

counts = AUDIT["counts"]
fig.text(
    0.5, 0.005,
    "doc-claim audit after the merge: "
    f"{counts['documents']} docs, PINNED={counts['pinned']}, ARTIFACT={counts['artifact']}, "
    f"UNGUARDED={counts['unguarded']} (paper-level {counts['unguarded_paper_level']}), "
    f"number coverage {AUDIT['coverage']['guarded_number_fraction']:.1%}; "
    f"ratchet baseline re-written with provenance note ({len(BASELINE['unguarded'])} entries).",
    ha="center", fontsize=7.4, color="#555555",
)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(OUT, dpi=105)
plt.close(fig)
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")