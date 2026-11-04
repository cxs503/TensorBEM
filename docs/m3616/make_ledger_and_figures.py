"""Regenerate the m3616 ledger and its two figures (zero hand-copied numbers)."""
from __future__ import annotations
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from tensorbem.benchmarks import lifting_doc_pins as pins

OUT = pathlib.Path(__file__).resolve().parent
text = pins.DOC.read_text(encoding="utf-8")
declared = pins.parse_document(text)
measured = pins.measure(declared)
verdicts = pins.compare(declared, measured)
ledger = pins.summary(verdicts)
ledger["declared_ladder"] = {
    str(n): {c: declared[f"refinement.{n}.{c}"].value for c in pins.REFINEMENT_COLUMNS}
    for n in pins.PANEL_COUNTS
}
(OUT / "lifting_doc_pins_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

panels = np.array(pins.PANEL_COUNTS, dtype=float)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
ax = axes[0]
for column, label, style in (
    ("potential_rms_error", r"rms $|\delta\phi|$", "o-"),
    ("relative_potential_error_percent", r"max $|\delta\phi|/\max|\phi_d|$  [%]", "s-"),
):
    m = np.array([measured[f"refinement.{n}.{c}"].value for n in pins.PANEL_COUNTS
                  for c in (column,)])
    for n, value in zip(pins.PANEL_COUNTS, m):
        ax.plot([n], [declared[f"refinement.{n}.{column}"].value], marker="x",
                color="k", ms=7, ls="none")
    ax.loglog(panels, m, style, label=label + " (measured)", lw=1.6)
ax.set_xlabel("panels N")
ax.set_ylabel("error")
ax.set_title("docs/lifting_foil.md §3.2 ladder\n(line = recomputed, × = document)", fontsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

ax = axes[1]
rows = [v for v in verdicts]
ratio = np.array([v.deviation / v.tolerance if v.tolerance else 0.0 for v in rows])
order = np.argsort(-ratio)
ax.bar(np.arange(len(rows)), ratio[order], color="#3b7dd8")
ax.axhline(1.0, color="crimson", lw=1.2, ls="--", label="printed-precision tolerance")
ax.set_yscale("log")
ax.set_xlabel(f"{len(rows)} documented claims (sorted)")
ax.set_ylabel(r"$|$measured − declared$|$ / tolerance")
ax.set_title("all claims inside the printed precision\n(bar ≥ 1 would be a red build)", fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "figures" / "lifting_doc_pins.png")
print(json.dumps({k: ledger[k] for k in ("verdict", "claims", "passed", "drifted")}))
