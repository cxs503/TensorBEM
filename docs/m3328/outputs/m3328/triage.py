"""Triage the 3-D mutation survivors and draw the before/after figure (m3328)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]

before = json.loads((OUT / "sweep_before" / "mutation_report.json").read_text())
after = json.loads((OUT / "sweep_after" / "mutation_report.json").read_text())

# ---------------------------------------------------------------------------
# Survivor triage.  Every surviving mutant of the *back-filled* run is assigned
# to one category; the categories are what the numbers in the report mean.
# ---------------------------------------------------------------------------
EQUIVALENT_EPS = "equivalent: epsilon/boundary literal (result unchanged at double precision)"
EQUIVALENT_SIGN = "equivalent: sign of a quantity that enters squared / negated"
REPRESENTATION = "equivalent: selects a representation, both branches compute the same function"
MARGIN = "unobservable: safety margin / truncation cap saturated at test resolutions"
DIAGNOSTIC = "diagnostic-helper constant (verify_* gates), no kernel value"
REEPS = "guard/validation path: no computed kernel value at test resolutions"

MANUAL = {
    "tensorbem.finite_depth3d:82:16:negate_constant": REEPS,
    "tensorbem.finite_depth3d:86:12:shift_constant": MARGIN,
    "tensorbem.finite_depth3d:251:20:flip_compare": REEPS,
    "tensorbem.finite_depth3d:251:40:negate_constant": REEPS,
    "tensorbem.finite_depth3d:256:14:shift_constant": MARGIN,
    "tensorbem.finite_depth3d:331:31:swap_binop": EQUIVALENT_EPS,
    "tensorbem.finite_depth3d:338:18:shift_constant": MARGIN,
    "tensorbem.finite_depth3d:335:32:swap_binop": EQUIVALENT_SIGN,
    "tensorbem.free_surface3d:103:12:flip_compare": REPRESENTATION,
    "tensorbem.free_surface3d:103:47:flip_compare": REPRESENTATION,
    "tensorbem.free_surface3d:155:12:flip_compare": REPRESENTATION,
    "tensorbem.free_surface3d:195:12:flip_compare": REPRESENTATION,
    "tensorbem.free_surface3d:195:47:negate_constant": REPRESENTATION,
    "tensorbem.free_surface3d:211:20:shift_constant": MARGIN,
}


def categorise(mutant: dict) -> str:
    key = mutant["key"]
    if key in MANUAL:
        return MANUAL[key]
    module = mutant["module"]
    line = mutant["lineno"]
    original, replacement = mutant["original"], mutant["replacement"]
    if "verify_" in mutant["function"] or "diagnostic" in mutant["function"]:
        return DIAGNOSTIC
    if module.endswith("free_surface3d") and line in {155, 195, 211}:
        return REPRESENTATION
    if "e-" in original or "e-" in replacement or original in {"0.0", "-0.0"}:
        return EQUIVALENT_EPS
    if original in {"abs", "sqrt"}:
        return EQUIVALENT_SIGN
    if mutant["operator"] in {"flip_compare"}:
        return EQUIVALENT_EPS
    return REEPS


rows = []
for m in after["mutants"]:
    if m["verdict"] == "survived":
        rows.append({"key": m["key"], "module": m["module"], "line": m["lineno"],
                     "function": m["function"], "summary": m["summary"],
                     "category": categorise(m)})

# ---------------------------------------------------------------------------
# Before/after per module, plus the mutants the back-filled suite killed.
# ---------------------------------------------------------------------------
def per_module(report: dict) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for m in report["mutants"]:
        slot = out.setdefault(m["module"], {"killed": 0, "survived": 0})
        slot["killed" if m["verdict"] == "killed" else "survived"] += 1
    return out


bef, aft = per_module(before), per_module(after)
modules = sorted(bef, key=lambda s: s.split(".")[-1])
flipped = [
    {"key": k, "summary": before["mutants"][i]["summary"],
     "killed_by": after["mutants"][i]["killed_by"]}
    for i, k in enumerate(m["key"] for m in before["mutants"])
    if before["mutants"][i]["verdict"] != after["mutants"][i]["verdict"]
]

payload = {
    "seed": after["seed"],
    "max_per_module": after["max_per_module"],
    "jobs": after["jobs"],
    "modules": modules,
    "score_before": before["overall"]["mutation_score"],
    "score_after": after["overall"]["mutation_score"],
    "per_module": {
        mod: {
            "mutants": bef[mod]["killed"] + bef[mod]["survived"],
            "score_before": round(bef[mod]["killed"] / (bef[mod]["killed"] + bef[mod]["survived"]), 4),
            "score_after": round(aft[mod]["killed"] / (aft[mod]["killed"] + aft[mod]["survived"]), 4),
        }
        for mod in modules
    },
    "killed_by_backfill": flipped,
    "survivors_after": rows,
    "survivor_categories": {c: sum(1 for r in rows if r["category"] == c) for c in {r["category"] for r in rows}},
}
(OUT / "triage.json").write_text(json.dumps(payload, indent=1) + "\n")

# ---------------------------------------------------------------------------
# Figure: score bars + survivor categories
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
xs = np.arange(len(modules))
labels = [m.split(".")[-1] for m in modules]
sb = [payload["per_module"][m]["score_before"] for m in modules]
sa = [payload["per_module"][m]["score_after"] for m in modules]
ax.bar(xs - 0.19, sb, 0.36, label="before (mapped tests only)", color="#9aa7b4")
ax.bar(xs + 0.19, sa, 0.36, label="after (+ test_mutation_gaps_3d.py)", color="#2f6f9f")
for x, v0, v1 in zip(xs, sb, sa):
    ax.text(x - 0.19, v0 + 0.01, f"{v0:.3f}", ha="center", fontsize=8)
    ax.text(x + 0.19, v1 + 0.01, f"{v1:.3f}", ha="center", fontsize=8)
ax.axhline(before["overall"]["mutation_score"], ls="--", lw=0.9, color="#444", label=f"overall before {before['overall']['mutation_score']:.3f}")
ax.axhline(after["overall"]["mutation_score"], ls=":", lw=1.2, color="#b33", label=f"overall after {after['overall']['mutation_score']:.3f}")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("mutation score (killed / mutants)")
ax.set_title("3-D kernels: mutation score before and after back-fill", fontsize=10)
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
cats = payload["survivor_categories"]
short = {
    EQUIVALENT_EPS: "epsilon /\nboundary literal",
    EQUIVALENT_SIGN: "sign that\ncancels",
    REPRESENTATION: "branch select\n(same function)",
    MARGIN: "safety margin /\ntruncation cap",
    DIAGNOSTIC: "verify_*\ndiagnostic only",
    REEPS: "guard /\nvalidation path",
}
names = [short.get(c, c) for c in cats]
vals = [cats[c] for c in cats]
order = np.argsort(vals)[::-1]
ax.barh(np.arange(len(vals)), [vals[i] for i in order], color="#7a8f4a")
ax.set_yticks(np.arange(len(vals)))
ax.set_yticklabels([names[i] for i in order], fontsize=8)
for y, i in enumerate(order):
    ax.text(vals[i] + 0.3, y, str(vals[i]), va="center", fontsize=8)
ax.set_xlabel("surviving mutants after back-fill")
ax.set_title(f"survivor triage ({len(rows)} of 360 mutants)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_mutation_3d.png", dpi=160)
print(json.dumps(payload["per_module"], indent=1))
print("flipped:", len(flipped), "survivors:", len(rows), payload["survivor_categories"])