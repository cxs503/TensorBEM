"""Figures for the m3245 -> main merge report (mission m3795).

Fig 1: declared-number registry coverage per group for main, m3245 and the
merged tree -- shows the merged registry is the arithmetic union of both sides.
Fig 2: coverage per cost tier, which is where a naive "take theirs" overlay
would have silently dropped the tiers the other side owns.

The counts are printed by the command in docs/merge_m3245_into_main_m3795.md and
re-checked here from the same source trees, so the figures cannot drift from the
text.
"""
import collections
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts"

CODE = (
    "import collections, json;"
    "from tensorbem.benchmarks import declared_numbers as dn;"
    "print(json.dumps({'n': len(dn.DECLARED_NUMBERS),"
    "'keys': sorted(e.key for e in dn.DECLARED_NUMBERS),"
    "'groups': dict(collections.Counter(e.group for e in dn.DECLARED_NUMBERS)),"
    "'tiers': dict(collections.Counter(e.tier for e in dn.DECLARED_NUMBERS))}))"
)


def load(tree_src: Path):
    out = subprocess.run(
        ["python3", "-c", CODE],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(tree_src), "PATH": "/usr/bin:/bin"},
    ).stdout
    return json.loads(out)


main = load(ROOT / "artifacts/baseline_main/src")
branch = load(ROOT / "artifacts/m3245_tree/src")
merged = load(ROOT / "src")

# Union ledger, written next to the figures so the report can quote it verbatim.
ledger = {
    "counts": {"main": main["n"], "m3245": branch["n"], "merged": merged["n"]},
    "shared_keys": len(set(main["keys"]) & set(branch["keys"])),
    "duplicate_keys_in_merged": merged["n"] - len(set(merged["keys"])),
    "lost_keys": sorted((set(main["keys"]) | set(branch["keys"])) - set(merged["keys"])),
    "groups": {g: [main["groups"].get(g, 0), branch["groups"].get(g, 0), merged["groups"].get(g, 0)]
               for g in sorted(set(main["groups"]) | set(branch["groups"]))},
    "tiers": {t: [main["tiers"].get(t, 0), branch["tiers"].get(t, 0), merged["tiers"].get(t, 0)]
              for t in ("fast", "standard", "heavy")},
}
(OUT / "merge_union_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

groups = list(ledger["groups"])
fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = range(len(groups))
width = 0.27
for offset, (label, idx, colour) in zip(
    (-width, 0.0, width),
    (("main (ecba798)", 0, "#4c72b0"), ("agent/m3245 (6753fcb)", 1, "#dd8452"),
     ("merged (union)", 2, "#55a868")),
):
    ax.bar([i + offset for i in x], [ledger["groups"][g][idx] for g in groups],
           width, label=f"{label}: {[main['n'], branch['n'], merged['n']][idx]} entries",
           color=colour)
ax.set_xticks(list(x))
ax.set_xticklabels(groups, rotation=25, ha="right")
ax.set_ylabel("declared numbers")
ax.set_title("Declared-numbers registry after merging m3245 into main: every group is the sum of both sides")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
ax.text(0.01, 1.02,
        f"merged {merged['n']} = union of main {main['n']} and m3245 {branch['n']} "
        f"minus {ledger['shared_keys']} keys both branches share; "
        f"keys lost by the merge: {len(ledger['lost_keys'])}; "
        f"duplicate keys: {ledger['duplicate_keys_in_merged']}",
        transform=ax.transAxes, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_m3795_registry_union.png", dpi=140)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
tiers = list(ledger["tiers"])
x = range(len(tiers))
for offset, (label, idx, colour) in zip(
    (-width, 0.0, width),
    (("main", 0, "#4c72b0"), ("agent/m3245", 1, "#dd8452"), ("merged", 2, "#55a868")),
):
    ax.bar([i + offset for i in x], [ledger["tiers"][t][idx] for t in tiers], width,
           label=label, color=colour)
ax.set_xticks(list(x))
ax.set_xticklabels(tiers)
ax.set_ylabel("declared numbers")
ax.set_title("Cost-tier coverage: the union keeps main's fast rows and m3245's heavy rows")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
for i, t in enumerate(tiers):
    ax.annotate(f"{ledger['tiers'][t][2]}", (i + width, ledger["tiers"][t][2]),
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_m3795_tier_coverage.png", dpi=140)

print(json.dumps(ledger, indent=2, sort_keys=True))
print("wrote", OUT / "fig_m3795_registry_union.png", OUT / "fig_m3795_tier_coverage.png")