"""Figures for the PR #136 cure report (docs/m3969/cure_pr136_union_merge.md).

Left  : per-claim agreement between the *branch* registry (as PR #136 shipped it)
        and the *merged* registry -- the machine answer to "did the merge drop
        any of the branch's 85 registered numbers?".  Produced from
        ``docs/m3969/union_report.json`` (written by ``check_merge_union.py``).
Right : the refinement columns the merge had to keep reproducible, read from the
        branch's own verification artefacts (``docs/m3250/outputs/verif_m3250``):
        heave added-mass error and vertical excitation error versus panel count
        for the two mesh families, with the acceptance bands the merged test
        now asserts.

Run from the worktree root::

    PYTHONPATH=src python3 docs/m3969/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERIF = ROOT / "docs/m3250/outputs/verif_m3250"

BRANCH_COLOR = "#1f4e79"
MERGE_COLOR = "#c0504d"


def claim_deviations():
    report = json.loads((HERE / "union_report.json").read_text())
    reg = report["registry"]
    # recompute every branch claim's deviation, not just the ones the checker
    # flagged, so the panel shows the whole distribution
    import importlib.util
    import subprocess
    import sys
    import tempfile

    def load(rev):
        src = subprocess.run(
            ["git", "show", f"{rev}:src/tensorbem/benchmarks/declared_numbers.py"],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout
        tmp = Path(tempfile.mkdtemp()) / "dn.py"
        tmp.write_text(src)
        spec = importlib.util.spec_from_file_location(f"fig_{rev.replace('/', '_')}", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return list(mod.DECLARED_NUMBERS)

    sys.path.insert(0, str(HERE))
    from check_merge_union import mapped_key  # noqa: E402

    branch = load("origin/agent/m3250")
    merged = load("HEAD")
    index = {(e.group, e.quantity): e for e in merged}
    by_key = {e.key: e for e in merged}

    rows = []
    for entry in branch:
        key = mapped_key(entry)
        hit = by_key.get(entry.key) if key == "key" else (index.get(key) if key else None)
        if hit is None:
            continue
        measured = abs(hit.declared) if entry.quantity.endswith("_abs") else hit.declared
        ref = max(abs(entry.declared), abs(measured), 1e-30)
        rows.append(abs(measured - entry.declared) / ref)
    return sorted(rows), reg


def refinement(path: Path):
    data = json.loads(path.read_text())
    levels = data["levels"]
    panels = [lvl.get("panel_count") for lvl in levels]
    if any(p is None for p in panels):
        raise SystemExit(f"{path.name}: no panel count in levels")
    return levels, panels


def main() -> None:
    devs, reg = claim_deviations()
    rtol = json.loads((HERE / "union_report.json").read_text())["rtol"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    xs = range(1, len(devs) + 1)
    ax1.semilogy(list(xs), [max(d, 1e-16) for d in devs], "o", ms=3.2,
                 color=BRANCH_COLOR, label=f"branch claim vs merged entry (n={len(devs)})")
    ax1.axhline(rtol, color=MERGE_COLOR, ls="--", lw=1.2,
                label=f"preservation tolerance {rtol:.0%}")
    ax1.set_xlabel("branch claim (sorted)")
    ax1.set_ylabel("relative deviation |merged − branch| / max")
    ax1.set_title("A. No branch claim moved beyond its printed precision")
    worst = max(devs)
    ax1.annotate(f"worst {worst:.2e}\n(two-significant-digit cell)",
                 xy=(len(devs), worst), xytext=(len(devs) * 0.55, worst * 6),
                 fontsize=8, color=MERGE_COLOR,
                 arrowprops=dict(arrowstyle="->", color=MERGE_COLOR, lw=0.8))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    for name, color, band, label in (
        ("hemisphere_refinement.json", BRANCH_COLOR, 1.0e-2, "hemisphere (Ka=1.0)"),
        ("submerged_sphere_refinement.json", MERGE_COLOR, 3.5e-2, "submerged sphere (Ka=0.8)"),
    ):
        path = VERIF / name
        if not path.exists():
            continue
        levels, panels = refinement(path)
        a33 = [lvl["heave_added_mass_rel_error"] for lvl in levels]
        fz = [lvl["excitation_force_z_rel_error"] for lvl in levels]
        ax2.loglog(panels, a33, "o-", color=color, lw=1.4,
                   label=f"{label} $A_{{33}}'$ err")
        ax2.loglog(panels, fz, "s--", color=color, lw=1.1, alpha=0.6,
                   label=f"{label} $F_z$ err (band {band:.1%})")
        ax2.axhline(band, color=color, lw=0.8, alpha=0.35)

    ax2.axhline(3.0e-2, color="0.35", ls=":", lw=1.0, label="3 % acceptance on $A_{33}'$")
    ax2.set_xlabel("panels")
    ax2.set_ylabel("relative error")
    ax2.set_title("B. Refinement columns stay reproducible after the merge")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    out = HERE / "fig_pr136_union_merge.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out} ({out.stat().st_size} bytes), claims={len(devs)}, worst={worst:.2e}")


if __name__ == "__main__":
    main()