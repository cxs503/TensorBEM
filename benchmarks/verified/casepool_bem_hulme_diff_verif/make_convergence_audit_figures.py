"""Figures for the third-party-route convergence audit (no BEM, reads the JSON).

Outputs (in this directory)::

    convergence_audit_ladders.png   error vs panel count, both mesh families
    convergence_audit_richardson.png  finest / corrected / pre-fix, 12 cells

Run::

    python benchmarks/verified/casepool_bem_hulme_diff_verif/make_convergence_audit_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "capytaine_convergence_audit.json"


def figure_ladders(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {"uv_family": ("o", "capytaine lat-lon sphere"), "octa_family": ("s", "repo octahedral mesh")}
    for family, (marker, label) in styles.items():
        block = data[family]
        for ka in (0.5, 1.0, 2.0):
            for dof, colour in (("surge", "tab:blue"), ("heave", "tab:red")):
                cell = block["cells"][f"{ka:.3f}/{dof}"]
                panels = [level["panels"] for level in cell["levels"]]
                errors = [level["relative_error"] for level in cell["levels"]]
                ax.loglog(
                    panels,
                    errors,
                    marker=marker,
                    color=colour,
                    alpha=0.75,
                    linestyle="-" if family == "octa_family" else "--",
                    label=f"{label}, ka={ka:g} {dof}",
                )
    panels = [256, 4096]
    ax.loglog(panels, [0.09 * math.sqrt(256 / p) for p in panels], "k:", lw=1.0,
              label=r"slope $h^1$")
    ax.set_xlabel("panels $N$ ( $h \\propto N^{-1/2}$ )")
    ax.set_ylabel(r"$|F - F_{\rm Hulme}| / |F_{\rm Hulme}|$")
    ax.set_title("Third-party route vs the Hulme series: error decays like $h$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    out = HERE / "convergence_audit_ladders.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_richardson(data: dict) -> Path:
    frozen = data["frozen_ladder_reanalysis"]
    names = sorted(frozen["cells"])
    finest = [frozen["cells"][n]["finest_relative_error"] * 100 for n in names]
    fixed = [frozen["cells"][n]["richardson_p2_relative_error"] * 100 for n in names]
    prefix = [frozen["cells"][n]["prefix_inverted_ratio_relative_error"] * 100 for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    w = 0.27
    ax.bar([i - w for i in x], finest, w, label="finest mesh (1024 panels)")
    ax.bar(list(x), fixed, w, label="Richardson, corrected ratio ($p=2$)")
    ax.bar([i + w for i in x], prefix, w, label="as published (inverted ratio)")
    ax.axhline(5.0, color="k", ls="--", lw=1.0, label="ledger gate 5 %")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("/", "\n") for n in names], fontsize=7)
    ax.set_ylabel("relative error  [%]")
    ax.set_title("Third-party agreement with the reference: three readings of one ladder")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    out = HERE / "convergence_audit_richardson.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    data = json.loads(AUDIT.read_text())
    for path in (figure_ladders(data), figure_richardson(data)):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())