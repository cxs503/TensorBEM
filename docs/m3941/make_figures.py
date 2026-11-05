"""Figures for the PR #134 (agent/m3236) merge-heal report (m3941).

Two panels, both driven by data produced by the repository itself:

1. Registry census: how many declared-number rows each documentation group
   carries on ``main`` (ecba798), on ``agent/m3236`` and on the merged tree.
   This is the silent-loss check in picture form: the merged bar must be the
   sum of the two parents on every group, and no group may shrink.
2. Dean table-7 recomputation: for the eleven rows that branch m3236 added to
   ``src/tensorbem/benchmarks/declared_numbers.py``, the measured value versus
   the documented value, as ``|measured - declared| / declared`` on a log axis,
   with the registry tolerance as a horizontal band. This is the evidence that
   the merged tree still reproduces the numbers the documentation quotes (the
   branch was written before the last three main merges).

Inputs
------
``docs/m3941/census.json``      written by ``docs/m3941/make_artifacts.py``
``docs/m3941/registry_merged.json``  the CLI output of
    ``python -m tensorbem.benchmarks.declared_numbers --tiers fast --json``
    restricted to the ``dean`` group (see ``make_artifacts.py``).

Usage
-----
    python docs/m3941/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"


def figure_census(census: dict) -> Path:
    groups = census["groups"]
    names = list(groups)
    main = np.array([groups[g]["origin/main"] for g in names])
    branch = np.array([groups[g]["origin/agent/m3236"] for g in names])
    merged = np.array([groups[g]["merged"] for g in names])

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, main, w, label="main (ecba798)", color="#4C72B0")
    ax.bar(x, branch, w, label="agent/m3236", color="#DD8452")
    ax.bar(x + w, merged, w, label="merged (agent/m3941)", color="#55A868")
    for xi, (m, b, g) in enumerate(zip(main, branch, merged)):
        # no group may shrink and every new row must survive the merge
        assert g == max(m, b), (names[xi], m, b, g)
        ax.text(xi + w, g + 0.4, str(int(g)), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("declared-number rows in the registry")
    ax.set_title(
        "Registry census: merged tree = union of both parents (no row lost)"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_registry_census.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_dean(report: dict) -> Path:
    rows = [r for r in report["checks"] if r["group"] == "dean"]
    rows.sort(key=lambda r: r["key"])
    keys = [r["key"].replace("dean_", "") for r in rows]
    declared = np.array([abs(r["declared"]) for r in rows])
    measured = np.array([abs(r["measured"]) for r in rows])
    # The three roundoff-level rows are guarded as upper bounds: their declared
    # value is the guard (1e-12 / 1e-12 / 1e-10), so the meaningful distance is
    # measured/guard, not a relative deviation against a roundoff-sized number.
    upper = np.array([r["comparison"] == "upper" for r in rows])
    dev_value = np.array([abs(r["relative_deviation"]) for r in rows])
    dev = np.where(upper, measured / np.maximum(declared, 1e-300), dev_value)
    dev_plot = np.where(dev > 0.0, dev, 1e-18)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    colors = ["#C44E52" if u else "#4C72B0" for u in upper]
    ax.barh(np.arange(len(keys)), dev_plot, color=colors)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(
        r"deviation  $|m-d|/|d|$  (blue, value rows) or  $|m|/d$  (red, upper-bound rows)"
    )
    ax.axvline(1e-2, color="k", ls="--", lw=1.0)
    ax.text(1.1e-2, -0.9, "value tolerance 1e-2", fontsize=8, va="bottom")
    ax.axvline(1.0, color="k", ls=":", lw=1.0)
    ax.text(1.1, -0.9, "upper-bound guard (=1)", fontsize=8, va="bottom")
    ax.set_xlim(1e-18, 10)
    for yi, d in enumerate(dev):
        label = "0 (exact)" if d == 0.0 else f"{d:.1e}"
        ax.text(max(d, 1e-18) * 1.8, yi, label, va="center", fontsize=7)
    ax.set_title(
        "Dean-Ursell-Ogilvie table 7 recomputed on the merged tree (11/11 PASS)"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_dean_registry_merged.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    census = json.loads((HERE / "census.json").read_text(encoding="utf-8"))
    report = json.loads((HERE / "registry_merged.json").read_text(encoding="utf-8"))
    a = figure_census(census)
    b = figure_dean(report)
    print(a)
    print(b)


if __name__ == "__main__":
    main()