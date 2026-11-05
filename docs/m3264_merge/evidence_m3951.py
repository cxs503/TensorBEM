"""Recompute the merged registry and plot declared vs measured (m3951, #135).

Evidence for the merge report: after resolving PR #135, every quoted number of
the merged ``declared_numbers`` registry is recomputed at its documented setting
and drawn against the value the document quotes. A merge that dropped a row, or
silently swapped a declaration for its neighbour's, shows up as a point off the
diagonal (the registry judges at ``DEFAULT_TOLERANCE = 1.5e-2``).

Outputs (both inside ``docs/m3264_merge/``):

``declared_numbers_recompute_m3951.json``
    The strict-JSON report of :func:`tensorbem.benchmarks.declared_numbers.check`
    on the ``standard`` tier, i.e. tables 10-12 and the two-dimensional rows.
``fig_m3951_parity.png``
    Left: measured against declared on log-log axes, equal-value rows only.
    Right: relative deviation of every row, sorted, with the 1.5 % band.

Run from the repository root::

    PYTHONPATH=src python3 docs/m3264_merge/evidence_m3951.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks.declared_numbers import DEFAULT_TOLERANCE, check

HERE = Path(__file__).resolve().parent


def main() -> int:
    report = check(tiers=("standard",))
    (HERE / "declared_numbers_recompute_m3951.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    rows = report["checks"]
    values = [r for r in rows if r["comparison"] == "value" and r["declared"] != 0.0]
    declared = np.array([r["declared"] for r in values], dtype=float)
    measured = np.array([r["measured"] for r in values], dtype=float)
    deviation = np.array(
        sorted(abs(r["relative_deviation"]) for r in values), dtype=float
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=140)

    positive = (declared > 0) & (measured > 0)
    groups = sorted({r["group"] for r, keep in zip(values, positive) if keep})
    colours = {name: colour for name, colour in zip(groups, plt.cm.tab10.colors)}
    for name in groups:
        sel = np.array([r["group"] == name for r, keep in zip(values, positive)]) & positive
        ax0.scatter(declared[sel], measured[sel], s=18, alpha=0.85, label=name,
                    color=colours[name], edgecolor="none")
    lims = [min(declared[positive].min(), measured[positive].min()) / 3.0,
            max(declared[positive].max(), measured[positive].max()) * 3.0]
    ax0.plot(lims, lims, "k-", lw=1.0, label="measured = declared")
    ax0.fill_between(lims, [v * (1 - DEFAULT_TOLERANCE) for v in lims],
                     [v * (1 + DEFAULT_TOLERANCE) for v in lims],
                     color="0.6", alpha=0.25, label=f"±{DEFAULT_TOLERANCE:.1%} band")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lims)
    ax0.set_ylim(lims)
    ax0.set_xlabel("value quoted by the documentation")
    ax0.set_ylabel("value recomputed from the merged registry")
    ax0.set_title(f"{len(values)} equality rows, {report['summary']['passed']} pass")
    ax0.legend(fontsize=6, loc="upper left", frameon=False)

    ax1.semilogy(np.arange(1, len(deviation) + 1), deviation, "o", ms=3, color="C0")
    ax1.axhline(DEFAULT_TOLERANCE, color="C3", ls="--", lw=1.0,
                label=f"tolerance {DEFAULT_TOLERANCE:.1%}")
    ax1.set_xlabel("row, sorted by |relative deviation|")
    ax1.set_ylabel("|measured − declared| / |declared|")
    ax1.set_title(
        f"worst {report['summary']['worst_value_relative_deviation']:.2e}, "
        f"drifted {report['summary']['drifted']}"
    )
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "fig_m3951_parity.png")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["drifted"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())