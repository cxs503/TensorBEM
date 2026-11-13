"""Figure for m17090: the reader graph and adjudication verdicts behind the debt table.

Every number comes from the tree: the verdict factors are read from
``benchmarks/verified/wigley3_recalled_ref/results.json`` and the reader counts
from ``tensorbem.reference_provenance.audit()``.  Nothing is typed in by hand.

Run: python make_report_figure_m17090.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorbem import reference_provenance as rp  # noqa: E402

ARTEFACT = ROOT / rp.ADJUDICATION_ARTEFACT
OUT = ROOT / "docs" / "fig_m17090_reference_provenance_readers.png"

BAND = 0.10  # the adjudicator's own tolerance, read back from the artefact below
COLOURS = {"REFUTED": "#c0392b", "FREQUENCY_MISMATCH": "#e67e22", "REPRODUCED": "#27ae60"}


def main() -> int:
    payload = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    band = float(payload.get("metadata", {}).get("band", BAND))
    records = [item for item in payload["adjudication"] if isinstance(item, dict)]

    result = rp.audit(ROOT)
    totals = result.summary()

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ---- left: computed/recalled factor per adjudicated record -------------
    order: list[str] = []
    rows: list[tuple[int, float, str, str]] = []
    for item in records:
        key = item["constant"]
        short = key.rsplit(".", 1)[-1]
        if short not in order:
            order.append(short)
        rows.append((order.index(short), float(item["factor"]), str(item["verdict"]), str(item.get("dof", ""))))

    # records with a vanishing factor cannot sit on a log axis; clip and mark
    clipped = [(y, max(x, 1e-6), verdict, dof) for y, x, verdict, dof in rows]
    for index, (y, x, verdict, dof) in enumerate(clipped):
        left.scatter(
            x,
            y + 0.06 * (index % 5 - 2),
            s=42,
            color=COLOURS.get(verdict, "#34495e"),
            marker="o" if verdict != "REPRODUCED" else "D",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    left.axvspan(1.0 - band, 1.0 + band, color="#bdc3c7", alpha=0.6, zorder=0, label=f"±{band:.0%} band")
    left.axvline(1.0, color="black", lw=1.0, ls="--", zorder=1)
    left.set_xscale("log")
    left.set_yticks(range(len(order)))
    left.set_yticklabels(order, fontsize=8)
    left.set_xlabel("computed / recalled  (TensorBEM and capytaine, finest mesh)")
    left.set_title("Recalled constants measured against two independent codes\n"
                   f"({len(records)} records, {len(order)} constants, values < 1e-6 clipped)", fontsize=10)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=colour, markeredgecolor="black", label=verdict)
        for verdict, colour in COLOURS.items()
        if any(v == verdict for _, _, v, _ in clipped)
    ]
    handles.append(plt.Line2D([], [], color="#bdc3c7", lw=8, label=f"±{band:.0%} band"))
    left.legend(handles=handles, fontsize=8, loc="lower right")
    left.grid(axis="x", which="both", alpha=0.25)

    # ---- right: what the gate now measures ---------------------------------
    adjudicated = totals["adjudicated"]
    unadjudicated = totals["unsourced"] - adjudicated
    labels = [
        "constants scanned",
        "CITED (published\ndigits resolve)",
        "UNSOURCED (debt)",
        "  -- adjudicated\nverdict on record",
        "  -- no verdict yet\n(no reader)",
        "cross-module readers",
        "UNSOURCED readers\noutside adjudicator",
    ]
    values = [
        totals["declarations"],
        totals["cited"],
        totals["unsourced"],
        adjudicated,
        unadjudicated,
        totals["readers"],
        totals["leaked"],
    ]
    colours = ["#2c3e50", "#27ae60", "#c0392b", "#e67e22", "#95a5a6", "#2980b9", "#8e44ad"]
    bars = right.barh(range(len(values)), values, color=colours, edgecolor="black", linewidth=0.4)
    for bar, value in zip(bars, values):
        right.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, str(value),
                   va="center", fontsize=9)
    right.set_yticks(range(len(labels)))
    right.set_yticklabels(labels, fontsize=8)
    right.invert_yaxis()
    right.set_xlabel("count")
    right.set_title(
        "Fail-closed surface after the reader gate\n"
        f"undeclared {totals['undeclared']}, stale {totals['stale']}, "
        f"unverified {totals['unverified']}, leaks {totals['leaked']}",
        fontsize=10,
    )
    right.grid(axis="x", alpha=0.25)
    right.set_xlim(0, max(values) * 1.25)

    fig.suptitle(
        "m17090  TensorBEM reference provenance: recalled numbers stay debt, and debt stays unread",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    print("totals", totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
