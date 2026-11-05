"""Build the artifacts for the mesh-table guard (m3449) and inject the doc section.

Everything that lands in ``docs/three_dimensional_bem.md`` is rendered by
``tensorbem.benchmarks.mesh_doc_pins.render_section`` from the check report --
no number is hand-copied, and ``tests/test_mesh_doc_pins.py`` asserts that the
section in the document is byte-identical to a fresh render.

Run with::

    PYTHONPATH=src python3 outputs/verif_m3449/build_artifacts.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks import mesh_doc_pins as pins

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DOC = REPO / "docs" / "three_dimensional_bem.md"
SECTION_ANCHOR = "\n### 3.2 闭合性与方向校验"


def build_json() -> dict:
    report = pins.check(DOC.read_text())
    (HERE / "mesh_table_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def inject(section: str) -> None:
    text = DOC.read_text()
    block = re.compile(
        re.escape(pins.BEGIN_MARKER) + r".*?" + re.escape(pins.END_MARKER), re.DOTALL
    )
    if block.search(text):
        # Idempotent: replacing the generated block reproduces it byte for byte.
        text = block.sub(lambda _: section, text, count=1)
    else:
        if SECTION_ANCHOR not in text:
            raise SystemExit("section 3.2 anchor not found")
        text = text.replace(SECTION_ANCHOR, "\n" + section + "\n" + SECTION_ANCHOR, 1)
    DOC.write_text(text)


def figure(report: dict) -> Path:
    measured = pins.measure()
    panels = sorted(measured)
    labels = [str(int(measured[k]["panels"])) for k in panels]
    x = np.arange(len(panels))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    area_ratio = [float(measured[k]["area_ratio"]) for k in panels]
    volume_ratio = [float(measured[k]["volume_ratio"]) for k in panels]
    ax.axhline(1.0, color="k", lw=1.0, ls=":", label="exact sphere (= 1.0)")
    ax.plot(x, area_ratio, "o-", label=r"area ratio $A/4\pi a^2$")
    ax.plot(x, volume_ratio, "s--", label=r"volume ratio $V/(4\pi a^3/3)$")
    for xi, value in zip(x, area_ratio):
        ax.annotate(f"{value:.5f}", (xi, value), textcoords="offset points", xytext=(0, -14), fontsize=8)
    for xi, value in zip(x, volume_ratio):
        ax.annotate(f"{value:.5f}", (xi, value), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("panels $N$")
    ax.set_ylabel("ratio to the sphere")
    ax.set_title("(a) inscribed polyhedron approaches the sphere from below")
    ax.set_ylim(0.5, 1.06)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for name in sorted({cell["quantity"] for cell in report["checks"] if cell["kind"] == "value"}):
        subset = [
            cell for cell in report["checks"] if cell["kind"] == "value" and cell["quantity"] == name
        ]
        xs = [int(round(math.log(int(cell["panels"]) / 20, 4))) for cell in subset]
        ys = [
            abs(cell["measured"] - cell["declared"]) / max(abs(cell["declared"]), abs(cell["measured"]))
            for cell in subset
        ]
        ax.semilogy(xs, ys, "o-", label=name, ms=4)
    ax.axhline(1.0, color="r", lw=1.0, ls="--", label="drift threshold (printed precision)")
    ax.set_xticks([0, 1, 2, 3], ["20", "80", "320", "1280"])
    ax.set_xlabel("panels $N$")
    ax.set_ylabel(r"$|\mathrm{declared}-\mathrm{measured}|/\mathrm{measured}$")
    ax.set_title("(b) every printed cell lands inside its own precision")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("TensorBEM: executable contract for Table 1 (icosphere mesh, a = 1)", fontsize=11)
    fig.tight_layout()
    path = HERE / "fig_mesh_table_guard.png"
    fig.savefig(path, dpi=85)
    plt.close(fig)
    return path


def main() -> int:
    report = build_json()
    print("verdict:", report["verdict"], "cells:", report["n_cells"], "failed:", report["n_failed"])
    inject(pins.render_section(report))
    print("injected:", DOC)
    print("figure:", figure(report))
    # Leave a machine-readable trace for the report step and fail loudly on drift.
    (HERE / "summary.txt").write_text(
        f"verdict={report['verdict']} cells={report['n_cells']} failed={report['n_failed']}\n"
    )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())