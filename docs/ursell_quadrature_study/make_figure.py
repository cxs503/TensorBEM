"""Figure for the Ursell quadrature study: node sensitivity and panel convergence.

Reads ``summary.json`` (built by ``build_summary.py`` from the raw benchmark
runs) and writes ``docs/figures/ursell_quadrature_study.png``.

Run: python docs/ursell_quadrature_study/make_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_FIGURE = _REPO / "docs" / "figures" / "ursell_quadrature_study.png"


def main() -> None:
    summary = json.loads((_HERE / "summary.json").read_text(encoding="utf-8"))
    reference = summary["reference"]["added_mass"]

    current = summary["node_sweep_current_panels128"]
    legacy = summary["node_sweep_legacy_panels128"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    left.plot(
        [row["panel_nodes"] for row in current],
        [row["panel_added_mass"] for row in current],
        "o-",
        label="current (analytic log primitive)",
    )
    left.plot(
        [row["panel_nodes"] for row in legacy],
        [row["panel_added_mass"] for row in legacy],
        "s--",
        label="legacy 878834a (singular quadrature)",
    )
    left.axhline(reference, color="k", linewidth=1.0, linestyle=":", label="Ursell multipole")
    left.set_xscale("log", base=2)
    left.set_xlabel("Gauss-Legendre nodes per panel (128 panels)")
    left.set_ylabel(r"added mass $A$ per unit span  [kg/m]")
    left.set_title("Quadrature is the difference, not the discretisation")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    panels = [row["panels"] for row in summary["panel_sweep_current_nodes24"]]
    errors = [row["panel_added_mass_relative_error"] for row in summary["panel_sweep_current_nodes24"]]
    damping = [row["panel_damping_relative_error"] for row in summary["panel_sweep_current_nodes24"]]
    right.loglog(panels, errors, "o-", label=r"added mass $A$")
    right.loglog(panels, damping, "s-", label=r"damping $B$")
    first_order = [errors[0] * (panels[0] / value) for value in panels]
    right.loglog(panels, first_order, "k:", linewidth=1.0, label=r"slope $-1$")
    right.set_xlabel("panels")
    right.set_ylabel("relative error vs Ursell")
    right.set_title("Remaining error is first order in panel size")
    right.legend(fontsize=8)
    right.grid(alpha=0.3, which="both")

    figure.tight_layout()
    _FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(_FIGURE, dpi=160)
    print(f"wrote {_FIGURE}")


if __name__ == "__main__":
    main()