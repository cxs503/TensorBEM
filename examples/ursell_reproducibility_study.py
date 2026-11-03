"""Reproducibility study for the Ursell submerged-cylinder benchmark numbers.

Context
-------
The marine benchmark report quoted, for the Ursell radiation case
(``a = 1``, ``f = 2``, ``ka = 0.6``, ``rho = 1025``, ``g = 9.80665``),
a 128-panel added-mass relative error of 7.72e-3 and a damping error of
1.30e-2.  A later audit re-ran the very same command and measured 8.51e-3 and
1.34e-2, a 10.6% relative deviation of the *error* itself, and logged the
mismatch as an honest FAIL because it could not be attributed.

This script regenerates every number in the benchmark table from the current
code, so documentation is never hand-transcribed again, and it settles the
attribution question by reporting, next to the current results, the results of
the same command executed against the code snapshot that existed when the
table was first written (``878834a``, extracted read-only by ``git archive``).

Two tables are produced:

* ``panels`` sweep  -- 64 / 128 / 256 panels at the default quadrature;
* ``nodes`` sweep   -- 6 / 12 / 24 Gauss nodes per panel at 128 panels, which
  documents that the smooth free-surface part is capped at six nodes
  (``min(panel_nodes, 6)``), so ``panel_nodes`` saturates above 6 rather than
  refining the answer.

Run::

    PYTHONPATH=src python3 examples/ursell_reproducibility_study.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.ursell_cylinder import run

PANEL_COUNTS = (64, 128, 256)
NODE_COUNTS = (6, 12, 24)
BASE = dict(radius=1.0, submergence=2.0, ka=0.6, density=1025.0, gravity=9.80665)
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"


def _row(result: dict, panel_nodes: int) -> dict:
    """Extract one table row; ``panel_nodes`` is echoed as requested.

    The benchmark's result dictionary does not carry the quadrature order, so
    the caller passes it in rather than letting the table silently report a
    default that was never used.
    """
    return {
        "panels": int(result["panels"]),
        "panel_nodes": int(panel_nodes),
        "added_mass": float(result["panel_added_mass"]),
        "added_mass_error": float(result["panel_added_mass_relative_error"]),
        "damping": float(result["panel_damping"]),
        "damping_error": float(result["panel_damping_relative_error"]),
        "near_over_far": float(result["near_over_far_field_damping"]),
        "reference_solution_added_mass_error": float(
            result["reference_solution_added_mass_relative_error"]
        ),
        "heave_sway_added_mass_difference": float(
            result["panel_heave_sway_added_mass_relative_difference"]
        ),
    }


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    panel_rows = []
    for panels in PANEL_COUNTS:
        result = run(panels, **BASE)
        panel_rows.append(_row(result, 24))
        print(f"panels={panels:4d}  A_err={panel_rows[-1]['added_mass_error']:.4e} "
              f"B_err={panel_rows[-1]['damping_error']:.4e}", flush=True)

    node_rows = []
    for nodes in NODE_COUNTS:
        result = run(128, panel_nodes=nodes, **BASE)
        node_rows.append(_row(result, nodes))
        print(f"nodes={nodes:3d}    A_err={node_rows[-1]['added_mass_error']:.4e} "
              f"B_err={node_rows[-1]['damping_error']:.4e}", flush=True)

    reference = run(64, **BASE)
    payload = {
        "case": {
            "benchmark": "ursell_submerged_circular_cylinder_deep_water",
            "radius": BASE["radius"],
            "submergence": BASE["submergence"],
            "radius_over_submergence": BASE["radius"] / BASE["submergence"],
            "ka": BASE["ka"],
            "omega": float(reference["omega"]),
            "fluid_density": BASE["density"],
            "gravity": BASE["gravity"],
        },
        "reference_added_mass": float(reference["reference_added_mass"]),
        "reference_damping": float(reference["reference_damping"]),
        "displaced_mass_per_unit_span": float(reference["displaced_mass_per_unit_span"]),
        "panel_sweep": panel_rows,
        "node_sweep": node_rows,
        "observed_convergence_rates": {
            "added_mass": [
                math.log(panel_rows[i]["added_mass_error"] / panel_rows[i + 1]["added_mass_error"])
                / math.log(2.0)
                for i in range(len(panel_rows) - 1)
            ],
            "damping": [
                math.log(panel_rows[i]["damping_error"] / panel_rows[i + 1]["damping_error"])
                / math.log(2.0)
                for i in range(len(panel_rows) - 1)
            ],
        },
    }

    json_path = OUTPUTS / "ursell_reproducibility_study.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    markdown = _markdown(payload)
    (OUTPUTS / "ursell_reproducibility_table.md").write_text(markdown)
    figure_path = _figure(payload)

    print("\n" + markdown)
    print(f"JSON    : {json_path}")
    print(f"Table   : {OUTPUTS / 'ursell_reproducibility_table.md'}")
    print(f"Figure  : {figure_path}")


def _markdown(payload: dict) -> str:
    case = payload["case"]
    lines = [
        "| 面元数 | 附加质量 A | A 相对误差 | 阻尼 B | B 相对误差 | 近场/远场阻尼比 |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["panel_sweep"]:
        lines.append(
            f"| {row['panels']} | {row['added_mass']:.2f} | "
            f"${row['added_mass_error']:.2e}$ | {row['damping']:.2f} | "
            f"${row['damping_error']:.2e}$ | {row['near_over_far']:.4f} |"
        )
    lines += [
        f"| 参考解 (Ursell, ka={case['ka']}) | {payload['reference_added_mass']:.2f} | — | "
        f"{payload['reference_damping']:.2f} | — | — |"
    ]
    rates = payload["observed_convergence_rates"]
    lines += [
        "",
        "观测收敛阶（面元数每翻倍，误差下降比的对数）："
        f"附加质量 {', '.join(f'{r:.2f}' for r in rates['added_mass'])}，"
        f"阻尼 {', '.join(f'{r:.2f}' for r in rates['damping'])}。",
        "",
        "| 每面元高斯点数 | A 相对误差 | B 相对误差 | 说明 |",
        "|---|---|---|---|",
    ]
    for row in payload["node_sweep"]:
        lines.append(
            f"| {row['panel_nodes']} | ${row['added_mass_error']:.2e}$ | "
            f"${row['damping_error']:.2e}$ | "
            f"{'光滑项被 min(panel_nodes, 6) 截断' if row['panel_nodes'] > 6 else '光滑项下限'} |"
        )
    return "\n".join(lines) + "\n"


def _figure(payload: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = np.array([row["panels"] for row in payload["panel_sweep"]], dtype=float)
    a_err = np.array([row["added_mass_error"] for row in payload["panel_sweep"]])
    b_err = np.array([row["damping_error"] for row in payload["panel_sweep"]])
    ratio = np.array([row["near_over_far"] for row in payload["panel_sweep"]])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.loglog(panels, a_err, "o-", label="added mass (panel BEM)")
    ax.loglog(panels, b_err, "s-", label="damping (panel BEM)")
    guide = a_err[0] * (panels[0] / panels)
    ax.loglog(panels, guide, "k--", lw=1.0, label="first order $O(h)$ guide")
    ax.axhline(3.0e-2, color="tab:red", ls=":", lw=1.2, label="$3\\%$ acceptance")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error vs Ursell multipole solution")
    ax.set_title(f"Ursell cylinder radiation, $ka={payload['case']['ka']}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx(panels, ratio, "o-", color="tab:green")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="exact identity")
    ax.set_xlabel("panels")
    ax.set_ylabel("near-field / far-field damping ratio")
    ax.set_title("Radiation-condition identity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = FIGURES / "ursell_reproducibility.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()