"""Quadrature-versus-discretisation study for the Ursell cylinder benchmark.

The benchmark prints a single number per panel count; this study separates the
two numerical errors that the number mixes, because they behave differently and
the difference matters for how the documented accuracy should be quoted.

* **Quadrature error** is the error of the source-potential evaluation for a
  *given* panel mesh: Gauss-Legendre nodes per panel, and the treatment of the
  ``log r`` singularity on the self panel. The current solver integrates the
  singular Rankine part analytically over each straight panel and quadratures
  only the smooth free-surface remainder (``min(panel_nodes, 6)`` nodes).
* **Discretisation error** is the error of the piecewise-constant source
  distribution itself. It is first order in the panel size for this kernel, and
  no amount of quadrature improves it.

If the panel result changes when ``panel_nodes`` changes, the computed number is
quadrature-limited and cannot be quoted as the accuracy of the method. If it is
insensitive to ``panel_nodes``, the residual error belongs to the panel count
and the refinement trend is the honest accuracy statement.

Run: python -m tensorbem.benchmarks.ursell_quadrature_study --output study.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks.ursell_cylinder import run

_RADIUS = 1.0
_SUBMERGENCE = 2.0
_KA = 0.6


def study(
    panel_counts: tuple[int, ...] = (32, 64, 128, 256),
    node_counts: tuple[int, ...] = (6, 12, 24, 48),
    node_panels: int = 128,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict:
    """Return the panel-count and panel-node sweeps as one JSON-ready record."""
    reference_mass, reference_damping = ursell_cylinder_added_mass(
        _RADIUS, _SUBMERGENCE, math.sqrt(_KA * gravity / _RADIUS), density
    )

    panel_sweep = []
    for panels in panel_counts:
        row = run(panels, _RADIUS, _SUBMERGENCE, _KA, density, gravity)
        panel_sweep.append(
            {
                "panels": panels,
                "panel_added_mass": row["panel_added_mass"],
                "panel_added_mass_relative_error": row["panel_added_mass_relative_error"],
                "panel_damping": row["panel_damping"],
                "panel_damping_relative_error": row["panel_damping_relative_error"],
                "near_far_field_relative_difference": row[
                    "near_far_field_relative_difference"
                ],
                "panel_condition_number": row["panel_condition_number"],
            }
        )

    node_sweep = []
    for nodes in node_counts:
        row = run(
            node_panels,
            _RADIUS,
            _SUBMERGENCE,
            _KA,
            density,
            gravity,
            panel_nodes=nodes,
        )
        node_sweep.append(
            {
                "panel_nodes": nodes,
                "panel_added_mass": row["panel_added_mass"],
                "panel_added_mass_relative_error": row[
                    "panel_added_mass_relative_error"
                ],
                "panel_damping": row["panel_damping"],
                "panel_damping_relative_error": row["panel_damping_relative_error"],
            }
        )

    spread_mass = max(r["panel_added_mass"] for r in node_sweep) - min(
        r["panel_added_mass"] for r in node_sweep
    )
    spread_damping = max(r["panel_damping"] for r in node_sweep) - min(
        r["panel_damping"] for r in node_sweep
    )

    return {
        "benchmark": "ursell_quadrature_study",
        "radius": _RADIUS,
        "submergence": _SUBMERGENCE,
        "ka": _KA,
        "fluid_density": density,
        "gravity": gravity,
        "reference_added_mass": reference_mass,
        "reference_damping": reference_damping,
        "panel_sweep": panel_sweep,
        "node_sweep": node_sweep,
        "node_sweep_spread_added_mass": spread_mass,
        "node_sweep_spread_damping": spread_damping,
        "node_sweep_spread_added_mass_relative": abs(spread_mass) / reference_mass,
        "node_sweep_spread_damping_relative": abs(spread_damping) / reference_damping,
        "node_sweep_is_converged": abs(spread_mass) / reference_mass < 1.0e-6,
    }


def to_markdown(record: dict) -> str:
    lines = [
        "| panels | A | err(A) | B | err(B) | near/far | kappa |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in record["panel_sweep"]:
        lines.append(
            "| {panels} | {a:.4f} | {ea:.3e} | {b:.4f} | {eb:.3e} | {nf:.3e} | {k:.2f} |".format(
                panels=row["panels"],
                a=row["panel_added_mass"],
                ea=row["panel_added_mass_relative_error"],
                b=row["panel_damping"],
                eb=row["panel_damping_relative_error"],
                nf=row["near_far_field_relative_difference"],
                k=row["panel_condition_number"],
            )
        )
    lines += [
        "",
        "| panel_nodes | A | err(A) | B | err(B) |",
        "|---|---|---|---|---|",
    ]
    for row in record["node_sweep"]:
        lines.append(
            "| {n} | {a:.6f} | {ea:.6e} | {b:.6f} | {eb:.6e} |".format(
                n=row["panel_nodes"],
                a=row["panel_added_mass"],
                ea=row["panel_added_mass_relative_error"],
                b=row["panel_damping"],
                eb=row["panel_damping_relative_error"],
            )
        )
    lines += [
        "",
        "reference: A={:.4f}, B={:.4f}".format(
            record["reference_added_mass"], record["reference_damping"]
        ),
        "node-sweep spread: dA={:.3e} ({:.2e} relative), dB={:.3e} ({:.2e} relative)".format(
            record["node_sweep_spread_added_mass"],
            record["node_sweep_spread_added_mass_relative"],
            record["node_sweep_spread_damping"],
            record["node_sweep_spread_damping_relative"],
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panels", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--panel-nodes", type=int, nargs="+", default=[6, 12, 24, 48])
    parser.add_argument("--node-panels", type=int, default=128)
    parser.add_argument(
        "--label",
        type=str,
        default="current",
        help="provenance tag written into the JSON record",
    )
    args = parser.parse_args()

    record = study(
        panel_counts=tuple(args.panels),
        node_counts=tuple(args.panel_nodes),
        node_panels=args.node_panels,
    )
    record["implementation_label"] = args.label
    payload = json.dumps(record, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        markdown_path = args.output.with_suffix(".md")
        markdown_path.write_text(to_markdown(record) + "\n", encoding="utf-8")
    print(payload)
    print()
    print(to_markdown(record))


if __name__ == "__main__":
    main()