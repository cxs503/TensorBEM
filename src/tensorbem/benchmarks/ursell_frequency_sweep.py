"""Frequency and submergence sweep of the Ursell deep-water radiation benchmark.

``docs/ursell_error_budget.md`` §5 records the open item that the accuracy of the
Ursell panel solution was established at a *single* non-dimensional frequency and
submergence, ``ka = 0.6`` with ``a/f = 0.5``:

    "仅审计了一个 (ka, a/f) 点：ka>=1 时自由液面项更强，收敛行为可能不同，未验证。"

This module closes that debt. It reuses nothing but the closed-form multipole
reference (:func:`tensorbem.analytic.ursell_cylinder_added_mass`) and the
radiation solver (:func:`tensorbem.free_surface_solver.solve_radiation_indirect`),
so every number it prints is a fresh comparison of the panel method against the
analytical solution at a grid of ``(ka, a/f)`` points:

* the panel discretisation error in added mass and damping,
* the observed order of convergence in the panel count,
* the near-field/far-field damping identity (the radiation-condition check),
* the truncation sensitivity of the analytical reference itself, which must be
  far below the discretisation error or the comparison is meaningless.

Only the 2-D deep-water problem is covered; no finite depth, no forward speed, no
experiment (the benchmark is analytical, so "error" always means relative error
against Ursell's multipole solution).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import far_field_damping, solve_radiation_indirect

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0

DEFAULT_KA_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5)
DEFAULT_A_OVER_F = (0.25, 0.5, 0.75)
DEFAULT_PANEL_COUNTS = (32, 64, 128)
REFERENCE_TERMS = 40


def _validate_point(ka: float, a_over_f: float) -> None:
    if not math.isfinite(ka) or ka <= 0.0:
        raise ValueError("ka must be a positive finite number")
    if not math.isfinite(a_over_f) or not 0.0 < a_over_f < 1.0:
        raise ValueError("a_over_f must lie strictly between 0 and 1")


def reference_coefficients(
    ka: float,
    a_over_f: float,
    *,
    terms: int = REFERENCE_TERMS,
    radius: float = RADIUS,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, float]:
    """Closed-form Ursell (1949) reference at one ``(ka, a/f)`` point."""
    _validate_point(ka, a_over_f)
    submergence = radius / a_over_f
    omega = math.sqrt(ka * gravity / radius)
    added_mass, damping = ursell_cylinder_added_mass(
        radius, submergence, omega, density, gravity=gravity, terms=terms
    )
    displaced = density * math.pi * radius * radius
    return {
        "ka": ka,
        "a_over_f": a_over_f,
        "terms": terms,
        "submergence": submergence,
        "omega": omega,
        "added_mass": added_mass,
        "damping": damping,
        "mu": added_mass / displaced,
        "nu": damping / (omega * displaced),
    }


def truncation_sensitivity(
    ka: float,
    a_over_f: float,
    terms: Sequence[int] = (REFERENCE_TERMS, 80, 120),
) -> dict[str, float]:
    """How far the analytical reference still moves when its truncation grows.

    If this is not negligible against the panel error, the comparison measures
    the reference, not the solver.
    """
    base = reference_coefficients(ka, a_over_f, terms=int(terms[0]))
    worst: dict[str, float] = {"added_mass": 0.0, "damping": 0.0}
    for count in terms[1:]:
        other = reference_coefficients(ka, a_over_f, terms=int(count))
        for key in ("added_mass", "damping"):
            delta = abs(other[key] - base[key]) / abs(base[key])
            worst[key] = max(worst[key], delta)
    worst["ka"] = ka
    worst["a_over_f"] = a_over_f
    worst["terms"] = float(max(terms))
    return worst


def panel_coefficients(
    panel_count: int,
    ka: float,
    a_over_f: float,
    *,
    panel_nodes: int = 24,
    radius: float = RADIUS,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, float]:
    """Panel-method added mass and damping at one ``(ka, a/f)`` point."""
    _validate_point(ka, a_over_f)
    if panel_count < 8:
        raise ValueError("panel_count must be at least 8")
    submergence = radius / a_over_f
    omega = math.sqrt(ka * gravity / radius)
    mesh = submerged_circle_mesh(panel_count, radius, submergence)
    result = solve_radiation_indirect(
        mesh, ka / radius, "heave", density=density, panel_nodes=panel_nodes
    )
    near = float(result.damping[1])
    far = far_field_damping(
        mesh,
        result.potential,
        result.normal_velocity,
        ka / radius,
        density,
        gravity=gravity,
    )
    return {
        "panels": panel_count,
        "added_mass": float(result.added_mass[1]),
        "damping": near,
        "far_field_damping": float(far),
        "near_over_far": near / float(far) if far != 0.0 else float("nan"),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
    }


def _relative_error(panel: float, reference: float) -> float:
    return abs(panel - reference) / abs(reference)


def sweep(
    ka_values: Iterable[float] = DEFAULT_KA_VALUES,
    a_over_f_values: Iterable[float] = DEFAULT_A_OVER_F,
    panel_counts: Iterable[int] = DEFAULT_PANEL_COUNTS,
    *,
    panel_nodes: int = 24,
    reference_terms: int = REFERENCE_TERMS,
    reference_terms_check: Sequence[int] = (REFERENCE_TERMS, 80, 120),
    radius: float = RADIUS,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, object]:
    """Run the full grid and return a JSON-ready record."""
    panel_counts = tuple(int(n) for n in panel_counts)
    if len(panel_counts) < 2:
        raise ValueError("at least two panel counts are needed for an order estimate")
    rows: list[dict[str, float]] = []
    for ka in ka_values:
        for ratio in a_over_f_values:
            reference = reference_coefficients(
                ka, ratio, terms=reference_terms, radius=radius,
                density=density, gravity=gravity,
            )
            truncation = truncation_sensitivity(
                ka, ratio, terms=tuple(reference_terms_check)
            )
            previous_left = None
            previous_error = None
            for panels in panel_counts:
                panel = panel_coefficients(
                    panels, ka, ratio, panel_nodes=panel_nodes,
                    radius=radius, density=density, gravity=gravity,
                )
                error_mass = _relative_error(panel["added_mass"], reference["added_mass"])
                error_damping = _relative_error(
                    panel["damping"], reference["damping"]
                )
                order_mass = order_damping = None
                if previous_left is not None:
                    order_mass = math.log2(previous_error[0] / error_mass)
                    order_damping = math.log2(previous_error[1] / error_damping)
                left = panels / (panels - 1.0)
                rows.append(
                    {
                        "ka": ka,
                        "a_over_f": ratio,
                        "panels": float(panels),
                        "reference_mu": reference["mu"],
                        "reference_nu": reference["nu"],
                        "reference_added_mass": reference["added_mass"],
                        "reference_damping": reference["damping"],
                        "reference_terms": float(reference["terms"]),
                        "truncation_added_mass": truncation["added_mass"],
                        "truncation_damping": truncation["damping"],
                        "panel_added_mass": panel["added_mass"],
                        "panel_damping": panel["damping"],
                        "error_added_mass": error_mass,
                        "error_damping": error_damping,
                        "order_added_mass": order_mass,
                        "order_damping": order_damping,
                        # N * error: constant for a first-order scheme.
                        "ne_added_mass": panels * error_mass,
                        "ne_damping": panels * error_damping,
                        "near_over_far": panel["near_over_far"],
                        "condition_number": panel["condition_number"],
                        "relative_residual": panel["relative_residual"],
                        "panel_nodes": float(panel_nodes),
                        "mesh_ratio": left,
                    }
                )
                previous_left = left
                previous_error = (error_mass, error_damping)

    finest = max(panel_counts)
    coarse = panel_counts[0]
    fine_rows = [row for row in rows if row["panels"] == float(finest)]
    coarse_rows = [row for row in rows if row["panels"] == float(coarse)]
    summary = {
        "points": len(fine_rows),
        "finest_panels": finest,
        "coarsest_panels": coarse,
        "max_error_added_mass": max(row["error_added_mass"] for row in fine_rows),
        "max_error_damping": max(row["error_damping"] for row in fine_rows),
        "max_error_added_mass_point": _argmax_point(fine_rows, "error_added_mass"),
        "max_error_damping_point": _argmax_point(fine_rows, "error_damping"),
        "min_error_added_mass": min(row["error_added_mass"] for row in fine_rows),
        "max_coarse_error_added_mass": max(
            row["error_added_mass"] for row in coarse_rows
        ),
        "max_near_over_far_deviation": max(
            abs(row["near_over_far"] - 1.0) for row in fine_rows
        ),
        "max_truncation_added_mass": max(row["truncation_added_mass"] for row in rows),
        "max_truncation_damping": max(row["truncation_damping"] for row in rows),
        "min_order_added_mass": min(
            row["order_added_mass"] for row in rows if row["order_added_mass"] is not None
        ),
        "max_order_added_mass": max(
            row["order_added_mass"] for row in rows if row["order_added_mass"] is not None
        ),
        "all_within_three_percent": all(
            row["error_added_mass"] < 3.0e-2 and row["error_damping"] < 3.0e-2
            for row in fine_rows
        ),
    }
    return {
        "benchmark": "ursell_submerged_circular_cylinder_deep_water_sweep",
        "settings": {
            "ka_values": [float(value) for value in ka_values],
            "a_over_f_values": [float(value) for value in a_over_f_values],
            "panel_counts": [float(value) for value in panel_counts],
            "panel_nodes": panel_nodes,
            "reference_terms": reference_terms,
            "reference_terms_check": [int(value) for value in reference_terms_check],
            "radius": radius,
            "density": density,
            "gravity": gravity,
            "motion": "heave",
            "depth": "infinite",
        },
        "rows": rows,
        "summary": summary,
    }


def _argmax_point(rows: Sequence[dict[str, float]], key: str) -> dict[str, float]:
    best = max(rows, key=lambda row: row[key])
    return {"ka": best["ka"], "a_over_f": best["a_over_f"], "value": best[key]}


def render_markdown(result: dict[str, object]) -> str:
    """Markdown tables generated from the sweep record (no hand-copied numbers)."""
    rows: Sequence[dict[str, float]] = result["rows"]  # type: ignore[assignment]
    settings: dict[str, object] = result["settings"]  # type: ignore[assignment]
    summary: dict[str, object] = result["summary"]  # type: ignore[assignment]
    lines = [
        "# Ursell frequency/submergence sweep",
        "",
        "Generated by `python -m tensorbem.benchmarks.ursell_frequency_sweep`.",
        "",
        "| setting | value |",
        "|---|---|",
        f"| ka values | {', '.join(str(v) for v in settings['ka_values'])} |",
        f"| a/f values | {', '.join(str(v) for v in settings['a_over_f_values'])} |",
        f"| panel counts | {', '.join(str(int(v)) for v in settings['panel_counts'])} |",
        f"| quadrature nodes per panel (requested) | {settings['panel_nodes']} |",
        f"| reference truncation | {settings['reference_terms']} terms "
        f"(checked against {', '.join(str(v) for v in settings['reference_terms_check'])}) |",
        "",
        f"## Panel error at {int(summary['finest_panels'])} panels",
        "",
        "| ka | a/f | A panel | A reference | rel err A | B panel | B reference | rel err B | near/far |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["panels"] != float(summary["finest_panels"]):
            continue
        lines.append(
            "| {ka:.2f} | {ratio:.2f} | {pa:.4f} | {ra:.4f} | {ea:.3e} | "
            "{pb:.4f} | {rb:.4f} | {eb:.3e} | {nf:.6f} |".format(
                ka=row["ka"],
                ratio=row["a_over_f"],
                pa=row["panel_added_mass"],
                ra=row["reference_added_mass"],
                ea=row["error_added_mass"],
                pb=row["panel_damping"],
                rb=row["reference_damping"],
                eb=row["error_damping"],
                nf=row["near_over_far"],
            )
        )
    lines += [
        "",
        "## Observed order in the panel count",
        "",
        "| ka | a/f | panel step | order (A) | order (B) |",
        "|---|---|---|---|---|",
    ]
    counts = [int(v) for v in settings["panel_counts"]]
    steps = list(zip(counts[:-1], counts[1:]))
    for row in rows:
        if row["order_added_mass"] is None:
            continue
        index = counts.index(int(row["panels"]))
        lines.append(
            "| {ka:.2f} | {ratio:.2f} | {left}->{right} | {oa:.2f} | {ob:.2f} |".format(
                ka=row["ka"],
                ratio=row["a_over_f"],
                left=steps[index - 1][0],
                right=steps[index - 1][1],
                oa=row["order_added_mass"],
                ob=row["order_damping"],
            )
        )
    lines += [
        "",
        "## Unrounded summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ka", type=float, nargs="+", default=list(DEFAULT_KA_VALUES))
    parser.add_argument(
        "--a-over-f", type=float, nargs="+", default=list(DEFAULT_A_OVER_F)
    )
    parser.add_argument("--panels", type=int, nargs="+", default=list(DEFAULT_PANEL_COUNTS))
    parser.add_argument("--panel-nodes", type=int, default=24)
    parser.add_argument("--output", type=Path, help="write the JSON record here")
    parser.add_argument("--summary", type=Path, help="write the markdown tables here")
    arguments = parser.parse_args(argv)

    result = sweep(
        arguments.ka,
        arguments.a_over_f,
        arguments.panels,
        panel_nodes=arguments.panel_nodes,
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(result)
    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(markdown + "\n")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())