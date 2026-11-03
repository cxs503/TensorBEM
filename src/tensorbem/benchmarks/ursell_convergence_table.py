"""Regenerate the Ursell BEM-versus-reference table published in the docs.

``docs/marine_bem_benchmarks.md`` quotes a refinement table for the submerged
circular cylinder at ``a = 1``, ``f = 2``, ``ka = 0.6``, ``rho = 1025``,
``g = 9.80665``: for 64/128/256 panels it lists the heave added mass ``A``, the
damping ``B``, their relative errors against Ursell's multipole solution, and
the near-field / far-field damping ratio that checks the radiation condition.

Those digits used to be transcribed by hand, and they silently stopped
reproducing when the singular part of the single-layer potential was
re-integrated analytically (commit 267d5a3): the panel results moved by 0.08 %
in ``A`` while the documentation kept the old numbers, which an audit then
reported as a 10.6 % discrepancy. This module removes the hand-copying: the
table is machine-generated from the shipped solver, and
``tests/test_ursell_doc_reproducibility.py`` pins the published rows so a future
change either updates both or fails loudly.

Run::

    python -m tensorbem.benchmarks.ursell_convergence_table --panels 64 128
    python -m tensorbem.benchmarks.ursell_convergence_table --panels 256 \
        --json-out /tmp/ursell256.json

The second form exists because the 256-panel case costs about a minute; the
JSON dumps can be merged by :func:`merge_rows` before formatting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import (
    far_field_damping,
    radiation_reference,
    solve_radiation_indirect,
)

_DEFAULTS = dict(
    radius=1.0,
    submergence=2.0,
    ka=0.6,
    density=1025.0,
    gravity=9.80665,
    panel_nodes=24,
    source_count=24,
)


def reference_coefficients(**overrides: float) -> dict[str, float]:
    """Ursell's analytic added mass and damping for the default configuration."""
    params = {**_DEFAULTS, **overrides}
    omega = math.sqrt(params["ka"] * params["gravity"] / params["radius"])
    added_mass, damping = ursell_cylinder_added_mass(
        params["radius"],
        params["submergence"],
        omega,
        params["density"],
        gravity=params["gravity"],
    )
    return {"omega": omega, "added_mass": float(added_mass), "damping": float(damping)}


def panel_row(panel_count: int, **overrides: float) -> dict[str, float]:
    """One refinement row: panel solution, its errors and the energy identity."""
    params = {**_DEFAULTS, **overrides}
    exact = reference_coefficients(**params)
    mesh = submerged_circle_mesh(
        panel_count, params["radius"], params["submergence"]
    )
    result = solve_radiation_indirect(
        mesh,
        params["ka"] / params["radius"],
        "heave",
        density=params["density"],
        panel_nodes=int(params["panel_nodes"]),
    )
    added_mass = float(result.added_mass[1])
    damping = float(result.damping[1])
    far_damping = far_field_damping(
        mesh,
        result.potential,
        result.normal_velocity,
        params["ka"] / params["radius"],
        params["density"],
        gravity=params["gravity"],
    )
    return {
        "panels": int(panel_count),
        "added_mass": added_mass,
        "added_mass_relative_error": abs(added_mass - exact["added_mass"])
        / abs(exact["added_mass"]),
        "damping": damping,
        "damping_relative_error": abs(damping - exact["damping"])
        / abs(exact["damping"]),
        "near_over_far_field_damping": damping / far_damping,
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
    }


def mfs_row(panel_count: int = 64, **overrides: float) -> dict[str, float]:
    """Method-of-fundamental-solutions row (same mesh, no singular quadrature)."""
    params = {**_DEFAULTS, **overrides}
    exact = reference_coefficients(**params)
    mesh = submerged_circle_mesh(
        panel_count, params["radius"], params["submergence"]
    )
    result = radiation_reference(
        mesh,
        params["ka"] / params["radius"],
        "heave",
        density=params["density"],
        source_count=int(params["source_count"]),
    )
    added_mass = float(result.added_mass[1])
    damping = float(result.damping[1])
    return {
        "panels": int(panel_count),
        "method": "mfs",
        "added_mass": added_mass,
        "added_mass_relative_error": abs(added_mass - exact["added_mass"])
        / abs(exact["added_mass"]),
        "damping": damping,
        "damping_relative_error": abs(damping - exact["damping"])
        / abs(exact["damping"]),
    }


def build_table(panel_counts: list[int], **overrides: float) -> dict:
    """Rows for every requested panel count plus the reference and MFS rows."""
    params = {**_DEFAULTS, **overrides}
    return {
        "benchmark": "ursell_submerged_circular_cylinder_deep_water",
        "parameters": params,
        "reference": reference_coefficients(**params),
        "rows": [panel_row(n, **params) for n in panel_counts],
        "mfs": [mfs_row(n, **params) for n in panel_counts],
    }


def merge_rows(*tables: dict) -> dict:
    """Merge several dumps (e.g. a fast run and a long 256-panel run)."""
    merged: dict | None = None
    rows: list[dict] = []
    mfs_rows: list[dict] = []
    for table in tables:
        if merged is None:
            merged = {
                k: v for k, v in table.items() if k not in ("rows", "mfs")
            }
        rows.extend(table.get("rows", []))
        mfs_rows.extend(table.get("mfs", []))
    if merged is None:
        raise ValueError("merge_rows needs at least one table")
    rows.sort(key=lambda row: row["panels"])
    mfs_rows.sort(key=lambda row: row["panels"])
    merged["rows"] = rows
    merged["mfs"] = mfs_rows
    return merged


def format_markdown(table: dict) -> str:
    """Markdown rows in the layout used by ``docs/marine_bem_benchmarks.md``."""
    def sci(value: float) -> str:
        mantissa, exponent = f"{value:.2e}".split("e")
        # LaTeX-inline delimiters used by docs/marine_bem_benchmarks.md
        return "\\(" + mantissa + "\\times10^{" + str(int(exponent)) + "}\\)"

    lines = [
        "| 面元数 | \\(A\\) | 相对误差 | \\(B\\) | 相对误差 | 近场/远场阻尼比 |",
        "|---|---|---|---|---|---|",
    ]
    for row in table["rows"]:
        lines.append(
            f"| {row['panels']} | {row['added_mass']:.2f} | "
            f"{sci(row['added_mass_relative_error'])} | {row['damping']:.2f} | "
            f"{sci(row['damping_relative_error'])} | "
            f"{row['near_over_far_field_damping']:.4f} |"
        )
    reference = table["reference"]
    lines.append(
        f"| 参考解（Ursell） | {reference['added_mass']:.2f} | — | "
        f"{reference['damping']:.2f} | — | — |"
    )
    for mfs in table.get("mfs", []):
        lines.append(
            f"| 基本解方法（MFS，配点网格 {mfs['panels']} 面元） | "
            f"{mfs['added_mass']:.2f} | {sci(mfs['added_mass_relative_error'])} | "
            f"{mfs['damping']:.2f} | {sci(mfs['damping_relative_error'])} | — |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--ka", type=float, default=_DEFAULTS["ka"])
    parser.add_argument("--radius", type=float, default=_DEFAULTS["radius"])
    parser.add_argument("--submergence", type=float, default=_DEFAULTS["submergence"])
    parser.add_argument("--panel-nodes", type=int, default=_DEFAULTS["panel_nodes"])
    parser.add_argument(
        "--json-out", type=Path, default=None, help="also dump the raw table"
    )
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown"
    )
    args = parser.parse_args()

    table = build_table(
        args.panels,
        ka=args.ka,
        radius=args.radius,
        submergence=args.submergence,
        panel_nodes=args.panel_nodes,
    )
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(table, indent=2) + "\n")
    if args.format == "json":
        print(json.dumps(table, indent=2))
    else:
        print(format_markdown(table))


if __name__ == "__main__":
    main()