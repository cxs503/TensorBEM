"""Two-route regime map of the 3-D finite-depth free-surface kernel.

Why this driver exists
----------------------
``docs/finite_depth3d_deepwater_arbitration.md`` closes with an explicit honest
limit: *"it does not add an independent evaluation of the finite-depth kernel"*.
Two independent representations of the same kernel ship in this package --

* the **John eigenseries** (:func:`tensorbem.finite_depth3d.green_function_3d`),
  which finds the propagating root ``k0 tanh(k0 h) = nu`` and every evanescent
  root ``mu_n tan(mu_n h) = -nu`` and normalises each mode by an analytic norm;
* the **wavenumber-quadrature** route
  (:func:`tensorbem.finite_depth3d_quadrature.green_function_finite_depth_3d`),
  which evaluates the wave integral with kernel subtraction and uses **no**
  evanescent root finder and **no** mode norm.

The m13811 off-axis fix compared them at one depth and one frequency
(``<= 3.3e-11``).  A point is not a map: the two routes fail in different ways
with ``k0 h`` (the eigenseries accumulates truncated evanescent modes as ``h``
falls; the quadrature route's ``k_max`` is set by the free-surface clearance
``-(z + zeta)``).  This driver publishes the map over
``(k0 h, z/h, k0 R)``, plus the two limits that the map alone cannot show:

* a **rule ladder** -- whether a discrepant cell is the quadrature route's panel
  rule (it must collapse as ``nodes`` grows) or the eigenseries (it must not);
* the **deep-water limit** at *fixed absolute elevations*, which is the only
  defensible way to compare a finite-depth point with the Kelvin kernel: at
  fixed ``z/h`` the sea bed approaches the points as ``h`` grows, so the bottom
  image -- genuine physics -- stays visible and the limit is not ``G -> G_deep``
  (measured: relative difference 0.77 at ``k0 h = 77``, ``z/h = -0.98``; the
  construct, not the kernel, was wrong).

Grid
----
``k0 = 1`` is pinned so every length is in ``1/k0``: ``depth = X`` forces
``nu = tanh(X)``.  ``X = k0 h`` runs shallow (0.5) to deep (77), field
elevations from just below the free surface (``-0.02 h``) to just above the bed
(``-0.98 h``), two ranges.  Source elevation is mid-depth.

Run::

    python -m tensorbem.benchmarks.finite_depth3d_route_map \\
        --output docs/finite_depth3d_route_map

Nothing in the generated JSON/markdown is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..finite_depth3d import green_function_3d as eigenseries_kernel
from ..finite_depth3d_quadrature import (
    green_function_finite_depth_3d as quadrature_kernel,
)
from ..free_surface3d import green_function_3d as deepwater_kernel

#: ``k0 h`` values of the map (shallow -> deep).
DEPTH_PARAMETERS = (0.5, 1.0, 3.0, 10.0, 30.0, 77.0)
#: First characters of the generated block, for the published-page gate.
BEGIN_MARKER_FRAGMENT = "<!-- finite-depth3d-route-map:begin"
#: Field-point elevations as a fraction of the depth.
ELEVATION_FRACTIONS = (-0.02, -0.5, -0.98)
#: Horizontal ranges in units of ``1/k0``.
RANGES = (0.5, 2.0)
#: Gauss-Legendre nodes of the quadrature route used for the published map.
#: The module default (16) leaves a near-surface residual of up to 4.4e-8; the
#: rule ladder below shows it collapsing with ``nodes``, which identifies it as
#: the quadrature route's panel rule rather than the eigenseries.
MAP_NODES = 32
#: Rule ladder for the cells that are discrepant at the default node count.
LADDER_NODES = (16, 32, 48)


def _depth_inputs(k0h: float) -> tuple[float, float]:
    """``(nu, depth)`` with ``k0 = 1`` pinned: ``nu = tanh(k0 h)``, ``h = k0 h``."""
    depth = float(k0h)
    return math.tanh(k0h), depth


def _cell(
    k0h: float, elevation_fraction: float, range_k: float, nodes: int = MAP_NODES,
) -> dict[str, Any]:
    """Compare the two routes in one ``(k0 h, z/h, k0 R)`` cell."""
    nu, depth = _depth_inputs(k0h)
    zeta = -0.5 * depth
    z = elevation_fraction * depth
    target = np.array([[range_k, 0.0, z]])
    source = np.array([[0.0, 0.0, zeta]])

    g_series = complex(eigenseries_kernel(target, source, nu, depth)[0])
    g_quad = complex(quadrature_kernel(target, source, nu, depth, nodes=nodes)[0])
    scale = abs(g_quad)
    difference = abs(g_series - g_quad)
    return {
        "k0h": k0h,
        "nu": nu,
        "depth": depth,
        "elevation_fraction": elevation_fraction,
        "range_k": range_k,
        "nodes": nodes,
        "source_z": zeta,
        "target_z": z,
        "g_eigenseries_real": g_series.real,
        "g_eigenseries_imag": g_series.imag,
        "g_quadrature_real": g_quad.real,
        "g_quadrature_imag": g_quad.imag,
        "abs_g_quadrature": scale,
        "abs_difference": difference,
        "relative_difference": difference / scale if scale > 0.0 else math.inf,
    }


def _rule_ladder(k0h: float, elevation_fraction: float, range_k: float) -> dict[str, Any]:
    """Hold the cell fixed and vary the quadrature rule of the second route."""
    rows = [_cell(k0h, elevation_fraction, range_k, nodes=n) for n in LADDER_NODES]
    return {
        "k0h": k0h,
        "elevation_fraction": elevation_fraction,
        "range_k": range_k,
        "relative_difference_by_nodes": {
            str(row["nodes"]): row["relative_difference"] for row in rows
        },
    }


def _deep_limit(k0h: float, target_z: float, source_z: float, range_k: float) -> dict[str, Any]:
    """Eigenseries at finite ``h`` vs the deep-water Kelvin kernel, *fixed* elevations.

    The elevations are absolute (in ``1/k0``), not fractions of ``h``: a fraction
    would drag the points towards the sea bed as ``h`` grows and keep the bottom
    image at a finite distance, so no limit would be measured.
    """
    nu, depth = _depth_inputs(k0h)
    if -target_z > depth or -source_z > depth:
        raise ValueError("elevations must lie inside the strip for this depth")
    target = np.array([[range_k, 0.0, target_z]])
    source = np.array([[0.0, 0.0, source_z]])
    g_series = complex(eigenseries_kernel(target, source, nu, depth)[0])
    g_deep = complex(deepwater_kernel(target, source, nu)[0])
    scale = abs(g_deep)
    difference = abs(g_series - g_deep)
    return {
        "k0h": k0h,
        "target_z": target_z,
        "source_z": source_z,
        "range_k": range_k,
        "abs_g_deepwater": scale,
        "abs_difference": difference,
        "relative_difference": difference / scale if scale > 0.0 else math.inf,
    }


def run_map(depth_parameters: Sequence[float] = DEPTH_PARAMETERS) -> dict[str, Any]:
    """Run the whole grid and return the machine-readable report."""
    cells = [
        _cell(k0h, elevation, range_k)
        for k0h in depth_parameters
        for elevation in ELEVATION_FRACTIONS
        for range_k in RANGES
    ]
    worst = max(cells, key=lambda c: c["relative_difference"])
    by_depth: dict[str, dict[str, float]] = {}
    for k0h in depth_parameters:
        subset = [c for c in cells if c["k0h"] == k0h]
        by_depth[f"{k0h:g}"] = {
            "cells": len(subset),
            "max_relative_difference": max(c["relative_difference"] for c in subset),
            "min_abs_g": min(c["abs_g_quadrature"] for c in subset),
        }

    ladder = [
        _rule_ladder(k0h, elevation, range_k)
        for k0h, elevation, range_k in (
            (0.5, -0.02, 0.5),
            (0.5, -0.02, 2.0),
            (1.0, -0.02, 0.5),
            (1.0, -0.98, 2.0),
        )
    ]

    deep_ladder = [
        _deep_limit(k0h, target_z, -1.0, range_k)
        for k0h in (3.0, 10.0, 30.0, 77.0)
        for target_z in (-0.2, -1.0)
        for range_k in RANGES
    ]
    deep_by_depth: dict[str, float] = {}
    for k0h in (3.0, 10.0, 30.0, 77.0):
        deep_by_depth[f"{k0h:g}"] = max(
            row["relative_difference"] for row in deep_ladder if row["k0h"] == k0h
        )

    return {
        "driver": "tensorbem.benchmarks.finite_depth3d_route_map",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "grid": {
            "depth_parameters_k0h": list(depth_parameters),
            "elevation_fractions": list(ELEVATION_FRACTIONS),
            "ranges_k": list(RANGES),
            "cells": len(cells),
            "nodes": MAP_NODES,
        },
        "cells": cells,
        "by_depth": by_depth,
        "worst_cell": worst,
        "worst_relative_difference": worst["relative_difference"],
        "rule_ladder": ladder,
        "rule_ladder_worst_by_nodes": {
            str(n): max(
                row["relative_difference_by_nodes"][str(n)] for row in ladder
            )
            for n in LADDER_NODES
        },
        "deep_water_ladder": deep_ladder,
        "deep_water_worst_by_depth": deep_by_depth,
        "routes": {
            "eigenseries": "tensorbem.finite_depth3d.green_function_3d",
            "quadrature": "tensorbem.finite_depth3d_quadrature.green_function_finite_depth_3d",
            "deep_water": "tensorbem.free_surface3d.green_function_3d",
        },
        "shared_inputs_declared": [
            "k0 = 1 is pinned by construction (nu = tanh(depth)), so the propagating root is not an independent piece of evidence",
            "both routes satisfy the same free-surface / sea-bed conditions by construction",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as the generated markdown block (no hand-typed numbers)."""
    lines = [
        "<!-- finite-depth3d-route-map:begin (generated by "
        "tensorbem.benchmarks.finite_depth3d_route_map) -->",
        "",
        f"Quadrature rule used for the map: `nodes = {report['grid']['nodes']}`.",
        "",
        "| `k0 h` | cells | worst relative difference | smallest `|G|` in row |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(report["by_depth"], key=float):
        row = report["by_depth"][key]
        lines.append(
            f"| {key} | {row['cells']} | {row['max_relative_difference']:.3e} "
            f"| {row['min_abs_g']:.3e} |"
        )
    worst = report["worst_cell"]
    lines += [
        "",
        f"Worst cell over the whole map: `{report['worst_relative_difference']:.3e}` "
        f"at `k0 h = {worst['k0h']:g}`, `z/h = {worst['elevation_fraction']:g}`, "
        f"`k0 R = {worst['range_k']:g}`.",
        "",
        "**Rule ladder** (same cell, quadrature panel rule):",
        "",
        "| `k0 h` | `z/h` | `k0 R` | "
        + " | ".join(f"nodes={n}" for n in LADDER_NODES)
        + " |",
        "| ---: | ---: | ---: | " + " | ".join("---:" for _ in LADDER_NODES) + " |",
    ]
    for row in report["rule_ladder"]:
        values = " | ".join(
            f"{row['relative_difference_by_nodes'][str(n)]:.3e}" for n in LADDER_NODES
        )
        lines.append(
            f"| {row['k0h']:g} | {row['elevation_fraction']:g} | {row['range_k']:g} | {values} |"
        )
    lines += [
        "",
        "**Deep-water limit at fixed absolute elevations** (source `z = -1`, targets "
        "`z = -0.2` and `z = -1`, `k0 = 1`):",
        "",
        "| `k0 h` | worst relative difference vs Kelvin kernel |",
        "| ---: | ---: |",
    ]
    for key in sorted(report["deep_water_worst_by_depth"], key=float):
        value = report["deep_water_worst_by_depth"][key]
        lines.append(f"| {key} | {value:.3e} |")
    lines += [
        "",
        "| `k0 h` | `z/h` | `k0 R` | relative difference |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for cell in report["cells"]:
        lines.append(
            f"| {cell['k0h']:g} | {cell['elevation_fraction']:g} | {cell['range_k']:g} "
            f"| {cell['relative_difference']:.3e} |"
        )
    lines += ["", "<!-- finite-depth3d-route-map:end -->", ""]
    return "\n".join(lines)


def make_figures(report: dict[str, Any], outdir: Path) -> list[Path]:
    """Plot the discrepancy map and the two ladders."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    for axis, range_k in zip(axes[:2], RANGES):
        for elevation in ELEVATION_FRACTIONS:
            rows = [
                c
                for c in report["cells"]
                if c["elevation_fraction"] == elevation and c["range_k"] == range_k
            ]
            axis.loglog(
                [c["k0h"] for c in rows],
                [max(c["relative_difference"], 1e-17) for c in rows],
                marker="o",
                label=f"z/h = {elevation:g}",
            )
        axis.axhline(1e-12, color="0.6", linestyle=":", linewidth=1.0)
        axis.set_xlabel(r"$k_0 h$")
        axis.set_ylabel("relative difference (eigenseries vs quadrature)")
        axis.set_title(f"route map, $k_0 R$ = {range_k:g}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)

    depths = sorted({row["k0h"] for row in report["deep_water_ladder"]})
    for target_z in (-0.2, -1.0):
        axes[2].loglog(
            depths,
            [
                max(
                    row["relative_difference"]
                    for row in report["deep_water_ladder"]
                    if row["k0h"] == k and row["target_z"] == target_z
                )
                for k in depths
            ],
            marker="s",
            label=f"target z = {target_z}",
        )
    axes[2].set_xlabel(r"$k_0 h$ (elevations fixed in $1/k_0$)")
    axes[2].set_ylabel("relative difference vs Kelvin kernel")
    axes[2].set_title("deep-water limit at fixed elevations")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=8)

    figure.suptitle(
        "3-D finite-depth kernel: two independent routes across the "
        r"$k_0h$ / elevation / range plane, and the deep-water limit"
    )
    figure.tight_layout()
    path = outdir / "finite_depth3d_route_map.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return [path]


def write_report(report: dict[str, Any], outdir: Path) -> list[Path]:
    """Write ``route_map.json``, the generated markdown block and the figure."""
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "route_map.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path = outdir / "route_map.md"
    md_path.write_text(render_markdown(report))
    return [json_path, md_path, *make_figures(report, outdir)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/finite_depth3d_route_map"),
        help="directory for route_map.json / route_map.md / the figure",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = run_map()
    paths = write_report(report, args.output)
    if not args.quiet:
        print(f"worst relative difference: {report['worst_relative_difference']:.3e}")
        print(f"rule ladder by nodes: {report['rule_ladder_worst_by_nodes']}")
        print(f"deep-water worst by depth: {report['deep_water_worst_by_depth']}")
        for path in paths:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())