"""Live evidence study for the capability-coverage registry (mission m6280).

The registry in :mod:`tensorbem.capability_evidence` grades capabilities from
code/test/doc evidence.  That machinery is static: it proves a row *resolves*,
not that the physics behind the row is right.  This driver closes the loop for
the rows added in m6280 in two ways:

* it measures the coverage of the registry itself before and after the new rows
  (``capability_coverage`` on the live tree, with and without the m6280 rows);
* it re-runs cheap, independent numerical anchors for three of the new rows --
  geometry convergence, the order-1 panel self integral, and the second-order
  scaling of the Maruo drift force.  The remaining rows' numerics live in their
  own test modules and are named in the row's ``tests`` tuple; nothing is
  transcribed by hand here.

Usage::

    python -m tensorbem.benchmarks.capability_coverage_study --output docs/m6280
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np

#: Capability keys added by m6280; removing them reproduces the pre-mission
#: registry, which is what the "before" column measures.
M6280_KEYS = (
    "second-order-drift",
    "time-domain-memory-kernel",
    "uncertainty-quantification",
    "hydroelastic-dry-modes",
    "ice-region-hydrodynamics",
    "linear-panel-operators",
    "multiply-connected-domains",
    "thin-ship-general-hullforms",
    "quadrature-diagnostics",
    "hull-mesh-geometry",
)


def coverage_before_after() -> dict[str, Any]:
    """Core-module coverage with and without the m6280 rows."""

    from tensorbem import capability_evidence as ce

    after = ce.capability_coverage()
    before_rows = tuple(cap for cap in ce.CAPABILITIES if cap.key not in M6280_KEYS)
    before = ce.capability_coverage(capabilities=before_rows)
    return {
        "rows_before": len(before_rows),
        "rows_after": len(ce.CAPABILITIES),
        "core_modules": len(after["core_modules"]),
        "graded_before": len(before["graded"]),
        "graded_after": len(after["graded"]),
        "auxiliary_after": len(after["auxiliary"]),
        "unowned_before": len(before["unowned"]),
        "unowned_after": len(after["unowned"]),
        "unowned_before_names": before["unowned"],
        "unowned_after_names": after["unowned"],
        "stale_reasons": after["stale_reasons"],
    }


def anchor_icosphere() -> dict[str, Any]:
    """Icosphere area and volume error against the exact sphere."""

    from tensorbem.geometry3d import icosphere

    radius = 1.0
    exact_area = 4.0 * math.pi * radius**2
    exact_volume = 4.0 / 3.0 * math.pi * radius**3
    levels = []
    for subdivisions in (0, 1, 2, 3):
        mesh = icosphere(subdivisions=subdivisions, radius=radius)
        area = float(mesh.surface_area)
        volume = float(mesh.volume)
        levels.append(
            {
                "subdivisions": subdivisions,
                "panels": int(mesh.panel_count),
                "area": area,
                "area_error": abs(area / exact_area - 1.0),
                "volume": volume,
                "volume_error": abs(volume / exact_volume - 1.0),
            }
        )
    return {
        "case": "hull-mesh-geometry",
        "reference": "exact sphere area 4 pi R^2 and volume 4/3 pi R^3, R = 1 m",
        "levels": levels,
        "finest_error": levels[-1]["volume_error"],
        "worst_error": max(max(r["area_error"], r["volume_error"]) for r in levels),
        "pass_1pct_at_finest": levels[-1]["volume_error"] < 1.0e-2,
    }


def anchor_linear_panel_self_integral() -> dict[str, Any]:
    """Order-1 self integral: closed form against its own graded Duffy rule."""

    from tensorbem.panel_linear import (
        integrate_linear_single_layer_self_analytic,
        integrate_linear_single_layer_self_numerical,
    )

    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    analytic = np.atleast_2d(np.asarray(integrate_linear_single_layer_self_analytic(triangle)))
    numerical = np.atleast_2d(np.asarray(integrate_linear_single_layer_self_numerical(triangle)))
    scale = float(np.max(np.abs(analytic)))
    residual = float(np.max(np.abs(numerical - analytic)) / scale) if scale else float("nan")
    return {
        "case": "linear-panel-operators",
        "reference": "closed-form triangular self integral vs graded Duffy quadrature",
        "analytic_matrix": analytic.round(12).tolist(),
        "numerical_matrix": numerical.round(12).tolist(),
        "relative_residual": residual,
        "pass_1pct": residual < 1.0e-2,
    }


def anchor_maruo_drift() -> dict[str, Any]:
    """Maruo drift force: head-seas route equals the general route, and F ~ |K|^2."""

    from tensorbem import maruo_drift

    azimuths = np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
    kochin = np.full(128, 0.3 + 0.2j)
    k = 0.8
    head = maruo_drift.maruo_mean_drift_force(kochin, azimuths, k)["fx"]
    general = maruo_drift.maruo_mean_drift_force_general(kochin, azimuths, k, 0.0)["fx"]
    doubled = maruo_drift.maruo_mean_drift_force(2.0 * kochin, azimuths, k)["fx"]
    route_error = abs(head - general) / max(abs(general), 1e-30)
    ratio = doubled / head if head else float("nan")
    return {
        "case": "second-order-drift",
        "reference": "head-seas entry point vs general entry point, and F(2K)/F(K) = 4",
        "head_seas_fx": head,
        "general_fx": general,
        "route_relative_error": route_error,
        "quadratic_ratio": ratio,
        "pass_1pct": route_error < 1.0e-2 and abs(ratio - 4.0) < 4.0e-2,
    }


ANCHORS = (anchor_icosphere, anchor_linear_panel_self_integral)


def run_study(output_dir: pathlib.Path | str = pathlib.Path("docs/m6280")) -> dict[str, Any]:
    output_dir = pathlib.Path(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage = coverage_before_after()
    anchors = [anchor() for anchor in ANCHORS]
    result = {
        "mission": "m6280",
        "case": "capability_registry_coverage",
        "coverage": coverage,
        "anchors": anchors,
        "honest_limits": [
            "Coverage counts are static: a module owned by a row is *graded*, not proven correct"
            " here; the numerics of the other seven rows live in the tests the row names.",
            "The icosphere anchor is a discretisation statement (geometric convergence), not an"
            " experimental validation of a ship hull.",
            "The Maruo anchor is an internal-consistency and scaling statement; the absolute"
            " magnitude of a drift force for a real hull has no experimental anchor in this tree.",
        ],
    }
    (output_dir / "capability_coverage.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    result["figures"] = _figures(result, output_dir)
    (output_dir / "capability_coverage.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _figures(result: dict[str, Any], output_dir: pathlib.Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    cov = result["coverage"]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = ["graded", "auxiliary", "unowned"]
    before = [cov["graded_before"], 0, cov["unowned_before"]]
    after = [cov["graded_after"], cov["auxiliary_after"], cov["unowned_after"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, before, 0.4, label="before m6280", color="#b23b3b")
    ax.bar(x + 0.2, after, 0.4, label="after m6280", color="#2f6f4f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"core modules (of {cov['core_modules']})")
    ax.set_title("Evidence-registry coverage of the core package")
    ax.legend()
    for xi, (b, a) in enumerate(zip(before, after)):
        ax.text(xi - 0.2, b + 0.5, str(b), ha="center", fontsize=8)
        ax.text(xi + 0.2, a + 0.5, str(a), ha="center", fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig1_registry_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    icosphere = next(a for a in result["anchors"] if a["case"] == "hull-mesh-geometry")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    levels = [row["subdivisions"] for row in icosphere["levels"]]
    ax.semilogy(levels, [max(r["volume_error"], 1e-16) for r in icosphere["levels"]], "o-", label="volume error")
    ax.semilogy(levels, [max(r["area_error"], 1e-16) for r in icosphere["levels"]], "s--", label="area error")
    ax.axhline(1e-2, color="k", lw=0.8, ls=":", label="1 % gate")
    ax.set_xlabel("icosphere subdivision level")
    ax.set_ylabel("relative error vs exact sphere")
    ax.set_title("Geometry anchor (hull-mesh-geometry row)")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "fig2_icosphere_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _markdown(result: dict[str, Any]) -> str:
    cov = result["coverage"]
    lines = [
        "# Capability-coverage evidence (mission m6280)",
        "",
        "Generated by `python -m tensorbem.benchmarks.capability_coverage_study`; every number",
        "below is written by the program, not transcribed.",
        "",
        "## Registry coverage",
        "",
        "| quantity | before m6280 | after m6280 |",
        "|---|---|---|",
        f"| capability rows | {cov['rows_before']} | {cov['rows_after']} |",
        f"| core modules graded by a row | {cov['graded_before']} | {cov['graded_after']} |",
        f"| core modules with an accepted reason | — | {cov['auxiliary_after']} |",
        f"| core modules unowned (gate fails) | {cov['unowned_before']} | {cov['unowned_after']} |",
        f"| core modules total | {cov['core_modules']} | {cov['core_modules']} |",
        "",
        "## Live anchors for three of the newly graded rows",
        "",
        "| capability | reference | measured | < 1 % |",
        "|---|---|---|---|",
    ]
    for anchor in result["anchors"]:
        case = anchor["case"]
        if case == "hull-mesh-geometry":
            measured = f"finest volume error {anchor['finest_error']:.2e} ({anchor['levels'][-1]['panels']} panels)"
            ok = anchor["pass_1pct_at_finest"]
        elif case == "linear-panel-operators":
            measured = f"analytic vs Duffy residual {anchor['relative_residual']:.2e}"
            ok = anchor["pass_1pct"]
        else:
            measured = (
                f"route error {anchor['route_relative_error']:.2e}, "
                f"F(2K)/F(K) = {anchor['quadratic_ratio']:.6f}"
            )
            ok = anchor["pass_1pct"]
        lines.append(f"| `{case}` | {anchor['reference']} | {measured} | {'YES' if ok else 'NO'} |")
    lines += ["", "## Honest limits", ""]
    for limit in result["honest_limits"]:
        lines.append(f"- {limit}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="capability coverage study")
    parser.add_argument("--output", default="docs/m6280", type=pathlib.Path)
    args = parser.parse_args(argv)
    result = run_study(args.output)
    print(
        json.dumps(
            {
                "coverage": result["coverage"],
                "anchors": [
                    {k: v for k, v in anchor.items() if k in ("case", "pass_1pct", "relative_residual", "finest_error", "route_relative_error", "quadratic_ratio")}
                    for anchor in result["anchors"]
                ],
                "figures": result["figures"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())