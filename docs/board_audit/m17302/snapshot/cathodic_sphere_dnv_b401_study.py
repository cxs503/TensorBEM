"""CASE-BEM-CATHODIC-SPHERE: DNV-RP-B401 conducting-sphere anchor study.

Runs the acceptance study for the case-pool anchor

    |BEM-CATHODIC-SPHERE| ref: 4 pi sigma a phi_a (DNV-RP-B401 spreading
    resistance of an isolated anode) | tolerance < 5 % (ledger) |

and writes every number the report quotes into
``artifacts/cathodic_sphere/results.json`` (no hand-copied digits).

Blocks
------
1. ``convergence``   -- icosphere levels 1..N: surface current vs ``4 pi sigma a phi_a``,
   observed order, exterior-field error.
2. ``routes``        -- at two levels: surface panel integral vs far-field
   (Gauss-law) recovery of the same total current.
3. ``scaling``       -- non-dimensional check ``I / (4 pi sigma a phi_a) = 1`` over
   conductivity / radius / potential points (two+ parameter points).

Run:
    PYTHONPATH=src python scripts/cathodic_sphere_dnv_b401_study.py [--subdiv 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.cathodic_protection import (
    SEAWATER_CONDUCTIVITY,
    evaluate_exterior_potential,
    far_field_current,
    fibonacci_sphere_directions,
    solve_cathodic_protection,
    verify_conducting_sphere,
    verify_conducting_sphere_routes,
)
from tensorbem.geometry3d import icosphere

LEDGER_TOLERANCE = 0.05   # |BEM-CATHODIC-SPHERE| ledger criterion
INTERNAL_TOLERANCE = 0.03  # module-internal stricter target


def _solve_sphere(subdivisions: int, radius: float, potential: float, conductivity: float):
    mesh = icosphere(int(subdivisions), radius=float(radius))
    panels = np.arange(mesh.panel_count, dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    result = solve_cathodic_protection(
        mesh,
        empty,
        empty,
        panels,
        protection_potential=float(potential),
        anode_potential=float(potential),
        conductivity=float(conductivity),
    )
    return mesh, result


def convergence_block(
    subdivisions: Sequence[int], radius: float, potential: float, conductivity: float
) -> dict[str, Any]:
    report = verify_conducting_sphere(
        subdivisions=tuple(int(s) for s in subdivisions),
        radius=radius,
        potential=potential,
        conductivity=conductivity,
    )
    levels = []
    for level in report["levels"]:
        levels.append({
            "subdivisions": level["subdivisions"],
            "n_panels": level["n_panels"],
            "h_m": float(4.0 * math.pi * radius ** 2 / max(level["n_panels"], 1)) ** 0.5,
            "current_A": level["current_A"],
            "current_rel_error": level["current_rel_error"],
            "max_potential_rel_error": level["max_potential_rel_error"],
            "relative_residual": level["relative_residual"],
        })
    return {
        "exact_current_A": report["exact_current_A"],
        "exact_spreading_resistance_ohm": report["exact_spreading_resistance_ohm"],
        "unscaled_current_error_percent": report["unscaled_current_error_percent"],
        "levels": levels,
        "observed_orders": report["observed_orders"],
        "ledger_tolerance": LEDGER_TOLERANCE,
        "internal_tolerance": INTERNAL_TOLERANCE,
        "pass_ledger": bool(report["finest_relative_current_error"] < LEDGER_TOLERANCE),
        "pass_internal": bool(
            report["finest_relative_current_error"] < INTERNAL_TOLERANCE
            and report["finest_max_relative_potential_error"] < INTERNAL_TOLERANCE
        ),
    }


def routes_block(subdivisions: int, radius: float, potential: float, conductivity: float) -> dict[str, Any]:
    routes = verify_conducting_sphere_routes(
        subdivisions=int(subdivisions),
        radius=radius,
        potential=potential,
        conductivity=conductivity,
    )
    return routes


def scaling_block(
    points: Sequence[tuple[float, float, float]],
    subdivisions: int,
) -> list[dict[str, Any]]:
    """Non-dimensional scaling: ``I / (4 pi sigma a phi) == 1`` for every point."""
    rows = []
    for conductivity, radius, potential in points:
        mesh, result = _solve_sphere(subdivisions, radius, potential, conductivity)
        exact = 4.0 * math.pi * conductivity * radius * potential
        numeric = float(result.anode_current_total)
        rows.append({
            "conductivity_S_m": float(conductivity),
            "radius_m": float(radius),
            "potential_V": float(potential),
            "n_panels": int(mesh.panel_count),
            "nondim_current": float(numeric / exact),          # -> 1
            "nondim_current_error": float(abs(numeric / exact - 1.0)),
            "spreading_resistance_numeric_ohm": float(-potential / numeric),
            "spreading_resistance_exact_ohm": float(1.0 / (4.0 * math.pi * conductivity * radius)),
            "within_ledger_tolerance": bool(abs(numeric / exact - 1.0) < LEDGER_TOLERANCE),
        })
    return rows


def decay_block(radius: float, potential: float, conductivity: float) -> dict[str, Any]:
    """Exterior potential decay ``phi(r) = phi_a a / r`` at two resolutions."""
    radii = np.array([1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    curves = {}
    for subdivisions in (1, 2):
        mesh, result = _solve_sphere(subdivisions, radius, potential, conductivity)
        directions = fibonacci_sphere_directions(16)
        phi = np.array([
            float(np.mean(evaluate_exterior_potential(mesh, result, float(r) * directions)))
            for r in radii
        ])
        exact = potential * radius / radii
        curves[str(subdivisions)] = {
            "n_panels": int(mesh.panel_count),
            "phi_numeric": phi.tolist(),
            "phi_exact": exact.tolist(),
            "rel_error": np.abs(phi - exact) / np.abs(exact),
        }
    return {
        "radii_m": radii.tolist(),
        "potential_V": float(potential),
        "radius_m": float(radius),
        "curves": curves,
    }


def make_figures(results: dict[str, Any], outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # --- Fig 1: convergence of the surface current error -------------------
    levels = results["convergence"]["levels"]
    n = np.array([l["n_panels"] for l in levels], dtype=float)
    err = np.array([l["current_rel_error"] for l in levels], dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(n, err, "o-", label="BEM surface current vs $4\\pi\\sigma a\\phi_a$")
    # second-order reference through the finest point
    ref = err[-1] * (n / n[-1]) ** -1.0
    ax.loglog(n, ref, "k--", label="slope $-1$ in $N$ (= 2nd order in $h$)")
    for xi, yi, lab in zip(n, err, [str(l["subdivisions"]) for l in levels]):
        ax.annotate(f"sub{lab}\n{yi:.2f}%", (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=7)
    ax.axhline(LEDGER_TOLERANCE * 100.0, color="tab:red", ls=":", label="ledger 5 %")
    ax.axhline(INTERNAL_TOLERANCE * 100.0, color="tab:orange", ls=":", label="internal 3 %")
    ax.set_xlabel("panels N")
    ax.set_ylabel("relative current error [%]")
    ax.set_title("CASE-BEM-CATHODIC-SPHERE: current convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = outdir / "convergence_current.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    written.append(str(path))

    # --- Fig 2: exterior decay -------------------------------------------
    decay = results["decay"]
    radii = np.array(decay["radii_m"])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(radii, np.array(decay["curves"]["2"]["phi_exact"]) * 1000.0, "k-",
              label="exact $\\phi_a a/r$")
    for key, style in (("1", "o--"), ("2", "s--")):
        curve = decay["curves"][key]
        ax.loglog(radii, np.array(curve["phi_numeric"]) * 1000.0, style,
                  label=f"BEM sub={key} ({curve['n_panels']} panels)")
    ax.set_xlabel("r [m]  (a = %.1f m)" % decay["radius_m"])
    ax.set_ylabel("$\\phi$ [mV]")
    ax.set_title("exterior potential decay, $\\phi_a$ = %.1f V" % decay["potential_V"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = outdir / "exterior_decay.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    written.append(str(path))

    # --- Fig 3: errors vs tolerances (levels + scaling points) -----------
    scale = results["scaling"]
    labels = [f"sub{l['subdivisions']}\nN={l['n_panels']}" for l in levels]
    vals = [100.0 * l["current_rel_error"] for l in levels]
    for row in scale:
        labels.append("σ=%.1f\na=%.2f\nφ=%.1f" % (
            row["conductivity_S_m"], row["radius_m"], row["potential_V"]))
        vals.append(100.0 * row["nondim_current_error"])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = ["tab:blue"] * len(levels) + ["tab:green"] * len(scale)
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(LEDGER_TOLERANCE * 100.0, color="tab:red", ls=":", label="ledger 5 %")
    ax.axhline(INTERNAL_TOLERANCE * 100.0, color="tab:orange", ls=":", label="internal 3 %")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relative error [%]")
    ax.set_title("convergence levels (blue) and non-dimensional scaling points (green)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, which="major", alpha=0.3)
    path = outdir / "error_budget.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subdiv", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--potential", type=float, default=-1.0)
    parser.add_argument("--conductivity", type=float, default=SEAWATER_CONDUCTIVITY)
    parser.add_argument("--scaling-subdiv", type=int, default=2)
    parser.add_argument("--routes-subdiv", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/cathodic_sphere"))
    parser.add_argument(
        "--figures-only", action="store_true",
        help="re-render figures from an existing results.json (no recomputation)",
    )
    args = parser.parse_args(argv)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        results = json.loads((outdir / "results.json").read_text())
        print("figures:", make_figures(results, outdir))
        print("verdict:", json.dumps(results["verdict"]))
        return 0 if results["verdict"]["overall_pass"] else 1

    convergence = convergence_block(
        args.subdiv, args.radius, args.potential, args.conductivity
    )
    print("convergence:", [
        "N=%d err=%.3f%%" % (l["n_panels"], 100.0 * l["current_rel_error"])
        for l in convergence["levels"]
    ], "orders=", ["%.2f" % o for o in convergence["observed_orders"]])

    routes = [
        routes_block(s, args.radius, args.potential, args.conductivity)
        for s in args.routes_subdiv
    ]
    for row in routes:
        print("routes sub=%d: surface %.4f%% far-field %.4f%% conservation %.2e" % (
            row["subdivisions"], 100.0 * row["surface_rel_error"],
            100.0 * row["far_field_rel_error"],
            row["far_field"]["far_field_vs_surface_rel_error"],
        ))

    points = [
        (args.conductivity, args.radius, args.potential),
        (2.0, args.radius, args.potential),
        (args.conductivity, 0.5 * args.radius, args.potential),
        (args.conductivity, args.radius, -2.0),
    ]
    scaling = scaling_block(points, args.scaling_subdiv)
    for row in scaling:
        print("scaling σ=%.2f a=%.2f φ=%.2f -> I/(4πσaφ)=%.6f" % (
            row["conductivity_S_m"], row["radius_m"], row["potential_V"],
            row["nondim_current"]))

    decay = decay_block(args.radius, args.potential, args.conductivity)

    results: dict[str, Any] = {
        "case": "CASE-BEM-CATHODIC-SPHERE",
        "reference": "DNV-RP-B401 isolated-anode spreading resistance R = 1/(4 pi sigma a), i.e. I = 4 pi sigma a phi_a",
        "ledger_tolerance": LEDGER_TOLERANCE,
        "internal_tolerance": INTERNAL_TOLERANCE,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "convergence": convergence,
        "routes": routes,
        "scaling": scaling,
        "decay": {
            "radii_m": decay["radii_m"],
            "radius_m": decay["radius_m"],
            "potential_V": decay["potential_V"],
            "curves": {
                key: {
                    "n_panels": curve["n_panels"],
                    "phi_numeric": curve["phi_numeric"],
                    "phi_exact": curve["phi_exact"],
                    "rel_error": np.asarray(curve["rel_error"]).tolist(),
                }
                for key, curve in decay["curves"].items()
            },
        },
    }
    results["verdict"] = {
        "finest_level_error_percent": 100.0 * convergence["levels"][-1]["current_rel_error"],
        "ledger_pass": convergence["pass_ledger"],
        "internal_pass": convergence["pass_internal"],
        "routes_pass": all(r["pass"] for r in routes),
        "scaling_pass": all(r["within_ledger_tolerance"] for r in scaling),
        "overall_pass": bool(
            convergence["pass_ledger"]
            and all(r["pass"] for r in routes)
            and all(r["within_ledger_tolerance"] for r in scaling)
        ),
    }

    results_path = outdir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", results_path)

    figures = make_figures(results, outdir)
    print("figures:", figures)
    print("verdict:", json.dumps(results["verdict"]))
    return 0 if results["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())