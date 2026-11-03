"""Error budget for the Ursell submerged-cylinder radiation benchmark.

``docs/marine_bem_benchmarks.md`` states that at ``ka = 0.6``, ``a/f = 0.5``
the 128-panel added-mass error against Ursell's multipole solution is
``7.7e-3`` and the damping error ``1.3e-2``.  An independent audit (m238)
remeasured ``8.5e-3`` and recorded a 10.6 % discrepancy -- of the *error*,
not of the physical coefficient -- as an open item.  A quoted accuracy figure
that cannot be reproduced is a verification debt: the number is either tied to
discretisation settings that were never written down, or the reference itself
moved.

This module turns the single quoted number into a transparent error budget.
Every knob that can change the answer is swept on its own, so the sweeps
answer three questions that a single ``--panels 128`` run cannot:

1. **Reference side.**  The multipole system is truncated at ``terms``
   harmonics.  Is the truncation error below the panel error by a safe margin,
   so that the panel error is genuinely a discretisation error and not a
   comparison against a partially converged reference?
2. **Discretisation side.**  The constant-panel indirect formulation has two
   numerical knobs: the panel count ``N`` and the Gauss nodes per panel used
   for the kernel quadrature.  Which of them sets the 128-panel error, and does
   the error converge at the documented first order?
3. **Independent side.**  The method of fundamental solutions has no singular
   quadrature and converges exponentially, so it can be pushed far below the
   panel method.  Agreement between the two, plus the near-field/far-field
   damping identity (Haskind), is what makes the reference comparison credible
   at the ``1e-3`` level.

The documented ``7.7e-3`` is then located (or not) inside the swept knob space:
:func:`reproduce_documented_claim` searches the ``(N, panel_nodes)`` grid for
cells whose added-mass error matches the quoted figure and reports the settings,
so the documentation can be either pinned to explicit settings or corrected.

Run::

    python -m tensorbem.benchmarks.ursell_error_budget --json artifacts/ursell_error_budget.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_coefficients
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import (
    far_field_damping,
    radiation_reference,
    solve_radiation_indirect,
)

RADIUS = 1.0
SUBMERGENCE = 2.0
KA = 0.6
DENSITY = 1025.0
GRAVITY = 9.80665
A_OVER_F = RADIUS / SUBMERGENCE
#: Harmonics used for the reference value.  The truncated system plateaus to
#: machine precision well before this (see :func:`truncation_study`); the point
#: of going this high is to *measure* the plateau instead of assuming it.
CONVERGED_TERMS = 120

#: The accuracy claim in ``docs/marine_bem_benchmarks.md`` that this audit tests.
DOCUMENTED = {
    "panels": 128,
    "ka": KA,
    "a_over_f": A_OVER_F,
    "added_mass_relative_error": 7.7e-3,
    "damping_relative_error": 1.3e-2,
    "source": "docs/marine_bem_benchmarks.md (benchmark 2 summary)",
}


def _wavenumber(ka: float = KA, radius: float = RADIUS) -> float:
    return ka / radius


def _nondimensional_panel(
    panel_count: int,
    *,
    ka: float = KA,
    panel_nodes: int = 24,
    direction: str = "heave",
) -> tuple[float, float]:
    """Dimensionless ``(mu, nu)`` from the panel solve, same convention as Ursell."""
    mesh = submerged_circle_mesh(panel_count, RADIUS, SUBMERGENCE)
    nu = _wavenumber(ka)
    result = solve_radiation_indirect(
        mesh, nu, direction, density=DENSITY, panel_nodes=panel_nodes
    )
    omega = math.sqrt(nu * GRAVITY)
    displaced = DENSITY * math.pi * RADIUS**2
    index = 1 if direction == "heave" else 0
    return (
        float(result.added_mass[index]) / displaced,
        float(result.damping[index]) / (displaced * omega),
    )


def _nondimensional_mfs(
    source_count: int, *, ka: float = KA, direction: str = "heave"
) -> tuple[float, float]:
    mesh = submerged_circle_mesh(128, RADIUS, SUBMERGENCE)
    nu = _wavenumber(ka)
    result = radiation_reference(
        mesh, nu, direction, source_count=source_count, density=DENSITY
    )
    omega = math.sqrt(nu * GRAVITY)
    displaced = DENSITY * math.pi * RADIUS**2
    index = 1 if direction == "heave" else 0
    return (
        float(result.added_mass[index]) / displaced,
        float(result.damping[index]) / (displaced * omega),
    )


def _relative_error(value: tuple[float, float], reference: tuple[float, float]) -> tuple[float, float]:
    return (
        abs(value[0] - reference[0]) / abs(reference[0]),
        abs(value[1] - reference[1]) / abs(reference[1]),
    )


def _panel_error_task(task: tuple[int, int]) -> dict[str, float]:
    """One ``(panel count, nodes per panel)`` solve, for the process pool."""
    panel_count, panel_nodes = task
    mu, nu = _nondimensional_panel(panel_count, panel_nodes=panel_nodes)
    reference = ursell_cylinder_coefficients(KA, A_OVER_F, CONVERGED_TERMS)
    error = _relative_error((mu, nu), reference)
    return {
        "panels": panel_count,
        "panel_nodes": panel_nodes,
        "mu": mu,
        "nu": nu,
        "added_mass_relative_error": error[0],
        "damping_relative_error": error[1],
    }


def _panel_sweep(
    tasks: list[tuple[int, int]], workers: int | None = None
) -> list[dict[str, float]]:
    """Evaluate independent panel solves in parallel.

    A dense ``N x N`` kernel assembly with Gauss nodes per panel costs
    ``O(N**2 * nodes)`` special-function evaluations, so the full sweep is a
    minute of wall clock serial and a few seconds spread over worker processes.
    """
    from concurrent.futures import ProcessPoolExecutor

    ordered = sorted(set(tasks))
    if len(ordered) <= 1 or (workers is not None and workers <= 1):
        return [_panel_error_task(task) for task in ordered]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_panel_error_task, ordered))


# --------------------------------------------------------------------------
# 1. Reference side: multipole truncation
# --------------------------------------------------------------------------


def truncation_study(
    ka_values: tuple[float, ...] = (0.2, 0.6, 1.0, 1.5, 2.0),
    terms_list: tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40),
    converged_terms: int = CONVERGED_TERMS,
) -> dict[str, object]:
    """Show how fast the truncated multipole series settles.

    The converged value is computed with ``converged_terms`` harmonics, far
    beyond the library default of 40, so that the residual change at the
    default is a measured quantity rather than an assumption.
    """
    rows = []
    for ka in ka_values:
        converged = ursell_cylinder_coefficients(ka, A_OVER_F, converged_terms)
        previous = None
        monotone = True
        for terms in terms_list:
            mu, nu = ursell_cylinder_coefficients(ka, A_OVER_F, terms)
            error = _relative_error((mu, nu), converged)
            if previous is not None and (error[0] > previous[0] * 1.0000001
                                         or error[1] > previous[1] * 1.0000001):
                monotone = False
            previous = error
            rows.append(
                {
                    "ka": ka,
                    "terms": terms,
                    "mu": mu,
                    "nu": nu,
                    "added_mass_relative_error": error[0],
                    "damping_relative_error": error[1],
                }
            )
    default_rows = [row for row in rows if row["terms"] == 40]
    return {
        "converged_terms": converged_terms,
        "monotone_convergence": monotone,
        "rows": rows,
        "default_terms_relative_error": {
            "max_added_mass": max(r["added_mass_relative_error"] for r in default_rows),
            "max_damping": max(r["damping_relative_error"] for r in default_rows),
        },
        "converged_values": {
            str(ka): ursell_cylinder_coefficients(ka, A_OVER_F, converged_terms)
            for ka in ka_values
        },
    }


# --------------------------------------------------------------------------
# 2. Discretisation side: panel nodes and panel count
# --------------------------------------------------------------------------


def panel_node_study(
    panel_count: int = 128,
    node_counts: tuple[int, ...] = (4, 6, 8, 12, 16, 24, 32, 48),
    workers: int | None = None,
) -> dict[str, object]:
    """Fix the panel count and vary only the per-panel kernel quadrature."""
    rows = [
        row for row in _panel_sweep([(panel_count, nodes) for nodes in node_counts], workers)
    ]
    for row in rows:
        del row["panels"]
    best = min(rows, key=lambda row: row["added_mass_relative_error"])
    saturated = [row for row in rows if row["panel_nodes"] >= 24]
    spread = (
        max(r["added_mass_relative_error"] for r in saturated)
        - min(r["added_mass_relative_error"] for r in saturated)
    )
    return {
        "panel_count": panel_count,
        "rows": sorted(rows, key=lambda row: row["panel_nodes"]),
        "best_panel_nodes": best["panel_nodes"],
        "saturated_added_mass_error_spread": spread,
    }


def panel_convergence(
    panel_counts: tuple[int, ...] = (24, 32, 48, 64, 96, 128, 192, 256, 384, 512),
    panel_nodes: int = 24,
    workers: int | None = None,
) -> dict[str, object]:
    """Refine the mesh and estimate the observed order of convergence."""
    rows = _panel_sweep([(count, panel_nodes) for count in panel_counts], workers)
    orders = []
    for previous, current in zip(rows, rows[1:]):
        ratio = current["panels"] / previous["panels"]
        for key, target in (
            ("added_mass_relative_error", "added_mass_observed_order"),
            ("damping_relative_error", "damping_observed_order"),
        ):
            if previous[key] > 0.0 and current[key] > 0.0:
                order = math.log(previous[key] / current[key]) / math.log(ratio)
                current[target] = order
                orders.append((order, current["panels"]))
    tail = [order for order, count in orders if count >= 128]
    return {
        "panel_nodes": panel_nodes,
        "rows": rows,
        "observed_order_tail_mean": float(np.mean(tail)) if tail else None,
        "documented_panels_row": next(
            (row for row in rows if row["panels"] == DOCUMENTED["panels"]), None
        ),
    }


# --------------------------------------------------------------------------
# 3. Independent side: MFS reference and the radiation condition
# --------------------------------------------------------------------------


def mfs_convergence(
    source_counts: tuple[int, ...] = (4, 6, 8, 12, 16, 24, 32, 48, 64),
) -> dict[str, object]:
    reference = ursell_cylinder_coefficients(KA, A_OVER_F, CONVERGED_TERMS)
    rows = []
    for count in source_counts:
        mu, nu = _nondimensional_mfs(count)
        error = _relative_error((mu, nu), reference)
        rows.append(
            {
                "sources": count,
                "mu": mu,
                "nu": nu,
                "added_mass_relative_error": error[0],
                "damping_relative_error": error[1],
            }
        )
    return {
        "rows": rows,
        "best": min(rows, key=lambda row: row["added_mass_relative_error"]),
    }


def energy_identity(
    panel_counts: tuple[int, ...] = (32, 64, 128, 256),
    panel_nodes: int = 24,
) -> dict[str, object]:
    """Near-field (pressure) damping against far-field (wave amplitude) damping."""
    rows = []
    nu = _wavenumber()
    omega = math.sqrt(nu * GRAVITY)
    displaced = DENSITY * math.pi * RADIUS**2
    for count in panel_counts:
        mesh = submerged_circle_mesh(count, RADIUS, SUBMERGENCE)
        result = solve_radiation_indirect(
            mesh, nu, "heave", density=DENSITY, panel_nodes=panel_nodes
        )
        far = far_field_damping(
            mesh, result.potential, result.normal_velocity, nu, DENSITY,
            gravity=GRAVITY,
        )
        near = float(result.damping[1])
        rows.append(
            {
                "panels": count,
                "near_field_nu": near / (displaced * omega),
                "far_field_nu": far / (displaced * omega),
                "relative_difference": abs(near - far) / abs(far),
            }
        )
    return {"rows": rows, "max_relative_difference": max(r["relative_difference"] for r in rows)}


# --------------------------------------------------------------------------
# 4. Physics limits that need no panel solve
# --------------------------------------------------------------------------


def deep_submergence_limit(
    submergences: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0),
    ka: float = KA,
) -> dict[str, object]:
    """As ``f/a -> infinity`` the added mass tends to the displaced mass (``mu -> 1``)."""
    rows = []
    for submergence in submergences:
        mu, nu = ursell_cylinder_coefficients(ka, RADIUS / submergence, CONVERGED_TERMS)
        rows.append(
            {
                "submergence_over_radius": submergence / RADIUS,
                "mu": mu,
                "nu": nu,
                "added_mass_deviation_from_displaced_mass": abs(mu - 1.0),
            }
        )
    return {"ka": ka, "rows": rows, "decays_to_unity": rows[-1]["added_mass_deviation_from_displaced_mass"]
            < rows[0]["added_mass_deviation_from_displaced_mass"]}


def heave_sway_symmetry(panel_count: int = 128, panel_nodes: int = 24) -> dict[str, object]:
    """In infinite depth the two motions share one added mass and one damping."""
    heave = _nondimensional_panel(panel_count, panel_nodes=panel_nodes, direction="heave")
    sway = _nondimensional_panel(panel_count, panel_nodes=panel_nodes, direction="sway")
    return {
        "panels": panel_count,
        "panel_nodes": panel_nodes,
        "heave": heave,
        "sway": sway,
        "added_mass_relative_difference": abs(heave[0] - sway[0]) / heave[0],
        "damping_relative_difference": abs(heave[1] - sway[1]) / heave[1],
    }


# --------------------------------------------------------------------------
# 5. Where did 7.7e-3 come from?
# --------------------------------------------------------------------------


def reproduce_documented_claim(
    panel_counts: tuple[int, ...] = (48, 64, 96, 128, 192),
    node_counts: tuple[int, ...] = (12, 16, 24, 32),
    tolerance: float = 0.02,
    workers: int | None = None,
) -> dict[str, object]:
    """Search the ``(N, panel_nodes)`` grid for the documented 7.7e-3 error.

    A cell matches when its added-mass relative error is within ``tolerance``
    (relative) of the documented value.  If nothing matches, the documented
    figure is not reproducible from any plausible setting of this formulation
    and the documentation must be corrected rather than explained.
    """
    target = DOCUMENTED["added_mass_relative_error"]
    cells = _panel_sweep(
        [(count, nodes) for count in panel_counts for nodes in node_counts], workers
    )
    for cell in cells:
        cell["relative_deviation_from_documented"] = (
            abs(cell["added_mass_relative_error"] - target) / target
        )
    matches = sorted(
        (cell for cell in cells if cell["relative_deviation_from_documented"] <= tolerance),
        key=lambda cell: cell["relative_deviation_from_documented"],
    )
    best = min(cells, key=lambda cell: cell["relative_deviation_from_documented"])
    at_documented = next(
        (cell for cell in cells
         if cell["panels"] == DOCUMENTED["panels"] and cell["panel_nodes"] == 24),
        None,
    )
    return {
        "target_added_mass_error": target,
        "target_damping_error": DOCUMENTED["damping_relative_error"],
        "tolerance": tolerance,
        "matches": matches,
        "best_cell": best,
        "cells_at_documented_panels_default_nodes": at_documented,
        "documented_claim_reproduced": bool(matches),
        "grid": {"panel_counts": list(panel_counts), "node_counts": list(node_counts)},
        "all_cells": cells,
    }


def convergence_extrapolation(panel_convergence_rows: list[dict[str, object]]) -> dict[str, object]:
    """Richardson-style estimate of the continuum value from the two finest meshes.

    With ``e ~ C N^(-p)`` the continuum limit is ``f_inf = f + (f_N - f_2N)/(2^p - 1)``;
    the residual to the analytic reference after extrapolation shows whether the
    first-order model of the error is adequate.
    """
    ordered = sorted(panel_convergence_rows, key=lambda row: row["panels"])
    if len(ordered) < 3:
        return {"applicable": False, "reason": "need at least three meshes"}
    coarse, fine = ordered[-2], ordered[-1]
    ratio = fine["panels"] / coarse["panels"]
    reference = ursell_cylinder_coefficients(KA, A_OVER_F, CONVERGED_TERMS)
    result = {"applicable": True, "ratio": ratio}
    for key, ref_key, name in (
        ("mu", "mu", "added_mass"),
        ("nu", "nu", "damping"),
    ):
        order = math.log(
            abs(coarse[key] - reference[0 if ref_key == "mu" else 1])
            / abs(fine[key] - reference[0 if ref_key == "mu" else 1])
        ) / math.log(ratio)
        extrapolated = fine[key] + (fine[key] - coarse[key]) / (ratio**order - 1.0)
        exact = reference[0 if ref_key == "mu" else 1]
        result[name] = {
            "observed_order": order,
            "fine_value": fine[key],
            "extrapolated": extrapolated,
            "reference": exact,
            "extrapolation_relative_error": abs(extrapolated - exact) / abs(exact),
        }
    return result


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(*, full: bool = True, workers: int | None = None) -> dict[str, object]:
    """Assemble the error budget.  ``full=False`` runs the cheap subset."""
    panel_counts = (
        (24, 32, 48, 64, 96, 128, 192, 256) if full else (48, 64, 96, 128)
    )
    convergence = panel_convergence(panel_counts, workers=workers)
    budget = {
        "benchmark": "ursell_submerged_circular_cylinder_error_budget",
        "parameters": {
            "radius": RADIUS,
            "submergence": SUBMERGENCE,
            "a_over_f": A_OVER_F,
            "ka": KA,
            "density": DENSITY,
            "gravity": GRAVITY,
        },
        "documented_claim": DOCUMENTED,
        "truncation": truncation_study(),
        "panel_nodes": panel_node_study(
            panel_count=128,
            node_counts=(8, 16, 24, 32) if full else (8, 16, 24, 32),
            workers=workers,
        ),
        "panel_convergence": convergence,
        "richardson": convergence_extrapolation(convergence["rows"]),
        "mfs_convergence": mfs_convergence(),
        "energy_identity": energy_identity(),
        "deep_submergence": deep_submergence_limit(),
        "heave_sway_symmetry": heave_sway_symmetry(),
    }
    if full:
        budget["documented_claim_reproduction"] = reproduce_documented_claim(workers=workers)
    return budget


def plot(budget: dict[str, object], path: Path) -> None:
    """Two-panel figure: panel/MFS convergence and knob sensitivity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    rows = budget["panel_convergence"]["rows"]
    panels = [row["panels"] for row in rows]
    axes[0].loglog(panels, [row["added_mass_relative_error"] for row in rows], "o-",
                   label="panel added mass")
    axes[0].loglog(panels, [row["damping_relative_error"] for row in rows], "s--",
                   label="panel damping")
    mfs_rows = budget["mfs_convergence"]["rows"]
    axes[0].loglog([row["sources"] for row in mfs_rows],
                   [row["added_mass_relative_error"] for row in mfs_rows], "^-",
                   label="MFS added mass")
    documented = budget["documented_claim"]
    axes[0].axhline(documented["added_mass_relative_error"], color="0.4", ls=":",
                    label=f"documented {documented['added_mass_relative_error']:.1e}")
    measured = budget["panel_convergence"]["documented_panels_row"]
    if measured is not None:
        axes[0].plot([measured["panels"]], [measured["added_mass_relative_error"]], "k*", ms=12,
                     label=f"measured @128 = {measured['added_mass_relative_error']:.2e}")
    axes[0].set_xlabel("panels $N$  /  MFS sources")
    axes[0].set_ylabel("relative error vs Ursell multipole")
    axes[0].set_title("discretisation convergence ($ka=0.6$, $a/f=0.5$)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    node_rows = budget["panel_nodes"]["rows"]
    axes[1].semilogy([row["panel_nodes"] for row in node_rows],
                     [row["added_mass_relative_error"] for row in node_rows], "o-",
                     label="added mass")
    axes[1].semilogy([row["panel_nodes"] for row in node_rows],
                     [row["damping_relative_error"] for row in node_rows], "s--",
                     label="damping")
    axes[1].axhline(documented["added_mass_relative_error"], color="0.4", ls=":",
                    label="documented")
    axes[1].set_xlabel("Gauss nodes per panel ($N=128$)")
    axes[1].set_ylabel("relative error vs Ursell multipole")
    axes[1].set_title("kernel quadrature sensitivity at $N=128$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def render_markdown(budget: dict[str, object]) -> str:
    """Render the budget as markdown tables straight from the JSON.

    Everything here is generated from the measured structure, so the document
    cannot drift from the artifact the way the original 7.7e-3 note did.
    """
    params = budget["parameters"]
    documented = budget["documented_claim"]
    lines: list[str] = []
    add = lines.append
    add("# Ursell submerged-cylinder radiation: error budget audit")
    add("")
    add("Generated by `python -m tensorbem.benchmarks.ursell_error_budget`; every number")
    add("below is read from `artifacts/ursell_error_budget.json`, not transcribed.")
    add("")
    add("## 1. Case and parameters")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    for key, value in params.items():
        add(f"| {key} | {value} |")
    add("")
    add("## 2. The claim under audit")
    add("")
    add("| quantity | documented |")
    add("|---|---|")
    add(f"| panels | {documented['panels']} |")
    add(f"| added-mass relative error | {documented['added_mass_relative_error']:.3e} |")
    add(f"| damping relative error | {documented['damping_relative_error']:.3e} |")
    add(f"| source | {documented['source']} |")
    add("")
    add("## 3. Reference side: multipole truncation")
    add("")
    add("| ka | terms | mu | nu | rel. err. added mass | rel. err. damping |")
    add("|---|---|---|---|---|---|")
    for row in budget["truncation"]["rows"]:
        add(
            f"| {row['ka']} | {row['terms']} | {row['mu']:.12f} | {row['nu']:.12f} "
            f"| {row['added_mass_relative_error']:.2e} | {row['damping_relative_error']:.2e} |"
        )
    truncation = budget["truncation"]["default_terms_relative_error"]
    add("")
    add(
        f"Worst truncation error at the default 40 harmonics: added mass "
        f"{truncation['max_added_mass']:.2e}, damping {truncation['max_damping']:.2e} "
        f"(reference computed with {budget['truncation']['converged_terms']} harmonics)."
    )
    add("")
    add("## 4. Discretisation side: panel quadrature at fixed mesh")
    add("")
    add("| Gauss nodes per panel | mu | nu | rel. err. added mass | rel. err. damping |")
    add("|---|---|---|---|---|")
    for row in budget["panel_nodes"]["rows"]:
        add(
            f"| {row['panel_nodes']} | {row['mu']:.12f} | {row['nu']:.12f} "
            f"| {row['added_mass_relative_error']:.3e} | {row['damping_relative_error']:.3e} |"
        )
    add("")
    add(
        f"Saturated (>= 24 nodes) added-mass error spread: "
        f"{budget['panel_nodes']['saturated_added_mass_error_spread']:.2e}."
    )
    add("")
    add("## 5. Mesh refinement and observed order")
    add("")
    add("| panels | mu | nu | rel. err. added mass | obs. order | rel. err. damping | obs. order |")
    add("|---|---|---|---|---|---|---|")
    for row in budget["panel_convergence"]["rows"]:
        add(
            f"| {row['panels']} | {row['mu']:.12f} | {row['nu']:.12f} "
            f"| {row['added_mass_relative_error']:.3e} "
            f"| {row.get('added_mass_observed_order', float('nan')):.2f} "
            f"| {row['damping_relative_error']:.3e} "
            f"| {row.get('damping_observed_order', float('nan')):.2f} |"
        )
    add("")
    order = budget["panel_convergence"]["observed_order_tail_mean"]
    add(f"Mean observed order over the tail (N >= 128): {order:.2f}.")
    richardson = budget["richardson"]
    if richardson.get("applicable"):
        add("")
        add("Richardson extrapolation from the two finest meshes:")
        add("")
        add("| quantity | fine value | extrapolated | reference | extrapolated rel. error |")
        add("|---|---|---|---|---|")
        for name in ("added_mass", "damping"):
            entry = richardson[name]
            add(
                f"| {name} | {entry['fine_value']:.12f} | {entry['extrapolated']:.12f} "
                f"| {entry['reference']:.12f} | {entry['extrapolation_relative_error']:.2e} |"
            )
    add("")
    add("## 6. Independent side: MFS reference and radiation condition")
    add("")
    add("| MFS sources | mu | nu | rel. err. added mass | rel. err. damping |")
    add("|---|---|---|---|---|")
    for row in budget["mfs_convergence"]["rows"]:
        add(
            f"| {row['sources']} | {row['mu']:.12f} | {row['nu']:.12f} "
            f"| {row['added_mass_relative_error']:.3e} | {row['damping_relative_error']:.3e} |"
        )
    add("")
    add("| panels | near-field nu | far-field nu | relative difference |")
    add("|---|---|---|---|")
    for row in budget["energy_identity"]["rows"]:
        add(
            f"| {row['panels']} | {row['near_field_nu']:.9f} | {row['far_field_nu']:.9f} "
            f"| {row['relative_difference']:.2e} |"
        )
    add("")
    add("## 7. Analytical limits")
    add("")
    add("| f/a | mu | nu | |mu - 1| |")
    add("|---|---|---|---|")
    for row in budget["deep_submergence"]["rows"]:
        add(
            f"| {row['submergence_over_radius']} | {row['mu']:.12f} | {row['nu']:.12f} "
            f"| {row['added_mass_deviation_from_displaced_mass']:.2e} |"
        )
    symmetry = budget["heave_sway_symmetry"]
    add("")
    add(
        f"Heave/sway symmetry at {symmetry['panels']} panels: added mass "
        f"{symmetry['added_mass_relative_difference']:.2e}, damping "
        f"{symmetry['damping_relative_difference']:.2e}."
    )
    reproduction = budget.get("documented_claim_reproduction")
    if reproduction:
        add("")
        add("## 8. Can the documented 7.7e-3 be reproduced?")
        add("")
        grid = reproduction["grid"]
        add(
            f"Searched N in {grid['panel_counts']} and nodes in {grid['node_counts']} "
            f"(tolerance {reproduction['tolerance']:.0%} of the documented error)."
        )
        add("")
        if reproduction["documented_claim_reproduced"]:
            add("Matching settings (panels, nodes, added-mass error, damping error):")
            add("")
            add("| panels | nodes | rel. err. added mass | rel. err. damping |")
            add("|---|---|---|---|")
            for cell in reproduction["matches"][:15]:
                add(
                    f"| {cell['panels']} | {cell['panel_nodes']} "
                    f"| {cell['added_mass_relative_error']:.3e} "
                    f"| {cell['damping_relative_error']:.3e} |"
                )
        else:
            best = reproduction["best_cell"]
            add(
                "**No setting in the searched grid reproduces the documented figure.** "
                f"The closest cell is N={best['panels']}, {best['panel_nodes']} nodes, "
                f"added-mass error {best['added_mass_relative_error']:.3e} "
                f"({best['relative_deviation_from_documented']:.1%} from the documented value)."
            )
        at_documented = reproduction["cells_at_documented_panels_default_nodes"]
        if at_documented:
            add("")
            add(
                f"At the documented setting (N={at_documented['panels']}, "
                f"{at_documented['panel_nodes']} nodes) the measured errors are "
                f"{at_documented['added_mass_relative_error']:.3e} (added mass) and "
                f"{at_documented['damping_relative_error']:.3e} (damping)."
            )
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the budget as JSON here")
    parser.add_argument("--figure", type=Path, help="write the convergence figure here")
    parser.add_argument("--markdown", type=Path, help="write a generated markdown report here")
    parser.add_argument("--quick", action="store_true", help="cheap subset for smoke tests")
    parser.add_argument("--workers", type=int, default=None, help="process pool size")
    args = parser.parse_args()

    budget = run(full=not args.quick, workers=args.workers)
    rendered = json.dumps(budget, indent=2, sort_keys=True, allow_nan=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    if args.figure:
        plot(budget, args.figure)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(budget) + "\n", encoding="utf-8")
    if not args.json and not args.figure and not args.markdown:
        print(rendered)


if __name__ == "__main__":
    main()