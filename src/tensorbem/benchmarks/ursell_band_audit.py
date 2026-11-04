"""Accuracy envelope of the Ursell radiation benchmark over the ``(ka, a/f)`` band.

Why this module exists
----------------------
:mod:`tensorbem.benchmarks.ursell_error_budget` audits a *single* configuration:
``a = 1`` m, ``f = 2`` m (``a/f = 0.5``), ``ka = 0.6``.  Its own limitations
section records the debt verbatim: "仅审计了一个 ``(ka, a/f)`` 点：``ka >= 1``
时自由液面项更强，收敛行为可能不同，未验证".  The 128-panel accuracy claim that
the documentation quotes for the benchmark is therefore anchored at one point of
a two-parameter space, and the panel count that is adequate there is not
automatically adequate anywhere else: the free-surface part of the Green
function oscillates over a length scale ``1/nu * nu^2 = nu`` in wavenumber, and
the depth ratio ``a/f`` sets how close the body comes to the free surface, i.e.
how strong the singular free-surface term is at the collocation points.

This module measures the error at every node of a grid of ``(ka, a/f)`` and
turns the answer into a decidable statement:

* the **observed order of convergence** at each grid node, from a refinement
  ladder ``N = 32, 64, 128, 256`` at fixed 24 Gauss nodes per panel;
* the **relative error against Ursell's multipole solution** at each ``N``, so
  the panel count needed for a given accuracy is read off the table instead of
  assumed;
* the **residual of the two-point power-law model**.  A two-point Richardson
  extrapolation whose order is *fitted* from the same two errors is algebraically
  exact for any pure power law, so the extrapolated limit agreeing with the
  reference is not by itself independent evidence -- what it measures is the
  **higher-order contamination**: a residual that stays at round-off over the
  whole ladder says the refinement sequence is a pure ``C/N`` law down to
  machine precision, and a residual that grows says the asymptotic regime has
  not been reached.  Quoting both is the honest form;
* the **near-field / far-field damping identity** (Haskind) at every node, which
  checks the radiation condition independently of the reference value;
* the **truncation gap of the reference itself** at every node, so a large
  panel error can never be excused by comparing against a partially converged
  multipole series.

The reference is Ursell's (1949) multipole solution for the submerged circular
cylinder, :func:`tensorbem.analytic.ursell_cylinder_coefficients`.  It shares no
code with the panel assembly, and its truncation is *measured* here (harmonic
count doubled until the coefficients stop moving), not assumed.

Conventions are those of the rest of the repository: ``z`` up, time factor
``exp(-i*omega*t)``, dimensionless coefficients

    mu = A / (rho * pi * a^2),     nu = B / (rho * pi * a^2 * omega),

so that the two numbers are directly comparable with the tabulated Ursell
values and with ``docs/marine_bem_benchmarks.md``.

Run::

    python -m tensorbem.benchmarks.ursell_band_audit --json artifacts/ursell_band_audit.json
    python -m tensorbem.benchmarks.ursell_band_audit --quick --figure /tmp/band.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_coefficients
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import far_field_damping, solve_radiation_indirect

RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
PANEL_NODES = 24

#: Refinement ladder.  256 panels is the finest mesh that stays inside a
#: minute of dense assembly, and the ladder is a factor two at every step so
#: the observed order needs no fitting.
DEFAULT_PANELS = (32, 64, 128, 256)

#: Free-surface parameter band.  ``ka`` spans long waves to the short-wave end
#: where the free-surface term dominates the kernel; ``a/f`` spans a deeply
#: submerged cylinder to one whose crown is a third of a radius below the
#: surface (``f = 4a/3``), which is as shallow as a fully submerged circular
#: section can go in this formulation.
KA_GRID = (0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5)
A_OVER_F_GRID = (0.25, 0.5, 0.75)

#: Multipole truncation is doubled from this seed until the dimensionless
#: coefficients stop moving by more than ``TRUNCATION_TOL``.
TRUNCATION_SEED = 20
TRUNCATION_TOL = 1.0e-12
TRUNCATION_MAX = 640

#: A damping coefficient is only compared *relatively* where it is physically
#: resolvable.  A deeply submerged cylinder in short waves radiates almost
#: nothing (``nu -> 0`` as ``ka`` grows at ``a/f = 0.25``), so there the relative
#: error divides by a number smaller than the panel method's own absolute noise
#: and carries no information.  A node is called *resolved* when the reference
#: damping is at least ``DAMPING_RESOLVED_FLOOR`` (dimensionless, i.e. of the
#: displaced-mass scale) **and** at least ``DAMPING_RESOLVED_MARGIN`` times the
#: measured absolute error of the finest panel solution.
DAMPING_RESOLVED_FLOOR = 1.0e-3
DAMPING_RESOLVED_MARGIN = 10.0

#: The accuracy figure the documentation quotes at ``(ka=0.6, a/f=0.5)``.
DOCUMENTED_PANELS = 128
DOCUMENTED_ADDED_MASS_ERROR = 8.5e-3
DOCUMENTED_DAMPING_ERROR = 1.34e-2


def submergence(a_over_f: float, radius: float = RADIUS) -> float:
    """Axis depth belonging to a depth ratio ``a/f``."""
    a_over_f = float(a_over_f)
    if not math.isfinite(a_over_f) or a_over_f <= 0.0:
        raise ValueError("a_over_f must be a positive finite number")
    if a_over_f >= 1.0:
        # The cylinder must be fully submerged: the crown is at -f + a.
        raise ValueError("a_over_f must be < 1 for a fully submerged cylinder")
    return float(radius) / a_over_f


def reference_coefficients(
    ka: float,
    a_over_f: float,
    *,
    seed: int = TRUNCATION_SEED,
    tolerance: float = TRUNCATION_TOL,
    max_terms: int = TRUNCATION_MAX,
) -> dict[str, float]:
    """Ursell's dimensionless ``(mu, nu)`` with the truncation *measured*.

    The harmonic count starts at ``seed`` and doubles until both coefficients
    move by less than ``tolerance`` in relative terms.  The returned
    ``truncation_gap`` is that last relative movement, so a panel error can be
    compared against a reference whose own error is known.
    """
    terms = int(seed)
    previous = None
    while True:
        current = ursell_cylinder_coefficients(ka, a_over_f, terms)
        if previous is not None:
            gap = max(
                abs(current[0] - previous[0]) / max(abs(current[0]), 1e-300),
                abs(current[1] - previous[1]) / max(abs(current[1]), 1e-300),
            )
            if gap <= tolerance or terms >= max_terms:
                return {
                    "added_mass": float(current[0]),
                    "damping": float(current[1]),
                    "terms": terms,
                    "truncation_gap": float(gap),
                }
        previous = current
        terms = min(terms * 2, int(max_terms))


def panel_coefficients(
    panel_count: int,
    ka: float,
    a_over_f: float,
    *,
    panel_nodes: int = PANEL_NODES,
    direction: str = "heave",
    radius: float = RADIUS,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, float]:
    """One dimensionless panel solve plus its radiation-condition identity."""
    depth = submergence(a_over_f, radius)
    mesh = submerged_circle_mesh(int(panel_count), radius, depth)
    nu_wave = float(ka) / float(radius)
    result = solve_radiation_indirect(
        mesh, nu_wave, direction, density=density, panel_nodes=panel_nodes
    )
    omega = math.sqrt(nu_wave * gravity)
    displaced = density * math.pi * radius**2
    index = 1 if direction == "heave" else 0
    added_mass = float(result.added_mass[index])
    damping = float(result.damping[index])
    far_field = far_field_damping(
        mesh,
        result.potential,
        result.normal_velocity,
        nu_wave,
        density,
        gravity=gravity,
    )
    return {
        "panels": int(panel_count),
        "added_mass": added_mass / displaced,
        "damping": damping / (displaced * omega),
        "near_over_far_field_damping": float(damping / far_field),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
    }


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def observed_order(coarse: float, fine: float) -> float:
    """Observed order of convergence between two meshes related by a factor 2."""
    if fine <= 0.0 or coarse <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(2.0))


def richardson_limit(coarse: float, fine: float, order: float = 1.0) -> float:
    """Limit of the refinement sequence for a known observed order."""
    if not math.isfinite(order) or order <= 0.0:
        return float("nan")
    return float(fine + (fine - coarse) / (2.0**order - 1.0))


def band_node(
    ka: float,
    a_over_f: float,
    panel_counts=DEFAULT_PANELS,
    *,
    panel_nodes: int = PANEL_NODES,
    reference: dict[str, float] | None = None,
) -> dict[str, object]:
    """Full refinement ladder at one ``(ka, a/f)`` node of the band."""
    reference = reference or reference_coefficients(ka, a_over_f)
    rows = [
        panel_coefficients(int(n), ka, a_over_f, panel_nodes=panel_nodes)
        for n in panel_counts
    ]
    for row in rows:
        row["added_mass_relative_error"] = _relative_error(
            row["added_mass"], reference["added_mass"]
        )
        row["damping_relative_error"] = _relative_error(
            row["damping"], reference["damping"]
        )
        # Absolute error on the displaced-mass scale (*omega for damping).  Well
        # conditioned everywhere, unlike the relative damping error.
        row["added_mass_scaled_error"] = abs(
            row["added_mass"] - reference["added_mass"]
        )
        row["damping_scaled_error"] = abs(row["damping"] - reference["damping"])
    pairs = []
    for coarse_row, fine_row in zip(rows[:-1], rows[1:]):
        pair_order_added_mass = observed_order(
            coarse_row["added_mass_relative_error"],
            fine_row["added_mass_relative_error"],
        )
        pair_order_damping = observed_order(
            coarse_row["damping_relative_error"],
            fine_row["damping_relative_error"],
        )
        pairs.append(
            {
                "coarse_panels": int(coarse_row["panels"]),
                "fine_panels": int(fine_row["panels"]),
                "order_added_mass": pair_order_added_mass,
                "order_damping": pair_order_damping,
                "power_law_residual_added_mass": _relative_error(
                    richardson_limit(
                        coarse_row["added_mass"],
                        fine_row["added_mass"],
                        pair_order_added_mass,
                    ),
                    reference["added_mass"],
                ),
                "power_law_residual_damping": _relative_error(
                    richardson_limit(
                        coarse_row["damping"],
                        fine_row["damping"],
                        pair_order_damping,
                    ),
                    reference["damping"],
                ),
                "error_times_panels_added_mass": float(
                    fine_row["added_mass_relative_error"] * fine_row["panels"]
                ),
                "error_times_panels_damping": float(
                    fine_row["damping_relative_error"] * fine_row["panels"]
                ),
            }
        )
    coarse, fine = rows[-2], rows[-1]
    order_added_mass = observed_order(
        coarse["added_mass_relative_error"], fine["added_mass_relative_error"]
    )
    order_damping = observed_order(
        coarse["damping_relative_error"], fine["damping_relative_error"]
    )
    limit_added_mass = richardson_limit(
        coarse["added_mass"], fine["added_mass"], order_added_mass
    )
    limit_damping = richardson_limit(
        coarse["damping"], fine["damping"], order_damping
    )
    finest_documented = next(
        (row for row in rows if row["panels"] == DOCUMENTED_PANELS), fine
    )
    reference_damping = float(reference["damping"])
    resolved = bool(
        reference_damping >= DAMPING_RESOLVED_FLOOR
        and reference_damping >= DAMPING_RESOLVED_MARGIN * fine["damping_scaled_error"]
    )
    return {
        "ka": float(ka),
        "a_over_f": float(a_over_f),
        "submergence": submergence(a_over_f),
        "reference": reference,
        "rows": rows,
        "pairs": pairs,
        "observed_order_added_mass": order_added_mass,
        "observed_order_damping": order_damping,
        "richardson_limit_added_mass": limit_added_mass,
        "richardson_limit_damping": limit_damping,
        "richardson_gap_added_mass": _relative_error(
            limit_added_mass, reference["added_mass"]
        ),
        "richardson_gap_damping": _relative_error(
            limit_damping, reference["damping"]
        ),
        "error_at_documented_panels": {
            "panels": int(finest_documented["panels"]),
            "added_mass": float(finest_documented["added_mass_relative_error"]),
            "damping": float(finest_documented["damping_relative_error"]),
            "damping_scaled": float(finest_documented["damping_scaled_error"]),
        },
        "reference_damping": reference_damping,
        "damping_resolved": resolved,
        "max_near_over_far_field_damping_deviation": max(
            abs(float(row["near_over_far_field_damping"]) - 1.0) for row in rows
        ),
        "max_condition_number": max(float(row["condition_number"]) for row in rows),
        "max_damping_scaled_error": max(
            float(row["damping_scaled_error"]) for row in rows
        ),
        "max_added_mass_scaled_error": max(
            float(row["added_mass_scaled_error"]) for row in rows
        ),
    }


def _node_task(task: tuple[float, float, tuple[int, ...], int]) -> dict[str, object]:
    ka, a_over_f, panels, panel_nodes = task
    return band_node(ka, a_over_f, panels, panel_nodes=panel_nodes)


def audit(
    *,
    ka_grid=KA_GRID,
    a_over_f_grid=A_OVER_F_GRID,
    panel_counts=DEFAULT_PANELS,
    panel_nodes: int = PANEL_NODES,
    workers: int | None = None,
) -> dict[str, object]:
    """Measure the whole band and summarise it into decidable quantities."""
    tasks = [
        (float(ka), float(ratio), tuple(int(n) for n in panel_counts), int(panel_nodes))
        for ka in ka_grid
        for ratio in a_over_f_grid
    ]
    if workers is None or workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            nodes = list(pool.map(_node_task, tasks))
    else:
        nodes = [_node_task(task) for task in tasks]

    worst_added_mass = max(
        nodes, key=lambda node: node["error_at_documented_panels"]["added_mass"]
    )
    resolved = [node for node in nodes if node["damping_resolved"]]
    unresolved = [node for node in nodes if not node["damping_resolved"]]
    worst_damping = (
        max(
            resolved,
            key=lambda node: node["error_at_documented_panels"]["damping"],
        )
        if resolved
        else None
    )
    orders = [
        node["observed_order_added_mass"]
        for node in nodes
        if math.isfinite(node["observed_order_added_mass"])
    ]
    orders += [
        node["observed_order_damping"]
        for node in resolved
        if math.isfinite(node["observed_order_damping"])
    ]
    return {
        "benchmark": "ursell_submerged_circular_cylinder_band_audit",
        "parameters": {
            "radius": RADIUS,
            "density": DENSITY,
            "gravity": GRAVITY,
            "panel_nodes": int(panel_nodes),
            "panel_counts": [int(n) for n in panel_counts],
            "ka_grid": [float(value) for value in ka_grid],
            "a_over_f_grid": [float(value) for value in a_over_f_grid],
            "truncation_tolerance": TRUNCATION_TOL,
            "damping_resolved_floor": DAMPING_RESOLVED_FLOOR,
            "damping_resolved_margin": DAMPING_RESOLVED_MARGIN,
        },
        "documented_claim": {
            "panels": DOCUMENTED_PANELS,
            "added_mass_relative_error": DOCUMENTED_ADDED_MASS_ERROR,
            "damping_relative_error": DOCUMENTED_DAMPING_ERROR,
        },
        "nodes": nodes,
        "summary": {
            "node_count": len(nodes),
            "resolved_node_count": len(resolved),
            "unresolved_nodes": [
                {
                    "ka": node["ka"],
                    "a_over_f": node["a_over_f"],
                    "reference_damping": node["reference_damping"],
                    "damping_scaled_error_at_documented_panels": node[
                        "error_at_documented_panels"
                    ]["damping_scaled"],
                }
                for node in unresolved
            ],
            "worst_ka_at_documented_panels_added_mass": worst_added_mass["ka"],
            "worst_a_over_f_at_documented_panels_added_mass": worst_added_mass[
                "a_over_f"
            ],
            "max_added_mass_error_at_documented_panels": worst_added_mass[
                "error_at_documented_panels"
            ]["added_mass"],
            "worst_ka_at_documented_panels_damping": (
                worst_damping["ka"] if worst_damping else None
            ),
            "worst_a_over_f_at_documented_panels_damping": (
                worst_damping["a_over_f"] if worst_damping else None
            ),
            "max_damping_error_at_documented_panels_resolved": (
                worst_damping["error_at_documented_panels"]["damping"]
                if worst_damping
                else None
            ),
            "min_observed_order": min(orders) if orders else float("nan"),
            "max_observed_order": max(orders) if orders else float("nan"),
            "max_power_law_residual_added_mass": max(
                pair["power_law_residual_added_mass"]
                for node in nodes
                for pair in node["pairs"]
            ),
            "max_power_law_residual_damping": max(
                pair["power_law_residual_damping"]
                for node in resolved
                for pair in node["pairs"]
            )
            if resolved
            else None,
            "max_richardson_gap_added_mass": max(
                node["richardson_gap_added_mass"] for node in nodes
            ),
            "max_richardson_gap_damping": max(
                node["richardson_gap_damping"] for node in resolved
            )
            if resolved
            else None,
            "max_near_over_far_field_deviation": max(
                node["max_near_over_far_field_damping_deviation"] for node in nodes
            ),
            "max_reference_truncation_gap": max(
                node["reference"]["truncation_gap"] for node in nodes
            ),
            "max_condition_number": max(
                node["max_condition_number"] for node in nodes
            ),
            "max_damping_scaled_error": max(
                node["max_damping_scaled_error"] for node in nodes
            ),
            "panels_needed_for_one_percent": _panels_needed(nodes, 1.0e-2),
        },
    }


def _panels_needed(nodes: list[dict[str, object]], target: float) -> dict[str, object]:
    """Smallest ladder rung whose error is below ``target`` at *every* judged node.

    Judged nodes are: all nodes for the added mass, and the damping-resolved
    nodes for the damping.  ``None`` means the finest rung of the ladder still
    misses the target somewhere in the band, which says the ladder must be
    extended rather than that the solver fails.
    """
    per_rung: dict[int, float] = {}
    for node in nodes:
        for row in node["rows"]:
            candidates = [float(row["added_mass_relative_error"])]
            if node["damping_resolved"]:
                candidates.append(float(row["damping_relative_error"]))
            worst = max(candidates)
            key = int(row["panels"])
            per_rung[key] = max(per_rung.get(key, 0.0), worst)
    feasible = sorted(panels for panels, worst in per_rung.items() if worst <= target)
    return {
        "target": float(target),
        "worst_error_by_panels": {str(key): value for key, value in sorted(per_rung.items())},
        "smallest_feasible_panels": feasible[0] if feasible else None,
    }


def plot(budget: dict[str, object], path: Path) -> None:
    """Two panels: refinement curves per band node, and order/error versus ``ka``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = budget["nodes"]
    ratios = sorted({node["a_over_f"] for node in nodes})
    colours = {0.25: "tab:blue", 0.5: "tab:orange", 0.75: "tab:green"}
    markers = {0.1: "o", 0.2: "^", 0.3: "s", 0.4: "D", 0.6: "v", 1.0: "P", 1.5: "X"}

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    def _marker(ka: float) -> str:
        return markers[min(markers, key=lambda key: abs(key - ka))]

    for axis, quantity in zip(axes[:2], ("added_mass", "damping")):
        for node in nodes:
            panels = [row["panels"] for row in node["rows"]]
            errors = [row[f"{quantity}_relative_error"] for row in node["rows"]]
            axis.loglog(
                panels,
                errors,
                marker=_marker(node["ka"]),
                color=colours[node["a_over_f"]],
                linewidth=1.0,
                markersize=4.0,
                label=f"ka={node['ka']:g}, a/f={node['a_over_f']:g}",
            )
        axis.set_xlabel("panels")
        axis.set_ylabel(
            "relative error vs Ursell" + (" (added mass)" if quantity == "added_mass" else " (damping)")
        )
        axis.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=5.2, loc="lower left", ncol=2)

    axis = axes[2]
    for ratio in ratios:
        subset = sorted(
            (node for node in nodes if node["a_over_f"] == ratio),
            key=lambda node: node["ka"],
        )
        axis.plot(
            [node["ka"] for node in subset],
            [node["observed_order_added_mass"] for node in subset],
            marker="o",
            color=colours[ratio],
            label=f"added mass, a/f={ratio:g}",
        )
        damping_subset = [node for node in subset if node["damping_resolved"]]
        axis.plot(
            [node["ka"] for node in damping_subset],
            [node["observed_order_damping"] for node in damping_subset],
            marker="s",
            linestyle="--",
            color=colours[ratio],
            alpha=0.7,
            label=f"damping (resolved), a/f={ratio:g}",
        )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle=":")
    axis.set_xlabel("ka")
    axis.set_ylabel("observed order between the two finest meshes")
    axis.set_ylim(0.0, 1.8)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=5.2, loc="lower left", ncol=2)

    figure.suptitle(
        "Ursell submerged cylinder: panel accuracy over the (ka, a/f) band, "
        f"24 nodes/panel, N <= {budget['parameters']['panel_counts'][-1]}"
    )
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_markdown(budget: dict[str, object]) -> str:
    """Machine-generated markdown, so no number in the report is hand-copied."""
    parameters = budget["parameters"]
    summary = budget["summary"]
    lines = [
        "# Ursell radiation: accuracy envelope over the (ka, a/f) band",
        "",
        "Generated by `python -m tensorbem.benchmarks.ursell_band_audit`; "
        "no number below is transcribed by hand.",
        "",
        "## Configuration",
        "",
        f"* radius `a = {parameters['radius']}` m, density `rho = {parameters['density']}` kg/m^3, "
        f"`g = {parameters['gravity']}` m/s^2",
        f"* panel nodes per panel: `{parameters['panel_nodes']}`",
        f"* refinement ladder: `{parameters['panel_counts']}`",
        f"* band: `ka = {parameters['ka_grid']}`, `a/f = {parameters['a_over_f_grid']}`",
        f"* reference: Ursell multipole, truncation doubled until the coefficients move by "
        f"< `{parameters['truncation_tolerance']:g}` (measured per node)",
        "",
        "## Error versus panel count",
        "",
        "| ka | a/f | submergence f/a | " + " | ".join(
            f"N={n} A err / B err" for n in parameters["panel_counts"]
        ) + " | order A / B |",
        "|---:|---:|---:|" + "---|" * len(parameters["panel_counts"]) + "---|",
    ]
    for node in budget["nodes"]:
        cells = " | ".join(
            f"{row['added_mass_relative_error']:.3e} / {row['damping_relative_error']:.3e}"
            for row in node["rows"]
        )
        lines.append(
            f"| {node['ka']:g} | {node['a_over_f']:g} | {1.0 / node['a_over_f']:.2f} | "
            f"{cells} | {node['observed_order_added_mass']:.2f} / "
            f"{node['observed_order_damping']:.2f} |"
        )
    lines += [
        "",
        "## Independent checks per node",
        "",
        "| ka | a/f | reference terms | reference truncation gap | "
        "Richardson gap A / B | max |near/far - 1| | max condition number |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    for node in budget["nodes"]:
        lines.append(
            f"| {node['ka']:g} | {node['a_over_f']:g} | {node['reference']['terms']} | "
            f"{node['reference']['truncation_gap']:.2e} | "
            f"{node['richardson_gap_added_mass']:.2e} / {node['richardson_gap_damping']:.2e} | "
            f"{node['max_near_over_far_field_damping_deviation']:.2e} | "
            f"{node['max_condition_number']:.1f} |"
        )
    feasible = summary["panels_needed_for_one_percent"]
    unresolved = summary["unresolved_nodes"]
    lines += [
        "",
        "Damping entries in parentheses are *absolute* (displaced-mass scale) "
        "errors at nodes where the reference damping is below "
        f"`{parameters['damping_resolved_floor']:g}` or below "
        f"`{parameters['damping_resolved_margin']:g}x` the measured absolute error; "
        "there the relative error divides by panel noise and is reported as "
        "unresolved (`n/r`) instead.  Unresolved nodes: "
        + (
            ", ".join(
                f"ka={node['ka']:g}/a:f={node['a_over_f']:g} "
                f"(nu_ref={node['reference_damping']:.1e}, "
                f"|dnu|={node['damping_scaled_error_at_documented_panels']:.1e})"
                for node in unresolved
            )
            or "none"
        ),
        "",
        "",
        "## Summary",
        "",
        f"* worst added-mass error at N = {budget['documented_claim']['panels']} panels: "
        f"`{summary['max_added_mass_error_at_documented_panels']:.3e}` "
        f"(at ka = {summary['worst_ka_at_documented_panels_added_mass']:g}, "
        f"a/f = {summary['worst_a_over_f_at_documented_panels_added_mass']:g})",
        f"* worst damping error at N = {budget['documented_claim']['panels']} panels "
        f"over the {summary['resolved_node_count']} resolved nodes: "
        + (
            f"`{summary['max_damping_error_at_documented_panels_resolved']:.3e}` "
            f"(at ka = {summary['worst_ka_at_documented_panels_damping']:g}, "
            f"a/f = {summary['worst_a_over_f_at_documented_panels_damping']:g})"
            if summary["max_damping_error_at_documented_panels_resolved"] is not None
            else "no resolved node on this ladder"
        ),
        f"* observed order range over the band: {summary['min_observed_order']:.2f} to "
        f"{summary['max_observed_order']:.2f}",
        f"* worst Richardson gap: {summary['max_richardson_gap_added_mass']:.2e} (added mass), "
        + (
            f"{summary['max_richardson_gap_damping']:.2e} (damping, resolved nodes)"
            if summary["max_richardson_gap_damping"] is not None
            else "damping: no resolved node"
        ),
        f"* worst near-field/far-field damping deviation: "
        f"{summary['max_near_over_far_field_deviation']:.2e}",
        f"* worst reference truncation gap: {summary['max_reference_truncation_gap']:.2e}",
        f"* worst absolute damping error over the whole band: "
        f"`{summary['max_damping_scaled_error']:.2e}`",
        f"* smallest ladder rung with every node below "
        f"{feasible['target']:g} in both coefficients: "
        f"`{feasible['smallest_feasible_panels']}` (worst error by rung: "
        + ", ".join(f"N={k}: {v:.2e}" for k, v in feasible["worst_error_by_panels"].items())
        + ")",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the audit as JSON here")
    parser.add_argument("--figure", type=Path, help="write the figure here")
    parser.add_argument("--markdown", type=Path, help="write the generated markdown here")
    parser.add_argument("--workers", type=int, default=None, help="process pool size")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="coarse band and short ladder, for smoke tests",
    )
    arguments = parser.parse_args()
    if arguments.quick:
        budget = audit(
            ka_grid=(0.2, 1.0),
            a_over_f_grid=(0.5,),
            panel_counts=(16, 32),
            workers=arguments.workers,
        )
    else:
        budget = audit(workers=arguments.workers)
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(budget, indent=2) + "\n")
    if arguments.figure:
        plot(budget, arguments.figure)
    if arguments.markdown:
        arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
        arguments.markdown.write_text(render_markdown(budget) + "\n")
    print(json.dumps(budget["summary"], indent=2))


if __name__ == "__main__":
    main()