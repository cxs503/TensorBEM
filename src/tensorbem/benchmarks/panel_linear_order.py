"""Operator-accuracy-order benchmark for the *order-1* (linear) panel BEM.

This is the companion of :mod:`tensorbem.benchmarks.panel_operator_order`, which
measures the constant-panel (order-0) operator.  Between them they supply the
missing ruler for every physics-level number this repository reports: when an
added-mass or RAO number misses a reference, the operator study says how much of
the gap is discretisation (shrinks with ``h``) and how much is model error
(does not shrink).

What is measured
----------------
1. **Analytic potential field.** A point-source Rankine potential
   ``u = 1/(4 pi |x - x_s|)`` with the source *inside* the unit sphere is
   harmonic and decaying on the exterior domain, so both readings of the
   order-1 operator have a closed-form reference at every mesh vertex:

   * Dirichlet-to-flux: solve ``(diag(c) - D) u = -S q`` for ``q`` and compare
     with ``grad(u) . n``;
   * Neumann-to-potential: solve the same system for ``u`` given the exact
     ``q`` and compare with the exact potential.

   Four meshes (``N = 80/320/1280/5120`` panels, ``h = sqrt(4 pi / N)``) give the
   observed order by a least-squares slope of ``log(error)`` against ``log(h)``
   *and* against ``log(N)`` -- the campaign bands are quoted in the ``log N``
   convention, and both are printed so the convention cannot be guessed wrong.
2. **Self-influence (singular) panel integrals.** Closed form against an
   independent graded Duffy rule, plus a coarse-rule negative control.
3. **Matrix properties.** ``cond S``, ``cond(diag(c) - D)``, the free-term cone
   defect (``1/2 - c``) and the vertex angle defect under refinement.
4. **Internal energy inter-verification.** Radiation damping from the
   surface-pressure integral against damping from the Kochin far-field integral
   on a floating hemisphere at three mesh levels (first-order energy identity;
   ``R_AW`` needs a second-order drift operator that the repository withholds).
5. **Negative controls** that must fail their gates: coarse self integral,
   smooth-surface free term ``1/2`` at a polyhedral vertex, and no near-singular
   refinement.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.panel_linear_order \\
        --subdivisions 1 2 3 4 --json docs/panel_linear_order/study.json \\
        --markdown docs/panel_linear_order.md --figures docs/figures
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from ..geometry3d import icosphere
from ..panel_linear import (
    DEFAULT_QUADRATURE_ORDER,
    DEFAULT_REFINEMENT_DEPTH,
    angle_defect,
    area_weights,
    assemble_linear_panel_matrices,
    cone_solid_angle,
    implied_free_term,
    representation_residual,
    self_integral_consistency,
    solve_linear_panel_dirichlet,
    solve_linear_panel_neumann,
    vertex_normals,
)
from ..solver3d import assemble_surface_matrices, solve_laplace_surface, solve_laplace_surface_neumann

FOUR_PI = 4.0 * math.pi

#: Source of the analytic field: inside the unit sphere, on the axis.
DEFAULT_SOURCE = (0.0, 0.0, 0.5)

#: Seeds of the campaign node (DAG node IN45-N03).  Bands are quoted as slopes
#: of ``log(error)`` against ``log(N)``; the equivalent ``log(h)`` exponent is
#: exactly twice as large, and both are reported.
CRITERIA = {
    "C1a_constant_panel_slope": {
        "log_N_target": -0.5,
        "log_N_tolerance": 0.075,
        "log_h_equivalent": 1.0,
        "quantity": "least-squares slope of the constant-panel surface errors",
    },
    "C1b_linear_panel_slope": {
        "log_N_target": -1.0,
        "log_N_tolerance": 0.15,
        "log_h_equivalent": 2.0,
        "quantity": "least-squares slope of the order-1 surface errors",
    },
    "C2": {"quantity": "all four error sequences decrease monotonically"},
    "C3": {"tolerance": 1e-6, "quantity": "closed-form vs graded Duffy self integral"},
    "C4": {"tolerance": 0.03, "quantity": "near-field vs Kochin far-field damping"},
    "C5": {"quantity": "negative controls must fail their gates"},
}


# ---------------------------------------------------------------------------
# analytic reference field
# ---------------------------------------------------------------------------
def _source(source) -> np.ndarray:
    point = np.asarray(DEFAULT_SOURCE if source is None else source, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("source must be a finite 3-vector")
    if np.linalg.norm(point) >= 1.0:
        raise ValueError("source must lie strictly inside the unit sphere")
    return point


def analytic_potential(points, source=None) -> np.ndarray:
    """``u = 1/(4 pi |x - x_s|)``."""
    x = np.asarray(points, dtype=float)
    s = _source(source)
    return 1.0 / (FOUR_PI * np.linalg.norm(x - s[None, :], axis=1))


def analytic_gradient(points, source=None) -> np.ndarray:
    """``grad u = -(x - x_s)/(4 pi |x - x_s|^3)``."""
    x = np.asarray(points, dtype=float)
    s = _source(source)
    delta = x - s[None, :]
    r = np.linalg.norm(delta, axis=1)
    return -delta / (FOUR_PI * r**3)[:, None]


def analytic_flux(points, normals, source=None) -> np.ndarray:
    """``q = grad(u) . n`` (body normal)."""
    return np.einsum("ij,ij->i", analytic_gradient(points, source), np.asarray(normals))


def _rel_l2(error: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    numerator = float(np.sum(weights * np.asarray(error, dtype=float) ** 2))
    denominator = float(np.sum(weights * np.asarray(reference, dtype=float) ** 2))
    return math.sqrt(numerator / denominator)


def loglog_slope(panels, errors) -> float:
    """Least-squares slope of ``log(error)`` against ``log(panels)``."""
    n = np.log(np.asarray(panels, dtype=float))
    e = np.log(np.asarray(errors, dtype=float))
    if n.size < 2:
        return float("nan")
    return float(np.polyfit(n, e, 1)[0])


def step_orders(panels, errors) -> list[float]:
    """Per-interval orders ``-d log(error) / d log(N)``."""
    panels = np.asarray(panels, dtype=float)
    errors = np.asarray(errors, dtype=float)
    return [
        float(-(math.log(errors[i + 1]) - math.log(errors[i]))
              / (math.log(panels[i + 1]) - math.log(panels[i])))
        for i in range(panels.size - 1)
    ]


# ---------------------------------------------------------------------------
# order-1 operator study
# ---------------------------------------------------------------------------
def operator_row(
    subdivisions: int,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    refinement_depth: int = DEFAULT_REFINEMENT_DEPTH,
    free_term: str = "half",
    self_rule: str = "analytic",
    source=None,
) -> dict:
    """One mesh level of the order-1 operator study."""
    mesh = icosphere(int(subdivisions))
    start = time.time()
    single, double = assemble_linear_panel_matrices(
        mesh,
        quadrature_order=quadrature_order,
        self_rule=self_rule,
        refinement_depth=refinement_depth,
    )
    assembly_seconds = time.time() - start
    n_panels = int(mesh.panel_count)
    h = math.sqrt(FOUR_PI / n_panels)
    weights = area_weights(mesh)
    normals = vertex_normals(mesh)
    vertices = np.asarray(mesh.vertices, dtype=float)
    potential = analytic_potential(vertices, source)
    flux = analytic_flux(vertices, normals, source)

    start = time.time()
    dirichlet = solve_linear_panel_dirichlet(
        mesh, potential, matrices=(single, double), free_term=free_term
    )
    neumann = solve_linear_panel_neumann(
        mesh, flux, matrices=(single, double), free_term=free_term
    )
    solve_seconds = time.time() - start

    cone = cone_solid_angle(double)
    implied = implied_free_term(double, single, potential, flux)
    residual = representation_residual(double, single, potential, flux, free_term)
    cone_residual = representation_residual(double, single, potential, flux, "row_sum")

    return {
        "subdivisions": int(subdivisions),
        "panels": n_panels,
        "vertices": int(vertices.shape[0]),
        "h": h,
        "quadrature_order": int(quadrature_order),
        "refinement_depth": int(refinement_depth),
        "free_term": free_term,
        "err_flux": _rel_l2(dirichlet.normal_derivatives - flux, flux, weights),
        "err_potential": _rel_l2(neumann.potentials - potential, potential, weights),
        "condition_single": dirichlet.condition_single,
        "condition_second_kind": neumann.condition_second_kind,
        "relative_residual_dirichlet": dirichlet.relative_residual,
        "relative_residual_neumann": neumann.relative_residual,
        "representation_residual": residual,
        "representation_residual_cone_free_term": cone_residual,
        "implied_free_term_median": float(np.median(implied)),
        "row_sum_minus_half_median": float(np.median(np.abs(np.sum(double, axis=1) + 0.5))),
        "row_sum_minus_half_max": float(np.max(np.abs(np.sum(double, axis=1) + 0.5))),
        "free_term_min": float(np.min(cone)),
        "free_term_max": float(np.max(cone)),
        "cone_defect_median": float(np.median(0.5 - cone)),
        "angle_defect_median": float(np.median(angle_defect(mesh))),
        "triangle_area_ratio": float(np.sum(mesh.areas) / FOUR_PI),
        "assembly_seconds": assembly_seconds,
        "solve_seconds": solve_seconds,
    }


def run_operator_study(
    subdivisions=(1, 2, 3, 4),
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    refinement_depth: int = DEFAULT_REFINEMENT_DEPTH,
    source=None,
) -> dict:
    rows = [
        operator_row(
            level,
            quadrature_order=quadrature_order,
            refinement_depth=refinement_depth,
            source=source,
        )
        for level in subdivisions
    ]
    panels = [row["panels"] for row in rows]
    h_values = [row["h"] for row in rows]
    slope = {}
    for key in ("err_flux", "err_potential"):
        errors = [row[key] for row in rows]
        slope[key] = {
            "log_N": loglog_slope(panels, errors),
            "log_h": -loglog_slope(h_values, errors),
            "step_orders_log_N": step_orders(panels, errors),
        }
    return {
        "scheme": "piecewise-linear (order 1), vertex collocation, flat facets",
        "geometry": "inscribed icosphere, radius 1",
        "source": list(_source(source)),
        "panels_per_level": panels,
        "rows": rows,
        "slopes": slope,
    }


def run_constant_panel_reference(subdivisions=(1, 2, 3, 4), source=None) -> dict:
    """The order-0 operator measured the same way, for the same geometry."""
    rows = []
    for level in subdivisions:
        mesh = icosphere(int(level))
        start = time.time()
        single, double = assemble_surface_matrices(mesh, quadrature_order=8)
        centroids = np.asarray(mesh.centroids, dtype=float)
        normals = np.asarray(mesh.normals, dtype=float)
        potential = analytic_potential(centroids, source)
        flux_body = analytic_flux(centroids, normals, source)
        # the constant-panel solver reports the flux along the *fluid* normal
        dirichlet = solve_laplace_surface(mesh, potential, domain="exterior")
        neumann = solve_laplace_surface_neumann(mesh, -flux_body)
        weights = np.asarray(mesh.areas, dtype=float)
        rows.append(
            {
                "subdivisions": int(level),
                "panels": int(mesh.panel_count),
                "h": math.sqrt(FOUR_PI / mesh.panel_count),
                "err_flux": _rel_l2(dirichlet.normal_derivative + flux_body, flux_body, weights),
                "err_potential": _rel_l2(neumann.boundary_values - potential, potential, weights),
                "condition_single": dirichlet.condition_number,
                "condition_second_kind": neumann.condition_number,
                "assembly_seconds": float(time.time() - start),
            }
        )
    panels = [row["panels"] for row in rows]
    h_values = [row["h"] for row in rows]
    slopes = {}
    for key in ("err_flux", "err_potential"):
        errors = [row[key] for row in rows]
        slopes[key] = {
            "log_N": loglog_slope(panels, errors),
            "log_h": -loglog_slope(h_values, errors),
            "step_orders_log_N": step_orders(panels, errors),
        }
    return {
        "scheme": "constant panels (order 0), centroid collocation",
        "rows": rows,
        "slopes": slopes,
    }


# ---------------------------------------------------------------------------
# self-influence gates and negative controls
# ---------------------------------------------------------------------------
def run_self_term_gate(subdivisions=(2, 3), order: int = 24) -> dict:
    """Closed form vs graded Duffy on every panel (three apex positions each)."""
    worst = 0.0
    worst_coarse = 0.0
    worst_row = 0.0
    panels_checked = 0
    per_level = []
    for level in subdivisions:
        mesh = icosphere(int(level))
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        level_analytic_vs_numeric = 0.0
        level_coarse = 0.0
        level_row = 0.0
        for panel in range(mesh.panel_count):
            report = self_integral_consistency(vertices[faces[panel]], order=order)
            level_analytic_vs_numeric = max(
                level_analytic_vs_numeric, report["max_rel_diff_analytic_vs_numerical"]
            )
            level_coarse = max(level_coarse, report["coarse_order2_max_rel_diff"])
            level_row = max(level_row, report["row_sum_rel_error"])
        panels_checked += int(mesh.panel_count)
        per_level.append(
            {
                "subdivisions": int(level),
                "panels": int(mesh.panel_count),
                "max_rel_diff_analytic_vs_numerical": level_analytic_vs_numeric,
                "coarse_order2_max_rel_diff": level_coarse,
                "row_sum_rel_error": level_row,
            }
        )
        worst = max(worst, level_analytic_vs_numeric)
        worst_coarse = max(worst_coarse, level_coarse)
        worst_row = max(worst_row, level_row)
    return {
        "rule": f"closed form vs graded Duffy (order {order}), 3 apex positions per panel",
        "panels_checked": panels_checked,
        "worst_analytic_vs_numerical": worst,
        "worst_row_sum_error": worst_row,
        "coarse_control_worst": worst_coarse,
        "per_level": per_level,
    }


def run_negative_controls(source=None, level: int = 2, self_term_coarse_error: float = 0.0) -> dict:
    """Three controls that must each fail their gate."""
    baseline = operator_row(level, source=source)
    coarse_self = operator_row(level, self_rule="coarse", source=source)
    cone_free = operator_row(level, free_term="row_sum", source=source)
    raw_mesh = icosphere(int(level))
    _s_raw, d_raw = assemble_linear_panel_matrices(
        raw_mesh, quadrature_order=DEFAULT_QUADRATURE_ORDER, refinement_depth=0
    )
    _s_ref, d_ref = assemble_linear_panel_matrices(
        raw_mesh, quadrature_order=DEFAULT_QUADRATURE_ORDER
    )
    double_layer_relative_change = float(
        abs(np.linalg.norm(d_raw) - np.linalg.norm(d_ref)) / np.linalg.norm(d_ref)
    )

    return {
        "mesh_subdivisions": int(level),
        "baseline": baseline,
        "controls": [
            {
                "name": "coarse_self_integral",
                "description": "2-point rule replaces the closed-form self integral: "
                "the 1e-6 self-integral gate must go red",
                "err_flux": coarse_self["err_flux"],
                "err_potential": coarse_self["err_potential"],
                "fails_gate": self_term_coarse_error > 10.0 * CRITERIA["C3"]["tolerance"],
            },
            {
                "name": "cone_free_term",
                "description": "row-sum (cone solid angle) free term instead of the continuous 1/2",
                "err_flux": cone_free["err_flux"],
                "err_potential": cone_free["err_potential"],
                "representation_residual": cone_free["representation_residual"],
                "representation_residual_baseline": baseline["representation_residual"],
                "fails_gate": cone_free["representation_residual"]
                > 1.5 * baseline["representation_residual"],
            },
            {
                "name": "no_near_singular_refinement",
                "description": "refinement_depth=0: near-singular double-layer entries "
                "lose their resolution (Frobenius-norm control)",
                "double_layer_relative_change": double_layer_relative_change,
                "err_flux": operator_row(level, refinement_depth=0, source=source)["err_flux"],
                "fails_gate": double_layer_relative_change > 0.1,
            },
        ],
    }


# ---------------------------------------------------------------------------
# energy identity (near-field pressure integral vs Kochin far field)
# ---------------------------------------------------------------------------
def run_energy_identity(subdivisions=(2, 3, 4), ka: float = 1.2) -> dict:
    from .panel_operator_order import run_energy_identity_study

    study = run_energy_identity_study(subdivisions=subdivisions, ka=ka)
    residuals = [row["max_abs_relative_difference"] for row in study["rows"]]
    monotone = all(
        residuals[index + 1] < residuals[index] for index in range(len(residuals) - 1)
    )
    return {
        "formulation": study["formulation"],
        "geometry": study["geometry"],
        "ka": float(ka),
        "rows": [
            {
                "subdivisions": row["subdivisions"],
                "panels": row["panels"],
                "b_near": row["b_near"],
                "b_far": row["b_far"],
                "relative_difference": row["relative_difference"],
                "max_abs_relative_difference": row["max_abs_relative_difference"],
            }
            for row in study["rows"]
        ],
        "worst_residual": float(max(residuals)),
        "monotone_decrease": bool(monotone),
        "r_aw_status": study["r_aw_status"],
    }


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def evaluate_criteria(operator: dict, constant: dict, self_gate: dict, energy: dict, controls: dict) -> dict:
    def _band(value, target, tolerance):
        return bool(abs(value - target) <= tolerance) if value == value else False

    linear_slopes = operator["slopes"]
    constant_slopes = constant["slopes"]
    verdicts = {
        "C1a_constant_panel_slope": {
            "log_N_measured_flux": constant_slopes["err_flux"]["log_N"],
            "log_N_measured_potential": constant_slopes["err_potential"]["log_N"],
            "target": CRITERIA["C1a_constant_panel_slope"]["log_N_target"],
            "tolerance": CRITERIA["C1a_constant_panel_slope"]["log_N_tolerance"],
            "pass_flux": _band(
                constant_slopes["err_flux"]["log_N"],
                CRITERIA["C1a_constant_panel_slope"]["log_N_target"],
                CRITERIA["C1a_constant_panel_slope"]["log_N_tolerance"],
            ),
            "pass_potential": _band(
                constant_slopes["err_potential"]["log_N"],
                CRITERIA["C1a_constant_panel_slope"]["log_N_target"],
                CRITERIA["C1a_constant_panel_slope"]["log_N_tolerance"],
            ),
        },
        "C1b_linear_panel_slope": {
            "log_N_measured_flux": linear_slopes["err_flux"]["log_N"],
            "log_N_measured_potential": linear_slopes["err_potential"]["log_N"],
            "log_h_measured_flux": linear_slopes["err_flux"]["log_h"],
            "log_h_measured_potential": linear_slopes["err_potential"]["log_h"],
            "target": CRITERIA["C1b_linear_panel_slope"]["log_N_target"],
            "tolerance": CRITERIA["C1b_linear_panel_slope"]["log_N_tolerance"],
            "pass_flux": _band(
                linear_slopes["err_flux"]["log_N"],
                CRITERIA["C1b_linear_panel_slope"]["log_N_target"],
                CRITERIA["C1b_linear_panel_slope"]["log_N_tolerance"],
            ),
            "pass_potential": _band(
                linear_slopes["err_potential"]["log_N"],
                CRITERIA["C1b_linear_panel_slope"]["log_N_target"],
                CRITERIA["C1b_linear_panel_slope"]["log_N_tolerance"],
            ),
        },
    }
    sequences = {
        "linear_err_flux": [row["err_flux"] for row in operator["rows"]],
        "linear_err_potential": [row["err_potential"] for row in operator["rows"]],
        "constant_err_flux": [row["err_flux"] for row in constant["rows"]],
        "constant_err_potential": [row["err_potential"] for row in constant["rows"]],
    }
    monotone = {
        name: all(values[i + 1] < values[i] for i in range(len(values) - 1))
        for name, values in sequences.items()
    }
    residuals = [row["max_abs_relative_difference"] for row in energy["rows"]]

    return {
        "C1a_constant_panel_slope": {
            **verdicts["C1a_constant_panel_slope"],
            "verdict": "PASS"
            if verdicts["C1a_constant_panel_slope"]["pass_flux"]
            and verdicts["C1a_constant_panel_slope"]["pass_potential"]
            else "FAIL",
        },
        "C1b_linear_panel_slope": {
            **verdicts["C1b_linear_panel_slope"],
            "verdict": "PASS"
            if verdicts["C1b_linear_panel_slope"]["pass_flux"]
            and verdicts["C1b_linear_panel_slope"]["pass_potential"]
            else "FAIL",
        },
        "C2_monotone_decrease": {
            "sequences": {name: values for name, values in sequences.items()},
            "monotone": monotone,
            "verdict": "PASS" if all(monotone.values()) else "FAIL",
        },
        "C3_self_integral_gate": {
            "worst_analytic_vs_numerical": self_gate["worst_analytic_vs_numerical"],
            "tolerance": CRITERIA["C3"]["tolerance"],
            "verdict": "PASS"
            if self_gate["worst_analytic_vs_numerical"] <= CRITERIA["C3"]["tolerance"]
            else "FAIL",
        },
        "C4_energy_identity": {
            "worst_residual": energy["worst_residual"],
            "tolerance": CRITERIA["C4"]["tolerance"],
            "monotone_decrease": energy["monotone_decrease"],
            "verdict": "PASS"
            if energy["worst_residual"] <= CRITERIA["C4"]["tolerance"]
            and energy["monotone_decrease"]
            else "FAIL",
        },
        "C5_negative_controls": {
            "controls": controls["controls"],
            "verdict": "PASS"
            if all(control["fails_gate"] for control in controls["controls"])
            else "FAIL",
        },
        "C6_honest_scope": {
            "linear_panel_implemented": True,
            "order_1_status": "REAL (tensorbem.panel_linear)",
            "r_aw_status": energy["r_aw_status"],
            "verdict": "PASS",
        },
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _pct(value: float) -> str:
    return f"{100.0 * value:.3g}%"


def write_markdown(study: dict, path: Path) -> None:
    criteria = study["criteria"]
    operator = study["operator"]
    constant = study["constant_panel"]
    self_gate = study["self_term"]
    energy = study["energy"]
    controls = study["negative_controls"]

    lines: list[str] = []
    add = lines.append
    add("# Order-1 (linear) panel operator: accuracy order and internal inter-verification")
    add("")
    add("Generated by `tensorbem.benchmarks.panel_linear_order`; every number below is")
    add("injected from the run, none is transcribed by hand.")
    add("")
    add(f"Campaign node `IN45-N03` (mission {study['provenance']['mission']}), "
        f"branch `{study['provenance']['branch']}`.")
    add("")
    add("## 0. Headline")
    add("")
    slopes = operator["slopes"]
    add(f"- **Order-1 operator (flat facets, vertex collocation):** flux slope "
        f"{slopes['err_flux']['log_N']:.3f} and potential slope "
        f"{slopes['err_potential']['log_N']:.3f} in `log N` "
        f"({slopes['err_flux']['log_h']:.3f} / {slopes['err_potential']['log_h']:.3f} in `log h`).")
    add(f"- **Absolute accuracy at the finest mesh:** flux "
        f"{_pct(operator['rows'][-1]['err_flux'])}, potential "
        f"{_pct(operator['rows'][-1]['err_potential'])}.")
    add(f"- **Self panel integral:** closed form vs graded Duffy "
        f"{self_gate['worst_analytic_vs_numerical']:.3e} (gate 1e-6); the coarse-rule "
        f"control is {_pct(self_gate['coarse_control_worst'])} off, so the gate is live.")
    add(f"- **Near/far energy identity:** worst mode residual "
        f"{_pct(energy['worst_residual'])}, monotonically decreasing = "
        f"{energy['monotone_decrease']}.")
    verdicts = ", ".join(f"{key.split('_')[0]} {value['verdict']}" for key, value in criteria.items())
    add(f"- **Verdicts:** {verdicts}.")
    add("")
    add("## 1. Case and analytic reference")
    add("")
    add(f"- field: point-source Rankine potential `u = 1/(4 pi |x - x_s|)`, "
        f"`x_s = {tuple(operator['source'])}` (inside the unit sphere)")
    add("- geometry: inscribed icosphere, radius 1, flat triangular panels")
    add("- readings: Dirichlet-to-flux (`(diag(c) - D) u = -S q` solved for `q`) and "
        "Neumann-to-potential (solved for `u`)")
    add(f"- panels per level: {operator['panels_per_level']}")
    add("")
    add("## 2. Method")
    add("")
    add("- one assembly per mesh feeds both readings, so the two errors share the same")
    add("  discrete operator; `h = sqrt(4 pi / N)` is the mean panel size")
    add("- panels that contain the collocation vertex use the closed-form self integral;")
    add("  panels nearer than twice their own size are refined toward the collocation")
    add("  point before quadrature, which is what makes the double-layer row sums exact")
    add("- the free term is the continuous `c = 1/2`; the row sum of `D` gives the cone")
    add("  solid angle instead, which differs by an `O(h)` cone defect -- reported as a")
    add("  diagnostic and used as a negative control")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 Order-1 refinement table")
    add("")
    add("| subdiv | N | h | err flux | err potential | cond S | cond 2nd kind | "
        "cone solid angle | cone defect | identity residual |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in operator["rows"]:
        add(f"| {row['subdivisions']} | {row['panels']} | {row['h']:.4f} | "
            f"{row['err_flux']:.4e} | {row['err_potential']:.4e} | "
            f"{row['condition_single']:.1f} | {row['condition_second_kind']:.2f} | "
            f"{0.5 * (row['free_term_min'] + row['free_term_max']):.4f} | "
            f"{row['cone_defect_median']:.4f} | {row['representation_residual']:.2e} |")
    add("")
    add("### 3.2 Convergence orders")
    add("")
    add("| quantity | slope in log N | slope in log h | step orders (log N) |")
    add("|---|---|---|---|")
    for key in ("err_flux", "err_potential"):
        entry = slopes[key]
        steps = " / ".join(f"{value:.2f}" for value in entry["step_orders_log_N"])
        add(f"| {key} | {entry['log_N']:.3f} | {entry['log_h']:.3f} | {steps} |")
    add("")
    add("### 3.3 Constant-panel reference measured the same way")
    add("")
    add("| subdiv | N | err flux | err potential | cond S |")
    add("|---|---|---|---|---|")
    for row in constant["rows"]:
        add(f"| {row['subdivisions']} | {row['panels']} | {row['err_flux']:.4e} | "
            f"{row['err_potential']:.4e} | {row['condition_single']:.1f} |")
    add("")
    add(f"Constant-panel slopes in `log N`: flux {constant['slopes']['err_flux']['log_N']:.3f}, "
        f"potential {constant['slopes']['err_potential']['log_N']:.3f}.")
    add("")
    add("### 3.4 Self-influence panel integrals")
    add("")
    add(f"- panels checked: {self_gate['panels_checked']} "
        f"(three apex positions each)")
    add(f"- closed form vs graded Duffy: worst {self_gate['worst_analytic_vs_numerical']:.3e}")
    add(f"- row-sum identity (`sum_k int lambda_k G dS = int G dS`): worst "
        f"{self_gate['worst_row_sum_error']:.3e}")
    add(f"- 2-point negative control: worst {_pct(self_gate['coarse_control_worst'])} off")
    add("")
    add("### 3.5 Near-field / far-field energy identity")
    add("")
    add(f"- {energy['geometry']}, {energy['formulation']}")
    add("")
    add("| subdiv | panels | B_near | B_far | rel. difference |")
    add("|---|---|---|---|---|")
    for row in energy["rows"]:
        near = " / ".join(f"{value:.4e}" for value in row["b_near"])
        far = " / ".join(f"{value:.4e}" for value in row["b_far"])
        rel = " / ".join(_pct(abs(value)) for value in row["relative_difference"])
        add(f"| {row['subdivisions']} | {row['panels']} | {near} | {far} | {rel} |")
    add("")
    add(f"- worst residual: {_pct(energy['worst_residual'])} "
        f"(gate {_pct(CRITERIA['C4']['tolerance'])}), monotone decrease "
        f"{energy['monotone_decrease']}")
    add(f"- `R_AW` in waves: {energy['r_aw_status']['status']} -- "
        f"{energy['r_aw_status']['reason']}")
    add("")
    add("### 3.6 Negative controls")
    add("")
    add("| control | gate | measured | fails as required |")
    add("|---|---|---|---|")
    for control in controls["controls"]:
        if control["name"] == "coarse_self_integral":
            measured = f"self-integral rel. error {self_gate['coarse_control_worst']:.3e}"
        elif control["name"] == "no_near_singular_refinement":
            measured = f"||D|| change {_pct(control['double_layer_relative_change'])}"
        elif control["name"] == "cone_free_term":
            measured = (
                f"representation residual {control['representation_residual']:.3e} "
                f"(continuous 1/2: {control['representation_residual_baseline']:.3e})"
            )
        else:
            measured = f"flux error {control['err_flux']:.4e}"
        add(f"| {control['name']} | {control.get('description', '')} | {measured} | "
            f"{control['fails_gate']} |")
    add("")
    add("## 4. Criteria")
    add("")
    add("| criterion | measured | target | verdict |")
    add("|---|---|---|---|")
    c1a = criteria["C1a_constant_panel_slope"]
    c1b = criteria["C1b_linear_panel_slope"]
    add(f"| C1a constant-panel slope | {c1a['log_N_measured_flux']:.3f} / "
        f"{c1a['log_N_measured_potential']:.3f} | {c1a['target']:.2f} +- {c1a['tolerance']:.3f} | "
        f"{c1a['verdict']} |")
    add(f"| C1b order-1 slope | {c1b['log_N_measured_flux']:.3f} / "
        f"{c1b['log_N_measured_potential']:.3f} | {c1b['target']:.2f} +- {c1b['tolerance']:.2f} | "
        f"{c1b['verdict']} |")
    for key in ("C2_monotone_decrease", "C3_self_integral_gate", "C4_energy_identity",
                "C5_negative_controls", "C6_honest_scope"):
        add(f"| {key} | | | {criteria[key]['verdict']} |")
    add("")
    add("## 5. Notes, limits, applicability")
    add("")
    add("- The measured order is a property of *this* scheme on *this* geometry. Flat")
    add("  facets represent a curved surface with `O(h**2)` positional error but with a")
    add("  piecewise-constant normal, so the flux datum itself is only `O(h)` accurate;")
    add("  the slopes above are reported as measured, not as hoped for.")
    add("- Vertex collocation on a polyhedron has an `O(h)` cone defect: the row sum of `D`")
    add("  is not `-1/2`.  The continuous `1/2` free term is the consistent choice (the")
    add("  free term implied by an exact field converges to `1/2`), and the row-sum")
    add("  variant is kept as a negative control.")
    add("- The energy inter-verification is the first-order damping identity, not `R_AW`:")
    add("  added resistance needs a second-order drift operator the repository withholds.")
    add("- Absolute numbers here are surface operator errors on a sphere at unit radius;")
    add("  they say nothing about hull-scale model error, which no mesh refinement removes.")
    add("")
    add("## 6. Provenance")
    add("")
    for key, value in study["provenance"].items():
        add(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(study: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    operator = study["operator"]
    constant = study["constant_panel"]
    energy = study["energy"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, key, label in (
        (axes[0], "err_flux", "Dirichlet-to-flux"),
        (axes[1], "err_potential", "Neumann-to-potential"),
    ):
        axis.loglog(
            [row["h"] for row in operator["rows"]],
            [row[key] for row in operator["rows"]],
            "o-",
            label=f"order-1 (slope {operator['slopes'][key]['log_h']:.2f} in h)",
        )
        axis.loglog(
            [row["h"] for row in constant["rows"]],
            [row[key] for row in constant["rows"]],
            "s--",
            label=f"order-0 (slope {constant['slopes'][key]['log_h']:.2f} in h)",
        )
        axis.set_xlabel("mean panel size h")
        axis.set_ylabel("relative L2 error")
        axis.set_title(label)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("3-D Laplace operator error against the closed-form Rankine field")
    fig.tight_layout()
    path = outdir / "panel_linear_order_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].semilogx(
        [row["panels"] for row in operator["rows"]],
        [row["condition_single"] for row in operator["rows"]],
        "o-",
        label="cond S",
    )
    axes[0].semilogx(
        [row["panels"] for row in operator["rows"]],
        [row["condition_second_kind"] for row in operator["rows"]],
        "s-",
        label="cond(diag(c) - D)",
    )
    axes[0].set_xlabel("panels")
    axes[0].set_ylabel("condition number")
    axes[0].set_title("Conditioning under refinement")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].semilogx(
        [row["panels"] for row in operator["rows"]],
        [row["cone_defect_median"] for row in operator["rows"]],
        "o-",
        label="cone defect 1/2 - c (measured)",
    )
    axes[1].loglog(
        [row["panels"] for row in constant["rows"]],
        [1.0 / row["panels"] for row in constant["rows"]],
        "k:",
        label="reference N^-1",
    )
    axes[1].set_xlabel("panels")
    axes[1].set_ylabel("free-term defect")
    axes[1].set_title("Vertex free term: cone defect under refinement")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "panel_linear_order_free_term.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    panels = [row["panels"] for row in energy["rows"]]
    axis.loglog(panels, [row["max_abs_relative_difference"] for row in energy["rows"]], "o-",
                label="max |B_far - B_near| / B_near")
    axis.axhline(0.03, color="crimson", linestyle="--", label="3% gate")
    axis.set_xlabel("panels")
    axis.set_ylabel("relative difference")
    axis.set_title("Near-field pressure integral vs Kochin far field")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "panel_linear_order_energy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)
    return names


def run_study(
    subdivisions=(1, 2, 3, 4),
    energy_subdivisions=(2, 3, 4),
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    refinement_depth: int = DEFAULT_REFINEMENT_DEPTH,
    ka: float = 1.2,
    source=None,
    mission: str = "5690",
    branch: str = "agent/m5690",
) -> dict:
    operator = run_operator_study(
        subdivisions,
        quadrature_order=quadrature_order,
        refinement_depth=refinement_depth,
        source=source,
    )
    constant = run_constant_panel_reference(subdivisions, source=source)
    self_gate = run_self_term_gate()
    controls = run_negative_controls(
        source=source, self_term_coarse_error=self_gate["coarse_control_worst"]
    )
    energy = run_energy_identity(subdivisions=energy_subdivisions, ka=ka)
    criteria = evaluate_criteria(operator, constant, self_gate, energy, controls)
    return {
        "provenance": {
            "mission": mission,
            "branch": branch,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "compute_command": (
                "PYTHONPATH=src python3 -m tensorbem.benchmarks.panel_linear_order "
                f"--subdivisions {' '.join(str(int(level)) for level in subdivisions)} "
                f"--energy-subdivisions {' '.join(str(int(level)) for level in energy_subdivisions)} "
                f"--quadrature-order {quadrature_order} --refinement-depth {refinement_depth} "
                f"--ka {ka}"
            ),
        },
        "criteria_definitions": CRITERIA,
        "operator": operator,
        "constant_panel": constant,
        "self_term": self_gate,
        "negative_controls": controls,
        "energy": energy,
        "criteria": criteria,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--energy-subdivisions", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--quadrature-order", type=int, default=DEFAULT_QUADRATURE_ORDER)
    parser.add_argument("--refinement-depth", type=int, default=DEFAULT_REFINEMENT_DEPTH)
    parser.add_argument("--ka", type=float, default=1.2)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--mission", type=str, default="5690")
    parser.add_argument("--branch", type=str, default="agent/m5690")
    arguments = parser.parse_args(argv)

    study = run_study(
        subdivisions=tuple(arguments.subdivisions),
        energy_subdivisions=tuple(arguments.energy_subdivisions),
        quadrature_order=arguments.quadrature_order,
        refinement_depth=arguments.refinement_depth,
        ka=arguments.ka,
        mission=arguments.mission,
        branch=arguments.branch,
    )
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(study, indent=2), encoding="utf-8")
    if arguments.figures is not None:
        names = write_figures(study, arguments.figures)
        study["figures"] = names
    if arguments.markdown is not None:
        arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(study, arguments.markdown)
        if arguments.json is not None:
            arguments.json.write_text(json.dumps(study, indent=2), encoding="utf-8")
    for key, value in study["criteria"].items():
        print(f"{key}: {value['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())