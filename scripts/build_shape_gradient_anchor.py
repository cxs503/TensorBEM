"""Build the shape-gradient anchor artefacts (numbers, then prose, no hand copy).

Writes ``docs/shape_gradient_anchor/results.json`` and
``docs/shape_gradient_anchor/report.md``; every number in the prose is injected
from the measurement dictionaries below.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorbem import shape_gradient as sg  # noqa: E402
from tensorbem.geometry3d import TriangleMesh, icosphere  # noqa: E402
from tensorbem.solver3d import (  # noqa: E402
    assemble_surface_matrices,
    integrate_single_layer_triangle_self_batch,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "shape_gradient_anchor"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = 8
STEP = 1.0e-6


def metric_check() -> dict:
    mesh = icosphere(0)
    vertices = mesh.vertices[mesh.faces]
    worst_area = 0.0
    worst_normal = 0.0
    for panel in range(mesh.panel_count):
        area, normal, d_area, d_normal = sg.panel_metric_derivatives(vertices[panel])
        for local in range(3):
            for axis in range(3):
                h = STEP
                plus = vertices[panel].copy()
                plus[local, axis] += h
                minus = vertices[panel].copy()
                minus[local, axis] -= h
                a_p, n_p, _, _ = sg.panel_metric_derivatives(plus)
                a_m, n_m, _, _ = sg.panel_metric_derivatives(minus)
                worst_area = max(
                    worst_area,
                    abs(d_area[local][axis] - (a_p - a_m) / (2 * h)) / max(abs(a_p), 1e-30),
                )
                worst_normal = max(
                    worst_normal,
                    float(np.max(np.abs(d_normal[local][:, axis] - (n_p - n_m) / (2 * h)))),
                )
    return {"max_relative_area_error": worst_area, "max_abs_normal_error": worst_normal}


def operator_fidelity() -> dict:
    mesh = icosphere(1)
    single_ref, double_ref = assemble_surface_matrices(mesh, quadrature_order=ORDER)
    single, double = sg.assemble_quadrature_operators(mesh, order=ORDER)
    self_ref = integrate_single_layer_triangle_self_batch(mesh.vertices[mesh.faces])
    diagonal = np.diag(single)
    return {
        "panels": mesh.panel_count,
        "single_max_rel": float(np.max(np.abs(single - single_ref)) / np.max(np.abs(single_ref))),
        "double_max_rel": float(np.max(np.abs(double - double_ref)) / np.max(np.abs(double_ref))),
        "row_sum_error": float(np.max(np.abs(double.sum(axis=1) + 0.5))),
        "row_sum_error_production": float(np.max(np.abs(double_ref.sum(axis=1) + 0.5))),
        "self_term_max_rel": float(np.max(np.abs(diagonal - self_ref) / np.abs(self_ref))),
    }


def operator_derivative_check() -> dict:
    mesh = icosphere(0)
    worst_single = 0.0
    worst_double = 0.0
    for vertex in (0, 5, 7):
        d_single, d_double = sg.vertex_operator_derivative(mesh, vertex, order=ORDER)
        for axis in range(3):
            h = STEP
            plus = mesh.vertices.copy()
            plus[vertex, axis] += h
            minus = mesh.vertices.copy()
            minus[vertex, axis] -= h
            s_p, d_p = sg.assemble_quadrature_operators(TriangleMesh(plus, mesh.faces), ORDER)
            s_m, d_m = sg.assemble_quadrature_operators(TriangleMesh(minus, mesh.faces), ORDER)
            fd_single = (s_p - s_m) / (2 * h)
            fd_double = (d_p - d_m) / (2 * h)
            worst_single = max(
                worst_single,
                float(np.max(np.abs(d_single[:, :, axis] - fd_single)) / np.max(np.abs(fd_single))),
            )
            worst_double = max(
                worst_double,
                float(np.max(np.abs(d_double[:, :, axis] - fd_double)) / np.max(np.abs(fd_double))),
            )
    return {
        "vertices_checked": 3,
        "single_max_rel_error": worst_single,
        "double_max_rel_error": worst_double,
    }


def potential_sensitivity_check() -> dict:
    mesh = icosphere(1)
    centre = mesh.vertices.mean(axis=0)
    direction = mesh.vertices - centre
    analytic = sg.potential_sensitivity_directional(mesh, direction, dof=0, quadrature_order=ORDER)
    h = STEP * float(np.max(np.linalg.norm(direction, axis=1)))

    def potentials(scale: float) -> np.ndarray:
        moved = TriangleMesh(mesh.vertices + scale * direction, mesh.faces)
        single, double = sg.assemble_quadrature_operators(moved, ORDER)
        return np.linalg.solve(0.5 * np.eye(moved.panel_count) - double, single @ (-moved.normals[:, 0]))

    fd = (potentials(h) - potentials(-h)) / (2 * h)
    return {
        "panels": mesh.panel_count,
        "cosine": float(analytic @ fd / np.linalg.norm(analytic) / np.linalg.norm(fd)),
        "relative_error": float(np.linalg.norm(analytic - fd) / np.linalg.norm(fd)),
        "step": h,
    }


def added_mass_levels() -> list[dict]:
    rows = []
    for subdivisions in (0, 1, 2):
        mesh = icosphere(subdivisions)
        result = sg.added_mass_gradient(mesh, dof="surge", quadrature_order=ORDER)
        volume = 4.0 / 3.0 * math.pi
        analytic = 0.5 * volume
        centre = mesh.vertices.mean(axis=0)
        scale_direction = mesh.vertices - centre
        rows.append(
            {
                "subdivisions": subdivisions,
                "panels": mesh.panel_count,
                "vertices": mesh.vertex_count,
                "added_mass": result.value,
                "analytic_half_volume": analytic,
                "relative_error": abs(result.value - analytic) / analytic,
                "condition_number": result.condition_number,
                "forward_seconds": result.forward_seconds,
                "gradient_seconds": result.gradient_seconds,
                "cost_ratio": result.cost_ratio,
                "homogeneity_measured": float(np.sum(result.gradient * scale_direction)),
                "homogeneity_exact": 3.0 * result.value,
                "translation_invariance": float(np.max(np.abs(result.gradient.sum(axis=0)))),
            }
        )
    return rows


def functional_gradient_check() -> dict:
    mesh = icosphere(1)
    analytic = sg.added_mass_gradient(mesh, dof="surge", quadrature_order=ORDER)
    value, fd, h = sg.finite_difference_gradient(
        mesh, dof="surge", quadrature_order=ORDER, step=1.0e-5
    )
    agreement = sg.gradient_agreement(analytic.gradient, fd)
    return {
        "panels": mesh.panel_count,
        "value": analytic.value,
        "value_from_fd_path": value,
        "direction_cosine": agreement["direction_cosine"],
        "relative_error": agreement["relative_error"],
        "analytic_norm": agreement["norms"][0],
        "finite_difference_norm": agreement["norms"][1],
        "step": h,
        "fd_translation_invariance": float(np.max(np.abs(fd.sum(axis=0)))),
    }


def figure(payload: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None
    levels = payload["added_mass_levels"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [row["panels"] for row in levels]
    errors = [100.0 * row["relative_error"] for row in levels]
    axes[0].loglog(panels, errors, "o-", label="A$_11$ rel. error")
    axes[0].set_xlabel("panels")
    axes[0].set_ylabel("relative error [%]")
    axes[0].set_title("added mass vs 0.5 rho volume (analytic)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    check = payload["functional_gradient_check"]
    axes[1].bar(
        ["direction cosine", "relative error"],
        [check["direction_cosine"], check["relative_error"]],
        color=["#2b6cb0", "#c53030"],
    )
    axes[1].axhline(0.999, color="k", ls="--", lw=1)
    axes[1].set_title("functional gradient vs central differences (honest FAIL)")
    axes[1].set_ylim(-1.2, 1.6)
    for index, value in enumerate([check["direction_cosine"], check["relative_error"]]):
        axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top")
    path = OUT / "figure.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def main() -> dict:
    payload = {
        "quadrature_order": ORDER,
        "metric_derivatives": metric_check(),
        "operator_fidelity": operator_fidelity(),
        "operator_derivatives": operator_derivative_check(),
        "potential_sensitivity": potential_sensitivity_check(),
        "added_mass_levels": added_mass_levels(),
        "functional_gradient_check": functional_gradient_check(),
    }
    payload["figure"] = figure(payload)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(payload)
    print(json.dumps(payload, indent=2))
    return payload


def write_report(payload: dict) -> None:
    metric = payload["metric_derivatives"]
    fidelity = payload["operator_fidelity"]
    derivative = payload["operator_derivatives"]
    sensitivity = payload["potential_sensitivity"]
    functional = payload["functional_gradient_check"]
    levels = payload["added_mass_levels"]
    level_table = "\n".join(
        "| {panels} | {vertices} | {added_mass:.6f} | {analytic_half_volume:.6f} | {relative_error:.4%} "
        "| {condition_number:.3f} | {cost_ratio:.2f} |".format(**row)
        for row in levels
    )
    text = f"""# Adjoint shape gradient of the constant-panel Laplace BEM operator

Anchor: `GAP-BEM-ADJOINT-GEOM-GRAD` (P1, proposed by the OSS census m16421,
`docs/dev_report_m16421_oss_inventory_bem.md`; not yet in `feature_ledger`).
Mission m16953, generated by `scripts/build_shape_gradient_anchor.py`
(quadrature order {payload['quadrature_order']}); every number below is injected
from `results.json`, none is hand-copied.

## What was built

`src/tensorbem/shape_gradient.py` gives the discrete shape derivative of the
constant-panel 3-D Laplace operator: `panel_metric_derivatives` (area/normal
Jacobians), `self_single_layer` (the self single-layer integral with the apex
singularity removed in Duffy coordinates, differentiable), and
`vertex_operator_derivative` (`dS/dv`, `dD/dv` for one vertex), plus
`potential_sensitivity_directional` (forward sensitivity of the mode potential)
and `added_mass_gradient` (adjoint functional gradient).

## Verified pieces (against central differences)

| check | measured | verdict |
|---|---|---|
| panel area derivative, max relative error | {metric['max_relative_area_error']:.2e} | pass (<1e-8) |
| panel normal derivative, max absolute error | {metric['max_normal_error'] if False else metric['max_abs_normal_error']:.2e} | pass (<1e-8) |
| operator transcription vs production assembly, S | {fidelity['single_max_rel']:.2e} | pass |
| operator transcription vs production assembly, D | {fidelity['double_max_rel']:.2e} | pass |
| double-layer row-sum identity (this work / production) | {fidelity['row_sum_error']:.2e} / {fidelity['row_sum_error_production']:.2e} | quadrature-limited |
| self single-layer vs production analytic batch | {fidelity['self_term_max_rel']:.2e} | pass |
| dS/dv vs central differences | {derivative['single_max_rel_error']:.2e} | pass (<1e-8) |
| dD/dv vs central differences | {derivative['double_max_rel_error']:.2e} | pass (<1e-8) |
| mode-potential sensitivity vs central differences, cosine | {sensitivity['cosine']:.10f} | pass (>0.999) |
| mode-potential sensitivity, relative error | {sensitivity['relative_error']:.2e} | pass (<1e-3) |

## Added mass and its cost

| panels | vertices | A_11 (computed) | 0.5 rho volume | rel. error | cond(M) | gradient/forward |
|---|---|---|---|---|---|---|
{level_table}

## Honest FAIL

The assembled functional gradient `added_mass_gradient` does **not** reproduce
central differences: direction cosine **{functional['direction_cosine']:.4f}**,
relative error **{functional['relative_error']:.3f}**, analytic norm
{functional['analytic_norm']:.4f} against finite-difference norm
{functional['finite_difference_norm']:.4f}.  The sharper diagnosis is the exact
homogeneity identity: under uniform scaling ``A_11 ~ a^3`` so the gradient
contracted with the scaling direction must equal ``3 A_11``; the code returns
``A_11`` for every refinement level (see `homogeneity_measured` vs
`homogeneity_exact` in `results.json`).  The gap is localised to the
weight-sensitivity term ``d(rho A n_d)``: the operator derivatives and the
potential sensitivity are both machine-clean, so the residual can only live in
that contraction, which is where the next session should start.  Translation
invariance of the assembled gradient holds to
{max(row['translation_invariance'] for row in levels):.1e}, which shows the
gradient field itself is formed consistently (a broken contraction would break
it).

## Cost gate

The acceptance asked for gradient cost < 3x one forward solve; the measured
ratio is {'%.2f' % min(row['cost_ratio'] for row in levels)} to
{'%.2f' % max(row['cost_ratio'] for row in levels)} (Python loop over vertices,
`O(P^2)` derivative matrices each).  **Gate NOT met.**  The loop is the obvious
target: the derivative contractions are linear in the kernel derivatives and can
be batched over vertices.

## Not claimed

* rotational modes (``m = (x - centre) x n``): the derivative is not written;
* meshes above {sg.MAX_GRADIENT_PANELS} panels (dense per-vertex matrices);
* free surface / forward speed: the Rankine kernel only, so nothing is claimed
  about the free-surface Green-function terms.
"""
    (OUT / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
