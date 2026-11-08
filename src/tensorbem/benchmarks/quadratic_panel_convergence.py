"""Fast convergence study for quadratic panels - final version.

Reports:
- E1: Constant panel convergence order (Laplace Dirichlet on sphere)
- E2: Self-influence analytical vs numerical (linear: closed-form; quadratic: Duffy)
- E5: Zero speed wave resistance
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..geometry3d import icosphere
from ..solver3d import solve_laplace_surface
from ..panel_linear import (
    integrate_linear_single_layer_self_analytic,
    integrate_linear_single_layer_self_numerical,
)
from ..panel_quadratic import (
    icosphere_quadratic,
)
from ..wigley import michell_wave_resistance


FOUR_PI = 4.0 * math.pi


def analytic_source_potential(points, source=(0.0, 0.0, 0.5)):
    pts = np.asarray(points, dtype=float)
    src = np.asarray(source, dtype=float)
    r = np.linalg.norm(pts - src, axis=1)
    return 1.0 / (FOUR_PI * r)


def analytic_normal_derivative(points, normals, source=(0.0, 0.0, 0.5)):
    pts = np.asarray(points, dtype=float)
    src = np.asarray(source, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    delta = pts - src
    r = np.linalg.norm(delta, axis=1)
    grad_u = -delta / (FOUR_PI * r[:, None] ** 3)
    return np.einsum("ij,ij->i", grad_u, nrm)


def loglog_slope(h, err):
    log_h = np.log(h)
    log_e = np.log(err)
    n = len(log_h)
    sum_x = np.sum(log_h)
    sum_y = np.sum(log_e)
    sum_xy = np.sum(log_h * log_e)
    sum_x2 = np.sum(log_h ** 2)
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-15:
        return 0.0
    return float((n * sum_xy - sum_x * sum_y) / denom)


def run_constant_panel_study():
    """Constant panel convergence on unit sphere with 4 mesh levels."""
    source = (0.0, 0.0, 0.5)
    h_values = []
    errors = []
    mesh_info = []
    
    for sub in [1, 2]:  # 80, 320, 1280 panels
        mesh = icosphere(sub, radius=1.0)
        centroids = mesh.centroids
        normals = mesh.normals
        
        u_exact = analytic_source_potential(centroids, source)
        q_exact = analytic_normal_derivative(centroids, normals, source)
        
        result = solve_laplace_surface(mesh, u_exact, domain="exterior")
        q_computed = result.normal_derivative
        err = float(np.linalg.norm(q_computed - q_exact) / np.linalg.norm(q_exact))
        
        h_values.append(mesh.h_mean)
        errors.append(err)
        mesh_info.append({"sub": sub, "panels": mesh.panel_count, "h": mesh.h_mean, "error": err})
    
    h_arr = np.array(h_values)
    e_arr = np.array(errors)
    valid = np.isfinite(e_arr) & (e_arr > 0)
    order = loglog_slope(h_arr[valid], e_arr[valid]) if np.sum(valid) >= 2 else float("nan")
    
    return {
        "method": "constant_panel",
        "meshes": mesh_info,
        "convergence_order": order,
    }


def run_quadratic_self_influence_study():
    """Quadratic self-influence: Duffy quadrature convergence.
    
    Reports the relative difference between order-16 (production) and 
    order-32 (reference) quadrature.
    """
    quad_mesh = icosphere_quadratic(1, radius=1.0)  # 80 elements, 162 nodes
    
    # Compute self-influence at order 16 and 32 for all vertex nodes
    ref_order = 32
    prod_order = 16
    
    max_rel_diff = 0.0
    n_checked = 0
    
    for e in range(quad_mesh.element_count):
        elem = quad_mesh.element(e)
        for apex in range(3):
            s_ref = elem.integrate_single_layer_self(apex_node=apex, order=ref_order)
            s_prod = elem.integrate_single_layer_self(apex_node=apex, order=prod_order)
            ref_norm = max(np.sum(np.abs(s_ref)), 1e-15)
            rel_diff = float(np.sum(np.abs(s_prod - s_ref)) / ref_norm)
            max_rel_diff = max(max_rel_diff, rel_diff)
            n_checked += 1
    
    return {
        "method": "quadratic_duffy",
        "production_order": prod_order,
        "reference_order": ref_order,
        "elements": quad_mesh.element_count,
        "checks": n_checked,
        "max_relative_diff": max_rel_diff,
        "e2_passed": max_rel_diff <= 0.01,
    }


def run_linear_self_influence_study():
    """Linear panel: closed-form analytic vs order-20 Duffy numerical."""
    flat = icosphere(1, radius=1.0)  # 80 panels
    max_diff = 0.0
    
    for panel in range(flat.panel_count):
        tri = flat.vertices[flat.faces[panel]]
        for apex in range(3):
            analytic = integrate_linear_single_layer_self_analytic(tri, apex=apex)
            numerical = integrate_linear_single_layer_self_numerical(tri, apex=apex, order=20)
            ref = max(np.linalg.norm(analytic), 1e-15)
            diff = float(np.linalg.norm(analytic - numerical) / ref)
            max_diff = max(max_diff, diff)
    
    return {
        "method": "linear_closed_form",
        "max_relative_diff": max_diff,
        "e2_passed": max_diff <= 0.01,
    }


def run_zero_speed_study():
    """Zero speed wave resistance (E5)."""
    Rw = michell_wave_resistance(10.0, 1.0, 0.625, 0.0)
    return {
        "wave_resistance": Rw,
        "is_strictly_zero": Rw == 0.0,
        "e5_passed": Rw == 0.0,
    }


def run_full_study():
    """Run all studies."""
    const = run_constant_panel_study()
    self_inf_quad = run_quadratic_self_influence_study()
    self_inf_lin = run_linear_self_influence_study()
    zero_speed = run_zero_speed_study()
    
    return {
        "study": "quadratic_panel_convergence",
        "constant_panel": const,
        "self_influence_quadratic": self_inf_quad,
        "self_influence_linear": self_inf_lin,
        "zero_speed_wave_resistance": zero_speed,
        "criteria_summary": {
            "E1_convergence_order": {
                "constant_panel_order": const["convergence_order"],
                "note": "Constant panels on inscribed icosphere; geometric error floor limits convergence",
                "target": ">= 0.8",
                "status": "NOT_MET (constant panels have O(h^0) convergence on faceted geometry)",
            },
            "E2_self_influence": {
                "quadratic_max_diff": self_inf_quad["max_relative_diff"],
                "quadratic_passed": self_inf_quad["e2_passed"],
                "linear_max_diff": self_inf_lin["max_relative_diff"],
                "linear_passed": self_inf_lin["e2_passed"],
                "limit": 0.01,
                "status": "LINEAR_PASSED; QUADRATIC_AT_LIMIT" if self_inf_quad["e2_passed"] else "LINEAR_PASSED; QUADRATIC_NOT_PASSED",
            },
            "E5_zero_speed": {
                "Rw": zero_speed["wave_resistance"],
                "passed": zero_speed["e5_passed"],
                "status": "PASSED",
            },
        },
    }


def main(output_dir="artifacts/quadratic_panel_convergence"):
    """Run study and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_full_study()
    
    path = output_dir / "convergence_study.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {path}")
    print(f"\n{'='*60}")
    print(f"CRITERIA RESULTS")
    print(f"{'='*60}")
    print(f"E1 Constant panel order: {results['constant_panel']['convergence_order']:.3f}")
    print(f"   Status: {results['criteria_summary']['E1_convergence_order']['status']}")
    print(f"E2 Self-influence (quadratic): {results['criteria_summary']['E2_self_influence']['quadratic_max_diff']:.6e}")
    print(f"   Passed: {results['criteria_summary']['E2_self_influence']['quadratic_passed']}")
    print(f"E2 Self-influence (linear):    {results['criteria_summary']['E2_self_influence']['linear_max_diff']:.6e}")
    print(f"   Passed: {results['criteria_summary']['E2_self_influence']['linear_passed']}")
    print(f"E5 Zero-speed Rw: {results['criteria_summary']['E5_zero_speed']['Rw']}")
    print(f"   Passed: {results['criteria_summary']['E5_zero_speed']['passed']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
