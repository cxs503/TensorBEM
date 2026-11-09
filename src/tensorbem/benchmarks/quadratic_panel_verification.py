"""Quadratic panel convergence verification against analytical solutions.

Measures the convergence order of the quadratic (6-node) panel solver for the
exterior 3-D Laplace Dirichlet problem on a unit sphere with an interior point source.
"""
from __future__ import annotations
import json, math, argparse
from pathlib import Path
import numpy as np
from ..panel_quadratic import icosphere_quadratic, solve_laplace_dirichlet_quadratic
from ..solver3d import solve_laplace_surface
from ..geometry3d import icosphere

FOUR_PI = 4.0 * math.pi
SOURCE = np.array([0.0, 0.0, 0.5])

def analytic_potential(pts):
    r = np.linalg.norm(pts - SOURCE, axis=1)
    return 1.0 / (FOUR_PI * r)

def analytic_flux(pts, normals):
    delta = pts - SOURCE
    r = np.linalg.norm(delta, axis=1)
    grad = -delta / (FOUR_PI * r[:, None]**3)
    return np.einsum('ij,ij->i', grad, normals)

def run_quadratic_study():
    results = []
    for sub in [0, 1, 2]:
        mesh = icosphere_quadratic(subdivisions=sub)
        u_exact = analytic_potential(mesh.vertices)
        result = solve_laplace_dirichlet_quadratic(mesh, u_exact)
        # Recover flux at element centroids for comparison
        elem_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
        elem_normals = np.zeros_like(elem_centers)
        for e in range(mesh.element_count):
            v0, v1, v2 = mesh.vertices[mesh.faces[e]]
            n = np.cross(v1 - v0, v2 - v0)
            elem_normals[e] = n / np.linalg.norm(n)
        q_exact = analytic_flux(elem_centers, elem_normals)
        # Quadratic q is at nodes; average to elements
        q_nodal = result.nodal_values
        q_elem = np.mean(q_nodal[mesh.faces], axis=1)
        err = np.linalg.norm(q_elem - q_exact) / np.linalg.norm(q_exact)
        results.append({
            "subdivisions": sub,
            "nodes": mesh.node_count,
            "elements": mesh.element_count,
            "h_mean": float(mesh.vertices.std()),
            "flux_rel_error": float(err),
            "condition_number": float(result.condition_number),
            "relative_residual": float(result.relative_residual_l2),
        })
    return results

def compute_order(results):
    orders = []
    for i in range(len(results) - 1):
        h1, h2 = results[i]["h_mean"], results[i+1]["h_mean"]
        e1, e2 = results[i]["flux_rel_error"], results[i+1]["flux_rel_error"]
        if e1 > 0 and e2 > 0 and h1 > h2:
            orders.append(math.log(e1 / e2) / math.log(h1 / h2))
    return orders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="outputs/m7534/quadratic_verification.json")
    ap.add_argument("--figure", type=str, default="outputs/m7534/fig_quadratic_convergence.png")
    args = ap.parse_args()
    
    results = run_quadratic_study()
    orders = compute_order(results)
    avg_order = float(np.mean(orders)) if orders else 0.0
    
    output = {"quadratic_study": results, "observed_orders": orders, "average_order": avg_order}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Quadratic panel verification: avg order = {avg_order:.2f}")
    print(f"Results written to {args.json}")

if __name__ == "__main__":
    main()
