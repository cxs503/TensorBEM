import numpy as np
from tensorbem.geometry import regular_polygon
from tensorbem.solver import solve_laplace_exterior_neumann, evaluate_exterior_field
from tensorbem.benchmarks.case_ice_gate_power_section import doubled_rectangle_mesh

def solve(mesh, flip_half):
    normals = mesh.normals
    centers = mesh.centers
    upper = centers[:, 1] > 0.0
    lower = centers[:, 1] < 0.0
    flux = -normals[:, 1].copy()
    if flip_half == "upper":
        flux[upper] = normals[upper, 1]
    elif flip_half == "lower":
        flux[lower] = normals[lower, 1]
    res = solve_laplace_exterior_neumann(mesh, flux)
    return res, flux

A, T = 10.0, 6.0
mesh = doubled_rectangle_mesh(A, T, 32)
pts = np.column_stack((np.linspace(A * 1.05, A * 4.0, 7), np.zeros(7)))
for flip in ("none", "upper", "lower"):
    res, flux = solve(mesh, flip)
    field = evaluate_exterior_field(mesh, res, pts)
    print("flip=%-6s net_flux=%+.4f  max|phi|_wall=%.4f  max|dphi/dy|_wall=%.4f  A=%.1f" % (
        flip, float(np.sum(flux * mesh.lengths)),
        float(np.max(np.abs(field.potential))),
        float(np.max(np.abs(field.gradient[:, 1]))),
        -1025.0 * np.sum(res.boundary_values[mesh.centers[:, 1] < 0] * mesh.normals[mesh.centers[:, 1] < 0, 1] * mesh.lengths[mesh.centers[:, 1] < 0]) / 1000.0))

# also the half-immersed circle (full circle mesh = doubled body of the half circle)
circ = regular_polygon(128, radius=1.0)
pts2 = np.column_stack((np.linspace(1.05, 4.0, 7), np.zeros(7)))
for flip in ("none", "upper"):
    res, flux = solve(circ, flip)
    field = evaluate_exterior_field(circ, res, pts2)
    print("circle flip=%-6s net_flux=%+.4f max|phi|_wall=%.4f max|dphi/dy|_wall=%.4f A=%.1f" % (
        flip, float(np.sum(flux * circ.lengths)),
        float(np.max(np.abs(field.potential))),
        float(np.max(np.abs(field.gradient[:, 1]))),
        -1025.0 * np.sum(res.boundary_values[circ.centers[:, 1] < 0] * circ.normals[circ.centers[:, 1] < 0, 1] * circ.lengths[circ.centers[:, 1] < 0])))