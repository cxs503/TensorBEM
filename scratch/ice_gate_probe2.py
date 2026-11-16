import numpy as np
from tensorbem.geometry import PanelMesh, regular_polygon
from tensorbem.solver import assemble_matrices, compute_added_mass_matrix_2d
from tensorbem.kernels import REFERENCE_LENGTH

rho = 1025.0

def rect_mesh(A, Hb, n_side):
    xs = np.linspace(-A, A, n_side + 1)
    ys = np.linspace(-Hb, Hb, n_side + 1)
    bottom = np.column_stack((xs[:-1], -Hb * np.ones(n_side)))
    right = np.column_stack((A * np.ones(n_side), ys[:-1]))
    top = np.column_stack((xs[::-1][:-1], Hb * np.ones(n_side)))
    left = np.column_stack((-A * np.ones(n_side), ys[::-1][:-1]))
    return PanelMesh(np.vstack([bottom, right, top, left]))

def half_heave(mesh, rho, symmetric=True):
    n = mesh.panel_count
    normals = mesh.normals
    centers = mesh.centers
    q = -normals[:, 1]          # fluid-domain normal derivative for V=+y (body half)
    q = q.copy()
    if not symmetric:
        lower = centers[:, 1] < 0.0
        q[lower] = normals[lower, 1]   # image body moves with -V
    single, double = assemble_matrices(mesh, reference_length=REFERENCE_LENGTH)
    system = 0.5 * np.eye(n) - double
    phi = np.linalg.solve(system, single @ q)
    upper = centers[:, 1] > 0.0
    a_half = -rho * np.sum(phi[upper] * normals[upper, 1] * mesh.lengths[upper])
    return a_half, phi, upper

# validation: circle -> symmetric half-domain = 0.5*pi*rho*a^2
for n in (64, 128, 256):
    m = regular_polygon(n, radius=1.0)
    a, _, _ = half_heave(m, rho, symmetric=True)
    print("circle sym   n=%4d A_half=%.6f exact=%.6f rel=%.2e" % (n, a, 0.5*np.pi*rho, abs(a-0.5*np.pi*rho)/(0.5*np.pi*rho)))
    a2, _, _ = half_heave(m, rho, symmetric=False)
    print("circle antisym n=%4d A_half=%.6f" % (n, a2))

A, Hb = 10.0, 6.0
for n in (8, 16, 32, 64, 128):
    m = rect_mesh(A, Hb, n)
    a0, _, _ = half_heave(m, rho, symmetric=True)
    ai, _, _ = half_heave(m, rho, symmetric=False)
    print("rect n=%4d  A0=%.3f t/m   Ainf=%.3f t/m" % (n, a0/1000, ai/1000))