"""Cross-check closed-form roll/pitch stiffness + body inertia against the mesh route."""
import sys

import numpy as np

sys.path.insert(0, "src")
from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.wigley_hydrostatics import wigley_hydrostatic_reference, wigley_wetted_hull_mesh

L, B, T = case.LENGTH, case.BEAM, case.DRAFT
rho, g = case.DENSITY, case.GRAVITY
ref = wigley_hydrostatic_reference(L, B, T, density=rho, gravity=g)
print("closed form: vol=%.6f KB=%.5f GM_T=%.5f GM_L=%.4f C33=%.2f C44=%.4f C55=%.1f"
      % (ref["volume"], ref["KB"], ref["GM_T"], ref["GM_L"], ref["C33"], ref["C55"], ref["C44"]))

for label, (ns, nl) in case.MESHES.items():
    mesh = wigley_wetted_hull_mesh(L, B, T, n_sections=ns, n_levels=nl)
    props = compute_hydrostatic_restoring_matrix(
        mesh, ref["volume"] * rho, np.array([0.0, 0.0, -T / 2])
    )
    c = np.asarray(props.restoring_matrix)
    print(f"mesh {label:6s} C33={c[2,2]:12.3f} C44={c[3,3]:10.4f} C55={c[4,4]:10.2f}"
          f"  C35={c[2,4]:.3e} C34={c[3,4]:.3e}")
    print("      rel err vs closed form:",
          f"C33 {abs(c[2,2]/ref['C33']-1):.2%}", f"C44(roll) {abs(c[3,3]/ref['C55']-1):.2%}",
          f"C55(pitch) {abs(c[4,4]/ref['C44']-1):.2%}")

# body inertia about the rotation centre (origin at waterline), uniform density
x = np.linspace(-L / 2, L / 2, 800)
z = np.linspace(-T, 0.0, 400)
X, Z = np.meshgrid(x, z, indexing="ij")
y = 0.5 * B * (1.0 - (2.0 * X / L) ** 2) * (1.0 - (Z / T) ** 2)
dx, dz = x[1] - x[0], z[1] - z[0]
vol = float(np.sum(2.0 * y) * dx * dz)
m = rho * vol
i_xx = rho * float(np.sum((2.0 / 3.0) * y**3 + 2.0 * y * Z**2) * dx * dz)
i_yy = rho * float(np.sum(2.0 * y * X**2 + 2.0 * y * Z**2) * dx * dz)
i_zz = rho * float(np.sum((2.0 / 3.0) * y**3 + 2.0 * y * X**2) * dx * dz)
print(f"uniform-density body: vol={vol:.6f} m={m:.4f} Ixx={i_xx:.5f} Iyy={i_yy:.5f} Izz={i_zz:.5f}")
print(f"  k_xx={np.sqrt(i_xx/m):.4f} m = {np.sqrt(i_xx/m)/B:.3f} B ; k_yy={np.sqrt(i_yy/m):.4f} m = {np.sqrt(i_yy/m)/L:.3f} L")