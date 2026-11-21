"""Reproduce Journée Table 10-III with the zero-speed 3-D panel solvers."""

import argparse
import csv
from pathlib import Path

import numpy as np

from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)
from tensorbem.rao import dynamic_stiffness
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh

HERE = Path(__file__).resolve().parent
L, RHO, G = 3.0, 1000.0, 9.81
MODES = [2, 4]
NAMES = ["Heave", "Pitch"]
GRIDS = [(12, 6), (16, 8), (20, 10)]


def tensorbem_coefficients(mesh, omega):
    k = omega**2 / G
    ops = assemble_free_surface_operators_3d(mesh, k)
    args = dict(operators=ops, density=RHO, gravity=G)
    rad = solve_radiation_3d(mesh, k, modes=MODES, **args)
    diff = solve_diffraction_3d(mesh, k, wave_angle=np.pi, **args)
    return rad.added_mass, rad.damping, diff.excitation_force[MODES]


def capytaine_coefficients(mesh):
    import capytaine as cpt
    from capytaine.bem.airy_waves import froude_krylov_force

    faces = np.column_stack([mesh.faces, mesh.faces[:, 0]])
    body = cpt.FloatingBody(mesh=cpt.Mesh(vertices=mesh.vertices.copy(), faces=faces))
    body.add_all_rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0))
    solver = cpt.BEMSolver()

    def coefficients(_, omega, forward_speed=0.0):
        common = dict(
            body=body,
            omega=omega,
            g=G,
            rho=RHO,
            forward_speed=forward_speed,
            wave_direction=np.pi,
        )
        rs = [
            solver.solve(cpt.RadiationProblem(**common, radiating_dof=d)) for d in NAMES
        ]
        problem = cpt.DiffractionProblem(**common)
        diff = solver.solve(problem)
        fk = froude_krylov_force(problem)
        return (
            np.array([[r.added_mass[d] for r in rs] for d in NAMES]),
            np.array([[r.radiation_damping[d] for r in rs] for d in NAMES]),
            np.array([diff.forces[d] + fk[d] for d in NAMES]),
        )

    return coefficients


def model_matrices(mesh, *, mass=RHO * 0.078, cg_z=0.170 - 0.1875):
    """Return the two constrained model matrices about the waterplane origin."""
    cg = np.array([0.0, 0.0, cg_z])
    # Only the pitch entry is retained after projecting the rigid-body matrix.
    inertia = np.diag([0.0, mass * 0.75**2, 0.0])
    block = np.ix_(MODES, MODES)
    m = compute_mass_matrix(mass, cg, inertia)[block]
    c = compute_hydrostatic_restoring_matrix(mesh, mass, cg, RHO, G).restoring_matrix[
        block
    ]
    return m, c


def calculate(solver, output, short_wave=False):
    measurements = np.genfromtxt(HERE / "table10.csv", delimiter=",", names=True)
    ratios = [0.5] if short_wave else np.unique(measurements["lambda_L"])
    grids = [(24, 12), (28, 14)] if short_wave else GRIDS
    with output.open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            "nx nz panels lambda_L omega M33 M55 C33 C35 C53 C55 A33 A35 A53 A55 B33 B35 B53 B55 F3_re F3_im F5_re F5_im heave heave_deg pitch pitch_deg".split()
        )
        for nx, nz in grids:
            mesh = wigley3_wetted_hull_mesh(nx=nx, nz=nz, alpha=0.2)
            m, c = model_matrices(mesh)
            coeffs = (
                tensorbem_coefficients
                if solver == "tensorbem"
                else capytaine_coefficients(mesh)
            )
            for ratio in ratios:
                omega = np.sqrt(G * 2 * np.pi / (L * ratio))
                a, b, f = coeffs(mesh, omega)
                xi = np.linalg.solve(dynamic_stiffness(omega, m, a, b, c), f)
                phase = -np.angle(xi, deg=True) % 360
                values = [
                    ratio,
                    omega,
                    *m.diagonal(),
                    *c.flat,
                    *a.flat,
                    *b.flat,
                    f[0].real,
                    f[0].imag,
                    f[1].real,
                    f[1].imag,
                    abs(xi[0]),
                    phase[0],
                    abs(xi[1]) * L / (2 * np.pi),
                    phase[1],
                ]
                writer.writerow(
                    [nx, nz, len(mesh.faces), *(f"{v:.12g}" for v in values)]
                )
                file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=("tensorbem", "capytaine"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--short-wave",
        action="store_true",
        help="refine the lambda/L=0.50 discrepancy on two additional grids",
    )
    args = parser.parse_args()
    suffix = "_half_length" if args.short_wave else ""
    calculate(
        args.solver, args.output or HERE / f"{args.solver}{suffix}.csv", args.short_wave
    )
