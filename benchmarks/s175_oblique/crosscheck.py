"""Independent Capytaine 3.0.0 spot checks, using its uniform forward-speed approximation."""

import numpy as np
from tensorbem.rao import dynamic_stiffness
from benchmarks.marine_experiments.io import write
from .run import model, calculate, ROOT, RHO, G, L, FN, INDEX

NAMES = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


def compare():
    import capytaine as cpt
    from capytaine.bem.airy_waves import froude_krylov_force

    mesh, m, c = model(6)
    faces = np.column_stack([mesh.faces, mesh.faces[:, 0]])
    body = cpt.FloatingBody(mesh=cpt.Mesh(vertices=mesh.vertices, faces=faces))
    body.add_all_rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0))
    solver = cpt.BEMSolver()
    rows = []
    for beta, f in ((60, 2.295), (90, 2.5), (150, 2.195)):
        w = f * np.sqrt(G / L)
        k = w**2 / G
        heading = np.deg2rad(beta)
        u = FN * np.sqrt(G * L)
        common = dict(
            body=body, omega=w, rho=RHO, g=G, forward_speed=u, wave_direction=heading
        )
        rs = [
            solver.solve(cpt.RadiationProblem(**common, radiating_dof=d)) for d in NAMES
        ]
        problem = cpt.DiffractionProblem(**common)
        diff = solver.solve(problem)
        fk = froude_krylov_force(problem)
        a = np.array([[r.added_mass[d] for r in rs] for d in NAMES])
        b = np.array([[r.radiation_damping[d] for r in rs] for d in NAMES])
        xi = np.linalg.solve(
            dynamic_stiffness(problem.encounter_omega, m, a, b, c),
            [diff.forces[d] + fk[d] for d in NAMES],
        )
        native = calculate((6, beta, f))
        for q, j in INDEX.items():
            v = abs(xi[j]) / (1.0 if j == 2 else k)
            pred = native[q]["predicted"]
            rows.append(
                dict(
                    heading_deg=beta,
                    omega_sqrt_L_g=f,
                    quantity=q,
                    tensorbem=pred,
                    capytaine=v,
                    difference=pred - v,
                )
            )
    write(ROOT / "capytaine.csv", list(rows[0]), rows)


if __name__ == "__main__":
    compare()
