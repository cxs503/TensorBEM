"""Three-dimensional passive-body mean drift against both original zero-speed tank series."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.integrate import trapezoid

from benchmarks.journee_wigley3_zero_speed.run import MODES, NAMES, model_matrices
from benchmarks.marine_experiments.io import write
from benchmarks.marine_experiments.wigley import HULLS, L, mesh_for
from tensorbem.drift import far_field_mean_drift_3d
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.rao import dynamic_stiffness
from .run import ROOT, RHO, G


def native(mesh, k, m, c):
    ops = assemble_free_surface_operators_3d(mesh, k)
    kwargs = dict(operators=ops, density=RHO, gravity=G)
    rad = solve_radiation_3d(mesh, k, modes=MODES, **kwargs)
    diff = solve_diffraction_3d(mesh, k, wave_angle=np.pi, **kwargs)
    xi = np.linalg.solve(
        dynamic_stiffness(diff.omega, m, rad.added_mass, rad.damping, c),
        diff.excitation_force[MODES],
    )

    def resistance(motion):
        r = far_field_mean_drift_3d(
            mesh, diff, radiation=rad, motion=motion, density=RHO, n_azimuth=144
        )
        return -r["force"][0]

    return resistance(xi), resistance(np.zeros(2)), abs(xi[0]), abs(xi[1]), None


def external(mesh, k, m, c):
    import capytaine as cpt
    import xarray as xr
    from capytaine.bem.airy_waves import froude_krylov_force
    from capytaine.post_pro.kochin import compute_kochin
    from capytaine.post_pro.mean_drift_force import far_field_mean_drift_force

    faces = np.column_stack((mesh.faces, mesh.faces[:, 0]))
    body = cpt.FloatingBody(mesh=cpt.Mesh(vertices=mesh.vertices.copy(), faces=faces))
    body.add_all_rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0))
    solver = cpt.BEMSolver()
    w = np.sqrt(G * k)
    common = dict(body=body, omega=w, g=G, rho=RHO, wave_direction=np.pi)
    radios = [
        solver.solve(cpt.RadiationProblem(**common, radiating_dof=d), method="indirect")
        for d in NAMES
    ]
    problem = cpt.DiffractionProblem(**common)
    diff = solver.solve(problem, method="indirect")
    fk = froude_krylov_force(problem)
    a = np.array([[r.added_mass[d] for r in radios] for d in NAMES])
    b = np.array([[r.radiation_damping[d] for r in radios] for d in NAMES])
    xi = np.linalg.solve(
        dynamic_stiffness(w, m, a, b, c), [diff.forces[d] + fk[d] for d in NAMES]
    )
    theta = np.linspace(0, 2 * np.pi, 145)
    hd = compute_kochin(diff, theta)
    hr = np.array([compute_kochin(r, theta) for r in radios])
    ds = xr.Dataset(
        coords=dict(omega=[w], wave_direction=[np.pi], theta=theta, radiating_dof=NAMES)
    )
    ds["rho"], ds["g"], ds["water_depth"], ds["wavenumber"] = RHO, G, np.inf, k
    ds["kochin_diffraction"] = (("omega", "wave_direction", "theta"), hd[None, None, :])
    ds["kochin_radiation"] = (("omega", "radiating_dof", "theta"), hr[None, :, :])

    def resistance(motion):
        x = xr.DataArray(
            motion[None, None, :],
            dims=("omega", "wave_direction", "radiating_dof"),
            coords={
                key: ds[key] for key in ("omega", "wave_direction", "radiating_dof")
            },
        )
        return (
            -far_field_mean_drift_force(x, ds)["drift_force_surge"].values.item().real
        )

    # Capytaine source strengths use its 1/(4 pi) Kochin convention.
    pattern = abs(hd + xi @ hr) ** 2
    reduced = 2 * np.pi * RHO * k * k * trapezoid(pattern * (1 + np.cos(theta)), theta)
    return resistance(xi), resistance(np.zeros(2)), abs(xi[0]), abs(xi[1]), reduced


def calculate(task):
    solver, hull, ratio, nx, nz = task
    mesh = mesh_for(hull, nx, nz)
    beam, mass, cg = HULLS[hull]
    m, c = model_matrices(mesh, mass=mass, cg_z=cg)
    scale = RHO * G * beam**2 / L
    force, fixed, heave, pitch, reduced = {"tensorbem": native, "capytaine": external}[
        solver
    ](mesh, 2 * np.pi / (L * ratio), m, c)
    return dict(
        solver=solver,
        hull=hull,
        lambda_L=ratio,
        panels=len(mesh.faces),
        free=force / scale,
        fixed=fixed / scale,
        heave=heave,
        pitch=pitch * L / (2 * np.pi),
        reduced=reduced / scale if reduced is not None else "",
    )


def run(solver, workers):
    with (ROOT / "trials.csv").open() as f:
        trials = [r for r in csv.DictReader(f) if r["source_table"] == "10"]
    tasks = [
        (solver, h, x, n, z)
        for n, z in ((12, 6), (16, 8), (20, 10))
        for h, x in sorted({(r["hull"], float(r["lambda_L"])) for r in trials})
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(calculate, tasks))
    scores = []
    for hull in ("III", "IV"):
        obs = [r for r in trials if r["hull"] == hull]
        y = np.array([float(r["observed"]) for r in obs])
        x = np.array([float(r["lambda_L"]) for r in obs])
        unique = np.unique(x)
        ip = np.argmax([y[x == v].mean() for v in unique])
        for n in sorted({r["panels"] for r in values}):
            coeffs = {
                r["lambda_L"]: r["free"]
                for r in values
                if r["panels"] == n and r["hull"] == hull
            }
            predicted = np.array([coeffs[v] for v in x])
            err = (y - predicted) / max(abs(y))
            rms = np.sqrt(np.mean(err**2))
            maximum = max(abs(err))
            peak = np.argmax([coeffs[v] for v in unique])
            scores.append(
                dict(
                    solver=solver,
                    hull=hull,
                    panels=n,
                    trials=len(y),
                    nrms=rms,
                    nmax=maximum,
                    measured_peak=unique[ip],
                    predicted_peak=unique[peak],
                    verdict=(
                        "PASS"
                        if rms <= 0.2 and maximum <= 0.4 and abs(ip - peak) <= 1
                        else "FAIL"
                    ),
                )
            )
    write(ROOT / ("drift_" + solver + ".csv"), list(values[0]), values)
    write(ROOT / ("drift_" + solver + "_scores.csv"), list(scores[0]), scores)
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("solver", choices=("tensorbem", "capytaine"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--require-pass", action="store_true")
    a = p.parse_args()
    scores = run(a.solver, a.workers)
    if a.require_pass and any(
        r["verdict"] == "FAIL" and r["panels"] == 760 for r in scores
    ):
        raise SystemExit(1)
