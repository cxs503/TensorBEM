"""S-175 oblique-wave RAOs from a six-mode advancing-ship BEM, scored against ITTC tank markers."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tensorbem.forward_speed import solve_forward_speed_modes
from tensorbem.geometry3d import WettedHullMesh
from tensorbem.hydrostatic import (
    compute_mass_matrix,
    compute_hydrostatic_restoring_matrix,
)
from tensorbem.rao import dynamic_stiffness
from benchmarks.marine_experiments.mesh import mesh_from_patch
from benchmarks.marine_experiments.io import write
from .import_data import ROOT, SCALE, GRIDS

G, RHO, FN = 9.81, 1025.0, 0.275
L = 175.0 / SCALE
MASS = 24609620.0 / SCALE**3
CG_X = -2.5474801 / SCALE
CG = np.array([0.0, 0.0, 0.0500002 / SCALE])
INERTIA = MASS * np.diag((np.array([8.331, 42.0, 42.0]) / SCALE) ** 2)
INDEX = {"heave": 2, "roll": 3, "pitch": 4}


def model(girth):
    with np.load(ROOT / "geometry.npz") as data:
        mesh = mesh_from_patch(data[f"patch{girth}"])
    mesh = WettedHullMesh(mesh.vertices - [CG_X, 0, 0], mesh.faces)
    mass = compute_mass_matrix(MASS, CG, INERTIA)
    c = compute_hydrostatic_restoring_matrix(mesh, MASS, CG, RHO, G).restoring_matrix
    return mesh, mass, c


def calculate(task):
    girth, beta, frequency = task
    mesh, mass, c = model(girth)
    omega = frequency * np.sqrt(G / L)
    k = omega**2 / G
    hyd = solve_forward_speed_modes(
        mesh,
        omega,
        FN * np.sqrt(G * L),
        wave_angle=np.deg2rad(beta),
        density=RHO,
        gravity=G,
    )
    z = dynamic_stiffness(hyd.encounter_omega, mass, hyd.added_mass, hyd.damping, c)
    motion = np.linalg.solve(z, hyd.excitation)
    # External MSS input (20% of critical roll damping) is shown as a sensitivity.
    # Use the current effective roll inertia and restoring in this diagnostic.
    extra = 0.4 * np.sqrt(max(0.0, (mass[3, 3] + hyd.added_mass[3, 3]) * c[3, 3]))
    z_damped = z.copy()
    z_damped[3, 3] -= 1j * hyd.encounter_omega * extra
    damped = np.linalg.solve(z_damped, hyd.excitation)
    # Nondimensionalize generalized displacements and loads prior to condition estimate.
    scale = np.diag([L, L, L, 1, 1, 1])
    meta = dict(
        panels=len(mesh.faces),
        heading_deg=beta,
        omega_sqrt_L_g=frequency,
        encounter_ratio=hyd.encounter_omega / omega,
        volume_rel=mesh.volume / (MASS / RHO) - 1,
        hydro_condition=hyd.condition_number,
        motion_condition=np.linalg.cond(scale @ z @ scale),
        bem_residual=hyd.relative_residual,
    )
    return {
        name: dict(
            **meta,
            predicted=abs(motion[j]) / (1 if j == 2 else k),
            roll_damping_sensitivity=abs(damped[j]) / (1 if j == 2 else k),
        )
        for name, j in INDEX.items()
    }


def run(workers):
    with (ROOT / "observations.csv").open() as f:
        obs = list(csv.DictReader(f))
    cases = sorted({(int(r["heading_deg"]), float(r["omega_sqrt_L_g"])) for r in obs})
    tasks = [(g, *case) for g, _ in GRIDS for case in cases]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        values = dict(zip(tasks, pool.map(calculate, tasks), strict=True))
    rows = []
    for g, _ in GRIDS:
        for r in obs:
            prediction = values[g, int(r["heading_deg"]), float(r["omega_sqrt_L_g"])][
                r["quantity"]
            ]
            rows.append(
                dict(
                    r,
                    **{k: v for k, v in prediction.items() if k not in r},
                    error=prediction["predicted"] - float(r["observed"]),
                )
            )
    scores = []
    for panels in sorted({r["panels"] for r in rows}):
        for q, h in sorted({(r["quantity"], int(r["heading_deg"])) for r in rows}):
            group = [
                r
                for r in rows
                if r["quantity"] == q
                and int(r["heading_deg"]) == h
                and r["panels"] == panels
            ]
            y = np.array([float(r["observed"]) for r in group])
            s = max(abs(y))
            for key in ("predicted", "roll_damping_sensitivity"):
                err = (np.array([r[key] for r in group]) - y) / s
                rms = np.sqrt(np.mean(err**2))
                maximum = max(abs(err))
                scores.append(
                    dict(
                        panels=panels,
                        quantity=q,
                        heading_deg=h,
                        route=key,
                        points=len(y),
                        scale=s,
                        nrms=rms,
                        nmax=maximum,
                        verdict=(
                            (
                                "PASS"
                                if rms <= 0.2
                                and maximum <= 0.4
                                and abs(group[0]["volume_rel"]) <= 0.01
                                else "FAIL"
                            )
                            if key == "predicted"
                            else "diagnostic"
                        ),
                    )
                )
    write(ROOT / "results.csv", list(rows[0]), rows)
    write(ROOT / "scores.csv", list(scores[0]), scores)
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    scores = run(args.workers)
    if args.require_pass and any(
        r["panels"] == max(s["panels"] for s in scores) and r["verdict"] == "FAIL"
        for r in scores
    ):
        raise SystemExit(1)
