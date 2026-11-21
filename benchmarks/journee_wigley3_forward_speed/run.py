"""Run from the repository root with python -m benchmarks.journee_wigley3_forward_speed.run."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from benchmarks.journee_wigley3_zero_speed.run import (
    G,
    GRIDS,
    L,
    RHO,
    capytaine_coefficients,
    model_matrices,
)
from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from tensorbem.rao import dynamic_stiffness
from tensorbem.stf_fullspeed import encounter_frequency
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh

ROOT = Path(__file__).resolve().parent


def calculate(solver, output):
    with (ROOT / "motions.csv").open() as f:
        cases = sorted(
            {(float(row["Fn"]), float(row["lambda_L"])) for row in csv.DictReader(f)}
        )
    # A failed sweep must leave the last complete set of predictions readable.
    with TemporaryDirectory(dir=output.parent) as directory:
        pending = Path(directory) / output.name
        with pending.open("w") as f:
            _write_sweep(solver, cases, f)
        pending.replace(output)


def _write_sweep(solver, cases, f):
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        "panels Fn lambda_L U k omega omega_e M33 M55 C33 C35 C53 C55 A33 A35 A53 A55 B33 B35 B53 B55 F3_re F3_im F5_re F5_im heave heave_deg pitch pitch_deg relative_residual condition_number".split()
    )
    for nx, nz in GRIDS:
        mesh = wigley3_wetted_hull_mesh(nx=nx, nz=nz)
        m, c = model_matrices(mesh)
        if solver == "capytaine":
            external = capytaine_coefficients(mesh)
        for fn, ratio in cases:
            speed = fn * np.sqrt(G * L)
            k = 2 * np.pi / (L * ratio)
            omega = np.sqrt(G * k)
            we = encounter_frequency(omega, speed, np.pi, G)
            if solver == "tensorbem":
                result = solve_forward_speed_heave_pitch(
                    mesh, omega, speed, density=RHO, gravity=G
                )
                a, b, force = result.added_mass, result.damping, result.excitation
                diagnostics = [result.relative_residual, result.condition_number]
            else:
                a, b, force = external(mesh, omega, forward_speed=speed)
                diagnostics = []
            xi = np.linalg.solve(dynamic_stiffness(we, m, a, b, c), force)
            phase = -np.angle(xi, deg=True) % 360
            values = [
                fn,
                ratio,
                speed,
                k,
                omega,
                we,
                *m.diagonal(),
                *c.flat,
                *a.flat,
                *b.flat,
                force[0].real,
                force[0].imag,
                force[1].real,
                force[1].imag,
                abs(xi[0]),
                phase[0],
                abs(xi[1]) * L / (2 * np.pi),
                phase[1],
                *diagnostics,
            ]
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Non-finite output at {solver}, Fn={fn}, lambda/L={ratio}"
                )
            writer.writerow(
                [
                    len(mesh.faces),
                    *(f"{v:.12g}" for v in values),
                    *([] if diagnostics else ["", ""]),
                ]
            )
            f.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("solver", choices=("tensorbem", "capytaine"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    calculate(args.solver, args.output or ROOT / (args.solver + ".csv"))
