"""Run the double-body Eulerian load approximation at every measured condition."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from benchmarks.journee_wigley3_zero_speed.run import GRIDS
from .assess import ROOT, GRAVITY, LENGTH, table
from .strip import COEFFICIENTS
from tensorbem.double_body_flow import prepare_double_body_flow
from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh


@lru_cache(maxsize=1)
def basis(nx, nz):
    mesh = wigley3_wetted_hull_mesh(nx=nx, nz=nz)
    return prepare_double_body_flow(mesh)


def evaluate(case):
    nx, nz, kind, fn, key = case
    flow = basis(nx, nz)
    speed = fn * np.sqrt(GRAVITY * LENGTH)
    if kind == "motion":
        omega = np.sqrt((2 * np.pi * GRAVITY) / (key * LENGTH))
    else:
        encounter = key * np.sqrt(GRAVITY / LENGTH)
        omega = 2 * encounter / (1 + np.sqrt(1 + 4 * speed * encounter / GRAVITY))
    r = solve_forward_speed_heave_pitch(
        flow.mesh, omega, speed, base_flow=flow, density=1000.0, gravity=GRAVITY
    )
    values = (
        r.encounter_omega,
        *r.added_mass.flat,
        *r.damping.flat,
        r.excitation[0].real,
        r.excitation[0].imag,
        r.excitation[1].real,
        r.excitation[1].imag,
        flow.relative_residual,
        flow.condition_number,
        flow.max_patch_condition,
        r.relative_residual,
        r.condition_number,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite result at {case}")
    return [
        flow.mesh.panel_count,
        kind,
        fn,
        key if kind == "motion" else "",
        key if kind == "forced" else "",
        *[f"{v:.12g}" for v in values],
    ]


def calculate(output, workers=1):
    cases = []
    for kind, file, coord in [
        ("motion", "motions", "lambda_L"),
        ("forced", "radiation", "w_nd"),
    ]:
        observed = table(ROOT, file)
        cases.extend(
            (kind, fn, key)
            for fn, key in sorted(set(zip(observed["Fn"], observed[coord])))
        )
    tasks = [(nx, nz, *case) for nx, nz in GRIDS for case in cases]
    with (
        TemporaryDirectory(dir=output.parent) as d,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        tmp = Path(d) / output.name
        with tmp.open("w") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "panels",
                    "kind",
                    "Fn",
                    "lambda_L",
                    "w_nd",
                    "omega_e",
                    *COEFFICIENTS,
                    "F3_re",
                    "F3_im",
                    "F5_re",
                    "F5_im",
                    "steady_residual",
                    "steady_condition",
                    "patch_condition",
                    "wave_residual",
                    "wave_condition",
                ]
            )
            for row in pool.map(evaluate, tasks):
                writer.writerow(row)
                stream.flush()
        tmp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "double_body_coefficients.csv"
    )
    args = parser.parse_args()
    calculate(args.output, args.workers)
