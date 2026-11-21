"""Predict Wigley III radiation from the measured hull geometry and wave frequency."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .assess import ROOT, GRAVITY, LENGTH, table
from tensorbem.section_radiation import (
    solve_section_heave,
    double_ended_strip_radiation,
)
from tensorbem.wigley3_hull import WIGLEY3_T, wigley3_half_breadth

GRIDS = ((8, 12), (12, 20), (20, 32))
COEFFICIENTS = "A33 A35 A53 A55 B33 B35 B53 B55".split()


def solve_frequency(args):
    grid, we, speed = args
    stations, panels = grid
    # Mirror each station about midships. Gauss integration on each half
    # avoids solving a degenerate section at the ends.
    t, w = np.polynomial.legendre.leggauss(stations // 2)
    x, weight = (t + 1) * LENGTH / 4, w * LENGTH / 4
    z = np.linspace(0, -WIGLEY3_T, panels // 2 + 1)
    mass, damping = [], []
    residual, cond, energy_defect = 0.0, 0.0, 0.0
    for station in x:
        y = wigley3_half_breadth(station, z)
        contour = np.column_stack((np.r_[-y, y[-2::-1]], np.r_[z, z[-2::-1]]))
        r = solve_section_heave(contour, we, density=1000.0, gravity=GRAVITY)
        mass.append(r.added_mass)
        damping.append(r.damping)
        residual = max(residual, r.relative_residual)
        cond = max(cond, r.condition_number)
        # Scale by the exact sectional displaced mass times frequency;
        # this stays well-defined when damping tends to zero.
        area = (4 / 3) * y[0] * WIGLEY3_T
        energy_defect = max(
            energy_defect, abs(r.energy_damping - r.damping) / (1000.0 * we * area)
        )

    def both(values):
        return np.r_[values[::-1], values]

    a, b = double_ended_strip_radiation(
        np.r_[-x[::-1], x], both(weight), both(mass), both(damping), we, speed
    )
    return [*a.flat, *b.flat, residual, cond, energy_defect]


def calculate(workers=1):
    baseline = table(ROOT, "tensorbem")
    predictions = baseline[baseline["panels"] == max(baseline["panels"])]
    forced = table(ROOT, "radiation")
    cases = [("motion", p["Fn"], p["lambda_L"], p["omega_e"]) for p in predictions]
    for fn, nd in sorted(set(zip(forced["Fn"], forced["w_nd"]))):
        cases.append(("forced", fn, nd, nd * np.sqrt(GRAVITY / LENGTH)))
    tasks = [
        (grid, we, fn * np.sqrt(GRAVITY * LENGTH))
        for grid in GRIDS
        for _, fn, _, we in cases
    ]
    with (
        TemporaryDirectory(dir=ROOT) as directory,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        dest = Path(directory) / "strip_coefficients.csv"
        with dest.open("w") as f:
            out = csv.writer(f, lineterminator="\n")
            out.writerow(
                [
                    "stations",
                    "panels_per_section",
                    "kind",
                    "Fn",
                    "lambda_L",
                    "w_nd",
                    "omega_e",
                    *COEFFICIENTS,
                    "max_section_residual",
                    "max_section_condition",
                    "max_energy_defect",
                ]
            )
            for (grid, _, _), (kind, fn, key, we), result in zip(
                tasks, cases * len(GRIDS), pool.map(solve_frequency, tasks)
            ):
                out.writerow(
                    [
                        *grid,
                        kind,
                        fn,
                        key if kind == "motion" else "",
                        key if kind == "forced" else "",
                        f"{we:.12g}",
                        *[f"{v:.12g}" for v in result],
                    ]
                )
                f.flush()
        dest.replace(ROOT / dest.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    calculate(p.parse_args().workers)
