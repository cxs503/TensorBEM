"""Run the unmodified 3-D solver at the frequencies of the forced-motion tank runs."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .assess import ROOT, GRAVITY, LENGTH, table
from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh


def calculate(output):

    mesh = wigley3_wetted_hull_mesh(nx=20, nz=10)
    records = table(ROOT, "radiation")
    with TemporaryDirectory(dir=output.parent) as d:
        pending = Path(d) / output.name
        with pending.open("w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow("Fn w_nd A33 A35 A53 A55 B33 B35 B53 B55".split())
            for fn, nd in sorted(set(zip(records["Fn"], records["w_nd"]))):
                speed = fn * np.sqrt(GRAVITY * LENGTH)
                we = nd * np.sqrt(GRAVITY / LENGTH)
                omega = 2 * we / (1 + np.sqrt(1 + 4 * speed * we / GRAVITY))
                r = solve_forward_speed_heave_pitch(
                    mesh, omega, speed, density=1000.0, gravity=GRAVITY
                )
                writer.writerow(
                    [
                        fn,
                        nd,
                        *[f"{x:.12g}" for x in (*r.added_mass.flat, *r.damping.flat)],
                    ]
                )
                f.flush()
        pending.replace(output)


if __name__ == "__main__":
    calculate(ROOT / "forced_baseline.csv")
