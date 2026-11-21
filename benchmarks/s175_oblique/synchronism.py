"""Sensitivity to rounding near the 30-degree pitch marker at nondimensional frequency 4.200."""

from benchmarks.marine_experiments.io import write
from .run import ROOT, calculate


def evaluate():
    rows = []
    for f in (4.195, 4.200, 4.205):
        predicted = calculate((8, 30, f))["pitch"]
        rows.append(
            {
                k: predicted[k]
                for k in (
                    "panels",
                    "heading_deg",
                    "omega_sqrt_L_g",
                    "encounter_ratio",
                    "predicted",
                    "motion_condition",
                    "bem_residual",
                )
            }
        )
    write(ROOT / "synchronism.csv", list(rows[0]), rows)


if __name__ == "__main__":
    evaluate()
