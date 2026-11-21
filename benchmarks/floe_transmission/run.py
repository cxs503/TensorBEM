"""Compare the flexible-plate BEM with regular-wave flume tests (energy, not amplitude)."""

import argparse
import csv
from pathlib import Path
import numpy as np
from tensorbem.floe_scattering import scatter_floe
from benchmarks.marine_experiments.io import write

ROOT = Path(__file__).resolve().parent
MATERIALS = {"PP": (1.6e9, 905.0), "PVC": (5.0e8, 569.0)}


def calculate():
    with (ROOT / "observations.csv").open() as f:
        obs = list(csv.DictReader(f))
    rows = []
    scores = []
    for n in (12, 24, 48, 72):
        for material, (young, density) in MATERIALS.items():
            for period in (0.8, 0.9, 1.0):
                sol = scatter_floe(
                    2 * np.pi / period,
                    length=1.0,
                    depth=0.9,
                    rigidity=young * 0.01**3 / (12 * (1 - 0.4**2)),
                    mass_per_area=density * 0.01,
                    panels=n,
                )
                for r in obs:
                    if r["material"] == material and float(r["period"]) == period:
                        pred = abs(sol.transmission) ** 2
                        rows.append(
                            dict(
                                **r,
                                panels=n,
                                predicted_energy_T=pred,
                                predicted_energy_R=abs(sol.reflection) ** 2,
                                error_T=pred - float(r["observed_energy_T"]),
                                error_R=abs(sol.reflection) ** 2
                                - float(r["observed_energy_R"]),
                                observed_energy_loss=1
                                - float(r["observed_energy_T"])
                                - float(r["observed_energy_R"]),
                                energy_defect=sol.energy_defect,
                                residual=sol.relative_residual,
                            )
                        )
        for conf in ("free", "moored", "barriers"):
            group = [r for r in rows if r["panels"] == n and r["configuration"] == conf]
            for quantity in ("T", "R"):
                errors = np.array([r["error_" + quantity] for r in group])
                rms = np.sqrt(np.mean(errors**2))
                maximum = max(abs(errors))
                scores.append(
                    dict(
                        panels=n,
                        configuration=conf,
                        quantity=quantity,
                        trials=len(group),
                        rms=rms,
                        maximum=maximum,
                        verdict="PASS" if rms <= 0.10 and maximum <= 0.20 else "FAIL",
                    )
                )
    write(ROOT / "results.csv", list(rows[0]), rows)
    write(ROOT / "scores.csv", list(scores[0]), scores)
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--require-pass", action="store_true")
    a = p.parse_args()
    scores = calculate()
    if a.require_pass and any(
        r["verdict"] == "FAIL" for r in scores if r["panels"] == 72
    ):
        raise SystemExit(1)
