"""Score both independent predictions against every original Table 10-III row."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def wrap(degrees):
    return (degrees + 180) % 360 - 180


def score(root=HERE):
    limits = json.loads((root / "limits.json").read_text())
    observed = np.genfromtxt(root / "table10.csv", delimiter=",", names=True)
    wavelengths = np.unique(observed["lambda_L"])
    results, errors = {}, []
    for solver in ("tensorbem", "capytaine"):
        prediction = np.genfromtxt(root / f"{solver}.csv", delimiter=",", names=True)
        results[solver] = {}
        for panels in np.unique(prediction["panels"]):
            curve = np.sort(
                prediction[prediction["panels"] == panels], order="lambda_L"
            )
            if not np.array_equal(curve["lambda_L"], wavelengths):
                raise ValueError(
                    f"Incomplete or repeated wavelength in {solver}, {panels} panels"
                )
            indices = np.searchsorted(wavelengths, observed["lambda_L"])
            level = results[solver][str(int(panels))] = {}
            for dof in ("heave", "pitch"):
                measured, predicted = observed[dof], curve[dof][indices]
                diff = predicted - measured
                phase = wrap(curve[f"{dof}_deg"][indices] - observed[f"{dof}_deg"])
                valid = (
                    np.minimum(measured, predicted)
                    >= limits["phase_min_amplitude_both"]
                )
                for row, idx in enumerate(indices):
                    errors.append(
                        [
                            solver,
                            int(panels),
                            row + 1,
                            wavelengths[idx],
                            observed["amplitude_cm"][row],
                            dof,
                            measured[row],
                            predicted[row],
                            diff[row],
                            observed[f"{dof}_deg"][row],
                            curve[f"{dof}_deg"][idx],
                            phase[row],
                            int(valid[row]),
                        ]
                    )
                means = [
                    np.mean(measured[observed["lambda_L"] == wl]) for wl in wavelengths
                ]
                obs_peak, calc_peak = int(np.argmax(means)), int(np.argmax(curve[dof]))
                metrics = {
                    "amplitude_rms": np.sqrt(np.mean(diff**2)),
                    "amplitude_max": max(abs(diff)),
                    "phase_rms_deg": (
                        np.sqrt(np.mean(phase[valid] ** 2)) if valid.any() else None
                    ),
                    "phase_max_deg": max(abs(phase[valid])) if valid.any() else None,
                    "peak_max_grid_steps": abs(calc_peak - obs_peak),
                }
                level[dof] = {
                    "metrics": metrics,
                    "passed": {
                        k: v is not None and bool(v <= limits[k])
                        for k, v in metrics.items()
                    },
                    "phase_rows": int(np.sum(valid)),
                    "peak_observed_omega": curve["omega"][obs_peak],
                    "peak_predicted_omega": curve["omega"][calc_peak],
                    "peak_observed_at_band_edge": obs_peak in (0, len(curve) - 1),
                    "peak_predicted_at_band_edge": calc_peak in (0, len(curve) - 1),
                }
    return results, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    results, errors = score()
    (HERE / "scores.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    with (HERE / "errors.csv").open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            "solver panels run lambda_L amplitude_cm dof measured predicted error measured_deg predicted_deg wrapped_error_deg phase_scored".split()
        )
        writer.writerows(errors)
    finest = results["tensorbem"][str(max(map(int, results["tensorbem"])))]
    if args.require_pass and not all(
        all(d["passed"].values()) for d in finest.values()
    ):
        raise SystemExit(1)
