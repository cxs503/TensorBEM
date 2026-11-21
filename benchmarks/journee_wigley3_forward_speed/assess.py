"""Compare coefficient, load and motion records without fitting the model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LIMITS = ROOT.parent / "journee_wigley3_zero_speed/limits.json"
NAMES = ("heave", "pitch", "force3", "moment5")
LENGTH, MASS, GRAVITY = 3.0, 78.0, 9.81
# Report 0909 p.7: these are the scales used in Table 6-III.
C33_REF, C55_REF = 6119.0, 2874.0


def table(root, name):
    return np.genfromtxt(root / (name + ".csv"), names=True, delimiter=",")


def component(curve, name):
    if name in ("heave", "pitch"):
        return curve[name], curve[name + "_deg"]
    mode = "3" if name == "force3" else "5"
    force = curve["F" + mode + "_re"] + 1j * curve["F" + mode + "_im"]
    scale = C33_REF if mode == "3" else C55_REF * curve["k"]
    return abs(force) / scale, -np.angle(force, deg=True) % 360


def compare(root=ROOT, *, predictions_by_solver=None):
    limits = json.loads(LIMITS.read_text())
    trials = {name: table(root, name) for name in ("motions", "loads")}
    scores, errors = [], []
    if predictions_by_solver is None:
        predictions_by_solver = {
            name: table(root, name) for name in ("tensorbem", "capytaine")
        }
    for solver, predictions in predictions_by_solver.items():
        for panels in np.unique(predictions["panels"]):
            for fn in np.unique(trials["motions"]["Fn"]):
                curve = predictions[
                    (predictions["panels"] == panels) & (predictions["Fn"] == fn)
                ]
                for name in NAMES:
                    is_motion = name in ("heave", "pitch")
                    measured = trials["motions" if is_motion else "loads"]
                    measured = measured[measured["Fn"] == fn]
                    grid = np.unique(measured["lambda_L"])
                    selected = np.sort(
                        curve[np.isin(curve["lambda_L"], grid)], order="lambda_L"
                    )
                    if not np.array_equal(selected["lambda_L"], grid) or (
                        is_motion and len(curve) != len(grid)
                    ):
                        raise ValueError(
                            f"Missing or repeated prediction at Fn={fn}, {panels} panels"
                        )
                    indices = np.searchsorted(grid, measured["lambda_L"])
                    amp, phase = component(selected, name)
                    err = amp[indices] - measured[name]
                    angle = (phase[indices] - measured[name + "_deg"] + 180) % 360 - 180
                    mask = (
                        np.minimum(amp[indices], measured[name])
                        >= limits["phase_min_amplitude_both"]
                    )
                    for i, row in enumerate(measured):
                        errors.append(
                            [
                                solver,
                                int(panels),
                                fn,
                                name,
                                row["lambda_L"],
                                int(row["source_line"]),
                                int(row["source_group"]),
                                row["wave_cm"],
                                row[name],
                                amp[indices[i]],
                                err[i],
                                row[name + "_deg"],
                                phase[indices[i]],
                                angle[i],
                                int(mask[i]),
                            ]
                        )
                    observed_peak = np.argmax(
                        [
                            np.mean(measured[name][measured["lambda_L"] == r])
                            for r in grid
                        ]
                    )
                    predicted_peak = np.argmax(amp)
                    arms, amax = np.sqrt(np.mean(err**2)), np.max(abs(err))
                    prms = np.sqrt(np.mean(angle[mask] ** 2)) if mask.any() else np.nan
                    pmax = np.max(abs(angle[mask])) if mask.any() else np.nan
                    passed = (
                        arms <= limits["amplitude_rms"]
                        and amax <= limits["amplitude_max"]
                        and prms <= limits["phase_rms_deg"]
                        and pmax <= limits["phase_max_deg"]
                        and abs(predicted_peak - observed_peak)
                        <= limits["peak_max_grid_steps"]
                    )
                    scores.append(
                        [
                            solver,
                            int(panels),
                            fn,
                            name,
                            len(measured),
                            sum(mask),
                            arms,
                            amax,
                            prms,
                            pmax,
                            grid[observed_peak],
                            grid[predicted_peak],
                            selected["omega_e"][observed_peak],
                            selected["omega_e"][predicted_peak],
                            (
                                ("PASS" if passed else "FAIL")
                                if is_motion
                                else "diagnostic"
                            ),
                        ]
                    )
    return scores, errors


def measured_matrices(rows, we):
    """Linear interpolation within each test range, averaging the two forced amplitudes."""
    nd = we / np.sqrt(GRAVITY / LENGTH)
    a, b = np.empty((2, 2)), np.empty((2, 2))
    for col, mode in enumerate([3, 5]):
        groups = [
            np.sort(
                rows[(rows["driven"] == mode) & (rows["source_group"] == i)],
                order="w_nd",
            )
            for i in (1, 2)
        ]
        if not all(len(q) and q["w_nd"][0] <= nd <= q["w_nd"][-1] for q in groups):
            return None
        for row in (0, 1):
            term = "_self" if row == col else "_cross"
            scale = MASS * LENGTH ** (row + col)
            for matrix, name, unit in [
                (a, "A", scale),
                (b, "B", scale * np.sqrt(GRAVITY / LENGTH)),
            ]:
                matrix[row, col] = unit * np.mean(
                    [np.interp(nd, q["w_nd"], q[name + term]) for q in groups]
                )
    return a, b


def coefficient_errors(root=ROOT):
    """Compare dimensional two-by-two radiation matrices within the test band."""
    radiation = table(root, "radiation")
    rows = []
    for solver in ("tensorbem", "capytaine"):
        predictions = table(root, solver)
        for p in predictions[predictions["panels"] == max(predictions["panels"])]:
            fn, we = p["Fn"], p["omega_e"]
            matrices = measured_matrices(radiation[radiation["Fn"] == fn], we)
            if matrices is None:
                continue
            for prefix, matrix in zip("AB", matrices):
                for i, row in enumerate("35"):
                    for j, col in enumerate("35"):
                        name = prefix + row + col
                        obs = matrix[i, j]
                        rows.append(
                            [
                                solver,
                                int(p["panels"]),
                                fn,
                                p["lambda_L"],
                                we,
                                name,
                                obs,
                                p[name],
                                p[name] - obs,
                            ]
                        )
    return rows


def load_substitutions(root=ROOT):
    """Replace measured inputs in isolation on the same matched experimental cases."""
    radiation, loads, motions = [
        table(root, s) for s in ("radiation", "loads", "motions")
    ]
    native = table(root, "tensorbem")
    rows, excluded = [], []
    for p in native[native["panels"] == max(native["panels"])]:
        fn, ratio, we = p["Fn"], p["lambda_L"], p["omega_e"]
        measured = loads[(loads["Fn"] == fn) & (loads["lambda_L"] == ratio)]
        if not len(measured):
            continue
        matrices = measured_matrices(radiation[radiation["Fn"] == fn], we)
        if matrices is None:
            excluded.append([fn, ratio, we])
            continue
        f_tank = np.array(
            [
                np.mean(
                    measured[name]
                    * scale
                    * np.exp(-1j * np.deg2rad(measured[name + "_deg"]))
                )
                for name, scale in [("force3", C33_REF), ("moment5", C55_REF * p["k"])]
            ]
        )
        f_model = np.array(
            [complex(p["F" + d + "_re"], p["F" + d + "_im"]) for d in ("3", "5")]
        )
        a, b, c = [
            np.array([[p[name + i + j] for j in "35"] for i in "35"]) for name in "ABC"
        ]
        m = np.diag([p["M33"], p["M55"]])
        trials = motions[(motions["Fn"] == fn) & (motions["lambda_L"] == ratio)]
        a55, b55 = a.copy(), b.copy()
        a55[1, 1], b55[1, 1] = matrices[0][1, 1], matrices[1][1, 1]
        for label, aa, bb, ff in [
            ("native", a, b, f_model),
            ("tank_load", a, b, f_tank),
            ("tank_A55", a55, b, f_model),
            ("tank_B55", a, b55, f_model),
            ("tank_radiation", *matrices, f_model),
            ("tank_both", *matrices, f_tank),
        ]:
            xi = np.linalg.solve(c - we**2 * (m + aa) - 1j * we * bb, ff)
            for i, name in enumerate(("heave", "pitch")):
                value = abs(xi[i]) * (LENGTH / (2 * np.pi) if i else 1)
                obs = np.mean(trials[name])
                rows.append(
                    [fn, ratio, we, label, name, len(trials), obs, value, value - obs]
                )
    return rows, excluded


def write(path, header, rows):
    with path.open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header.split())
        for row in rows:
            w.writerow(
                [
                    format(v, ".10g") if isinstance(v, (np.floating, float)) else v
                    for v in row
                ]
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    scores, errors = compare()
    write(
        ROOT / "scores.csv",
        "solver panels Fn quantity trials phase_trials rms max phase_rms_deg phase_max_deg observed_peak_lambda_L predicted_peak_lambda_L observed_peak_we predicted_peak_we verdict",
        scores,
    )
    write(
        ROOT / "errors.csv",
        "solver panels Fn quantity lambda_L source_line source_group wave_cm observed predicted error observed_deg predicted_deg wrapped_error_deg phase_scored",
        errors,
    )
    write(
        ROOT / "coefficient_errors.csv",
        "solver panels Fn lambda_L omega_e coefficient observed_interpolated predicted error",
        coefficient_errors(),
    )
    rows, excluded = load_substitutions()
    write(
        ROOT / "substitutions.csv",
        "Fn lambda_L omega_e model quantity trials observed_mean predicted error",
        rows,
    )
    write(ROOT / "outside_radiation_band.csv", "Fn lambda_L omega_e", excluded)
    finest = max(row[1] for row in scores if row[0] == "tensorbem")
    if args.require_pass and any(
        row[-1] == "FAIL" for row in scores if row[:2] == ["tensorbem", finest]
    ):
        raise SystemExit(1)
