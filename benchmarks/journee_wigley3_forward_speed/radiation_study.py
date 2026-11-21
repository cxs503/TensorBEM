"""Compare physical strip predictions and endpoint-calibrated predictions with the tank."""

import argparse
import csv

import numpy as np

from .assess import (
    C33_REF,
    C55_REF,
    GRAVITY,
    LENGTH,
    MASS,
    ROOT,
    compare,
    measured_matrices,
    table,
    write,
)
from .strip import COEFFICIENTS, GRIDS
from tensorbem.rao import dynamic_stiffness


def endpoint_weights(fn):
    """Fn 0.30 is held out from both the radiation and load tables."""
    if not 0.20 <= fn <= 0.40:
        raise ValueError("outside the two calibration speeds")
    if fn in (0.2, 0.4):
        return [(fn, 1.0)]
    right = (fn - 0.20) / 0.20
    return [(0.20, 1 - right), (0.40, right)]


def endpoint_radiation(training, fn, we):
    result = np.zeros((2, 2, 2))
    for anchor, weight in endpoint_weights(fn):
        matrices = measured_matrices(training[training["Fn"] == anchor], we)
        if matrices is None:
            return None
        result += weight * np.asarray(matrices)
    return result


def endpoint_load(training, fn, ratio, k):
    force = np.zeros(2, dtype=complex)
    for anchor, weight in endpoint_weights(fn):
        rows = training[training["Fn"] == anchor]
        x = np.unique(rows["lambda_L"])
        if x.size == 0 or not x[0] <= ratio <= x[-1]:
            return None
        for i, (d, scale) in enumerate((("force3", C33_REF), ("moment5", C55_REF * k))):
            z = rows[d] * np.exp(-1j * np.deg2rad(rows[d + "_deg"]))
            mean = np.array([z[rows["lambda_L"] == r].mean() for r in x])
            force[i] += weight * scale * np.interp(ratio, x, mean)
    return force


def matrices(row):
    return np.array(
        [[float(row[p + i + j]) for j in "35"] for p in "AB" for i in "35"]
    ).reshape(2, 2, 2)


def solve_motion(row, ab, force):
    c = np.array([[row["C" + i + j] for j in "35"] for i in "35"])
    m = np.diag([row["M33"], row["M55"]])
    xi = np.linalg.solve(dynamic_stiffness(row["omega_e"], m, *ab, c), force)
    for field, value in zip(COEFFICIENTS, np.ravel(ab)):
        row[field] = value
    for dof, (name, mode, scale) in enumerate(
        (("heave", "3", 1), ("pitch", "5", LENGTH / (2 * np.pi)))
    ):
        row["F" + mode + "_re"], row["F" + mode + "_im"] = (
            force[dof].real,
            force[dof].imag,
        )
        row[name], row[name + "_deg"] = (
            abs(xi[dof]) * scale,
            (-np.angle(xi[dof], deg=True)) % 360,
        )


def workflows(root=ROOT):
    radiation, loads, native = [
        table(root, name) for name in ("radiation", "loads", "tensorbem")
    ]
    # Partition before any interpolation. Motion observations are read only by compare().
    training = radiation[np.isin(radiation["Fn"], (0.2, 0.4))]
    training_loads = loads[np.isin(loads["Fn"], (0.2, 0.4))]
    fine = native[native["panels"] == max(native["panels"])]
    with (root / "strip_coefficients.csv").open() as f:
        strip = {
            (
                int(r["stations"]),
                r["kind"],
                float(r["Fn"]),
                float(r["lambda_L"] or r["w_nd"]),
            ): r
            for r in csv.DictReader(f)
        }
    predictions = {"baseline": fine}
    sources = {}
    strip_models = {f"strip_{n}x{p}": n for n, p in GRIDS}
    for name in [*strip_models, "endpoint_radiation", "endpoint_radiation_load"]:
        curve = fine.copy()
        for r in curve:
            fn, ratio, we = r["Fn"], r["lambda_L"], r["omega_e"]
            force = np.array(
                [complex(r["F" + d + "_re"], r["F" + d + "_im"]) for d in "35"]
            )
            load_source = "native"
            if name in strip_models:
                n = strip_models[name]
                ab = matrices(strip[n, "motion", fn, ratio])
                source = "section_BEM"
            else:
                ab = endpoint_radiation(training, fn, we)
                source = "endpoint_tank"
                if ab is None:
                    ab = matrices(r)
                    source = "native_outside_band"
                if name == "endpoint_radiation_load":
                    measured_force = endpoint_load(training_loads, fn, ratio, r["k"])
                    if measured_force is not None:
                        force = measured_force
                        load_source = "endpoint_tank"
                    else:
                        load_source = "native_outside_band"
            solve_motion(r, ab, force)
            sources[name, fn, ratio] = (source, load_source)
        predictions[name] = curve
    return predictions, sources, strip, training


def forced_comparisons(root, strip, training):
    native = {(r["Fn"], r["w_nd"]): r for r in table(root, "forced_baseline")}
    rows = []
    for r in table(root, "radiation"):
        fn, nd, col = r["Fn"], r["w_nd"], int(r["driven"])
        estimator = endpoint_radiation(training, fn, nd * np.sqrt(GRAVITY / LENGTH))
        for code in COEFFICIENTS:
            prefix, i, j = code
            if int(j) != col:
                continue
            measured = r[prefix + ("_self" if i == j else "_cross")]
            scale = MASS * LENGTH ** ((int(i) - 3) // 2 + (int(j) - 3) // 2)
            if prefix == "B":
                scale *= np.sqrt(GRAVITY / LENGTH)
            baseline = native[fn, nd][code] / scale
            sections = [
                float(strip[n, "forced", fn, nd][code]) / scale for n, _ in GRIDS
            ]
            estimate = (
                baseline
                if estimator is None
                else estimator["AB".index(prefix), "35".index(i), "35".index(j)] / scale
            )
            rows.append(
                [
                    fn,
                    nd,
                    col,
                    int(r["source_line"]),
                    int(r["source_group"]),
                    code,
                    measured,
                    baseline,
                    *sections,
                    estimate,
                    "native_outside_band" if estimator is None else "endpoint_tank",
                ]
            )
    return rows


def assess(root=ROOT):
    predictions, sources, strip, training = workflows(root)
    scores, _ = compare(root, predictions_by_solver=predictions)
    write(
        root / "candidate_scores.csv",
        "model reference_mesh_panels Fn quantity trials phase_trials rms max phase_rms_deg phase_max_deg observed_peak_lambda_L predicted_peak_lambda_L observed_peak_we predicted_peak_we verdict",
        scores,
    )
    fields = (
        "omega_e",
        *COEFFICIENTS,
        "F3_re",
        "F3_im",
        "F5_re",
        "F5_im",
        "heave",
        "heave_deg",
        "pitch",
        "pitch_deg",
    )
    rows = []
    for name, curve in predictions.items():
        for r in curve:
            fn, ratio = r["Fn"], r["lambda_L"]
            rows.append(
                [
                    name,
                    fn,
                    ratio,
                    *sources.get((name, fn, ratio), ("native", "native")),
                    *[r[k] for k in fields],
                ]
            )
    write(
        root / "candidate_motions.csv",
        "model Fn lambda_L radiation_source load_source " + " ".join(fields),
        rows,
    )
    forced = forced_comparisons(root, strip, training)
    write(
        root / "forced_comparison.csv",
        "Fn w_nd driven source_line source_group coefficient measured native strip_8x12 strip_12x20 strip_20x32 endpoint_radiation endpoint_source",
        forced,
    )
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--require-pass",
        choices=[
            "baseline",
            "strip_20x32",
            "endpoint_radiation",
            "endpoint_radiation_load",
        ],
    )
    args = p.parse_args()
    scores = assess()
    if args.require_pass and any(
        r[-1] == "FAIL" for r in scores if r[0] == args.require_pass
    ):
        raise SystemExit(1)
