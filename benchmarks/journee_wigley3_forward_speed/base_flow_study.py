"""Score the geometry-only base-flow calculation and isolate measured radiation terms."""

import argparse
import csv
from itertools import combinations
from math import factorial

import numpy as np

from .assess import (
    ROOT,
    LENGTH,
    MASS,
    GRAVITY,
    compare,
    measured_matrices,
    table,
    write,
)
from .radiation_study import matrices, solve_motion
from .strip import COEFFICIENTS

TARGETS = ("A35", "A53", "B55")
SCORE_HEADER = "model panels Fn quantity trials phase_trials rms max phase_rms_deg phase_max_deg observed_peak_lambda_L predicted_peak_lambda_L observed_peak_we predicted_peak_we verdict"


def load_coefficients(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def wave_force(row):
    return np.array(
        [
            complex(float(row["F" + d + "_re"]), float(row["F" + d + "_im"]))
            for d in "35"
        ]
    )


def predictions(root, records):
    baseline = table(root, "tensorbem")
    motion = {
        (int(r["panels"]), float(r["Fn"]), float(r["lambda_L"])): r
        for r in records
        if r["kind"] == "motion"
    }
    curves = {"uniform": baseline}
    for name in ("double_body_radiation", "double_body_load", "double_body_both"):
        curve = baseline.copy()
        for row in curve:
            candidate = motion[int(row["panels"]), row["Fn"], row["lambda_L"]]
            r = row if name == "double_body_load" else candidate
            f = row if name == "double_body_radiation" else candidate
            solve_motion(row, matrices(r), wave_force(f))
        curves[name] = curve
    return curves


def forced_errors(root, records):
    reference = load_coefficients(root / "forced_comparison.csv")
    forced = {
        (int(r["panels"]), float(r["Fn"]), float(r["w_nd"])): r
        for r in records
        if r["kind"] == "forced"
    }
    panels = sorted({k[0] for k in forced})
    rows = []
    for row in reference:
        fn, nd, name = float(row["Fn"]), float(row["w_nd"]), row["coefficient"]
        scale = MASS * LENGTH ** (name.count("5"))
        if name[0] == "B":
            scale *= np.sqrt(GRAVITY / LENGTH)
        values = [float(forced[g, fn, nd][name]) / scale for g in panels]
        rows.append(
            [
                fn,
                nd,
                row["driven"],
                row["source_line"],
                row["source_group"],
                name,
                float(row["measured"]),
                float(row["native"]),
                float(row["strip_20x32"]),
                *values,
            ]
        )
    return panels, rows


def replace_terms(base, measured, terms):
    result = base.copy()
    for term in terms:
        kind, row, col = term
        index = ("AB".index(kind), "35".index(row), "35".index(col))
        result[index] = measured[index]
    return result


def measured_peak_window(trials, quantity):
    x = np.unique(trials["lambda_L"])
    peak = np.argmax(
        [np.mean(trials[quantity][trials["lambda_L"] == ratio]) for ratio in x]
    )
    return x[max(0, peak - 1) : peak + 2]


def component_effects(root, curves):
    """Exact marginal reductions in squared amplitude error over all replacement orders."""
    native = curves["uniform"]
    native = native[native["panels"] == max(native["panels"])]
    strip = {
        (float(r["Fn"]), float(r["lambda_L"])): r
        for r in load_coefficients(root / "strip_coefficients.csv")
        if r["kind"] == "motion" and r["stations"] == "20"
    }
    db = curves["double_body_radiation"]
    db = db[db["panels"] == max(db["panels"])]
    tank, motions = table(root, "radiation"), table(root, "motions")
    subsets = [s for n in range(4) for s in combinations(TARGETS, n)]
    records = []
    for original, corrected in zip(native, db, strict=True):
        fn, ratio = original["Fn"], original["lambda_L"]
        trial_speed = motions[motions["Fn"] == fn]
        runs = trial_speed[trial_speed["lambda_L"] == ratio]
        measured = measured_matrices(tank[tank["Fn"] == fn], original["omega_e"])
        if measured is None:
            continue
        for model, source in [
            ("strip", strip[fn, ratio]),
            ("uniform", original),
            ("double_body", corrected),
        ]:
            base = matrices(source)
            outputs = {}
            for subset in subsets:
                row = original.copy()
                solve_motion(
                    row,
                    replace_terms(base, np.asarray(measured), subset),
                    wave_force(original),
                )
                outputs[subset] = row
            for q in ("heave", "pitch"):
                is_peak = ratio in measured_peak_window(trial_speed, q)
                for trial in runs:
                    errors = {s: (r[q] - trial[q]) ** 2 for s, r in outputs.items()}
                    effects = []
                    for name in TARGETS:
                        total = 0.0
                        for s in subsets:
                            if name in s:
                                continue
                            with_name = tuple(t for t in TARGETS if t == name or t in s)
                            weight = factorial(len(s)) * factorial(2 - len(s)) / 6
                            total += weight * (errors[s] - errors[with_name])
                        effects.append(total)
                    records.append(
                        [
                            model,
                            fn,
                            ratio,
                            q,
                            int(trial["source_line"]),
                            int(trial["source_group"]),
                            int(is_peak),
                            errors[()],
                            errors[TARGETS],
                            *effects,
                        ]
                    )
    return records


def assess(root=ROOT):
    records = load_coefficients(root / "double_body_coefficients.csv")
    curves = predictions(root, records)
    scores, errors = compare(root, predictions_by_solver=curves)
    write(root / "double_body_scores.csv", SCORE_HEADER, scores)
    write(
        root / "double_body_errors.csv",
        "model panels Fn quantity lambda_L source_line source_group wave_cm measured predicted error measured_deg predicted_deg wrapped_error_deg phase_scored",
        [
            r
            for r in errors
            if r[0] == "double_body_both" and r[1] == max(curves["uniform"]["panels"])
        ],
    )
    panels, rows = forced_errors(root, records)
    write(
        root / "double_body_forced.csv",
        "Fn w_nd driven source_line source_group coefficient measured uniform strip "
        + " ".join("db" + str(g) for g in panels),
        rows,
    )
    effects = component_effects(root, curves)
    write(
        root / "double_body_attribution.csv",
        "model Fn lambda_L quantity source_line source_group in_peak_window base_sq_error replaced_sq_error A35_mse_reduction A53_mse_reduction B55_mse_reduction",
        effects,
    )
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    scores = assess()
    finest = max(r[1] for r in scores)
    if args.require_pass and any(
        r[-1] == "FAIL" for r in scores if r[0] == "double_body_both" and r[1] == finest
    ):
        raise SystemExit(1)
