"""Match Journée's forced-test restoring convention and isolate coupled motion phases."""

import argparse
import json

import numpy as np

from tensorbem.rao import dynamic_stiffness

from .assess import (
    C33_REF,
    C55_REF,
    LENGTH,
    LIMITS,
    ROOT,
    compare,
    measured_matrices,
    table,
    write,
)
from .base_flow_study import SCORE_HEADER, load_coefficients, wave_force
from .radiation_study import matrices, measured_load
from .speed_parity import parity_motions

REPORT_C = np.diag([C33_REF, C55_REF])
ERROR_HEADER = "model panels Fn quantity lambda_L source_line source_group wave_cm observed predicted error observed_deg predicted_deg wrapped_error_deg phase_scored"


def paired_parity(root):
    """Use the restoring used to reduce forced tests only when their radiation is used."""
    original, _ = parity_motions(root)
    paired, _ = parity_motions(root, measured_restoring=REPORT_C)
    return {"previous": original, "report_paired": paired}


def schur_response(z, f, i):
    """Eliminate the other DOF from Z xi = F and retain the complex numerator and denominator."""
    j = 1 - i
    coupling = z[i, j] / z[j, j]
    induced = coupling * f[j]
    effective_force = f[i] - induced
    impedance = z[i, i] - coupling * z[j, i]
    scale = abs(f[i]) + abs(induced)
    cancellation_ratio = abs(effective_force) / scale if scale else np.nan
    return effective_force, impedance, cancellation_ratio


def lag(z):
    return -np.angle(z, deg=True) % 360


def phase_substitutions(root):
    """Compare both independent component swaps on the common band at the trial speed."""
    phase_threshold = json.loads(LIMITS.read_text())["phase_min_amplitude_both"]
    db = load_coefficients(root / "double_body_coefficients.csv")
    base, rad, loads, trials = [
        table(root, name) for name in ("tensorbem", "radiation", "loads", "motions")
    ]
    fine = base[base["panels"] == max(base["panels"])]
    lookup = {
        (float(r["Fn"]), float(r["lambda_L"])): r
        for r in db
        if r["kind"] == "motion" and int(r["panels"]) == fine["panels"][0]
    }
    output, coverage = [], []
    for r in fine:
        fn, ratio, we = r["Fn"], r["lambda_L"], r["omega_e"]
        physical = lookup[fn, ratio]
        m = np.diag([r["M33"], r["M55"]])
        c = np.array([[r["C" + i + j] for j in "35"] for i in "35"])
        speed_loads = loads[loads["Fn"] == fn]
        force = measured_load(speed_loads, ratio, r["k"])
        ab = measured_matrices(rad[rad["Fn"] == fn], we)
        load_source = (
            "unavailable"
            if force is None
            else "sampled" if ratio in speed_loads["lambda_L"] else "interpolated"
        )
        coverage.append([fn, ratio, int(ab is not None), load_source])
        if ab is None or force is None:
            continue
        observed = trials[(trials["Fn"] == fn) & (trials["lambda_L"] == ratio)]
        for radiation_source, coeff, restoring in (
            ("double_body", matrices(physical), c),
            ("tank", ab, REPORT_C),
        ):
            z = dynamic_stiffness(we, m, *coeff, restoring)
            for excitation_source, f in (
                ("double_body", wave_force(physical)),
                ("tank", force),
            ):
                for i, name in enumerate(("heave", "pitch")):
                    effective, impedance, ratio_left = schur_response(z, f, i)
                    response = effective / impedance
                    amplitude = abs(response) * (LENGTH / (2 * np.pi) if i else 1)
                    phase = lag(response)
                    for t in observed:
                        error = (phase - t[name + "_deg"] + 180) % 360 - 180
                        output.append(
                            [
                                fn,
                                ratio,
                                name,
                                int(t["source_line"]),
                                int(t["source_group"]),
                                radiation_source,
                                excitation_source,
                                load_source,
                                t[name],
                                amplitude,
                                t[name + "_deg"],
                                phase,
                                error,
                                int(min(t[name], amplitude) >= phase_threshold),
                                abs(effective),
                                lag(effective),
                                abs(impedance),
                                lag(impedance),
                                ratio_left,
                            ]
                        )
    return output, coverage


def level_sensitivity(root):
    """Keep the original two forced amplitudes and two wave-load levels separate at the worst point."""
    fn, ratio = 0.4, 1.384
    base, rad, loads = [
        table(root, name) for name in ("tensorbem", "radiation", "loads")
    ]
    row = base[
        (base["panels"] == max(base["panels"]))
        & (base["Fn"] == fn)
        & (base["lambda_L"] == ratio)
    ][0]
    m = np.diag([row["M33"], row["M55"]])
    rows = []
    for level in (1, 2):
        ab = measured_matrices(rad[rad["Fn"] == fn], row["omega_e"], levels=(level,))
        z = dynamic_stiffness(row["omega_e"], m, *ab, REPORT_C)
        for wave_level in (1, 2):
            group = loads[(loads["Fn"] == fn) & (loads["source_group"] == wave_level)]
            f = measured_load(group, ratio, row["k"])
            xi = np.linalg.solve(z, f)
            rows.append(
                [
                    fn,
                    ratio,
                    level,
                    wave_level,
                    abs(xi[0]),
                    lag(xi[0]),
                    abs(xi[1]) * LENGTH / (2 * np.pi),
                    lag(xi[1]),
                ]
            )
    return rows


def assess(root=ROOT):
    scores, errors = compare(root, predictions_by_solver=paired_parity(root))
    write(
        root / "paired_scores.csv",
        SCORE_HEADER,
        [r for r in scores if r[3] in ("heave", "pitch")],
    )
    write(
        root / "paired_trials.csv",
        ERROR_HEADER,
        [r for r in errors if r[3] in ("heave", "pitch")],
    )
    phase, coverage = phase_substitutions(root)
    write(
        root / "phase_substitutions.csv",
        "Fn lambda_L quantity source_line source_group radiation excitation load_support observed predicted observed_deg predicted_deg wrapped_error_deg phase_scored effective_force_abs effective_force_lag_deg impedance_abs impedance_lag_deg cancellation_ratio",
        phase,
    )
    write(
        root / "phase_coverage.csv",
        "Fn lambda_L radiation_in_band load_support",
        coverage,
    )
    write(
        root / "pitch_level_sensitivity.csv",
        "Fn lambda_L radiation_group load_group heave heave_deg pitch pitch_deg",
        level_sensitivity(root),
    )
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    scores = assess()
    if args.require_pass and any(
        r[0] == "report_paired" and r[-1] == "FAIL" for r in scores
    ):
        raise SystemExit(1)
