"""Evaluate a reversal-symmetric interpolation of endpoint radiation coefficients."""

import argparse
import csv
from hashlib import sha256
from zipfile import ZipFile
from pathlib import Path

import numpy as np

from .assess import ROOT, compare, measured_matrices, table, write
from .import_tables import SHA256
from .radiation_study import endpoint_load, matrices, solve_motion
from .strip import COEFFICIENTS


def speed_weights(fn, lower, upper, cross, *, parity):
    """Return endpoint weights at fixed encounter frequency, for signed ship speed.

    On a fore/aft-symmetric hull diagonal coefficients are even, while the
    heave/pitch cross terms are odd. Interpolate Q or Q/Fn linearly in Fn².
    """
    if not 0 < lower < upper or not lower <= abs(fn) <= upper:
        raise ValueError("speed is outside the nonzero endpoint band")
    if not parity:
        if fn < 0:
            raise ValueError("linear reference uses positive speeds")
        right = (fn - lower) / (upper - lower)
        return 1 - right, right
    right = (fn**2 - lower**2) / (upper**2 - lower**2)
    if cross:
        return (1 - right) * fn / lower, right * fn / upper
    return 1 - right, right


def endpoint_matrices(training, fn, we):
    a, b = 0.2, 0.4
    if fn in (a, b):
        return measured_matrices(training[training["Fn"] == fn], we)
    endpoints = [measured_matrices(training[training["Fn"] == s], we) for s in (a, b)]
    if any(x is None for x in endpoints):
        return None
    result = np.empty((2, 2, 2))
    for i, j in np.ndindex(2, 2):
        w0, w1 = speed_weights(fn, a, b, i != j, parity=True)
        result[:, i, j] = (
            w0 * np.asarray(endpoints[0])[:, i, j]
            + w1 * np.asarray(endpoints[1])[:, i, j]
        )
    return result


def parity_motions(root=ROOT):
    native = table(root, "tensorbem")
    rows = native[native["panels"] == max(native["panels"])].copy()
    radiation = table(root, "radiation")
    train = radiation[np.isin(radiation["Fn"], (0.2, 0.4))]
    loads = table(root, "loads")
    train_load = loads[np.isin(loads["Fn"], (0.2, 0.4))]
    audit = []
    for r in rows:
        fn, we, ratio = r["Fn"], r["omega_e"], r["lambda_L"]
        rad = endpoint_matrices(train, fn, we)
        force = endpoint_load(train_load, fn, ratio, r["k"])
        use_rad, use_load = rad is not None, force is not None
        if not use_rad:
            rad = matrices(r)
        if not use_load:
            force = [complex(r["F" + d + "_re"], r["F" + d + "_im"]) for d in "35"]
        solve_motion(r, rad, force)
        audit.append(
            [
                fn,
                ratio,
                int(use_rad),
                int(use_load),
                *[r[k] for k in COEFFICIENTS],
                force[0].real,
                force[0].imag,
                force[1].real,
                force[1].imag,
                *[r[k] for k in ("heave", "heave_deg", "pitch", "pitch_deg")],
            ]
        )
    return rows, audit


def extract(archive, output):
    """Preserve the one- and two-amplitude forced tests from the other two hulls."""
    if sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("expected Journée's original report 0909 archive")
    records = []
    with ZipFile(archive) as z:
        for hull in ("I", "IV"):
            for mode, driven in (("02", 3), ("03", 5)):
                for speed in (20, 30, 40):
                    lines = (
                        z.read(f"{mode}-{hull}-{speed}.exp")
                        .decode("ascii")
                        .splitlines()
                    )
                    for lineno, line in enumerate(lines, 1):
                        fields = line.split()
                        if not fields:
                            continue
                        expected = 5 if hull == "IV" and speed == 40 else 10
                        if len(fields) != expected or any(
                            float(v) == -1111 for v in fields
                        ):
                            raise ValueError("unexpected forced-oscillation record")
                        for j in range(0, expected, 5):
                            records.append(
                                [
                                    hull,
                                    speed / 100,
                                    driven,
                                    lineno,
                                    j // 5 + 1,
                                    *fields[j : j + 5],
                                ]
                            )
    write(
        output,
        "hull Fn driven source_line source_group w_nd A_self B_self A_cross B_cross",
        records,
    )


def cross_hull(root=ROOT):
    with (root / "additional_forced_hulls.csv").open() as f:
        records = list(csv.DictReader(f))
    with (root / "radiation.csv").open() as f:
        records.extend(dict(r, hull="III") for r in csv.DictReader(f))
    output = []
    for target in records:
        if float(target["Fn"]) != 0.30:
            continue
        hull, driven, nd = target["hull"], int(target["driven"]), float(target["w_nd"])
        for grp in ("self", "cross"):
            cross = grp == "cross"
            response = 8 - driven if cross else driven
            for prefix in "AB":
                name = prefix + "_" + grp
                values = []
                for anchor in (0.2, 0.4):
                    train = [
                        r
                        for r in records
                        if r["hull"] == hull
                        and int(r["driven"]) == driven
                        and float(r["Fn"]) == anchor
                    ]
                    anchor_values = []
                    for level in sorted({r["source_group"] for r in train}):
                        series = sorted(
                            (r for r in train if r["source_group"] == level),
                            key=lambda r: float(r["w_nd"]),
                        )
                        x = [float(r["w_nd"]) for r in series]
                        if not x[0] <= nd <= x[-1]:
                            break
                        anchor_values.append(
                            np.interp(nd, x, [float(r[name]) for r in series])
                        )
                    else:
                        if anchor_values:
                            values.append(np.mean(anchor_values))
                covered = len(values) == 2
                estimates = (
                    [
                        np.dot(values, speed_weights(0.3, 0.2, 0.4, cross, parity=p))
                        for p in (False, True)
                    ]
                    if covered
                    else ["", ""]
                )
                output.append(
                    [
                        hull,
                        nd,
                        driven,
                        target["source_line"],
                        target["source_group"],
                        f"{prefix}{response}{driven}",
                        target[name],
                        int(covered),
                        *estimates,
                    ]
                )
    return output


def assess(root=ROOT):
    pred, audit = parity_motions(root)
    scores, _ = compare(root, predictions_by_solver={"speed_parity": pred})
    write(
        root / "speed_parity_scores.csv",
        "model panels Fn quantity trials phase_trials rms max phase_rms_deg phase_max_deg observed_peak_lambda_L predicted_peak_lambda_L observed_peak_we predicted_peak_we verdict",
        scores,
    )
    write(
        root / "speed_parity_motions.csv",
        "Fn lambda_L endpoint_radiation endpoint_load "
        + " ".join(COEFFICIENTS)
        + " F3_re F3_im F5_re F5_im heave heave_deg pitch pitch_deg",
        audit,
    )
    write(
        root / "speed_parity_cross_hull.csv",
        "hull w_nd driven source_line source_group coefficient observed in_band linear parity",
        cross_hull(root),
    )
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--require-held-speed-pass", action="store_true")
    args = parser.parse_args()
    if args.archive:
        extract(args.archive, ROOT / "additional_forced_hulls.csv")
    scores = assess()
    if args.require_held_speed_pass and any(
        r[-1] == "FAIL" for r in scores if r[2] == 0.3
    ):
        raise SystemExit(1)
