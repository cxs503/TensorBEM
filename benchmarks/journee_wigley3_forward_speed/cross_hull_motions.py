"""Test the frozen endpoint procedure on Wigley I and IV free motions at Fn = 0.30."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmarks.journee_wigley3_zero_speed.run import model_matrices
from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from benchmarks.marine_experiments.wigley import mesh_for
from tensorbem.rao import dynamic_stiffness
from tensorbem.stf_fullspeed import encounter_frequency

from .assess import (
    C33_REF,
    C55_REF,
    GRAVITY,
    LENGTH,
    LIMITS,
    ROOT,
    series_statistics,
    write,
)
from .import_tables import SHA256
from .radiation_study import endpoint_components
from .speed_parity import endpoint_matrices
from .strip import COEFFICIENTS


@dataclass(frozen=True)
class Hull:
    beam: float
    mass: float
    cg_z: float
    upper_levels: tuple
    restoring: tuple


HULLS = {
    "I": Hull(0.3, 94.6, 0.170 - 0.1875, (1, 2), (C33_REF, C55_REF)),
    "IV": Hull(0.6, 156.0, 0.0, (1,), (12238.0, 5748.0)),
}
FN = 0.30


def hull_mesh(hull, nx=20, nz=10):
    return mesh_for(hull, nx, nz)


def extract(archive, root):
    if sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Expected Journée's original report 0909 archive")
    with ZipFile(archive) as source:
        for prefix, speeds, width, filename, fields in (
            (
                "06",
                (20, 40),
                5,
                "transfer_loads",
                "force3 force3_deg moment5 moment5_deg",
            ),
            ("08", (30,), 6, "transfer_motions", "heave heave_deg pitch pitch_deg"),
        ):
            out = []
            for hull in HULLS:
                for speed in speeds:
                    member = f"{prefix}-{hull}-{speed}.exp"
                    lines = source.read(member).decode("ascii").splitlines()
                    for line, text in enumerate(lines, 1):
                        nums = text.split()
                        if not nums:
                            continue
                        groups = 1 if (hull, speed) in {("I", 30), ("IV", 40)} else 2
                        if len(nums) != 1 + groups * width:
                            raise ValueError(f"Unexpected width at {member}:{line}")
                        for level in range(groups):
                            begin = 1 + level * width
                            values = nums[begin : begin + 5]
                            if all(float(v) == -1111 for v in values):
                                continue
                            # Preserve partially missing force/phase channels (06-I-40).
                            values = [v if float(v) != -1111 else "" for v in values]
                            if not values[0] or (prefix == "08" and not all(values)):
                                raise ValueError(
                                    f"Incomplete wave run at {member}:{line}"
                                )
                            out.append(
                                [hull, speed / 100, nums[0], line, level + 1, *values]
                            )
            write(
                root / (filename + ".csv"),
                "hull Fn lambda_L source_line source_group wave_cm " + fields,
                out,
            )


def read_hulls(path):
    with path.open() as stream:
        names = stream.readline().strip().split(",")
    # Force phases are whole degrees, but an empty phase must remain NaN.
    integer_fields = {"source_line", "source_group", "driven"}
    types = [
        "U4" if n == "hull" else int if n in integer_fields else float for n in names
    ]
    return np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        dtype=list(zip(names, types, strict=True)),
        encoding="utf-8",
    )


def predict(hull, ratios, rad, loads):
    spec = HULLS[hull]
    mesh = hull_mesh(hull)
    mass, native_c = model_matrices(mesh, mass=spec.mass, cg_z=spec.cg_z)
    speed = FN * np.sqrt(GRAVITY * LENGTH)
    rows = []
    for ratio in ratios:
        k = 2 * np.pi / (LENGTH * ratio)
        omega = np.sqrt(GRAVITY * k)
        we = encounter_frequency(omega, speed, np.pi, GRAVITY)
        ab = endpoint_matrices(
            rad, FN, we, mass=spec.mass, upper_levels=spec.upper_levels
        )
        f = endpoint_components(loads, FN, ratio, k, spec.restoring)
        native_rad, native_load = ab is None, ~np.isfinite(f)
        c = native_c if native_rad else np.diag(spec.restoring)
        residual = condition = ""
        if native_rad or np.any(native_load):
            computed = solve_forward_speed_heave_pitch(
                mesh, omega, speed, density=1000.0, gravity=GRAVITY
            )
            if native_rad:
                ab = computed.added_mass, computed.damping
            f[native_load] = computed.excitation[native_load]
            residual, condition = computed.relative_residual, computed.condition_number
        xi = np.linalg.solve(dynamic_stiffness(we, mass, *ab, c), f)
        entry = dict(
            hull=hull,
            Fn=FN,
            lambda_L=ratio,
            omega_e=we,
            radiation="native" if native_rad else "endpoint",
            F3_source="native" if native_load[0] else "endpoint",
            F5_source="native" if native_load[1] else "endpoint",
            heave=abs(xi[0]),
            heave_deg=-np.angle(xi[0], deg=True) % 360,
            pitch=abs(xi[1]) * LENGTH / (2 * np.pi),
            pitch_deg=-np.angle(xi[1], deg=True) % 360,
        )
        entry.update(zip(("M33", "M55"), mass.diagonal(), strict=True))
        entry.update(zip(("C33", "C35", "C53", "C55"), c.flat, strict=True))
        entry.update(zip(COEFFICIENTS, np.ravel(ab), strict=True))
        entry.update(
            F3_re=f[0].real,
            F3_im=f[0].imag,
            F5_re=f[1].real,
            F5_im=f[1].imag,
            native_residual=residual,
            native_condition=condition,
        )
        rows.append(entry)
    return rows


def assess(root=ROOT):
    trials = read_hulls(root / "transfer_motions.csv")
    rad = read_hulls(root / "additional_forced_hulls.csv")
    loads = read_hulls(root / "transfer_loads.csv")
    limits = json.loads(LIMITS.read_text())
    scores, points, errors = [], [], []
    for hull in HULLS:
        measured = trials[(trials["hull"] == hull) & (trials["Fn"] == FN)]
        ratios = np.unique(measured["lambda_L"])
        component_rows = [
            t[(t["hull"] == hull) & np.isin(t["Fn"], (0.2, 0.4))] for t in (rad, loads)
        ]
        rows = predict(hull, ratios, *component_rows)
        points.extend(rows)
        for name in ("heave", "pitch"):
            amp, phase = np.array([(r[name], r[name + "_deg"]) for r in rows]).T
            s = series_statistics(ratios, measured, name, amp, phase, limits)
            scores.append(
                [
                    hull,
                    FN,
                    name,
                    len(measured),
                    sum(s.phase_scored),
                    s.rms,
                    s.maximum,
                    s.phase_rms,
                    s.phase_max,
                    ratios[s.observed_peak],
                    ratios[s.predicted_peak],
                    "PASS" if s.passed else "FAIL",
                ]
            )
            for j, row in enumerate(measured):
                n = s.trial_indices[j]
                errors.append(
                    [
                        hull,
                        FN,
                        name,
                        row["lambda_L"],
                        row["source_line"],
                        row["source_group"],
                        row["wave_cm"],
                        rows[n]["radiation"],
                        rows[n]["F3_source"],
                        rows[n]["F5_source"],
                        row[name],
                        amp[n],
                        s.error[j],
                        row[name + "_deg"],
                        phase[n],
                        s.angle[j],
                        int(s.phase_scored[j]),
                    ]
                )
    fields = tuple(points[0])
    write(
        root / "transfer_coefficients.csv",
        " ".join(fields),
        ([r[k] for k in fields] for r in points),
    )
    write(
        root / "transfer_scores.csv",
        "hull Fn quantity trials phase_trials rms max phase_rms_deg phase_max_deg observed_peak_lambda_L predicted_peak_lambda_L verdict",
        scores,
    )
    write(
        root / "transfer_trials.csv",
        "hull Fn quantity lambda_L source_line source_group wave_cm radiation F3_source F5_source observed predicted error observed_deg predicted_deg wrapped_error_deg phase_scored",
        errors,
    )
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    if args.archive:
        extract(args.archive, ROOT)
    scores = assess()
    if args.require_pass and any(s[-1] == "FAIL" for s in scores):
        raise SystemExit(1)
