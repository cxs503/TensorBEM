"""Forward-speed head-sea added resistance against the original Journée Tables 8 and 10."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import numpy as np

from benchmarks.journee_wigley3_forward_speed.import_tables import SHA256
from benchmarks.journee_wigley3_zero_speed.run import model_matrices
from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from tensorbem.added_resistance import radiated_resistance, bow_reflection
from benchmarks.marine_experiments.io import write
from benchmarks.marine_experiments.wigley import HULLS, L, T, sections, mesh_for
from tensorbem.rao import dynamic_stiffness
from tensorbem.section_radiation import solve_section_heave
from tensorbem.wigley3_hull import wigley3_half_breadth

ROOT = Path(__file__).resolve().parent
G, RHO = 9.81, 1000.0
# 3-D stations/depth panels, sectional longitudinal intervals/contour panels.
GRIDS = ((12, 6, 12, 16), (16, 8, 20, 24))
TRIAL_FIELDS = "source_table hull Fn lambda_L source_line source_group wave_cm heave heave_deg pitch pitch_deg observed".split()


def extract(archive, output):
    if sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Expected original report 0909 ZIP")
    result = []
    with ZipFile(archive) as z:
        for hull in HULLS:
            for speed in {"II": (20,), "IV": (20, 30)}.get(hull, (20, 30, 40)):
                for line, raw in enumerate(
                    z.read(f"08-{hull}-{speed}.exp").decode("ascii").splitlines(), 1
                ):
                    a = raw.split()
                    if not a:
                        continue
                    if len(a) != (7 if hull == "I" else 13):
                        raise ValueError("Invalid Table 8 width")
                    for group, offset in enumerate(range(1, len(a), 6), 1):
                        vals = a[offset : offset + 6]
                        if all(float(v) == -1111 for v in vals):
                            continue
                        if any(float(vals[i]) == -1111 for i in (0, 1, 3, 5)):
                            raise ValueError("Missing trial amplitude or resistance")
                        vals = [v if float(v) != -1111 else "" for v in vals]
                        result.append(
                            dict(
                                zip(
                                    TRIAL_FIELDS,
                                    [8, hull, speed / 100, a[0], line, group, *vals],
                                    strict=True,
                                )
                            )
                        )
        for hull in ("III", "IV"):
            for line, raw in enumerate(
                z.read(f"10-{hull}-00.exp").decode("ascii").splitlines(), 1
            ):
                a = raw.split()
                if not a:
                    continue
                if len(a) != 11 or "-1111" in a:
                    raise ValueError("Invalid Table 10 row")
                # Report p.57: wave, surge force + phase, heave + phase,
                # pitch + phase, added resistance, station 17 motion + phase.
                row = [10, hull, 0.0, a[0], line, 1, a[1], *a[4:9]]
                result.append(dict(zip(TRIAL_FIELDS, row, strict=True)))
    write(output, TRIAL_FIELDS, result)


def calculate(task):
    hull, fn, ratio, trials, grid = task
    nx, nz, n, panels = grid
    beam, body_mass, cg = HULLS[hull]
    speed = fn * np.sqrt(G * L)
    k = 2 * np.pi / (ratio * L)
    omega = np.sqrt(G * k)
    we = omega + k * speed
    mesh = mesh_for(hull, nx, nz)
    m, c = model_matrices(mesh, mass=body_mass, cg_z=cg)
    hyd = solve_forward_speed_heave_pitch(mesh, omega, speed, density=RHO, gravity=G)
    xi = np.linalg.solve(
        dynamic_stiffness(we, m, hyd.added_mass, hyd.damping, c), hyd.excitation
    )
    x = np.linspace(-L / 2, L / 2, n + 1)
    z = np.linspace(-T, 0, panels // 2 + 1)
    mass = np.zeros(n + 1)
    damp = np.zeros(n + 1)
    attenuation = np.ones(n + 1)
    maxres = 0.0
    for i in range(n // 2, n):
        y = sections(hull, x[i], z)
        contour = np.column_stack((np.r_[-y[::-1], y[1:]], np.r_[z[::-1], z[1:]]))
        sol = solve_section_heave(contour, we, density=RHO, gravity=G)
        # Project the local incident vertical velocity over the section's
        # horizontal width (signed projection also handles non-monotonic sides).
        width_mean = (
            np.sum(
                np.diff(y)
                * np.exp(k * z[:-1])
                * np.expm1(k * np.diff(z))
                / (k * np.diff(z))
            )
            / y[-1]
        )
        for j in (i, n - i):
            mass[j], damp[j], attenuation[j] = sol.added_mass, sol.damping, width_mean
        maxres = max(maxres, sol.relative_residual)
    # End sections collapse in breadth. Use the limit of the local shape
    # for the section average; M' contributes to the last cell.
    y = wigley3_half_breadth(0, z, B=beam)
    attenuation[[0, -1]] = (
        np.sum(
            np.diff(y)
            * np.exp(k * z[:-1])
            * np.expm1(k * np.diff(z))
            / (k * np.diff(z))
        )
        / y[-1]
    )
    incident = -1j * omega * attenuation * np.exp(-1j * k * x)
    scale = RHO * G * beam**2 / L

    def coefficient(motion):
        return (
            radiated_resistance(
                x, mass, damp, incident, motion, k=k, omega_e=we, speed=speed
            )
            / scale
        )

    bow_x = np.linspace(L / 2, 0, n + 1)
    bow = np.column_stack((bow_x, sections(hull, bow_x, 0.0)))
    reflection = (
        bow_reflection(bow, k=k, draft=T, froude=fn, density=RHO, gravity=G) / scale
    )
    predicted = coefficient(xi)
    common = dict(
        reflection=reflection,
        with_reflection=predicted + reflection,
        panels=len(mesh.faces),
        stations=n + 1,
        section_panels=panels,
        predicted=predicted,
        restrained=coefficient((0, 0)),
        native_heave=abs(xi[0]),
        native_pitch=abs(xi[1]) * L / (2 * np.pi),
        hull_residual=hyd.relative_residual,
        section_residual=maxres,
    )
    rows = []
    for t in trials:
        motion = (
            None
            if not (t["heave_deg"] and t["pitch_deg"])
            else [
                float(t[key]) * fac * np.exp(-1j * np.deg2rad(float(t[key + "_deg"])))
                for key, fac in (("heave", 1.0), ("pitch", 2 * np.pi / L))
            ]
        )
        rows.append(
            {
                **t,
                **common,
                "with_observed_motion": (
                    coefficient(motion) if motion is not None else ""
                ),
            }
        )
    return rows


def score(rows):
    scores = []
    groups = sorted({(r["hull"], float(r["Fn"]), r["panels"]) for r in rows})
    for hull, fn, panels in groups:
        group = [
            r
            for r in rows
            if (r["hull"], float(r["Fn"]), r["panels"]) == (hull, fn, panels)
        ]
        measured = np.array([float(r["observed"]) for r in group])
        scale = max(abs(measured))
        ratios = np.array([float(r["lambda_L"]) for r in group])
        x = np.unique(ratios)
        peak = np.argmax([measured[ratios == v].mean() for v in x])
        for key in (
            "predicted",
            "with_reflection",
            "with_observed_motion",
            "restrained",
        ):
            valid = np.array([r[key] != "" for r in group])
            pred = np.array([r[key] for r in group if r[key] != ""])
            err = pred - measured[valid]
            rms = np.sqrt(np.mean(err**2)) / scale
            maximum = max(abs(err)) / scale
            available = np.unique(ratios[valid])
            ip = np.argmax([pred[ratios[valid] == v].mean() for v in available])
            scores.append(
                dict(
                    hull=hull,
                    Fn=fn,
                    panels=panels,
                    route=key,
                    trials=len(pred),
                    peak_scale=scale,
                    nrms=rms,
                    nmax=maximum,
                    measured_peak=x[peak],
                    predicted_peak=available[ip],
                    verdict=(
                        (
                            "PASS"
                            if rms <= 0.20 and maximum <= 0.40 and abs(peak - ip) <= 1
                            else "FAIL"
                        )
                        if key in ("predicted", "with_reflection")
                        else "diagnostic"
                    ),
                )
            )
    return scores


def main(workers):
    with (ROOT / "trials.csv").open() as f:
        trials = list(csv.DictReader(f))
    groups = {}
    for r in trials:
        groups.setdefault((r["hull"], float(r["Fn"]), float(r["lambda_L"])), []).append(
            r
        )
    tasks = [(*key, values, grid) for grid in GRIDS for key, values in groups.items()]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(calculate, tasks):
            rows.extend(batch)
    scores = score(rows)
    write(ROOT / "results.csv", list(rows[0]), rows)
    write(ROOT / "scores.csv", list(scores[0]), scores)
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    if args.archive:
        extract(args.archive, ROOT / "trials.csv")
    scores = main(args.workers)
    if args.require_pass and any(
        r["route"] == "predicted"
        and r["panels"] == max(q["panels"] for q in scores)
        and r["verdict"] == "FAIL"
        for r in scores
    ):
        raise SystemExit(1)
