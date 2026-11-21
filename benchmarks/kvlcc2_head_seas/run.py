"""Geometry-only heave, pitch and radiated-energy added resistance of the KVLCC2 tanker in head seas."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from tensorbem.hydrostatic import (
    compute_mass_matrix,
    compute_hydrostatic_restoring_matrix,
)
from tensorbem.geometry3d import WettedHullMesh
from tensorbem.section_radiation import solve_section_heave
from tensorbem.rao import dynamic_stiffness
from tensorbem.added_resistance import radiated_resistance, bow_reflection
from benchmarks.marine_experiments.io import write
from .geometry import GRIDS, ROOT, L, SCALE, mesh_from_patch

RHO, G = 1000.0, 9.81
LCG = 0.0348 * L
U = 15.5 * (1852 / 3600) / np.sqrt(SCALE)
B = 58.0 / SCALE


def calculate(task):
    ratio, (n, g, p) = task
    with np.load(ROOT / "geometry.npz") as archive:
        patch = archive[f"patch{n}"]
        mesh = mesh_from_patch(patch)
        waterline = patch[-1, :, :2]
        bow = waterline[: np.argmax(waterline[:, 1]) + 1]
        stations = archive[f"xs{n}"] - LCG
        sections = archive[f"sections{n}"]
    mesh = WettedHullMesh(mesh.vertices - [LCG, 0, 0], mesh.faces)
    mass = RHO * 312622.0 / SCALE**3
    cg = np.array([0.0, 0.0, (18.56 - 20.8) / SCALE])
    block = np.ix_([2, 4], [2, 4])
    c = compute_hydrostatic_restoring_matrix(mesh, mass, cg, RHO, G).restoring_matrix[
        block
    ]
    k = 2 * np.pi / (L * ratio)
    omega = np.sqrt(G * k)
    we = omega + U * k
    hyd = solve_forward_speed_heave_pitch(mesh, omega, U, density=RHO, gravity=G)
    a = []
    b = []
    atten = []
    residual = 0.0
    for section in sections:
        y, z = section.T
        contour = np.column_stack((np.r_[-y[::-1], y[1:]], np.r_[z[::-1], z[1:]]))
        sec = solve_section_heave(contour, we, density=RHO, gravity=G)
        a.append(sec.added_mass)
        b.append(sec.damping)
        # Average exp(kz) on each straight segment, including a flat keel.
        dz = k * np.diff(z)
        avg = np.ones_like(dz)
        np.divide(np.expm1(dz), dz, out=avg, where=dz != 0)
        atten.append(np.sum(np.diff(y) * np.exp(k * z[:-1]) * avg) / y[-1])
        residual = max(residual, sec.relative_residual)
    # Consistent with NMRI's 2.5 m dry-ended transom extension. The
    # sectional correction stops at FP; submerged bulb still enters 3D BEM.
    x = np.r_[(-8.0 - 160.0) / SCALE - LCG, stations, 160.0 / SCALE - LCG]
    incident = -1j * omega * np.exp(-1j * k * x) * np.r_[atten[0], atten, atten[-1]]
    reflection = bow_reflection(
        bow, k=k, draft=20.8 / SCALE, froude=U / np.sqrt(G * L), density=RHO, gravity=G
    ) / (RHO * G * B**2 / L)
    values = {}
    for gyration in (0.24, 0.25, 0.26):
        m = compute_mass_matrix(
            mass, cg, np.diag([0.0, mass * (gyration * L) ** 2, 0.0])
        )[block]
        xi = np.linalg.solve(
            dynamic_stiffness(we, m, hyd.added_mass, hyd.damping, c), hyd.excitation
        )
        force = radiated_resistance(
            x, np.r_[0, a, 0], np.r_[0, b, 0], incident, xi, k=k, omega_e=we, speed=U
        )
        values[gyration] = [abs(xi[0]), abs(xi[1]) / k, force / (RHO * G * B**2 / L)]
    rows = []
    for i, key in enumerate(("heave", "pitch", "resistance")):
        rows.append(
            dict(
                panels=len(mesh.faces),
                lambda_L=ratio,
                quantity=key,
                predicted=values[0.25][i],
                kyy_024=values[0.24][i],
                kyy_026=values[0.26][i],
                volume_rel=mesh.volume / (mass / RHO) - 1,
                hull_residual=hyd.relative_residual,
                section_residual=residual,
                hull_condition=hyd.condition_number,
            )
        )
    rows.append(
        dict(
            rows[-1],
            quantity="resistance_plus_reflection",
            predicted=rows[-1]["predicted"] + reflection,
            kyy_024=rows[-1]["kyy_024"] + reflection,
            kyy_026=rows[-1]["kyy_026"] + reflection,
        )
    )
    return rows


def main(workers):
    with (ROOT / "observations.csv").open() as f:
        obs = list(csv.DictReader(f))
    ratios = sorted({float(r["lambda_L"]) for r in obs})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        raw = [
            r
            for rows in pool.map(
                calculate, [(v, grid) for grid in GRIDS for v in ratios]
            )
            for r in rows
        ]
    observed = {
        (r["quantity"], float(r["lambda_L"])): float(r["observed"]) for r in obs
    }
    rows = []
    for r in raw:
        ref = observed.get(
            (r["quantity"].removesuffix("_plus_reflection"), r["lambda_L"])
        )
        if ref is not None:
            rows.append(dict(**r, observed=ref, error=r["predicted"] - ref))
    scores = []
    for panels in sorted({r["panels"] for r in rows}):
        for q in ("heave", "pitch", "resistance", "resistance_plus_reflection"):
            group = [r for r in rows if r["quantity"] == q and r["panels"] == panels]
            errs = np.array([r["error"] for r in group])
            scale = (
                max(abs(r["observed"]) for r in group)
                if q.startswith("resistance")
                else 1.0
            )
            rms = np.sqrt(np.mean(errs**2)) / scale
            maximum = max(abs(errs)) / scale
            ie = np.argmax([r["observed"] for r in group])
            ip = np.argmax([r["predicted"] for r in group])
            thresholds = (0.2, 0.4) if q.startswith("resistance") else (0.1, 0.2)
            passed = (
                rms <= thresholds[0]
                and maximum <= thresholds[1]
                and abs(ie - ip) <= 1
                and abs(group[0]["volume_rel"]) <= 0.01
            )
            scores.append(
                dict(
                    panels=panels,
                    quantity=q,
                    points=len(group),
                    scale=scale,
                    rms=rms,
                    maximum=maximum,
                    measured_peak=group[ie]["lambda_L"],
                    predicted_peak=group[ip]["lambda_L"],
                    verdict="PASS" if passed else "FAIL",
                )
            )
    write(ROOT / "results.csv", list(rows[0]), rows)
    write(ROOT / "scores.csv", list(scores[0]), scores)
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--require-pass", action="store_true")
    a = p.parse_args()
    scores = main(a.workers)
    if a.require_pass and any(
        r["verdict"] == "FAIL" and r["panels"] == max(s["panels"] for s in scores)
        for r in scores
    ):
        raise SystemExit(1)
