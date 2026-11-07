"""Verification driver for the second-order mean wave drift force (IN21-B2).

Runs three independent verifications and dumps every number to JSON so that the
report is generated from artefacts rather than retyped:

1. **Analytic anchor (2-D vertical wall).**  Full reflection of a regular wave of
   amplitude ``a`` gives the closed form ``F = rho g a^2 / 2`` per unit length.
   The same near-field assembly that is used in 3-D is applied to the exact
   standing-wave field, so the anchor pins the coefficients of the pressure and
   waterline terms.
2. **Near/far-field cross-check (3-D vertical cylinder).**  The Pinkster surface
   integral and the Maruo Kochin-function integral are evaluated on the *same*
   diffraction solution; they share no formula, so their agreement is a genuine
   mutual verification.
3. **Long-wave asymptotics.**  For a bottom-mounted cylinder ``F ~ (k a)^3`` as
   ``k a -> 0``; the log-log slope of both estimators is fitted on the small
   ``k a`` window and compared with ``3 +- 0.2``.

Usage
-----
``python -m tensorbem.benchmarks.drift_second_order --output runs/drift_second_order``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..drift import (
    bottom_mounted_cylinder_mesh,
    drift_force_3d,
    far_field_mean_drift_3d,
    near_field_mean_drift_3d,
    vertical_wall_mean_drift,
    vertical_wall_near_field,
)

DENSITY = 1025.0
GRAVITY = 9.81
RADIUS = 1.0


# ---------------------------------------------------------------------------
# 1. analytic anchor
# ---------------------------------------------------------------------------
def wall_anchor_study(
    wavenumbers: Sequence[float] = (0.2, 0.4, 0.7, 1.0, 1.6),
    amplitude: float = 1.0,
) -> dict:
    rows = []
    for k in wavenumbers:
        res = vertical_wall_near_field(
            amplitude, k, density=DENSITY, gravity=GRAVITY
        )
        exact = float(res["exact"][0])
        rows.append(
            {
                "wavenumber": float(k),
                "pressure_term": float(res["pressure_term"][0]),
                "waterline_term": float(res["waterline_term"][0]),
                "near_field_total": float(res["total"][0]),
                "closed_form": exact,
                "relative_error": float(abs(res["total"][0] - exact) / abs(exact)),
                "pressure_fraction_of_waterline": float(
                    abs(res["pressure_term"][0]) / abs(res["waterline_term"][0])
                ),
            }
        )
    worst = max(row["relative_error"] for row in rows)
    return {
        "anchor": (
            "2-D vertical wall at x = 0, fluid at x > 0, incident wave travelling in "
            "-x: closed form F_x = -rho g a^2 / 2 per unit length (signed check)"
        ),
        "amplitude": float(amplitude),
        "density": DENSITY,
        "gravity": GRAVITY,
        "rows": rows,
        "max_relative_error": float(worst),
        "tolerance": 1.0e-3,
        "passed": bool(worst <= 1.0e-3),
    }


# ---------------------------------------------------------------------------
# 2/3. cylinder sweep, near vs far, long-wave slope
# ---------------------------------------------------------------------------
@dataclass
class CylinderCase:
    ka: float
    n_theta: int
    n_z: int
    growth: float = 1.5
    depth_factor: float = 5.0
    n_azimuth: int = 180
    wall_seconds: float = 0.0
    near: dict = field(default_factory=dict)
    far: dict = field(default_factory=dict)
    extra_far: dict = field(default_factory=dict)


def cylinder_case(
    ka: float,
    *,
    radius: float = RADIUS,
    n_theta: int = 24,
    n_z: int = 8,
    growth: float = 1.5,
    depth_factor: float = 5.0,
    n_azimuth: int = 180,
) -> CylinderCase:
    k = float(ka) / radius
    mesh = bottom_mounted_cylinder_mesh(
        radius, depth_factor / k, n_theta=n_theta, n_z=n_z, growth=growth
    )
    started = time.time()
    out = drift_force_3d(mesh, k, n_azimuth=n_azimuth)
    elapsed = time.time() - started
    near = out["near"]
    far = out["far"]
    far_proj = far_field_mean_drift_3d(
        mesh, out["diffraction"], density=DENSITY, n_azimuth=n_azimuth,
        weight="cos_projection",
    )
    scale = DENSITY * GRAVITY * radius  # A = 1 m
    return CylinderCase(
        ka=float(ka),
        n_theta=int(n_theta),
        n_z=int(n_z),
        growth=float(growth),
        depth_factor=float(depth_factor),
        n_azimuth=int(n_azimuth),
        wall_seconds=float(elapsed),
        near={
            "mean_force_x": float(near.mean_force[0]),
            "mean_force_y": float(near.mean_force[1]),
            "pressure_term_x": float(near.pressure_term[0]),
            "waterline_term_x": float(near.waterline_term[0]),
            "coefficient": float(near.mean_force[0] / scale),
            "waterline_edges": int(near.waterline_count),
            "panel_count": int(mesh.panel_count),
            "relative_residual": float(near.relative_residual),
        },
        far={
            "mean_force_x": float(far["force"][0]),
            "mean_force_y": float(far["force"][1]),
            "coefficient": float(far["force"][0] / scale),
            "kochin_abs_max": float(np.max(np.abs(far["kochin"]))),
        },
        extra_far={
            "cos_projection_x": float(far_proj["force"][0]),
            "isotropic_term_x": float(far["isotropic_term"][0]),
            "projection_term_x": float(far["projection_term"][0]),
        },
    )


def loglog_slope(ka: Sequence[float], values: Sequence[float]) -> dict:
    x = np.asarray(ka, dtype=float)
    y = np.asarray(values, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("all values must be positive for a log-log fit")
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    predicted = np.exp(intercept) * x**slope
    residual = np.log(y) - np.log(predicted)
    return {
        "slope": float(slope),
        "intercept_log": float(intercept),
        "max_log_residual": float(np.max(np.abs(residual))),
        "ka_window": [float(x.min()), float(x.max())],
        "points": int(x.size),
    }


def summarise_cases(
    cases: Sequence[CylinderCase],
    *,
    n_theta: int,
    n_z: int,
    slope_window: tuple[float, float] = (0.2, 0.65),
    agreement_tolerance: float = 0.15,
) -> dict:
    """Turn a list of solved cases into the JSON payload of the cylinder study."""
    window = [c for c in cases if slope_window[0] <= c.ka <= slope_window[1]]
    near_slope = loglog_slope(
        [c.ka for c in window], [abs(c.near["mean_force_x"]) for c in window]
    )
    far_slope = loglog_slope(
        [c.ka for c in window], [abs(c.far["mean_force_x"]) for c in window]
    )
    agreements = []
    for c in cases:
        near_x = c.near["mean_force_x"]
        far_x = c.far["mean_force_x"]
        denom = max(abs(near_x), abs(far_x))
        agreements.append(
            {
                "ka": c.ka,
                "near": near_x,
                "far": far_x,
                "relative_difference": float(abs(near_x - far_x) / denom),
                "sign_agree": bool(near_x * far_x > 0.0),
            }
        )
    worst_agree = max(a["relative_difference"] for a in agreements)
    passed_slope = (
        abs(near_slope["slope"] - 3.0) <= 0.2 and abs(far_slope["slope"] - 3.0) <= 0.2
    )
    return {
        "geometry": {
            "body": "bottom-mounted vertical circular cylinder",
            "radius_m": RADIUS,
            "draft": "truncated at 5/k below the mean free surface",
            "mesh": {"n_theta": n_theta, "n_z": n_z, "growth": 1.5},
            "amplitude_m": 1.0,
        },
        "cases": [
            {
                "ka": c.ka,
                "n_theta": c.n_theta,
                "n_z": c.n_z,
                "n_azimuth": c.n_azimuth,
                "wall_seconds": round(c.wall_seconds, 2),
                "near": c.near,
                "far": c.far,
                "far_weight_variants": c.extra_far,
            }
            for c in cases
        ],
        "slope": {
            "near": near_slope,
            "far": far_slope,
            "target": 3.0,
            "tolerance": 0.2,
            "passed": bool(passed_slope),
        },
        "agreement": {
            "tolerance": agreement_tolerance,
            "rows": agreements,
            "worst_relative_difference": float(worst_agree),
            "passed": bool(worst_agree <= agreement_tolerance),
        },
    }


def cylinder_study(
    ka_values: Iterable[float] = (0.2, 0.3, 0.45, 0.65, 0.9, 1.2, 1.6),
    *,
    n_theta: int = 24,
    n_z: int = 8,
    slope_window: tuple[float, float] = (0.2, 0.65),
    agreement_tolerance: float = 0.15,
) -> dict:
    cases = []
    for ka in ka_values:
        case = cylinder_case(ka, n_theta=n_theta, n_z=n_z)
        cases.append(case)
        print(
            f"ka={case.ka:.2f} panels={case.near['panel_count']} "
            f"C_near={case.near['coefficient']:+.4f} "
            f"C_far={case.far['coefficient']:+.4f} ({case.wall_seconds:.0f}s)",
            flush=True,
        )
    return summarise_cases(
        cases,
        n_theta=n_theta,
        n_z=n_z,
        slope_window=slope_window,
        agreement_tolerance=agreement_tolerance,
    )


def mesh_convergence(
    ka: float = 0.5,
    resolutions: Sequence[tuple[int, int]] = ((16, 6), (24, 8), (32, 10)),
) -> dict:
    rows = []
    for n_theta, n_z in resolutions:
        case = cylinder_case(ka, n_theta=n_theta, n_z=n_z)
        rows.append(
            {
                "n_theta": n_theta,
                "n_z": n_z,
                "panel_count": case.near["panel_count"],
                "near_x": case.near["mean_force_x"],
                "far_x": case.far["mean_force_x"],
                "relative_difference": float(
                    abs(case.near["mean_force_x"] - case.far["mean_force_x"])
                    / max(abs(case.far["mean_force_x"]), 1e-12)
                ),
                "wall_seconds": round(case.wall_seconds, 2),
            }
        )
        print(
            f"convergence ka={ka} n_theta={n_theta} n_z={n_z} "
            f"near={case.near['mean_force_x']:.1f} far={case.far['mean_force_x']:.1f}",
            flush=True,
        )
    return {"ka": float(ka), "rows": rows}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_wall_anchor(anchor: dict, path: str) -> str:
    plt = _pyplot()
    ks = [row["wavenumber"] for row in anchor["rows"]]
    total = [row["near_field_total"] for row in anchor["rows"]]
    pressure = [row["pressure_term"] for row in anchor["rows"]]
    waterline = [row["waterline_term"] for row in anchor["rows"]]
    exact = [row["closed_form"] for row in anchor["rows"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ks, exact, "k-", lw=2.0, label=r"closed form $\rho g a^2/2$")
    ax.plot(ks, total, "o", ms=6, label="near-field total")
    ax.plot(ks, waterline, "s--", ms=5, label="waterline term")
    ax.plot(ks, pressure, "^--", ms=5, label="pressure term")
    ax.set_xlabel("wavenumber k [1/m]")
    ax.set_ylabel("mean drift force / length [N/m]")
    ax.set_title("2-D vertical wall anchor: near-field vs closed form")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_coefficient(study: dict, path: str) -> str:
    plt = _pyplot()
    kas = [c["ka"] for c in study["cases"]]
    near = [c["near"]["coefficient"] for c in study["cases"]]
    far = [c["far"]["coefficient"] for c in study["cases"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(kas, near, "o-", label="near field (Pinkster)")
    ax.plot(kas, far, "s-", label="far field (Maruo/Kochin)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("ka (k = wavenumber, a = cylinder radius)")
    ax.set_ylabel(r"$C = \bar F_x / (\rho g A^2 a)$")
    ax.set_title("Mean drift force on a bottom-mounted cylinder")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_slope(study: dict, path: str) -> str:
    plt = _pyplot()
    kas = np.array([c["ka"] for c in study["cases"]])
    near = np.abs([c["near"]["mean_force_x"] for c in study["cases"]])
    far = np.abs([c["far"]["mean_force_x"] for c in study["cases"]])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(kas, near, "o-", label="near field")
    ax.loglog(kas, far, "s-", label="far field")
    ref = near[0] * (kas / kas[0]) ** 3
    ax.loglog(kas, ref, "k--", lw=1.2, label=r"slope 3 reference")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$|\bar F_x|$ [N]")
    ax.set_title("Long-wave asymptotics: slope 3")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_kochin(ka: float, n_theta: int, n_z: int, path: str, n_azimuth: int = 360) -> dict:
    plt = _pyplot()
    k = ka / RADIUS
    mesh = bottom_mounted_cylinder_mesh(RADIUS, 5.0 / k, n_theta=n_theta, n_z=n_z)
    out = drift_force_3d(mesh, k, n_azimuth=n_azimuth)
    far = out["far"]
    theta = np.asarray(far["azimuth"])
    pattern = np.abs(far["kochin"]) ** 2
    fig = plt.figure(figsize=(5.6, 5.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, pattern / pattern.max(), lw=1.5)
    ax.set_title(rf"$|K(\theta)|^2$ normalised, ka = {ka:g}", pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"path": path, "peak_to_mean": float(pattern.max() / pattern.mean())}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_all(
    output_dir: str,
    *,
    n_theta: int = 24,
    n_z: int = 8,
    ka_values: Sequence[float] = (0.2, 0.3, 0.45, 0.65, 0.9, 1.2, 1.6),
    convergence: bool = True,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print("[1/4] wall anchor", flush=True)
    anchor = wall_anchor_study()
    print("[2/4] cylinder sweep", flush=True)
    study = cylinder_study(ka_values, n_theta=n_theta, n_z=n_z)
    print("[3/4] mesh convergence", flush=True)
    conv = mesh_convergence() if convergence else {"rows": [], "skipped": True}
    print("[4/4] figures", flush=True)
    figs = {
        "wall_anchor": figure_wall_anchor(
            anchor, os.path.join(figures_dir, "drift_wall_anchor.png")
        ),
        "coefficient": figure_coefficient(
            study, os.path.join(figures_dir, "drift_cylinder_coefficient.png")
        ),
        "slope": figure_slope(
            study, os.path.join(figures_dir, "drift_small_ka_slope.png")
        ),
        "kochin": figure_kochin(
            float(ka_values[len(ka_values) // 2]),
            n_theta,
            n_z,
            os.path.join(figures_dir, "drift_kochin_pattern.png"),
        ),
    }
    payload = {
        "mission": "IN21-B2",
        "repository": "TensorBEM",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "density": DENSITY,
        "gravity": GRAVITY,
        "wall_anchor": anchor,
        "cylinder": study,
        "mesh_convergence": conv,
        "figures": figs,
        "interface_contract": {
            "drift_module": "tensorbem.drift",
            "near_field": "near_field_mean_drift_3d(mesh, Diffraction3DResult) -> DriftForce3DResult",
            "far_field": "far_field_mean_drift_3d(mesh, Diffraction3DResult) -> dict(force=(Fx,Fy))",
            "upstream_dependency": "tensorbem.free_surface_solver3d.solve_diffraction_3d / kochin_function_3d",
            "units": "SI, forces in newtons, coefficient normalised by rho g A^2 a",
            "amplitudes": "complex amplitudes with exp(-i omega t); <f^2> = |f_hat|^2/2",
        },
    }
    json_path = os.path.join(output_dir, "drift_second_order.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    payload["json_path"] = json_path
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/drift_second_order")
    parser.add_argument("--n-theta", type=int, default=24)
    parser.add_argument("--n-z", type=int, default=8)
    parser.add_argument("--ka", default="0.2,0.3,0.45,0.65,0.9,1.2,1.6")
    parser.add_argument("--no-convergence", action="store_true")
    args = parser.parse_args(argv)
    ka_values = [float(v) for v in args.ka.split(",") if v.strip()]
    payload = run_all(
        args.output,
        n_theta=args.n_theta,
        n_z=args.n_z,
        ka_values=ka_values,
        convergence=not args.no_convergence,
    )
    print(json.dumps(
        {
            "wall_anchor_passed": payload["wall_anchor"]["passed"],
            "wall_anchor_max_rel_err": payload["wall_anchor"]["max_relative_error"],
            "near_slope": payload["cylinder"]["slope"]["near"]["slope"],
            "far_slope": payload["cylinder"]["slope"]["far"]["slope"],
            "slope_passed": payload["cylinder"]["slope"]["passed"],
            "near_far_worst": payload["cylinder"]["agreement"]["worst_relative_difference"],
            "json": payload["json_path"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())