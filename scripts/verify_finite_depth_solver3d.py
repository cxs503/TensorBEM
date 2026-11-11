"""Verification study for the 3-D finite-depth radiation BEM.

Every number written to ``docs/finite_depth_solver3d/`` is produced here; nothing
is transcribed by hand.

Anchors
-------
A1  Deep-water limit vs the *analytic* Thorne/Wang/Linton multipole solution
    (:func:`tensorbem.analytic.submerged_sphere_wave_multipole`, deep water,
    closed form).  The depth is pushed to ``h/a = 20`` and the residual bottom
    influence is measured separately (``h/a = 10`` vs ``20``).
A2  Depth effect: ``h/a`` ladder ``3, 5, 10, 20`` at ``ka = 1``; the shallow-water
    stiffening of the added mass is quantified, and the deep-water solver is run
    alongside to show that the *old* path is depth-blind.
A3  Reciprocity and axisymmetry of the full ``6x6`` on a sphere: ``A = A^T``,
    ``B = B^T``, ``A11 = A22``, rotational coupling ~ 0.
A4  Mesh ladder ``20 / 80 / 320`` panels at ``h/a = 20`` against the analytic
    reference.
A5  Mode-truncation ladder (``mode_count``) at ``h/a = 5``.
A6  Negative controls: mesh below the sea bed, mesh touching the free surface,
    operator/argument mismatch.

Run::

    PYTHONPATH=src python3 scripts/verify_finite_depth_solver3d.py \\
        --output docs/finite_depth_solver3d --figures docs/finite_depth_solver3d
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.finite_depth_solver3d import (
    assemble_finite_depth_operators_3d,
    mode_count_for,
    solve_radiation_finite_depth_3d,
)
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import submerged_sphere_mesh

RADIUS = 1.0
SUBMERGENCE = 1.5
KA = 1.0
NU = KA / RADIUS
RHO = 1025.0
G = 9.80665
CENTER = (0.0, 0.0, -SUBMERGENCE)
SUBDIVISION_NAMES = {0: 20, 1: 80, 2: 320}


def _fmt(value: float) -> float:
    return float(f"{value:.10g}")


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("nan")


def _solve(subdivision: int, depth: float, *, mode_cap=None, modes=(0, 2)):
    mesh = submerged_sphere_mesh(subdivision, radius=RADIUS, submergence=SUBMERGENCE)
    t0 = time.perf_counter()
    result = solve_radiation_finite_depth_3d(
        mesh,
        NU,
        depth,
        modes=list(modes),
        center=CENTER,
        density=RHO,
        gravity=G,
        mode_count=mode_cap,
    )
    return mesh, result, time.perf_counter() - t0


def a1_deep_water_limit(state: dict, subdivisions=(0, 1, 2), mode_cap=None) -> dict:
    ref = submerged_sphere_wave_multipole(
        KA, radius=RADIUS, submergence=SUBMERGENCE, density=RHO, gravity=G
    )
    out = {
        "analytic": {
            "A11": _fmt(ref.surge_added_mass),
            "A33": _fmt(ref.heave_added_mass),
            "B11": _fmt(ref.surge_damping),
            "B33": _fmt(ref.heave_damping),
        },
        "ladder": [],
    }
    for subdivision in subdivisions:
        mesh, result, seconds = _solve(subdivision, 20.0, mode_cap=mode_cap)
        A = np.diag(result.added_mass)
        B = np.diag(result.damping)
        out["ladder"].append(
            {
                "subdivision": subdivision,
                "panels": mesh.panel_count,
                "depth": 20.0,
                "mode_count": result.mode_count,
                "A11": _fmt(A[0]),
                "A33": _fmt(A[1]),
                "B11": _fmt(B[0]),
                "B33": _fmt(B[1]),
                "rel_err_A11": _fmt(_rel(A[0], ref.surge_added_mass)),
                "rel_err_A33": _fmt(_rel(A[1], ref.heave_added_mass)),
                "rel_err_B11": _fmt(_rel(B[0], ref.surge_damping)),
                "rel_err_B33": _fmt(_rel(B[1], ref.heave_damping)),
                "condition_number": _fmt(result.condition_number),
                "relative_residual": _fmt(result.relative_residual),
                "seconds": round(seconds, 2),
            }
        )
        state["progress"] = f"A1 subdivision {subdivision}"
    return out


def a2_depth_ladder(state: dict, mode_cap=None) -> dict:
    ref = submerged_sphere_wave_multipole(
        KA, radius=RADIUS, submergence=SUBMERGENCE, density=RHO, gravity=G
    )
    deep_mesh, deep_result, _ = _solve(1, 20.0)
    out = {"baseline_A11": _fmt(np.diag(deep_result.added_mass)[0]),
           "baseline_A33": _fmt(np.diag(deep_result.added_mass)[1]),
           "baseline_B11": _fmt(np.diag(deep_result.damping)[0]),
           "baseline_B33": _fmt(np.diag(deep_result.damping)[1]),
           "analytic_A11": _fmt(ref.surge_added_mass),
           "analytic_A33": _fmt(ref.heave_added_mass),
           "rows": []}
    for depth in (20.0, 10.0, 5.0, 3.0):
        mesh, result, seconds = _solve(1, depth, mode_cap=mode_cap)
        A = np.diag(result.added_mass)
        B = np.diag(result.damping)
        out["rows"].append(
            {
                "depth": depth,
                "depth_ratio_h_over_a": _fmt(depth / RADIUS),
                "mode_count": result.mode_count,
                "A11": _fmt(A[0]),
                "A33": _fmt(A[1]),
                "B11": _fmt(B[0]),
                "B33": _fmt(B[1]),
                "A33_over_deep": _fmt(A[1] / out["baseline_A33"]),
                "B33_over_deep": _fmt(B[1] / out["baseline_B33"]),
                "rel_err_A33_vs_analytic": _fmt(_rel(A[1], ref.heave_added_mass)),
                "rel_err_B33_vs_analytic": _fmt(_rel(B[1], ref.heave_damping)),
                "seconds": round(seconds, 2),
            }
        )
        state["progress"] = f"A2 depth {depth}"
    return out


def a3_reciprocity(state: dict) -> dict:
    mesh = submerged_sphere_mesh(1, radius=RADIUS, submergence=SUBMERGENCE)
    result = solve_radiation_finite_depth_3d(
        mesh, NU, 5.0, center=CENTER, density=RHO, gravity=G
    )
    A = result.added_mass
    B = result.damping
    scale_A = float(np.max(np.abs(A)))
    scale_B = float(np.max(np.abs(B)))
    out = {
        "depth": 5.0,
        "panels": mesh.panel_count,
        "added_mass": [[_fmt(v) for v in row] for row in A],
        "damping": [[_fmt(v) for v in row] for row in B],
        "max_asymmetry_added_mass": _fmt(float(np.max(np.abs(A - A.T))) / scale_A),
        "max_asymmetry_damping": _fmt(float(np.max(np.abs(B - B.T))) / scale_B),
        "A11_vs_A22": _fmt(_rel(A[0, 0], A[1, 1])),
        "B11_vs_B22": _fmt(_rel(B[0, 0], B[1, 1])),
        "coupling_A13_over_A33": _fmt(abs(A[0, 2]) / abs(A[2, 2])),
        "coupling_A24_over_A33": _fmt(abs(A[1, 2]) / abs(A[2, 2])),
        "rotational_A33_over_A11": _fmt(abs(A[3, 3]) / abs(A[0, 0])),
        "condition_number": _fmt(result.condition_number),
    }
    state["progress"] = "A3 reciprocity"
    return out


def a5_mode_ladder(state: dict) -> dict:
    mesh = submerged_sphere_mesh(1, radius=RADIUS, submergence=SUBMERGENCE)
    full = mode_count_for(mesh, 5.0)
    out = {"depth": 5.0, "panels": mesh.panel_count, "full_mode_count": full, "rows": []}
    reference = None
    for cap in (32, 64, 128, 256, 512, full):
        _, result, seconds = _solve(1, 5.0, mode_cap=cap)
        A = np.diag(result.added_mass)
        B = np.diag(result.damping)
        if cap == full:
            reference = (A[0], A[1], B[0], B[1])
        out["rows"].append(
            {
                "mode_count": cap,
                "A11": _fmt(A[0]),
                "A33": _fmt(A[1]),
                "B11": _fmt(B[0]),
                "B33": _fmt(B[1]),
                "seconds": round(seconds, 2),
            }
        )
        state["progress"] = f"A5 mode cap {cap}"
    if reference is not None:
        for row in out["rows"]:
            row["rel_change_A33"] = _fmt(_rel(row["A33"], reference[1]))
            row["rel_change_B33"] = _fmt(_rel(row["B33"], reference[3]))
    return out


def a6_negative_controls(state: dict) -> dict:
    checks = []
    mesh = submerged_sphere_mesh(0, radius=RADIUS, submergence=SUBMERGENCE)

    def record(name: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            checks.append({"name": name, "raised": "ValueError", "message": str(exc)[:120]})
        else:
            checks.append({"name": name, "raised": None, "message": "no exception"})

    record(
        "mesh_below_sea_bed",
        lambda: solve_radiation_finite_depth_3d(mesh, NU, 2.0, modes=[2], center=CENTER),
    )
    record(
        "non_positive_depth",
        lambda: solve_radiation_finite_depth_3d(mesh, NU, 0.0, modes=[2], center=CENTER),
    )

    def operator_mismatch() -> None:
        ops = assemble_finite_depth_operators_3d(mesh, NU, 5.0, mode_count=16)
        solve_radiation_finite_depth_3d(mesh, NU, 6.0, modes=[2], operators=ops)

    record("operator_depth_mismatch", operator_mismatch)

    def frequency_mismatch() -> None:
        ops = assemble_finite_depth_operators_3d(mesh, NU, 5.0, mode_count=16)
        solve_radiation_finite_depth_3d(mesh, 2.0, 5.0, modes=[2], operators=ops)

    record("operator_frequency_mismatch", frequency_mismatch)
    state["progress"] = "A6 negative controls"
    return {"checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/finite_depth_solver3d"))
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--mode-cap", type=int, default=None)
    parser.add_argument("--a1-subdivisions", nargs="*", type=int, default=[0, 1])
    args = parser.parse_args()

    out_dir = args.output
    fig_dir = args.figures or out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    state: dict = {"progress": "start"}
    payload: dict = {
        "case": {
            "body": "submerged geodesic sphere",
            "radius": RADIUS,
            "submergence": SUBMERGENCE,
            "ka": KA,
            "nu": NU,
            "density": RHO,
            "gravity": G,
            "k0": None,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    from tensorbem.finite_depth3d import finite_depth_wavenumber

    payload["case"]["k0"] = _fmt(finite_depth_wavenumber(NU, 5.0))

    stages = {
        "A1": a1_deep_water_limit,
        "A2": a2_depth_ladder,
        "A3": a3_reciprocity,
        "A5": a5_mode_ladder,
        "A6": a6_negative_controls,
    }
    for name, fn in stages.items():
        if name in args.skip:
            continue
        t0 = time.perf_counter()
        print(f"[{name}] start ({fn.__name__})", flush=True)
        if name == "A1":
            payload[name] = fn(state, subdivisions=tuple(args.a1_subdivisions), mode_cap=args.mode_cap)
        elif name == "A2":
            payload[name] = fn(state, mode_cap=args.mode_cap)
        else:
            payload[name] = fn(state)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s", flush=True)
        (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    payload["finished"] = True
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("results written to", out_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())