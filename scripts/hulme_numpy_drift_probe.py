#!/usr/bin/env python3
"""Stage-by-stage fingerprint of one ``BEM-HULME-HEMISPHERE`` cell.

Motivation
----------
Dev report m21674 recorded a blocking reproducibility failure: the same commit
and the same mesh (``condition_number`` bit-identical) produced
``B33 = 0.10425`` under numpy 2.4.6 on the control plane and ``B33 = 0.05079``
under numpy 2.0.1 on a cluster node -- a factor 2.05.  A residual/condition
audit cannot separate the two candidate causes:

* **(A) assembly drift** -- the Green-function / Bessel / quadrature path
  evaluates differently under the two numpy builds, so the node solved a
  *different* system;
* **(B) solver drift** -- the assembled system is the same, but the complex
  least-squares solve (or the linear functional that turns the potentials into
  ``A33/B33``) is ill-conditioned enough that two backward-stable drivers
  disagree.

This probe answers that question with bitwise fingerprints of every stage, so
the comparison does not have to trust any single number:

1. geometry (vertices, panel centroids, areas, volume, waterplane);
2. kernel tables used by the free-surface quadrature (Bessel/Struve/Ei,
   Gauss-Legendre and Gauss-Laguerre nodes, and ``wave_function_3d`` on a
   fixed ``(X, Y)`` grid);
3. the assembled operators (single layer, chief single layer, system matrix);
4. the solution and its audit (residual, stationarity, driver, transfer
   coefficient of ``A33``/``B33``), plus the same coefficients obtained from a
   *real-embedded* driver (``dgelss`` instead of ``zgelss``) and under tiny
   right-hand-side perturbations -- a local sensitivity estimate that needs no
   second machine.

Every fingerprint is SHA-256 of ``array.tobytes()`` so two machines can be
compared byte for byte.  The 1e-13 RHS-perturbation spread is the quantity that
separates (A) from (B): if the spread is orders of magnitude below the observed
cross-machine difference, the difference cannot be a rounding effect inside the
solve and the drift must already be in the assembly.

Usage
-----
    python scripts/hulme_numpy_drift_probe.py --subdivisions 3 4 \
        --out benchmarks/verified/casepool_bem_hulme_hemisphere/\
evidence/numpy_drift_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import (
    bessel_j01_y01,
    exponential_integral,
    struve_h01,
)
from tensorbem.free_surface_solver3d import (
    _resolve_panel_areas,
    _solve_system,
    assemble_free_surface_operators_3d,
    rigid_body_normals_3d,
)
from tensorbem.free_surface3d import wave_function_3d
from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.quadrature import gauss_legendre

RADIUS = 1.0
DENSITY = 1025.0
MODES = np.array([0, 2])
WAVE_X = np.array([1e-3, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 40.0])
WAVE_Y = np.array([1e-3, 0.05, 0.5, 1.5, 3.0, 4.4, 4.6, 8.0, 16.0, 30.0])


def _sha(array: Any) -> str:
    """SHA-256 of the raw bytes of an array (dtype and shape folded in)."""
    arr = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}{arr.shape}".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def kernel_fingerprint() -> dict[str, Any]:
    """Kernel/quadrature fingerprints that do not depend on the mesh."""
    x = np.linspace(1e-3, 60.0, 601)
    j0, j1, y0, y1 = bessel_j01_y01(x)
    h0, h1 = struve_h01(x)
    ei = np.array([exponential_integral(float(v)) for v in np.linspace(1e-3, 30.0, 61)])
    xg, wg = gauss_legendre(40)
    from numpy.polynomial.laguerre import laggauss

    xl, wl = laggauss(40)
    xx, yy = np.meshgrid(WAVE_X, WAVE_Y)
    f_val, fx_val, fy_val = wave_function_3d(xx, yy)
    return {
        "bessel_j0_sha": _sha(j0),
        "bessel_j1_sha": _sha(j1),
        "bessel_y0_sha": _sha(y0),
        "bessel_y1_sha": _sha(y1),
        "struve_h0_sha": _sha(h0),
        "struve_h1_sha": _sha(h1),
        "exponential_integral_sha": _sha(ei),
        "gauss_legendre_nodes_sha": _sha(xg),
        "gauss_legendre_weights_sha": _sha(wg),
        "laguerre_nodes_sha": _sha(xl),
        "laguerre_weights_sha": _sha(wl),
        "wave_function_F_sha": _sha(f_val),
        "wave_function_Fx_sha": _sha(fx_val),
        "wave_function_Fy_sha": _sha(fy_val),
        "wave_function_F_probe": [
            float(f_val[i, j]) for i, j in ((3, 3), (7, 6), (9, 9))
        ],
        "bessel_j0_probe": [float(v) for v in j0[[10, 300, 600]]],
        "laguerre_node_probe": [float(v) for v in xl[[0, 20, 39]]],
    }


def _coefficients(solution: np.ndarray, weighted: np.ndarray) -> tuple[float, float]:
    """``(A33, B33)`` up to the (irrelevant) density factor."""
    impedance = -(weighted.T @ solution)
    return float(impedance.real[1, 1]), float(impedance.imag[1, 1] / math.sqrt(2.0))


def cell_fingerprint(ka: float, subdivisions: int) -> dict[str, Any]:
    nu = ka / RADIUS
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    vertices = np.asarray(mesh.vertices, dtype=float)
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    n6 = rigid_body_normals_3d(mesh)
    n_modes = n6[:, MODES]
    q_ext = (-n_modes).astype(np.complex128)
    weighted = _resolve_panel_areas(mesh, None)[:, None] * n_modes

    sol, res, method, stat = _solve_system(ops, q_ext)
    a33, b33 = _coefficients(sol, weighted)

    # Independent driver: embed the complex system in a real one so LAPACK uses
    # dgelss instead of zgelss -- a *different* code path inside the same
    # interpreter.  Agreement to rounding means the complex solve is not the
    # weak link; disagreement is driver drift reproduced on one machine.
    mat = ops.system_matrix
    rhs = np.vstack(
        (ops.single_layer @ q_ext, ops.chief_single_layer @ q_ext)
        if ops.chief_points.shape[0] > 0
        else (ops.single_layer @ q_ext,)
    )
    real_mat = np.block([[mat.real, -mat.imag], [mat.imag, mat.real]])
    real_rhs = np.concatenate([rhs.real, rhs.imag])
    xr, *_ = np.linalg.lstsq(real_mat, real_rhs, rcond=None)
    sol_real = xr[: mat.shape[1]] + 1j * xr[mat.shape[1] :]
    a33_real, b33_real = _coefficients(sol_real, weighted)
    resid_real = float(np.linalg.norm(mat @ sol_real - rhs))
    stat_real = float(
        np.linalg.norm(mat.conj().T @ (mat @ sol_real - rhs))
        / (np.linalg.norm(mat) * max(resid_real, 1e-300))
    )

    # Local RHS-perturbation sensitivity at 1e-13 relative -- the scale of the
    # rounding a different LAPACK build would introduce.
    rng = np.random.default_rng(20261105)
    pert: list[dict[str, float]] = []
    for _ in range(4):
        delta = rng.standard_normal(rhs.shape) + 1j * rng.standard_normal(rhs.shape)
        delta *= 1.0e-13 * np.linalg.norm(rhs) / np.linalg.norm(delta)
        sol_p, *_ = np.linalg.lstsq(mat, rhs + delta, rcond=None)
        a33_p, b33_p = _coefficients(sol_p, weighted)
        pert.append({"A33": a33_p, "B33": b33_p})
    spread_a = (max(p["A33"] for p in pert) - min(p["A33"] for p in pert)) / abs(a33)
    spread_b = (max(p["B33"] for p in pert) - min(p["B33"] for p in pert)) / abs(b33)

    return {
        "ka": ka,
        "subdivisions": subdivisions,
        "panels": int(mesh.panel_count),
        "geom": {
            "vertices_sha": _sha(vertices),
            "volume": float(mesh.volume),
            "waterplane_area": float(mesh.waterplane_area),
        },
        "operators": {
            "condition_number": float(ops.condition_number),
            "cond_2norm": float(np.linalg.cond(mat)),
            "single_layer_sha": _sha(ops.single_layer),
            "chief_single_layer_sha": _sha(ops.chief_single_layer),
            "system_matrix_sha": _sha(mat),
            "q_ext_sha": _sha(q_ext),
            "rhs_sha": _sha(rhs),
        },
        "solve": {
            "method": str(method),
            "relative_residual": float(res),
            "stationarity": float(stat),
            "A33": a33,
            "B33": b33,
            "potentials_sha": _sha(sol),
        },
        "real_driver": {
            "stationarity": stat_real,
            "A33": a33_real,
            "B33": b33_real,
            "delta_A33": abs(a33_real - a33) / abs(a33),
            "delta_B33": abs(b33_real - b33) / abs(b33),
        },
        "rhs_perturbation": {
            "scale": 1.0e-13,
            "spread_A33": spread_a,
            "spread_B33": spread_b,
            "samples": pert,
        },
    }


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "host": platform.node(),
    }
    try:  # the code revision is part of the comparability contract
        import subprocess

        env["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or None
        env["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except Exception:  # pragma: no cover - git is not always present
        env["commit"] = None
    try:  # scipy is optional for this repo; record it when present
        import scipy

        env["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover - scipy is not a dependency
        env["scipy"] = None
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ka", type=float, default=2.0)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {
        "purpose": (
            "isolate the numpy-version coefficient drift (m21674 / "
            "BUG-BEM-NUMPY-VERSION-DRIFT) into assembly vs solve by bitwise "
            "stage fingerprints"
        ),
        "environment": environment(),
        "kernel": kernel_fingerprint(),
        "cells": [cell_fingerprint(args.ka, s) for s in args.subdivisions],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())