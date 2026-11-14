#!/usr/bin/env python3
"""Stage-wise numeric fingerprint of one Hulme-hemisphere BEM cell.

Why this exists
---------------
Two runs of the *same* function, at the *same* mesh, at ``ka = 2`` produced two
different answers:

===========================  ============  ============  ==========  ==========
artifact (ka=2, s=4, 1024 p) A11           B11           A33         B33
===========================  ============  ============  ==========  ==========
``results.json`` (ctrl plane) 0.249848     0.340432      0.386337    0.104249
``inhouse_ladder_ka2_rung4_``  0.235658     0.337733      0.366607    0.050792
``node.json`` (cluster node)
reference (Hulme series)      0.249279     0.342410      0.388378    0.103065
===========================  ============  ============  ==========  ==========

Both rows report the *identical* mesh (``mesh_volume = 2.0821017981243184``)
and the *identical* reported condition number (to 1e-15), so the assembled
operator is the same object; yet the heave damping differs by 2.05x.  A
disagreement of that size cannot be attributed to the 5% gate or to mesh
resolution -- it has to be localised to one stage of the pipeline:

    mesh  ->  operator assembly  ->  linear solve  ->  force post-processing.

This script records a digest of every one of those stages plus the full
floating-point environment, so two machines can be diffed stage by stage
instead of by arguing about the final number::

    PYTHONPATH=src python scripts/hulme_cell_fingerprint.py \
        --ka 2.0 --subdivisions 4 --out fingerprints/ctrl_ka2_s4.json

    PYTHONPATH=src python scripts/hulme_cell_fingerprint.py \
        --compare fingerprints/node_ka2_s4.json fingerprints/ctrl_ka2_s4.json

The comparison prints the first stage whose content differs beyond roundoff;
every later stage is then downstream of a known divergence, not evidence.

Cost: ``subdivisions=2`` ~0.3 s, ``3`` ~2 s, ``4`` ~25 s (one BLAS thread).
Send ``subdivisions=5`` through the compute gate.

Fingerprints are diagnostics, not evidence of correctness: agreement between
two fingerprints proves the two runs are the same computation, not that the
computation is right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.benchmarks import hulme_hemisphere_case as case
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Rounding used before hashing. 10 decimal digits on a quantity of order 1 is
# ~1e-10 relative -- three orders of magnitude tighter than the 5% physics gate
# and two orders looser than float64 roundoff, so a hash mismatch means a real
# difference in the assembled object, not a last-bit difference.
_HASH_DECIMALS = 10

# Health threshold for the linear solve. ``lstsq`` on the CHIEF-augmented
# system leaves a residual of ~2e-4 at ka=2 / 1024 panels on a healthy
# BLAS/LAPACK (the CHIEF rows are physically inconsistent at that order), while
# a node whose complex least-squares path is broken was measured at 4.9e-1 with
# the *same* assembled matrix (see docs/bug_bem_numpy_version_drift.md). 1e-2
# sits two orders above the healthy value and two orders below the broken one.
SOLVE_RESIDUAL_MAX = 1.0e-2


# --- digests ---------------------------------------------------------------
def _array_digest(arr: np.ndarray) -> dict[str, Any]:
    """Shape/dtype/norms/rounded hash of one array."""
    a = np.asarray(arr)
    digest: dict[str, Any] = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "n_elements": int(a.size),
    }
    if a.size:
        absa = np.abs(a)
        digest["max_abs"] = float(absa.max())
        digest["sum_abs"] = float(absa.sum())
        digest["frobenius"] = float(np.linalg.norm(a))
        digest["mean"] = float(np.mean(a.real))
        digest["mean_imag"] = float(np.mean(a.imag)) if np.iscomplexobj(a) else 0.0
        rounded = np.round(a, _HASH_DECIMALS)
        digest["sha256_rounded"] = hashlib.sha256(
            np.ascontiguousarray(rounded).tobytes()
        ).hexdigest()
    else:
        for key in ("max_abs", "sum_abs", "frobenius", "mean", "mean_imag"):
            digest[key] = 0.0
        digest["sha256_rounded"] = hashlib.sha256(b"").hexdigest()
    return digest


def _vector_digest(name: str, values: Any) -> dict[str, Any]:
    return {"name": name, "digest": _array_digest(np.asarray(values))}


# --- provenance ------------------------------------------------------------
def _environment() -> dict[str, Any]:
    config = np.show_config(mode="dicts") if hasattr(np, "show_config") else {}
    blas = {}
    try:  # numpy >= 2 reports build dependencies under build / host sections
        for section in ("build", "host"):
            deps = config.get(section, {}).get("dependencies", {})
            for key in ("blas", "lapack"):
                if key in deps:
                    blas[f"{section}_{key}"] = {
                        k: deps[key].get(k) for k in ("name", "version", "pc file")
                    }
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        blas = {"error": "numpy build config not machine readable"}
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
        "blas_lapack": blas,
    }


def _provenance() -> dict[str, Any]:
    """Where the imported TensorBEM code actually came from (not where it should)."""
    import tensorbem

    files: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("tensorbem"):
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        files[name] = path
    solver_path = files.get("tensorbem.free_surface_solver3d", "")
    digest = None
    if solver_path and Path(solver_path).exists():
        digest = hashlib.sha256(Path(solver_path).read_bytes()).hexdigest()
    return {
        "tensorbem_file": getattr(tensorbem, "__file__", None),
        "tensorbem_version": getattr(tensorbem, "__version__", None),
        "module_files": files,
        "free_surface_solver3d_sha256": digest,
    }


# --- stages ----------------------------------------------------------------
def fingerprint(ka: float, subdivisions: int, *, chief: bool = True) -> dict[str, Any]:
    """Run one cell and digest every stage of it."""
    payload: dict[str, Any] = {
        "ka": ka,
        "subdivisions": subdivisions,
        "chief": chief,
        "environment": _environment(),
        "provenance": _provenance(),
    }

    mesh = hemisphere_hull_mesh(subdivisions, radius=case.RADIUS)
    nu = ka / case.RADIUS
    payload["mesh"] = {
        "panel_count": int(mesh.panel_count),
        "volume": float(mesh.volume),
        "waterplane_area": float(mesh.waterplane_area),
        "centroids": _array_digest(np.asarray(mesh.centroids)),
        "areas": _array_digest(np.asarray(mesh.areas)),
        "vertices": _array_digest(np.asarray(mesh.vertices)),
    }

    ops = assemble_free_surface_operators_3d(
        mesh, nu, chief_points="auto" if chief else None
    )
    system = np.asarray(ops.system_matrix)
    sv = np.linalg.svd(system, compute_uv=False)
    sv_max = float(sv[0]) if sv.size else 0.0
    eps = float(np.finfo(np.float64).eps)
    cutoff = eps * max(system.shape) * sv_max
    payload["operators"] = {
        "wavenumber": float(ops.wavenumber),
        "reported_condition_number": float(ops.condition_number),
        "single_layer": _array_digest(np.asarray(ops.single_layer)),
        "double_layer": _array_digest(np.asarray(ops.double_layer)),
        "chief_points": _array_digest(np.asarray(ops.chief_points)),
        "chief_single_layer": _array_digest(np.asarray(ops.chief_single_layer)),
        "chief_double_layer": _array_digest(np.asarray(ops.chief_double_layer)),
        "system_matrix": _array_digest(system),
    }
    payload["spectrum"] = {
        "n_rows": int(system.shape[0]),
        "n_cols": int(system.shape[1]),
        "sv_max": sv_max,
        "sv_min": float(sv[-1]) if sv.size else 0.0,
        "sv_smallest_5": [float(v) for v in sv[-5:]],
        "sv_largest_5": [float(v) for v in sv[:5]],
        # lstsq(rcond=None) truncates below eps*max(shape)*sv_max: this is the
        # rank the solve *actually* uses, which the reported condition number of
        # the body-only matrix does not describe.
        "lstsq_cutoff": cutoff,
        "effective_rank": int(np.sum(sv > cutoff)),
        "n_cols_rank_deficient": int(system.shape[1] - np.sum(sv > cutoff)),
        "cond_of_solved_system": float(sv_max / sv[-1]) if sv.size and sv[-1] > 0 else math.inf,
    }

    rad = solve_radiation_3d(
        mesh,
        nu,
        modes=(0, 2),
        density=case.DENSITY,
        gravity=case.GRAVITY,
        operators=ops,
    )
    payload["solve_radiation"] = {
        "relative_residual": float(rad.relative_residual),
        "reported_condition_number": float(rad.condition_number),
        "n_chief_points": int(np.asarray(rad.chief_points_used).shape[0]),
        "modes": _vector_digest("modes", rad.modes),
        "potential_surge": _array_digest(np.asarray(rad.potentials)[0]),
        "potential_heave": _array_digest(np.asarray(rad.potentials)[1]),
        "impulse_reactance": _array_digest(np.asarray(rad.potentials)),
    }

    diff = solve_diffraction_3d(
        mesh,
        nu,
        amplitude=case.AMPLITUDE,
        wave_angle=0.0,
        density=case.DENSITY,
        gravity=case.GRAVITY,
        operators=ops,
    )

    vol = (2.0 / 3.0) * math.pi * case.RADIUS ** 3
    norm_a = case.DENSITY * vol
    norm_b = case.DENSITY * math.sqrt(case.GRAVITY * nu) * vol
    scale_f = case.DENSITY * case.GRAVITY * case.AMPLITUDE * case.RADIUS ** 2
    fx = complex(diff.excitation_force[0]) / scale_f
    fz = complex(diff.excitation_force[2]) / scale_f
    payload["solve_health"] = {
        "relative_residual": float(rad.relative_residual),
        "residual_max_accepted": SOLVE_RESIDUAL_MAX,
        "healthy": bool(rad.relative_residual <= SOLVE_RESIDUAL_MAX),
        "note": (
            "a residual above the threshold means the environment did not solve "
            "the assembled system; no coefficient from this run is usable"
        ),
    }
    payload["results"] = {
        "A11_dimless": float(rad.added_mass[0, 0]) / norm_a,
        "A33_dimless": float(rad.added_mass[1, 1]) / norm_a,
        "B11_dimless": float(rad.damping[0, 0]) / norm_b,
        "B33_dimless": float(rad.damping[1, 1]) / norm_b,
        "Fx_dimless": abs(fx),
        "Fz_dimless": abs(fz),
        "phase_Fx_deg": math.degrees(math.atan2(fx.imag, fx.real)),
        "phase_Fz_deg": math.degrees(math.atan2(fz.imag, fz.real)),
        "added_mass": _array_digest(np.asarray(rad.added_mass)),
        "damping": _array_digest(np.asarray(rad.damping)),
        "excitation_force": _array_digest(np.asarray(diff.excitation_force)),
        "reference": case.reference(ka),
    }
    return payload


# --- comparison ------------------------------------------------------------
_SCALAR_STAGES: dict[str, tuple[str, ...]] = {
    "mesh": ("volume", "waterplane_area"),
    "operators": (
        "single_layer.max_abs",
        "double_layer.max_abs",
        "chief_single_layer.max_abs",
        "system_matrix.max_abs",
    ),
    "spectrum": ("sv_max", "sv_min", "effective_rank", "cond_of_solved_system"),
    "solve_radiation": ("relative_residual",),
    "results": (
        "A11_dimless",
        "A33_dimless",
        "B11_dimless",
        "B33_dimless",
        "Fx_dimless",
        "Fz_dimless",
    ),
}
_HASH_STAGES: dict[str, tuple[str, ...]] = {
    "mesh": ("centroids", "areas", "vertices"),
    "operators": (
        "single_layer",
        "double_layer",
        "chief_points",
        "chief_single_layer",
        "system_matrix",
    ),
    "solve_radiation": ("potential_surge", "potential_heave"),
    "results": ("added_mass", "damping", "excitation_force"),
}


def _dig(payload: dict[str, Any], dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def compare(a: dict[str, Any], b: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Stage-by-stage diff of two fingerprints. Returns (rows, first_divergence)."""
    rows: list[dict[str, Any]] = []
    first = True
    for stage, scalars in _SCALAR_STAGES.items():
        for key in scalars:
            va, vb = _dig(a, f"{stage}.{key}"), _dig(b, f"{stage}.{key}")
            if va is None or vb is None:
                rows.append({"stage": stage, "key": key, "status": "missing"})
                continue
            if isinstance(va, int) and isinstance(vb, int):
                same = va == vb
                rows.append(
                    {
                        "stage": stage,
                        "key": key,
                        "a": va,
                        "b": vb,
                        "rel": 0.0 if same else math.inf,
                        "status": "same" if same else "DIFFERS",
                        "downstream_of_divergence": not first,
                    }
                )
                first = first and same
                continue
            denom = max(abs(va), abs(vb), 1e-300)
            rel = abs(va - vb) / denom
            same = rel <= 1e-10
            rows.append(
                {
                    "stage": stage,
                    "key": key,
                    "a": va,
                    "b": vb,
                    "rel": rel,
                    "status": "same" if same else "DIFFERS",
                    "downstream_of_divergence": not first,
                }
            )
            first = first and same
        for key in _HASH_STAGES.get(stage, ()):
            va, vb = _dig(a, f"{stage}.{key}.sha256_rounded"), _dig(b, f"{stage}.{key}.sha256_rounded")
            same = va == vb and va is not None
            rows.append(
                {
                    "stage": stage,
                    "key": f"{key}::sha256",
                    "a": (va or "")[:16],
                    "b": (vb or "")[:16],
                    "rel": 0.0 if same else math.inf,
                    "status": "same" if same else "DIFFERS",
                    "downstream_of_divergence": not first,
                }
            )
            first = first and same
    return rows, first


def _print_comparison(rows: list[dict[str, Any]], label_a: str, label_b: str) -> None:
    print(f"{'stage':<18}{'key':<34}{'A':>14}{'B':>14}{'rel':>11}  status")
    print("-" * 104)
    for row in rows:
        if row["status"] == "missing":
            print(f"{row['stage']:<18}{row['key']:<34}{'-':>14}{'-':>14}{'-':>11}  missing")
            continue
        a, b = row["a"], row["b"]
        a_s = f"{a:.6g}" if isinstance(a, float) else str(a)
        b_s = f"{b:.6g}" if isinstance(b, float) else str(b)
        rel_s = "inf" if row["rel"] == math.inf else f"{row['rel']:.3e}"
        mark = row["status"]
        if mark == "DIFFERS" and row.get("downstream_of_divergence"):
            mark += " (downstream)"
        print(f"{row['stage']:<18}{row['key']:<34}{a_s:>14}{b_s:>14}{rel_s:>11}  {mark}")
    print("-" * 104)
    print(f"A = {label_a}\nB = {label_b}")
    diverging = [r for r in rows if r["status"] == "DIFFERS" and not r.get("downstream_of_divergence")]
    if diverging:
        stage = diverging[0]["stage"]
        print(f"\nFIRST DIVERGENCE: stage '{stage}' ({len(diverging)} field(s) in that stage)")
        print("Every later stage is downstream of it; only this stage is causal evidence.")
    else:
        print("\nNO DIVERGENCE: the two runs are the same computation to 1e-10.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ka", type=float, default=2.0)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--no-chief", action="store_true", help="disable interior CHIEF rows")
    parser.add_argument("--out", type=Path, help="write the fingerprint JSON here")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A_JSON", "B_JSON"),
        help="compare two fingerprints instead of computing one",
    )
    args = parser.parse_args(argv)

    if args.compare:
        path_a, path_b = (Path(p) for p in args.compare)
        a = json.loads(path_a.read_text())
        b = json.loads(path_b.read_text())
        rows, same = compare(a, b)
        _print_comparison(rows, str(path_a), str(path_b))
        print(
            "\nenvironment A: numpy {} on {}\nenvironment B: numpy {} on {}".format(
                a.get("environment", {}).get("numpy"),
                a.get("environment", {}).get("host"),
                b.get("environment", {}).get("numpy"),
                b.get("environment", {}).get("host"),
            )
        )
        return 0 if same else 1

    payload = fingerprint(args.ka, args.subdivisions, chief=not args.no_chief)
    out = args.out or Path(
        f"fingerprints/fingerprint_ka{args.ka:g}_s{args.subdivisions}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    res = payload["results"]
    spec = payload["spectrum"]
    health = payload["solve_health"]
    if not health["healthy"]:
        print(
            "WARNING: relative_residual {:.3e} > {:.0e} -- this environment did not "
            "solve the system; the coefficients below are not usable.".format(
                health["relative_residual"], SOLVE_RESIDUAL_MAX
            )
        )
    print(
        "ka={ka:g} s={s} panels={p} numpy={npv} host={h}\n"
        "  A11={A11:.6f} B11={B11:.6f} A33={A33:.6f} B33={B33:.6f} "
        "Fx={Fx:.6f} Fz={Fz:.6f}\n"
        "  solved-system cond={cond:.3e} effective-rank={er}/{nc} (deficient {defi})\n"
        "  written to {out}".format(
            ka=args.ka,
            s=args.subdivisions,
            p=payload["mesh"]["panel_count"],
            npv=np.__version__,
            h=platform.node(),
            A11=res["A11_dimless"],
            B11=res["B11_dimless"],
            A33=res["A33_dimless"],
            B33=res["B33_dimless"],
            Fx=res["Fx_dimless"],
            Fz=res["Fz_dimless"],
            cond=spec["cond_of_solved_system"],
            er=spec["effective_rank"],
            nc=spec["n_cols"],
            defi=spec["n_cols_rank_deficient"],
            out=out,
        )
    )
    return 0 if health["healthy"] else 3


if __name__ == "__main__":
    raise SystemExit(main())