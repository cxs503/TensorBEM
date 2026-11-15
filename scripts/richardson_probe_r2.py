#!/usr/bin/env python3
"""Can the 6.7 % own-mesh deviation be extrapolated away?  A probe, not a claim.

The obvious repair for a residual that is "just discretisation error" is
Richardson extrapolation of both codes' mesh ladders to ``h -> 0`` and a
comparison of the limits.  With three or four levels, a constant observed order
and no noise, that is legitimate.  This probe asks whether the committed
ladders support it, and records the answer either way.

Model fitted per matrix entry, per code: ``f(h) = f0 + C h**p`` with
``h = 1/sqrt(n_faces)``, ``f0``/``C`` by linear least squares for each trial
``p`` on a fixed grid, ``p`` chosen to minimise the residual.  The fit has as
many free parameters as the hemisphere ladder has points (three), so ``p`` is
determined by round-off whenever the data are the least bit noisy.

The artifact reports, per case and quantity: the fraction of entries whose
fitted order sits on a search bound (non-identifiable), the median and spread
of the fitted orders, and the worst extrapolated-limit deviation -- which is
only meaningful if the orders are identified.

Usage::

    PYTHONPATH=src python scripts/richardson_probe_r2.py
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from tensorbem.benchmarks.capytaine_crosscheck_r2 import (
    ARTIFACT_ROOT as R2_ROOT,
    DOF_ORDER,
    NULL_FLOOR,
    NU_GRID,
    RESULT_FILE as R2_RESULT,
    invariant_null_dofs,
    physical_scale,
    reference_scale,
)

OUT = pathlib.Path("benchmarks/verified/capytaine_same_surface_crossval") / "richardson_probe.json"
P_BOUNDS = (0.2, 6.0)
P_GRID = np.linspace(P_BOUNDS[0], P_BOUNDS[1], 2901)


def fit_limit(hs: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Least-squares ``f0 + C h**p``; returns the limit, the order and the SSE.

    Vectorised over the order grid: the 2x2 normal equations of ``[1, h**p]``
    are solved in closed form for every trial order at once.
    """
    hs = np.asarray(hs, dtype=float)
    values = np.asarray(values, dtype=float)
    n = hs.size
    H = hs[None, :] ** P_GRID[:, None]              # (M, n)
    s0 = float(n)
    s1 = H.sum(axis=1)
    s2 = (H * H).sum(axis=1)
    t0 = float(values.sum())
    t1 = H @ values
    det = s0 * s2 - s1 * s1
    with np.errstate(divide="ignore", invalid="ignore"):
        f0 = np.where(np.abs(det) > 0.0, (s2 * t0 - s1 * t1) / det, np.inf)
        coef = np.where(np.abs(det) > 0.0, (s0 * t1 - s1 * t0) / det, np.inf)
    residual = f0[:, None] + coef[:, None] * H - values[None, :]
    sse = np.where(np.isfinite(residual).all(axis=1), (residual**2).sum(axis=1), np.inf)
    best = int(np.argmin(sse))
    return {"limit": float(f0[best]), "order": float(P_GRID[best]), "sse": float(sse[best])}


def main() -> int:
    root = pathlib.Path(R2_ROOT) / R2_RESULT
    study = json.loads(root.read_text())
    tensorbem = study["tensorbem"]
    capytaine = json.loads((pathlib.Path(R2_ROOT) / "capytaine_ref_r2.json").read_text())["cases"]
    rows = []
    for case, tb_case in tensorbem.items():
        cap_case = capytaine[case]
        null_index = {DOF_ORDER.index(d) for d in invariant_null_dofs(case)}
        h_tb = np.asarray([1.0 / math.sqrt(l["n_faces"]) for l in tb_case["levels"]])
        h_cap = np.asarray([1.0 / math.sqrt(l["n_faces"]) for l in cap_case["levels"]])
        for quantity, key in (("added_mass", "added_mass"), ("damping", "damping")):
            for nu in NU_GRID:
                nkey = f"{nu:.3f}"
                for i in range(6):
                    for j in range(6):
                        if i in null_index or j in null_index:
                            continue
                        v_tb = np.asarray([
                            l["matrices"][nkey][key][i][j] for l in tb_case["levels"]
                        ])
                        v_cap = np.asarray([
                            l["matrices"][nkey][key][i][j] for l in cap_case["levels"]
                        ])
                        cap_ref = np.asarray(cap_case["reference"][nkey][key], dtype=float)
                        scale, degenerate = reference_scale(cap_ref, i, j)
                        if (
                            not math.isfinite(scale)
                            or degenerate
                            or scale <= NULL_FLOOR * physical_scale(i, j)
                            or abs(cap_ref[i, j]) <= NULL_FLOOR * scale
                        ):
                            continue
                        fit_tb = fit_limit(h_tb, v_tb)
                        fit_cap = fit_limit(h_cap, v_cap)
                        rows.append(
                            {
                                "case": case,
                                "quantity": quantity,
                                "nu": nu,
                                "row": DOF_ORDER[i],
                                "column": DOF_ORDER[j],
                                "order_tensorbem": fit_tb["order"],
                                "order_capytaine": fit_cap["order"],
                                "limit_tensorbem": fit_tb["limit"],
                                "limit_capytaine": fit_cap["limit"],
                                "finest_relative": float(
                                    (v_tb[-1] - v_cap[-1]) / abs(v_cap[-1])
                                ),
                                "limit_relative": (
                                    float((fit_tb["limit"] - fit_cap["limit"])
                                          / abs(fit_cap["limit"]))
                                    if fit_cap["limit"] != 0.0
                                    else float("nan")
                                ),
                            }
                        )
    at_bound = [
        r for r in rows
        if r["order_tensorbem"] <= P_BOUNDS[0] + 1e-9
        or r["order_tensorbem"] >= P_BOUNDS[1] - 1e-9
        or r["order_capytaine"] <= P_BOUNDS[0] + 1e-9
        or r["order_capytaine"] >= P_BOUNDS[1] - 1e-9
    ]
    orders = np.asarray([r["order_tensorbem"] for r in rows] + [r["order_capytaine"] for r in rows])
    limits = np.asarray([abs(r["limit_relative"]) for r in rows if math.isfinite(r["limit_relative"])])
    payload = {
        "model": "f(h) = f0 + C h**p, h = 1/sqrt(n_faces), p by grid search",
        "p_bounds": list(P_BOUNDS),
        "n_entries": len(rows),
        "n_orders_on_bound": len(at_bound),
        "fraction_on_bound": len(at_bound) / len(rows) if rows else 0.0,
        "order_median": float(np.median(orders)),
        "order_p05": float(np.percentile(orders, 5)),
        "order_p95": float(np.percentile(orders, 95)),
        "worst_finest_relative": float(max(abs(r["finest_relative"]) for r in rows)),
        "worst_limit_relative": float(limits.max()) if limits.size else float("nan"),
        "limit_relative_p95": float(np.percentile(limits, 95)) if limits.size else float("nan"),
        "entries": rows,
        "verdict": (
            "not identifiable: the order hits a search bound on more than half the "
            "entries, so the extrapolated limit is a fit artefact, not a mesh limit"
            if len(at_bound) > 0.5 * len(rows)
            else "identifiable"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(
        f"entries {payload['n_entries']}, orders on bound "
        f"{payload['n_orders_on_bound']} ({payload['fraction_on_bound']:.0%}), "
        f"order median {payload['order_median']:.2f} "
        f"[{payload['order_p05']:.2f}, {payload['order_p95']:.2f}], "
        f"finest worst {payload['worst_finest_relative']:.2%}, "
        f"limit worst {payload['worst_limit_relative']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())