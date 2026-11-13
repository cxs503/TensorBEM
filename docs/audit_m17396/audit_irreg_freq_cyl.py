#!/usr/bin/env python3
"""Board audit m17396 — independent re-measurement of CASE-BEM-IRREG-FREQ-CYL (mission m17374).

This script deliberately does NOT import the mission's own scripts.  It re-derives every
number from the library API in this worktree and from an external 30-digit reference
(mpmath), so that agreement cannot come from re-reading the mission's artifacts.

Checks
------
C1  The three Bessel literals inside ``tensorbem.irregular_frequency`` vs mpmath j_{m,1}
    computed here to 30 digits; plus the first/second zeros of J_0 (coverage check).
C2  The mission's headline metric is an identity: with nu_pred = j / R_wp(mesh) and
    nu_ref = j / a(true radius), the relative error equals (a / R_wp) - 1 for every mode.
    We verify that numerically (max deviation of err_m / ident - 1 over m).
C3  Waterplane equivalent radius vs the inscribed-regular-polygon model
    R_eq/a = sqrt(A_polygon / (pi a^2)) ~ 1 - pi^2/(3 N^2); the panel error must follow it.
C4  Route B (free-surface-consistent) eigenvalue tan(k_z T) = k_z / nu for the same drafts:
    the gap between the shipped predictor (Route A, rigid lid) and Route B.
C5  Route C (the solver's own BEM): reproduce the observed spurious band (conditioning /
    damping blow-up) on two meshes and compare it with the predictor.
C6  Mode coverage of the public API: n_modes > 3 cannot reach j_{0,2}.

Outputs (written next to this file)
----------------------------------
audit_metrics.json, fig_geom_scaling.png, fig_route_abc.png
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import mpmath as mp  # noqa: E402

from tensorbem import irregular_frequency as irr  # noqa: E402
from tensorbem.diffraction_drift_maturity import vertical_cylinder_hull_mesh  # noqa: E402
from tensorbem.benchmarks.irregular_frequency_cylinder_exact import (  # noqa: E402
    exact_cylinder_modes,
    observed_spurious_band,
)

OUT = Path(__file__).resolve().parent
G = 9.80665
mp.mp.dps = 30


def bessel_zeros_mpmath(order: int, count: int = 2) -> list[float]:
    """First ``count`` positive zeros of J_order at 30-digit precision (independent route)."""
    zeros = []
    prev = mp.mpf("0.1")
    for k in range(1, count + 1):
        guess = mp.mpf(str((k + order / 2.0 - 0.25) * math.pi))
        z = mp.findroot(lambda x: mp.besselj(order, x), guess)
        zeros.append(float(z))
        prev = z
    return zeros


def waterplane_radius(mesh) -> float:
    """Recompute the equivalent waterplane radius here (shoelace over waterline vertices)."""
    edges = mesh.waterline_edges
    idx = np.unique(edges.ravel())
    xy = mesh.vertices[idx][:, :2]
    order = np.arctan2(xy[:, 1], xy[:, 0])
    xy = xy[np.argsort(order)]  # canonical CCW ordering, independent of mesh ordering
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))
    return math.sqrt(area / math.pi), area


def ladder() -> list[dict]:
    """Panel ladder for the anchor geometry (R=1, T/R=0.5) and the R=2, T/R=1 variant."""
    configs = [("R1_T0.5", 1.0, 0.5), ("R2_T1.0", 2.0, 1.0)]
    rows = []
    for label, radius, ratio in configs:
        for n_theta, n_levels in ((16, 3), (32, 5), (64, 5), (128, 8)):
            mesh = vertical_cylinder_hull_mesh(radius, radius * ratio,
                                               n_theta=n_theta, n_levels=n_levels)
            pred = irr.predict_irregular_frequencies(mesh, gravity=G, n_modes=3)
            r_audit, area_audit = waterplane_radius(mesh)
            zeros = [irr._BESSEL_J01, irr._BESSEL_J11, irr._BESSEL_J21]
            nu_ref = [z / radius for z in zeros]
            err = [(p - r) / r for p, r in zip(pred.predicted_nu, nu_ref)]
            ident = radius / pred.waterplane_radius - 1.0
            rows.append(
                {
                    "label": label,
                    "radius_m": radius,
                    "draft_ratio": ratio,
                    "n_theta": n_theta,
                    "n_levels": n_levels,
                    "panels": int(mesh.faces.shape[0]),
                    "r_wp_library_m": float(pred.waterplane_radius),
                    "r_wp_audit_m": r_audit,
                    "r_wp_rel_diff": abs(r_audit - pred.waterplane_radius) / r_audit,
                    "area_rel_err": abs(area_audit - math.pi * radius ** 2) / (math.pi * radius ** 2),
                    "polygon_model_err": math.pi ** 2 / (3.0 * n_theta ** 2),
                    "nu_pred": [float(v) for v in pred.predicted_nu],
                    "nu_ref_route_A": nu_ref,
                    "nu_err": [float(e) for e in err],
                    "max_nu_err": float(max(abs(e) for e in err)),
                    "identity": float(ident),
                    "identity_residual": float(max(abs(e / ident - 1.0) for e in err)),
                }
            )
    return rows


def main() -> int:
    t0 = time.time()
    metrics: dict = {"mission": 17374, "audit": 17396, "case": "CASE-BEM-IRREG-FREQ-CYL"}

    # ── C1 constants vs 30-digit reference ────────────────────────────────────
    ref = {m: bessel_zeros_mpmath(m, 2) for m in (0, 1, 2)}
    literals = {"j01": irr._BESSEL_J01, "j11": irr._BESSEL_J11, "j21": irr._BESSEL_J21}
    const_cmp = {
        "j01": {"literal": literals["j01"], "mpmath": ref[0][0],
                "abs_diff": abs(literals["j01"] - ref[0][0])},
        "j11": {"literal": literals["j11"], "mpmath": ref[1][0],
                "abs_diff": abs(literals["j11"] - ref[1][0])},
        "j21": {"literal": literals["j21"], "mpmath": ref[2][0],
                "abs_diff": abs(literals["j21"] - ref[2][0])},
        "j02_second_zero_J0": {"value": ref[0][1], "reachable_by_api": False},
    }
    metrics["C1_constants"] = {
        "table": const_cmp,
        "max_abs_diff": max(v["abs_diff"] for v in const_cmp.values() if "abs_diff" in v),
        "where_declared": "src/tensorbem/irregular_frequency.py:82-86 (module literals)",
    }

    # ── C6 API mode coverage ──────────────────────────────────────────────────
    mesh = vertical_cylinder_hull_mesh(1.0, 0.5, n_theta=64, n_levels=5)
    wide = irr.predict_irregular_frequencies(mesh, gravity=G, n_modes=6)
    metrics["C6_modes"] = {
        "requested": 6,
        "returned": len(wide.mode_labels),
        "labels": list(wide.mode_labels),
        "j02_rad_per_m": ref[0][1] / 1.0,
        "note": "public API caps at the three first-order zeros; j_{0,2}=5.520078... unreachable",
    }

    # ── C2/C3 panel ladder ────────────────────────────────────────────────────
    rows = ladder()
    metrics["C2C3_ladder"] = rows
    metrics["C2_error_is_geometry_identity"] = {
        "max_identity_residual": max(r["identity_residual"] for r in rows),
        "claim": "err_nu(m) == a/R_wp - 1 for all m; the j's cancel exactly",
    }

    # ── C4 Route A vs Route B ─────────────────────────────────────────────────
    route_b = []
    for ratio in (0.5, 1.0):
        for mode in exact_cylinder_modes(1.0, ratio, angular_orders=(0, 1, 2)):
            route_b.append(
                {
                    "draft_ratio": ratio,
                    "m": mode.angular_order,
                    "nu_route_A": mode.radial_wavenumber,
                    "nu_route_B": mode.nu,
                    "gap_pct": 100.0 * (mode.radial_wavenumber - mode.nu) / mode.nu,
                }
            )
    metrics["C4_route_A_vs_B"] = route_b

    # ── C5 Route C: reproduce the solver's own spurious band (T/R = 1) ────────
    route_c = []
    for label, n_theta, n_levels in (("112_panels", 16, 3), ("264_panels", 24, 5)):
        band = observed_spurious_band(1.0, 1.0, mode=2, n_theta=n_theta, n_levels=n_levels)
        nu_a = irr._BESSEL_J01 / 1.0
        route_c.append(
            {
                "mesh": label,
                "panels": band["mesh"]["panels"],
                "nu_observed": band["nu_observed"],
                "nu_route_A_predicted": nu_a,
                "err_vs_route_A_pct": 100.0 * (band["nu_observed"] - nu_a) / nu_a,
                "passivity_violation": band["passivity_violation"],
                "min_damping": band["min_damping"],
                "max_condition_number": band["max_condition_number"],
                "samples": [{k: r[k] for k in ("nu", "damping", "condition_number")}
                            for r in band["samples"]],
            }
        )
    metrics["C5_route_C"] = route_c

    metrics["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "audit_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[C1] max |literal - mpmath(tabulated zero)| = {metrics['C1_constants']['max_abs_diff']:.3e}")
    print(f"[C2] max identity residual over modes/configs = "
          f"{metrics['C2_error_is_geometry_identity']['max_identity_residual']:.3e}")
    for r in rows:
        print(f"[C3] {r['label']:8s} n_theta={r['n_theta']:3d} panels={r['panels']:5d} "
              f"max_nu_err={100*r['max_nu_err']:.4f}% polygon_model={100*r['polygon_model_err']:.4f}%")
    for m in route_b:
        print(f"[C4] T/R={m['draft_ratio']} m={m['m']} A={m['nu_route_A']:.4f} "
              f"B={m['nu_route_B']:.4f} gap={m['gap_pct']:+.2f}%")
    for c in route_c:
        print(f"[C5] {c['mesh']:11s} nu_obs={c['nu_observed']:.4f} vs A={c['nu_route_A_predicted']:.4f} "
              f"({c['err_vs_route_A_pct']:+.2f}%) passivity_violation={c['passivity_violation']}")
    print(f"[done] runtime {metrics['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())