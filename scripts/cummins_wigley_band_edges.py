"""Measure the two band *edges* of the Wigley III radiation sweep (CASE-BEM-CUMMINS-WIGLEY).

Why this exists
---------------
Every Kramers-Kronig / Cummins number the case pool quotes leans on two
*modelling* statements about what the tabulated band does not see:

* below ``omega_min`` the production kernel either holds ``B`` constant
  (``low_frequency_model="zero"``) or integrates a dipole closure ``B ~ c w^3``;
* above ``omega_max`` :func:`tensorbem.causality.kkr_added_mass` drops the tail
  because "the damping vanishes exponentially fast" (its docstring).

Neither statement had ever been checked against panel data below 0.30 rad/s:
no run of this case had a node there.  The medium sweep solved for this round
does have them (0.10, 0.15, 0.20, 0.25 rad/s) plus 12.5 rad/s, so the two
exponents

    q = d ln B / d ln w   (w -> 0)        p = -d ln B / d ln w   (w -> max)

are *measured*, and the size of the omitted-band integrals is bounded with
them.  Nothing here redefines a gate: the numbers are diagnostics with the
same normalisation the gates use, and they are the bound the gate discussion
needs.

Usage::

    PYTHONPATH=src python3 scripts/cummins_wigley_band_edges.py \\
        --points-dir artifacts/case_bem_cummins_wigley_ext/points \\
        --level medium --out artifacts/case_bem_cummins_wigley_ext/band_edges.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kk_causality as case

#: Frequencies used for each edge fit (rad/s).  Fixed before looking at the
#: data: the four sub-0.30 nodes the extended grid adds, plus the lowest round-2
#: node; the top of the band for the high-frequency edge.
LOW_BAND = (0.10, 0.15, 0.20, 0.25, 0.30)
HIGH_BAND = (4.70, 12.50)


def _loglog_slope(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares exponent of ``y = C x**q`` (or ``y = C x**-p``)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y) & (y > 0.0)
    if mask.sum() < 2:
        return {"exponent": float("nan"), "coefficient": float("nan"),
                "r_squared": float("nan"), "points": int(mask.sum())}
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    return {
        "exponent": float(slope),
        "coefficient": float(math.exp(intercept)),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "points": int(mask.sum()),
    }


def edge_report(points: dict[float, dict], level: str) -> dict:
    grid = sorted(points)
    sweep = case.sweep_from_points(points, grid)
    low = [w for w in LOW_BAND if round(w, 9) in points]
    high = [w for w in HIGH_BAND if round(w, 9) in points]
    rows = []
    for dof, label in enumerate(case.DOF_LABELS):
        added = np.asarray(sweep["added_mass"])[:, dof, dof]
        damping = np.asarray(sweep["damping"])[:, dof, dof]
        lookup = {round(float(w), 9): i for i, w in enumerate(grid)}
        low_idx = [lookup[round(w, 9)] for w in low]
        high_idx = [lookup[round(w, 9)] for w in high]
        q_fit = _loglog_slope([grid[i] for i in low_idx],
                              [damping[i] for i in low_idx])
        p_fit = _loglog_slope([grid[i] for i in high_idx],
                              [damping[i] for i in high_idx])
        p_fit["exponent"] = -p_fit["exponent"]  # report as a decay exponent
        # ---- size of the band the case does not see -----------------------
        omega_min = grid[0]
        omega_max = grid[-1]
        # (a) sub-band contribution to K(0) = (2/pi) int_0^inf B dw, closed with
        #     the *measured* power law instead of the shipped constant model.
        k_band = float(np.sum(0.5 * (damping[:-1] + damping[1:]) * np.diff(grid)))
        q = q_fit["exponent"]
        c_low = q_fit["coefficient"]
        if np.isfinite(q) and q > -1.0:
            k_below_power = float(c_low * omega_min ** (q + 1.0) / (q + 1.0))
        else:
            k_below_power = float("nan")
        k_below_constant = float(damping[0] * omega_min)
        # (b) high-frequency tail of the *added-mass* Hilbert integral at the
        #     uppermost sample: int_{wmax}^inf B(s)/(s^2 - w^2) ds with
        #     B = c s^-p, evaluated in closed form for p > 1.
        p = p_fit["exponent"]
        c_high = p_fit["coefficient"]
        w_last = 0.5 * (grid[-2] + grid[-1])
        if np.isfinite(p) and p > 1.0 and np.isfinite(c_high):
            # int_{W}^inf s^-p/(s^2-w^2) ds <= int_W^inf s^-(p+2) ds / (1-(w/W)^2)
            tail_bound = (c_high * omega_max ** (-(p + 1.0)) / (p + 1.0)
                          / max(1.0 - (w_last / omega_max) ** 2, 1e-12))
        else:
            tail_bound = float("nan")
        a_inf, tail_c = _a_inf_from_top(grid, added, high_idx)
        rows.append({
            "dof": label,
            "low_nodes": low,
            "high_nodes": high,
            "B_at_low": [float(damping[i]) for i in low_idx],
            "B_at_high": [float(damping[i]) for i in high_idx],
            "low_exponent_q": q_fit,
            "high_decay_exponent_p": p_fit,
            "K_zero_band_integral": k_band,
            "K_zero_below_power_law": k_below_power,
            "K_zero_below_constant_model": k_below_constant,
            "K_zero_below_power_fraction": (
                float(k_below_power / (k_band + k_below_power))
                if np.isfinite(k_below_power) and k_band != 0.0 else float("nan")),
            "K_zero_below_constant_fraction": (
                float(k_below_constant / (k_band + k_below_constant))
                if k_band != 0.0 else float("nan")),
            "added_mass_inf": a_inf,
            "added_mass_tail_coefficient": tail_c,
            "KK_tail_bound_at_top_midpoint": tail_bound,
            "KK_scale_max_abs_A_minus_Ainf": float(
                np.max(np.abs(added - a_inf))) if np.isfinite(a_inf) else float("nan"),
        })
    return {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "level": level,
        "nodes": grid,
        "n_nodes": len(grid),
        "predeclared_low_band": list(LOW_BAND),
        "predeclared_high_band": list(HIGH_BAND),
        "rows": rows,
        "claims_under_test": {
            "shipped_low_frequency_closure": "B held constant below omega_min",
            "dipole_closure": "B ~ c w^3",
            "kkr_added_mass_tail_claim": "the damping vanishes exponentially fast "
                                         "above omega_max (docstring)",
        },
    }


def _a_inf_from_top(grid, added, high_idx):
    """Two-term ``A = A_inf + c/w^2`` fit on the declared top nodes."""
    if len(high_idx) < 2:
        return float("nan"), float("nan")
    w = np.asarray([grid[i] for i in high_idx], float)
    a = np.asarray([added[i] for i in high_idx], float)
    design = np.column_stack((np.ones_like(w), 1.0 / w ** 2))
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    return float(coef[0]), float(coef[1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", default="artifacts/case_bem_cummins_wigley_ext/points")
    parser.add_argument("--level", default="medium")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    points = case.load_points(pathlib.Path(args.points_dir), args.level)
    if not points:
        raise SystemExit(f"no {args.level} points under {args.points_dir}")
    report = edge_report(points, args.level)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.points_dir).parent / "band_edges.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"nodes: {['%.3f' % w for w in report['nodes']]}")
    print(f"{'DOF':6s} {'q(low)':>8s} {'R2':>7s} {'p(high)':>8s} {'R2':>7s} "
          f"{'below/K0 %':>11s} {'const %':>8s}")
    for row in report["rows"]:
        print(f"{row['dof']:6s} {row['low_exponent_q']['exponent']:8.3f} "
              f"{row['low_exponent_q']['r_squared']:7.4f} "
              f"{row['high_decay_exponent_p']['exponent']:8.3f} "
              f"{row['high_decay_exponent_p']['r_squared']:7.4f} "
              f"{100*row['K_zero_below_power_fraction']:11.4f} "
              f"{100*row['K_zero_below_constant_fraction']:8.4f}")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())