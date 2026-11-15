"""Resolution-convergence study: D2Q9 LBM lid-driven cavity at Re = 100.

Anchor: feature_ledger ``GAP-DIFFLBM-CAV-RES`` (JOUR-DIFFLBM).
Acceptance probed here: max-point error vs n on a log-log axis with observed
order p >= 1.8 and max-point error < 3 % at n >= 97.

Writes ``results/cavity_resolution_study.json`` (+ profiles npz) and prints a
markdown table.  Every number in the published report is read from that JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2q9_cavity import observed_order, run_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[65, 97, 129])
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u_lid", type=float, default=0.1)
    ap.add_argument("--collision", default="rlbm")
    ap.add_argument("--resid_tol", type=float, default=1e-9)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 = automatic (n-th dependent cap)")
    ap.add_argument("--out", default="results/cavity_resolution_study.json")
    args = ap.parse_args()

    rows = []
    profiles = {}
    for n in args.ns:
        cap = args.max_steps or {33: 60_000, 49: 80_000, 65: 120_000, 81: 140_000,
                                 97: 160_000, 129: 220_000}.get(n, 240_000)
        t0 = time.time()
        res = run_case(n=n, re=args.re, collision=args.collision,
                       u_lid=args.u_lid, max_steps=cap,
                       resid_tol=args.resid_tol, verbose=True)
        wall = time.time() - t0
        row = {
            "n": n, "L": res.L, "h": res.h, "re": args.re, "tau": res.tau,
            "collision": args.collision, "u_lid": args.u_lid,
            "steps": res.steps, "resid": res.resid, "converged": res.converged,
            "wall_seconds": wall,
            "errors": res.errors, "errors_legacy_v": res.errors_legacy,
            "errors_at_ghia_points": res.errors_pts,
        }
        rows.append(row)
        profiles[n] = np.vstack([res.y, res.u, res.x, res.v])
        print(json.dumps({k: row[k] for k in ("n", "steps", "resid", "errors")},
                         indent=2), flush=True)

    hs = [r["h"] for r in rows]
    fits = {}
    if len(rows) >= 2:
        for key in ("max_u", "max_v", "max_uv", "l2_u", "l2_v"):
            fits[key] = observed_order(hs, [r["errors"][key] for r in rows])
        for key in ("max_u", "max_v", "max_uv", "l2_u", "l2_v"):
            fits["legacy_" + key] = observed_order(
                hs, [r["errors_legacy_v"][key] for r in rows])
        fits["max_uv_pts"] = observed_order(
            hs, [r["errors_at_ghia_points"]["max_uv_pts"] for r in rows])

    out = {
        "anchor": "GAP-DIFFLBM-CAV-RES",
        "case": "D2Q9 lid-driven cavity, Re=100, u_lid=0.1, Ghia et al. (1982)",
        "reference_provenance": {
            "table_I_II": "benchmarks/anchors/ghia1982_provenance.json (TensorLBM)",
            "ocr_witness_sha256":
                "3637ef9ef72a76bb090f4e31765ed2870d18db67f0f99611730e61927e060f5b",
            "pdf_sha256": "1d76e4636d24d0fad41c7bdb5ed383e9b224b0bfe3aa0317ce7dfc50619f2d93",
            "note": "Table II uses its own x-stations, different from Table I y-stations",
        },
        "rows": rows,
        "order_fits": fits,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez("results/cavity_profiles.npz", **{f"n{n}": v for n, v in profiles.items()})

    print("\n| n | h | steps | resid | max u | max v | max uv | L2 u | L2 v |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e = r["errors"]
        print(f"| {r['n']} | 1/{r['L']} | {r['steps']} | {r['resid']:.1e} | "
              f"{100*e['max_u']:.2f}% | {100*e['max_v']:.2f}% | "
              f"{100*e['max_uv']:.2f}% | {100*e['l2_u']:.3f}% | "
              f"{100*e['l2_v']:.3f}% |")
    if fits:
        print("\nobserved order (log-log slope):")
        for k, v in fits.items():
            print(f"  {k:12s} p = {v['slope']:.3f}  pairs: "
                  + ", ".join(f"p={q['p']:.2f}@h={q['h_mean']:.4f}" for q in v["pairs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
