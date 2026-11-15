"""Cross-check the NumPy cavity solver against the TensorLBM (torch) solver.

Two implementations of the same physics are never bit-comparable until the
*lattice direction convention* is pinned.  That convention is discovered here
empirically rather than assumed:

  TensorLBM's ``_stream`` implements ``f_i(x) <- f_i(x + c_i)``, i.e. the
  population labelled ``i`` advects along ``-c_i``.  The scheme stays
  self-consistent (equilibrium, momentum and the Ladd momentum-transfer term all
  use the same labelled ``c_i``), which is equivalent to running the *mirror*
  image of the intended problem (lid moving in -x) and reporting the negated
  velocity.  Its centreline profiles are therefore the physical ones, while the
  full 2-D field it reports is ``u(x,y) -> u(L-x,y)``, ``v(x,y) -> -v(L-x,y)``.

This script checks all three statements:

  1. centreline profiles agree (the only quantity the TensorLBM verification
     gates consume);
  2. the full field agrees after applying the mirror/sign map above;
  3. the raw (unmapped) full field does *not* agree, so the map is not vacuous.

Usage
-----
    TENSORLBM_SRC=<...>/TensorLBM/src python crosscheck_tensorlbm.py --n 33 --steps 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2q9_cavity import CavityConfig, CavitySolver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u_lid", type=float, default=0.1)
    ap.add_argument("--collision", default="rlbm", choices=["rlbm", "trt", "bgk"])
    ap.add_argument("--out", default="results/crosscheck_tensorlbm.json")
    args = ap.parse_args()

    src = os.environ.get("TENSORLBM_SRC")
    if not src:
        raise SystemExit("set TENSORLBM_SRC to the TensorLBM 'src' directory")
    sys.path.insert(0, src)

    from tensorlbm.verification.lid_driven_cavity.solver import (
        CavityConfig as TConfig, CavitySolver as TSolver,
    )

    tcfg = TConfig(n=args.n, re=args.re, u_lid=args.u_lid, max_steps=args.steps)
    tcfg.use_rlbm = args.collision == "rlbm"
    tcfg.use_trt = args.collision == "trt"
    ts = TSolver(tcfg)
    for _ in range(args.steps):
        ts.step()
    _, tux, tuy = ts._macroscopic(ts.f)
    tux = tux.numpy()
    tuy = tuy.numpy()
    ty, tu, tx, tv = ts.get_profiles()
    tu = np.asarray(tu)
    tv = np.asarray(tv)

    cfg = CavityConfig(n=args.n, re=args.re, u_lid=args.u_lid,
                       collision=args.collision, max_steps=args.steps)
    ns = CavitySolver(cfg)
    for _ in range(args.steps):
        ns.step()
    _, nux, nuy = ns.macroscopic()
    ny, nu, nx, nv = ns.profiles()

    fluid = ns.fluid
    # map: TensorLBM reported field -> physical field
    mapped_u = tux[::-1, :].copy()        # u(x,y) = u_t(L-x, y)  (mirror in x)
    mapped_v = -tuy[:, ::-1].copy()       # v(x,y) = -v_t(L-x, y)

    out = {
        "n": args.n, "steps": args.steps, "re": args.re, "u_lid": args.u_lid,
        "collision": args.collision, "tensorlbm_src": src,
        "centreline_max_abs_du": float(np.max(np.abs(nu - tu))),
        "centreline_max_abs_dv": float(np.max(np.abs(nv - tv))),
        "u_at_centreline_mid": [float(nu[len(nu) // 2]), float(tu[len(tu) // 2])],
        "mapped_field_max_abs_du": float(np.abs(nux - mapped_u)[fluid].max()),
        "mapped_field_max_abs_dv": float(np.abs(nuy - mapped_v)[fluid].max()),
        "raw_field_max_abs_du": float(np.abs(nux - tux)[fluid].max()),
        "raw_field_max_abs_dv": float(np.abs(nuy - tuy)[fluid].max()),
        "u_lid": args.u_lid,
        "verdict": None,
    }
    tol = 1e-10 * max(args.u_lid, 1e-30)
    ok = (out["centreline_max_abs_du"] < tol
          and out["centreline_max_abs_dv"] < tol
          and out["mapped_field_max_abs_du"] < tol
          and out["mapped_field_max_abs_dv"] < tol
          and out["raw_field_max_abs_du"] > 1e-3 * args.u_lid)
    out["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())