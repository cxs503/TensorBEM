#!/usr/bin/env python3
"""Environment probe: is the cluster-node hemisphere cell wrong because of the
interpreter stack or because of a stale checkout?

The refinement ladder artifact ``inhouse_ladder_ka2.json`` mixes two hosts: its
``subdivisions`` 2/3/4 rows come from the control plane (numpy 2.4.6) and its
``subdivisions = 5`` row from a cluster node (numpy 2.0.1).  A separate node run
of the *same* cell at 1024 panels reported ``A33 = 0.3666 / B33 = 0.0508``
against ``0.3863 / 0.1042`` on the control plane, with the *same* assembled
condition number (2.398615) and the same mesh volume -- so the divergence is
downstream of assembly and the two candidate causes are

1. a broken LAPACK/special-function path in the node's numpy build (this repo
   already documents ``leggauss`` raising ``LinAlgError`` for ``n = 40`` on
   loongarch64 under numpy 2.0.1, see ``_legendre_gauss_golub_welsch``), or
2. a stale checkout on the node, i.e. the ladder rung was produced by older
   solver code.

This script answers that by running *the worktree's own code* on the node and
independently validating the quadrature rules the wave kernel depends on.  If
the node reproduces the control-plane numbers and all rules pass their exact
moment checks, the divergence is (2); otherwise it is (1).

Usage
-----
    PYTHONPATH=src python scripts/hulme_env_probe.py --subdivisions 3 4
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

from tensorbem.benchmarks.hulme_hemisphere_case import run_cell
from tensorbem.free_surface3d import (
    _laguerre_gauss_golub_welsch,
    _legendre_gauss_golub_welsch,
)

# Exact moments: int_0^inf x^k e^{-x} dx = k! and int_0^2 x^k dx = 2^(k+1)/(k+1).
LAGUERRE_MOMENTS = (1.0, 1.0, 2.0, 6.0, 24.0, 120.0)


def quadrature_checks() -> dict[str, object]:
    nodes, weights = _laguerre_gauss_golub_welsch(40)
    lag = [
        float(abs(np.sum(weights * nodes**k) - LAGUERRE_MOMENTS[k]))
        for k in range(len(LAGUERRE_MOMENTS))
    ]
    lnodes, lweights = _legendre_gauss_golub_welsch(40)
    scaled = lweights * 1.0  # [-1, 1] -> rescale below for [0, 2]
    leg = []
    for k in range(8):
        lo, hi = 0.0, 2.0
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        val = half * np.sum(scaled * (mid + half * lnodes) ** k)
        exact = (hi ** (k + 1) - lo ** (k + 1)) / (k + 1)
        leg.append(float(abs(val - exact)))
    # np.polynomial path: the one this repo had to stop trusting.
    try:
        from numpy.polynomial.legendre import leggauss

        np_nodes, np_weights = leggauss(40)
        np_err = float(abs(np.sum(np_weights * np_nodes**19) - 2.0 / 20.0))
        np_status: dict[str, object] = {"ok": True, "max_dev_2_25": np_err}
    except Exception as exc:  # noqa: BLE001 - diagnostic, report verbatim
        np_status = {"ok": False, "exception": f"{type(exc).__name__}: {exc}"}
    return {
        "laguerre40_max_moment_error": max(lag),
        "legendre40_on_0_2_max_moment_error": max(leg),
        "numpy_leggauss40": np_status,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ka", type=float, default=2.0)
    ap.add_argument("--subdivisions", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    payload: dict[str, object] = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "host": platform.node(),
        },
        "quadrature": quadrature_checks(),
        "cells": [],
    }
    keys = (
        "subdivisions",
        "panels",
        "condition_number",
        "mesh_volume",
        "A11_dimless",
        "B11_dimless",
        "A33_dimless",
        "B33_dimless",
        "Fx_dimless",
        "Fz_dimless",
        "phase_Fx_deg",
        "phase_Fz_deg",
        "wall_seconds",
    )
    for sd in args.subdivisions:
        cell = run_cell(args.ka, sd)
        payload["cells"].append({k: cell[k] for k in keys})
    text = json.dumps(payload, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())