"""m16109 repro: FMMOperator fail-open contract before setup().

m16000 limitation #2 registered: ``far_fraction()`` returned 0.0 on the
linearity-bench geometry, contradicting the measured 2.46 far-field-cache
error.  Root cause found here: the bench (scripts/fmm_linearity_bench.py:74)
calls ``far_fraction`` on a *fresh, never-set-up* operator, and the public API
is fail-open:

* ``matvec`` on an unset operator silently returns an all-zero vector;
* ``far_fraction`` on an unset operator silently returns 0.0,
  indistinguishable from "0% of pairs are far".

This script prints the observed behaviour; it is run once before the fix
(silent values) and once after (RuntimeError expected).
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "src")
from tensorbem.fmm import FMMOperator, direct_laplace_matvec  # noqa: E402


def _geometry(seed: int = 42):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(200, 3)) * 0.5
    targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    charges = rng.normal(size=200)
    return sources, targets, charges


def main() -> None:
    sources, targets, charges = _geometry()
    out: dict = {}

    # 1. far_fraction on a fresh (never set up) operator
    op = FMMOperator(p=4, max_leaf=25)
    try:
        out["far_fraction_unset"] = {"raised": None, "value": float(op.far_fraction(targets))}
    except RuntimeError as exc:
        out["far_fraction_unset"] = {"raised": "RuntimeError", "message": str(exc)}

    # 2. matvec on a fresh (never set up) operator
    op2 = FMMOperator(p=4, max_leaf=25)
    try:
        got = op2.matvec(targets, sources, charges)
        exact = direct_laplace_matvec(targets, sources, charges)
        rel = float(np.max(np.abs(got - exact)) / np.max(np.abs(exact)))
        out["matvec_unset"] = {
            "raised": None,
            "max_abs": float(np.max(np.abs(got))),
            "rel_err_vs_direct": rel,
        }
    except RuntimeError as exc:
        out["matvec_unset"] = {"raised": "RuntimeError", "message": str(exc)}

    # 3. control: set-up operator must keep working (sanity, both eras)
    op3 = FMMOperator(p=4, max_leaf=25)
    op3.setup(sources)
    got3 = op3.matvec(targets, sources, charges)
    exact3 = direct_laplace_matvec(targets, sources, charges)
    out["control_set_up"] = {
        "far_fraction": float(op3.far_fraction(targets)),
        "rel_err_vs_direct": float(np.max(np.abs(got3 - exact3)) / np.max(np.abs(exact3))),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
