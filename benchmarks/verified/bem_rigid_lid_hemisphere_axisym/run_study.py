#!/usr/bin/env python3
"""m17547 — rigid-lid hemisphere added mass: reference pin + independent MFS route.

Benchmark-pool case ``CASE-BEM-RIGID-LID-HEMISPHERE`` (delivered by m17243) paid
two verification debts here:

  1. the ``ka -> 0`` reference is pinned with exact rational Legendre
     coefficients + Richardson extrapolation (no Gauss quadrature of a kinked
     integrand);
  2. an independent solution route (axisymmetric MFS ring sources, exact
     elliptic-integral kernel, no singular integrals) reproduces the reference.

Produces (zero hand-copied numbers, everything from the single payload):
    benchmarks/verified/bem_rigid_lid_hemisphere_axisym/{results.json,
    summary.md, figure.png}

Run:
    python3 benchmarks/verified/bem_rigid_lid_hemisphere_axisym/run_study.py
    python3 benchmarks/verified/bem_rigid_lid_hemisphere_axisym/make_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for parent in HERE.parents:
        if (parent / "src" / "tensorbem" / "__init__.py").exists():
            return parent
    raise RuntimeError("repo root not found above " + str(HERE))


sys.path.insert(0, str(_repo_root() / "src"))

from tensorbem.benchmarks.rigid_lid_reference_pin import (  # noqa: E402
    render_summary,
    run_study,
)


def main() -> int:
    payload = run_study()
    payload["repro_cmd"] = (
        "python3 benchmarks/verified/bem_rigid_lid_hemisphere_axisym/run_study.py "
        "&& python3 benchmarks/verified/bem_rigid_lid_hemisphere_axisym/make_figure.py"
    )
    (HERE / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (HERE / "summary.md").write_text(render_summary(payload), encoding="utf-8")
    print(json.dumps(payload["criteria"], indent=2))
    print("verdict:", payload["verdict"])
    return 0 if payload["verdict"] == "ALL PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())