#!/usr/bin/env python3
"""Is the C9 tail-exponent ``p(20) ~ 3.0`` reproducible from the data we have?

Context
-------
Gate ``C9`` (:func:`tensorbem.benchmarks.case_bem_cummins_wigley.truncation_decidability_gate`)
decides whether surge / sway / yaw become *certifiable* once the radiation band is
widened from 12 to 20 rad/s.  The decision rule is one sided: a DOF is certifiable
only when the measured damping slope at the band top satisfies ``p > 2.05``
(``slope_bar``), because the omitted Kramers-Kronig integral
``int^inf B(s) ds/(s^2 - omega^2)`` converges only for ``p > 2``.  The slope itself
is :func:`tensorbem.causality.local_tail_exponent` -- a log-log fit over the last
``window`` (default 3) samples.

The round-9 plan (``dev_report_m22423_cummins_wigley_round9.md`` §7) *asserted*
``p(20) ~ 3.0`` for the horizontal modes and concluded from that number that
widening the band would make them decidable.  Nothing in the repository tested
that assertion.  This script does: it recomputes the slope on every extended-band
artifact that exists, for several window lengths, and reports how much the verdict
moves.

What it shows (2026-11-16, artifacts as committed)
--------------------------------------------------
The estimate is **not** a well determined number:
per DOF, ``p(20)`` changes by O(1)--O(10) between the two probe meshes and between
adjacent windows, and the admissibility verdict itself flips (pitch: ``-2.03`` on
the coarse probe, ``+5.93`` on the medium one; yaw: ``1.97`` vs ``5.58``).  So
"``p(20) ~ 3.0`` makes surge/sway/yaw decidable" is an artefact of one particular
three-sample fit on one particular mesh, not a property of the physics.  Gate C9
keeps its meaning (it still audits that the extension loses nothing), but its
single-window slope cannot be quoted as evidence on its own.

The probe meshes violate ``k*sqrt(A_panel) <= 1`` above ~10 / ~12 rad/s (gate C2),
so these numbers do not certify anything either -- they only show that the
**estimator** is fragile.  The certification that matters needs the
C2-admissible 3840-panel sweep (``sweep_extended.npz``); until that artifact is
assembled, a C9 verdict must be reported as ``not_run``, not as "widened".

Usage::

    PYTHONPATH=src python3 scripts/case_cummins_wigley_tail_estimator_robustness.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402
from tensorbem.causality import (  # noqa: E402
    high_band_truncation_budget,
    local_tail_exponent,
)

CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
REPORT = CASE_DIR / "tail_estimator_robustness.json"
#: ``slope_bar`` of gate C9 (``truncation_decidability_gate``).
EXPONENT_BAR = 2.05
WINDOWS = (3, 4, 5)


def evaluate(path: pathlib.Path) -> dict:
    sweep = case.load_sweep(path)
    rows = []
    for index, name in enumerate(case.DOF_NAMES):
        diag = sweep.damping[:, index, index]
        slopes = {}
        for window in WINDOWS:
            exponent, _ = local_tail_exponent(sweep.omega, diag, window=window)
            slopes[f"p_window{window}"] = float(exponent)
        budget = high_band_truncation_budget(sweep.omega, diag)
        slopes["budget_admissible"] = bool(budget.admissible)
        slopes["verdict_stable_over_windows"] = (
            len({v > EXPONENT_BAR for v in slopes.values() if isinstance(v, float)}) == 1
        )
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "band_top_B": float(diag[-1]),
                **slopes,
            }
        )
    return {
        "artifact": path.name,
        "mesh": {"n_sections": sweep.n_sections, "n_levels": sweep.n_levels,
                 "panel_count": sweep.panel_count,
                 "max_panel_scale": sweep.max_panel_scale},
        "band": [float(sweep.omega[0]), float(sweep.omega[-1])],
        "n_omega": int(sweep.omega.size),
        "per_dof": rows,
    }


def main() -> int:
    artifacts = []
    for name in (
        "sweep_extended_probe_coarse.npz",
        "sweep_extended_probe_medium.npz",
        "sweep_medium.npz",
    ):
        path = CASE_DIR / name
        if path.exists():
            artifacts.append(evaluate(path))

    # The asserted plan value vs what the artifacts actually give.
    spread = {}
    for index, name in enumerate(case.DOF_NAMES):
        values = [
            row["per_dof"][index]["p_window3"]
            for row in artifacts
            if math_finite(row["per_dof"][index]["p_window3"])
        ]
        spread[name] = {
            "p_window3_range": [min(values), max(values)] if values else None,
            "n_artifacts": len(values),
            "verdict_stable_across_meshes": (
                len({v > EXPONENT_BAR for v in values}) == 1 if values else False
            ),
        }

    report = {
        "question": (
            "Is the round-9 assertion p(20) ~ 3.0 (=> surge/sway/yaw decidable on "
            "the widened band) reproducible from the extended-band artifacts?"
        ),
        "answer": (
            "No. The three-sample log-log slope at the band top is not a stable "
            "estimator on these artifacts: it moves by O(1)-O(10) between meshes and "
            "between adjacent windows, and the admissibility verdict flips with the "
            "mesh for heave/pitch/yaw. p ~ 3.0 is one sample of that spread, not a "
            "property of the physics."
        ),
        "exponent_bar": EXPONENT_BAR,
        "windows": list(WINDOWS),
        "artifacts": artifacts,
        "cross_artifact_spread": spread,
        "c2_caveat": (
            "both probe meshes violate k*sqrt(A_panel) <= 1 above ~10 (coarse) resp. "
            "~12 (medium) rad/s, so none of them can certify C9; the certification "
            "needs the 3840-panel sweep_extended.npz."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=1) + "\n")

    print(f"{'artifact':32s} {'dof':6s} {'p(w3)':>8s} {'p(w4)':>8s} {'p(w5)':>8s} adm")
    for block in artifacts:
        for row in block["per_dof"]:
            print(
                f"{block['artifact']:32s} {row['dof_name']:6s} "
                f"{row['p_window3']:8.3f} {row['p_window4']:8.3f} "
                f"{row['p_window5']:8.3f} {row['budget_admissible']}"
            )
    print()
    for name, block in spread.items():
        print(f"{name:6s} p(w3) over artifacts: {block['p_window3_range']}  "
              f"stable = {block['verdict_stable_across_meshes']}")
    print(f"\nwrote {REPORT}")
    return 0


def math_finite(value: float) -> bool:
    return bool(np.isfinite(value))


if __name__ == "__main__":
    sys.exit(main())