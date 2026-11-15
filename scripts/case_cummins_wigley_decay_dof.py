"""Publish the roll/pitch free-decay audit of CASE-BEM-CUMMINS-WIGLEY.

Replays gate C5 (time-domain Cummins decay vs the frequency-domain damping
ratio) on every restoring mode of the case from the committed frequency sweep --
no BEM system is solved here -- and writes
``benchmarks/verified/casepool_bem_cummins_wigley/decay_dof_audit.json``.

Usage::

    PYTHONPATH=src python3 scripts/case_cummins_wigley_decay_dof.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

SWEEP = "benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz"
OUTPUT = "benchmarks/verified/casepool_bem_cummins_wigley/decay_dof_audit.json"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sweep = case.load_sweep(root / SWEEP)
    started = time.time()
    audit = case.dof_decay_audit(sweep)
    audit["artifact"] = {
        "case": case.CASE_SLUG,
        "sweep": SWEEP,
        "sweep_summary": sweep.summary(),
        "elapsed_s": time.time() - started,
        "replay": "PYTHONPATH=src python3 scripts/case_cummins_wigley_decay_dof.py",
    }
    target = root / OUTPUT
    target.write_text(json.dumps(audit, indent=2, default=float) + "\n")
    print(f"wrote {target} in {audit['artifact']['elapsed_s']:.1f} s")
    print(f"criterion: {audit['criterion']}")
    for name, row in audit["modes"].items():
        values = row["values"]
        rel = values.get("relative_error")
        attrib = row["attribution"]
        tail = attrib.get("tail_closure", {}).get("relative_error")
        window = attrib.get("window", {}).get("relative_error")
        print(
            f"  {name:6s} grade={row['grade']:12s} zeta_decay={values.get('zeta_decay'):.5f} "
            f"zeta_fd={values.get('zeta_frequency_domain'):.5f} rel={rel:.3%} "
            f"omega_n={values.get('natural_frequency_rad_s'):.3f} rad/s "
            f"| tail-closed {tail:.3%} | wider window {window:.3%}"
        )
    print(f"within bar: {audit['within_bar']}  beyond bar: {audit['beyond_bar']}  "
          f"undecidable: {audit['undecidable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())