"""Publish the C11 (state-space reduction) evidence for the committed sweep.

Writes ``benchmarks/verified/casepool_bem_cummins_wigley/state_space_reduction_c11.json``
from the committed ``sweep_medium.npz``; every number is programmatic.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case

OUT = Path("benchmarks/verified/casepool_bem_cummins_wigley")
START = time.time()
sweep = case.load_sweep(OUT / "sweep_medium.npz")

payload = {
    "case": case.CASE_SLUG,
    "mission": "m23122",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "platform": platform.platform(),
    "numpy": np.__version__,
    "sweep": sweep.summary(),
    "gate_id": "C11",
    "scored_dof": "heave",
    "why": (
        "C4/M3 certify B->K->B and C5/C5b/C5c certify the convolution route; none "
        "of them touches the substitution an operational code makes, replacing the "
        "O(N^2) memory convolution by a finite pole sum. C11 scores exactly that "
        "substitution, with the reduction order capped."
    ),
    "audit": case.state_space_audit(sweep),
}
payload["gate"] = case.state_space_gate(sweep, dof=2)
payload["elapsed_s"] = time.time() - START
target = OUT / "state_space_reduction_c11.json"
target.write_text(json.dumps(payload, indent=2, default=str) + "\n")
print(target, payload["elapsed_s"])
print(json.dumps(payload["audit"], indent=1, default=str)[:1400])