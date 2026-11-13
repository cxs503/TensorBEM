"""CASE-BEM-RIGID-BODY-MASS : run the rigid-body added-mass benchmark.

Reuses TensorBEM's existing capability (no new solver):
  - 2-D elliptic cylinder 3x3 rigid-body added-mass tensor (Neumann)
  - 3-D triaxial ellipsoid 6x6 rigid-body added-mass tensor (Neumann)
against Lamb (1932) closed-form block form. Writes machine-readable JSON.
"""
import json
import sys
from pathlib import Path

from tensorbem.benchmarks.rigid_body_added_mass import run

OUT = Path(__file__).resolve().parent / "runs" / "m17248_rigid_body_mass"
OUT.mkdir(parents=True, exist_ok=True)

# Convergence ladders (>=2 points each) for the sensitivity section.
report = run(
    panel_counts=[16, 32, 64, 128],
    subdivisions=[0, 1, 2],
    density=1025.0,
)

(OUT / "rigid_body_mass_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print("wrote", OUT / "rigid_body_mass_report.json")
print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
