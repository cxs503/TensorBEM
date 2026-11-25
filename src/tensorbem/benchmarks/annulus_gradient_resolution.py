"""Measure near-hole gradient accuracy for the constant-panel Laplace solver.

Run python -m tensorbem.benchmarks.annulus_gradient_resolution > report.json.
Distances are radial clearances from the analytic circular hole, in units
where the outer radius is one. The polygonal boundary is inscribed. The
reported resolution is the smallest sampled clearance for which that ring
and all larger sampled clearances satisfy the relative max-norm criterion.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .. import kernels, multiloop
from . import annulus_laplace as study


CLEARANCES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
TOLERANCE = 0.03


def measure(case, inner_panels):
    mesh = multiloop.annulus_mesh(2 * inner_panels, inner_panels)
    solution = multiloop.solve_laplace_multiloop(mesh, case.potential(mesh.centers))
    theta = 2 * np.pi * np.arange(8 * inner_panels) / (8 * inner_panels)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    rings = []
    for distance in CLEARANCES:
        points = (study.INNER_RADIUS + distance) * directions
        # Directional derivatives along the two Cartesian basis vectors give
        # the exact vector field without differencing a sampled potential.
        exact = np.column_stack([
            case.normal_derivative(points, np.broadcast_to(axis, points.shape))
            for axis in np.eye(2)
        ])
        grad = multiloop.evaluate_multiloop_field(mesh, solution, points, gradient=True)
        error = np.linalg.norm(grad - exact, axis=1)
        worst = int(error.argmax())
        rings.append({"clearance": distance, "worst_angle_rad": float(theta[worst]),
                      "relative_linf": float(error[worst] / np.linalg.norm(exact, axis=1).max())})
    resolved = None
    for ring in reversed(rings):
        if not ring["relative_linf"] <= TOLERANCE:
            break
        resolved = ring["clearance"]
    return {"case": case.name, "inner_panels": inner_panels, "outer_panels": 2 * inner_panels,
            "angles_per_ring": len(theta), "rings": rings, "smallest_resolved_clearance": resolved}


def main():
    files = [Path(path).resolve() for path in
             (__file__, study.__file__, multiloop.__file__, kernels.__file__)]
    report = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "commit": subprocess.check_output(["git", "-C", str(files[0].parent),
                                                 "rev-parse", "HEAD"], text=True).strip(),
              "relative_linf_limit": TOLERANCE,
              "levels": [measure(case, n) for case in study.CASES for n in (16, 32, 64)]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
