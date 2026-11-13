# Overset verification: independent acceptance audit and mesh refinement limits

Read with the original [execution report](report.md). That report is preserved byte for byte in the producer manifest. This addendum clarifies classification and refinement, and records the separate postprocessing audit.

## Classification and refinement

The executed producer includes an additional solid intersection check: every Cartesian cell box intersecting the radius-1 sphere becomes HOLE, including possible cells whose centroids are outside the sphere. The closest-point box/sphere test is exact for the Cartesian background. Remaining split and fringe masks use centroid radii; this is not a cut-cell discretization.

The three angular subdivision levels are **1, 2, 2**, with 80, 320 and 320 surface triangles. Background resolutions are 14³, 18³ and 22³; radial shell layers are 10, 12 and 14. Both interpolation errors decrease, but this is a joint-grid interpolation diagnostic. The final level does not further refine the angular surface. No uniform-h order, second-order claim, PDE mesh convergence or physical flow accuracy is established.

| Level | Shell → Cartesian relative L2 | Cartesian → shell relative L2 |
|---|---:|---:|
| 1 | 0.416069% | 0.230837% |
| 2 | 0.108864% | 0.133129% |
| 3 | 0.101185% | 0.080714% |

These errors compare prescribed nonlinear **point values**, not conservatively transferred cell averages or a numerical flow solution. Each direction and each individual vector component is tested against 3% separately.

## Independent audit

[`independent-audit/result.json`](independent-audit/result.json) is produced by [`scripts/audit_overset3d.py`](../../scripts/audit_overset3d.py), using the shared `tensorfvm.verification.core.Metric` / `evaluate` acceptance interface. It does not import the connectivity builder or locator. It checks every producer artifact SHA-256, then independently reconstructs:

- Constant and affine reproduction, coordinate reproduction, weight signs and sums.
- Nonlinear reference fields, mapped fields and receiver-volume-weighted relative L2, including all three components.
- ACTIVE donor and containing-cell classifications.
- Cartesian solid intersection blanking.
- Native convex-facet containment and sphere tangent-halfspace separation of all four donor centres.
- Strictly decreasing two-way errors over the three archived levels.

The audit binds the original producer manifest hash and has its own source snapshot and manifest. The numerical producer SHA-256 at launch and after execution is `c74cfcbdc0ac725a2558f08a93b063a496fba2d8501de9f2398eb2a876525379`. The repository plotting driver subsequently received wording clarifications; its actual executed bytes remain in `source-snapshot/scripts/verify_overset3d.py`.

Reproduce the audit with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_overset3d.py --directory RUN_DIRECTORY` where the producer manifest is valid and `independent-audit` does not exist. The audit requires a fresh destination to preserve prior results.

**No conservative overset Navier–Stokes solve or FVM/LBM time integration was executed.**
