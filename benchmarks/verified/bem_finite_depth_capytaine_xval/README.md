# Finite-depth capytaine cross-validation

Cross-validation of TensorBEM's 3-D finite-depth BEM solver against capytaine
on a submerged sphere geometry.

## Files

- `run_xval.py` — Producer script: runs both solvers and compares results
- `xval_results.json` — Frozen results (updated when computation completes)
- `README.md` — This file

## Usage

```bash
# Run the full cross-validation (needs capytaine installed)
PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py

# Run tests (reads frozen results)
PYTHONPATH=src python -m pytest tests/test_bem_finite_depth_capytaine_xval.py -v
```

## Status

Computation submitted to cluster (job cj2147). Results pending.

## Geometry

Submerged sphere, radius 1 m, centre at z = -2 m.

## Depth ladder

h/R = 3, 5, 8, 12, 20, ∞

## Frequency grid

ka = 0.3, 0.6, 1.0, 1.5, 2.0
