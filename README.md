# TensorBEM

**TensorBEM** is the boundary-element solver in the Tensor scientific-computing ecosystem. Its goal is to provide verifiable boundary-integral solvers that can run independently and later be orchestrated by TensorPlatform agents.

## First milestone

The initial milestone is a transparent 2-D Laplace BEM implementation with closed polygon geometry, constant panels, the 2-D fundamental solution, collocation assembly, a manufactured unit-disk benchmark, and deterministic tests.

This is a research prototype, not yet a validated ship-hydrodynamics or acoustics package.

## Quick start

Requires Python 3.10+ and NumPy.

```bash
python -m pip install -e ".[dev]"
pytest
python -m tensorbem.benchmarks.unit_disk --panels 64
```

The benchmark writes JSON to stdout. The manufactured field is \(u(r,\theta)=r\cos\theta\), with exact outward normal derivative \(q=\cos\theta\) on the unit circle. The polygon approximates the circle, so results should be assessed by refinement trends, not by treating the polygon as an exact circle.

## Project layout

- `src/tensorbem/geometry.py`: panel geometry and validation
- `src/tensorbem/kernels.py`: 2-D Laplace kernels
- `src/tensorbem/solver.py`: collocation assembly and solve
- `src/tensorbem/benchmarks/unit_disk.py`: manufactured-solution benchmark
- `tests/`: unit and numerical regression tests
- `docs/verification.md`: equations, conventions, and verification plan

## Numerical conventions

The fundamental solution is \(G(x,y)=-\frac{1}{2\pi}\log(|x-y|/2)\), satisfying \(-\nabla^2G=\delta\). In 2-D it is defined up to an additive constant; the reference length of 2 avoids the unit-capacity degeneracy of the single-layer matrix on a unit circle. Panels are ordered counter-clockwise with outward normals. The baseline uses constant boundary elements and midpoint collocation; diagonal single-layer integrals are analytic while off-diagonal integrals use midpoint quadrature. This is a correctness-oriented baseline, not a high-order production method.

## Roadmap

1. Establish reproducible Laplace benchmarks and convergence/error studies.
2. Improve geometry, quadrature, and conditioning diagnostics.
3. Add 3-D surface elements and potential-flow applications after verification.
4. Expose a stable task/result contract for TensorPlatform.

## License

Apache-2.0.
