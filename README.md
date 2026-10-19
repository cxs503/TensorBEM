# TensorBEM

**TensorBEM** is the boundary-element solver in the Tensor scientific-computing ecosystem. Its goal is to provide verifiable boundary-integral solvers that can be run independently and later orchestrated by TensorPlatform agents.

## First milestone

The initial milestone is a small, inspectable 2-D Laplace BEM implementation with:

- closed polygon boundary geometry and constant panels;
- the 2-D Laplace fundamental solution;
- collocation assembly for prescribed Dirichlet boundary values;
- a manufactured unit-disk benchmark and convergence checks;
- deterministic tests and machine-readable benchmark output.

This is a research prototype, not yet a validated ship-hydrodynamics or acoustics package.

## Quick start

Requires Python 3.10+ and NumPy.

```bash
python -m pip install -e ".[dev]"
pytest
python -m tensorbem.benchmarks.unit_disk --panels 64
```

The benchmark writes JSON results to stdout. The reference field for the first manufactured solution is \(u(r,\theta)=r\cos\theta\), whose exact outward normal derivative on the unit circle is \(q=\cos\theta\). The polygonal boundary approximates the circle, so the report includes discretization and geometry limitations.

## Project layout

- `src/tensorbem/geometry.py`: panel geometry and validation
- `src/tensorbem/kernels.py`: 2-D Laplace kernels
- `src/tensorbem/solver.py`: collocation assembly and solve
- `src/tensorbem/benchmarks/unit_disk.py`: manufactured-solution benchmark
- `tests/`: unit and numerical regression tests
- `docs/verification.md`: equations, sign conventions, and verification plan

## Numerical conventions

The fundamental solution is \(G(x,y)=-\frac{1}{2\pi}\log|x-y|\), satisfying \(-\nabla^2G=\delta\). Boundary panels are ordered counter-clockwise, and their normals point outward. The initial method uses constant boundary elements and midpoint collocation; diagonal single-layer integrals are evaluated analytically. This discretization is intended as a transparent baseline, not as a high-order production method.

## Roadmap

1. Establish unit tests and a reproducible Laplace benchmark.
2. Add independent convergence/error studies and more manufactured solutions.
3. Improve geometry, quadrature, and conditioning diagnostics.
4. Add 3-D surface elements and potential-flow applications only after verification.
5. Expose a stable task/result contract for TensorPlatform.

## License

Apache-2.0.