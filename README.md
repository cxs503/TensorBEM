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
python -m tensorbem.benchmarks.cylinder_added_mass --panels 64
python -m tensorbem.benchmarks.refinement_study --panels 16 32 64 128 --output artifacts/refinement-study.json
python -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128 --output artifacts/analytical-study.json
```

The benchmark writes JSON to stdout; the refinement study can also persist a machine-readable JSON report. The manufactured field is \\(u(r,\\theta)=r\\cos\\theta\\), with exact outward normal derivative \\(q=\\cos\\theta\\) on the unit circle. The polygon approximates the circle, so results should be assessed by refinement trends, not by treating the polygon as an exact circle.

The circular-cylinder benchmark reports the added mass per unit span for a translating cylinder, with the analytical reference \\(\\rho\\pi a^2\\). It is a restricted boundary-trace verification case, not evidence that TensorBEM supports general exterior-flow or wave-body problems. See [the marine BEM benchmark report](docs/marine_bem_benchmarks.md) for the benchmark landscape, derivation, and limitations.

CI runs both benchmark families, executes the refinement study at 16/32/64/128 panels, and uploads the resulting JSON evidence as a workflow artifact. This is a reproducibility gate, not a claim of general exterior-flow validation.

The additional analytical suite covers a cubic harmonic field on a disk, a quadratic field on an ellipse, a separated solution on a square, and a singular field at an L-shaped re-entrant corner. It reports boundary-flux errors, fixed-probe field errors, flux balance, conditioning, and observed convergence orders. See [the analytical verification report](docs/verification.md) for case definitions, measured results, and limitations. CI also archives this suite's JSON evidence.

Install the optional plotting dependency and generate unit-disk field cloudplots (BEM potential, analytical \\(u=x\\), and absolute error) over the interior region \\(r\\le0.9\\), avoiding the near-boundary region where midpoint field evaluation is less accurate:

```bash
python -m pip install -e ".[visualization]"
python -m tensorbem.benchmarks.unit_disk --panels 64 --plot unit_disk_comparison.png
python -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128 --plot analytical_convergence.svg --output artifacts/analytical-study.json
```

## Project layout

- `src/tensorbem/geometry.py`: panel geometry and validation
- `src/tensorbem/kernels.py`: 2-D Laplace kernels
- `src/tensorbem/solver.py`: collocation assembly and solve
- `src/tensorbem/benchmarks/unit_disk.py`: manufactured-solution benchmark
- `src/tensorbem/benchmarks/cylinder_added_mass.py`: translating-cylinder added-mass benchmark
- `src/tensorbem/benchmarks/refinement_study.py`: reproducible multi-resolution JSON report
- `src/tensorbem/benchmarks/analytical_suite.py`: four additional analytical cases and convergence plots
- `tests/`: unit and numerical regression tests
- `docs/verification.md`: analytical verification cases, equations, numerical results, and limitations
- `docs/marine_bem_benchmarks.md`: marine BEM benchmark survey and numerical report

## Numerical conventions

The fundamental solution is \\(G(x,y)=-\\frac{1}{2\\pi}\\log(|x-y|/2)\\), satisfying \\(-\\nabla^2G=\\delta\\). In 2-D it is defined up to an additive constant; the reference length of 2 avoids the unit-capacity degeneracy of the single-layer matrix on a unit circle. Panels are ordered counter-clockwise with outward normals. The baseline uses constant boundary elements and midpoint collocation; both single- and double-layer straight-panel integrals are analytical, including off-diagonal and interior-field evaluation terms. This is a correctness-oriented baseline, not a high-order production method.

## Roadmap

1. Establish reproducible Laplace benchmarks and convergence/error studies.
2. Improve geometry, quadrature, and conditioning diagnostics.
3. Add 3-D surface elements and potential-flow applications after verification.
4. Expose a stable task/result contract for TensorPlatform.

## License

Apache-2.0.
