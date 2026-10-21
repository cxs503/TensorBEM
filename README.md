# TensorBEM

**TensorBEM** is the boundary-element solver in the Tensor scientific-computing ecosystem. Its goal is to provide verifiable boundary-integral solvers that can run independently and later be orchestrated by TensorPlatform agents. The current baseline includes a 2-D polygonal constant-panel Laplace solver and a 3-D triangular-surface constant-panel Laplace solver with interior and exterior Dirichlet formulations.

## Current milestones

The first milestone is a transparent 2-D Laplace BEM implementation with closed polygon geometry, constant panels, the 2-D fundamental solution, exact straight-panel integrals, collocation assembly, a manufactured unit-disk benchmark, and deterministic tests.

The second milestone adds a 3-D Laplace surface-element baseline: closed triangular meshes, constant panels, centroid collocation with the \(c=1/2\) free term, analytic double-layer solid angles, Duffy single-layer quadrature, an exact centroid self-panel integral, and interior/exterior Dirichlet formulations. The sphere added-mass benchmark, three cross-verification cases, and a refinement study are included.

This is a research prototype, not yet a validated ship-hydrodynamics, free-surface, acoustics, or 3-D ship-flow package.

## Quick start

Requires Python 3.10+ and NumPy.

```bash
python -m pip install -e ".[dev]"
pytest
python -m tensorbem.benchmarks.unit_disk --panels 64
python -m tensorbem.benchmarks.cylinder_added_mass --panels 64
python -m tensorbem.benchmarks.refinement_study --panels 16 32 64 128 --output artifacts/refinement-study.json
python -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128 --output artifacts/analytical-study.json
python -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2
python -m tensorbem.benchmarks.sphere_velocity --subdivisions 2
python -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3 --output artifacts/three-dimensional-study.json
```

The benchmark writes JSON to stdout; the refinement study can also persist a machine-readable JSON report. The manufactured field is \\(u(r,\\theta)=r\\cos\\theta\\), with exact outward normal derivative \\(q=\\cos\\theta\\) on the unit circle. The polygon approximates the circle, so results should be assessed by refinement trends, not by treating the polygon as an exact circle.

The circular-cylinder benchmark reports the added mass per unit span for a translating cylinder, with the analytical reference \\(\\rho\\pi a^2\\). With exact straight-panel integration, its relative error falls from \\(2.62\\times10^{-2}\\) at 16 panels to \\(4.02\\times10^{-4}\\) at 128 panels. It is a restricted boundary-trace verification case, not evidence that TensorBEM supports general exterior-flow or wave-body problems. See [the marine BEM benchmark report](docs/marine_bem_benchmarks.md) for the benchmark landscape, derivation, and limitations.

The 3-D sphere benchmark reports the added mass of a translating sphere, with analytical reference \\(\\tfrac12\\rho V\\). The surface is a flat-faceted icosphere, the boundary trace is the sphere solution at panel-centroid angular coordinates, and the solve uses the exterior Dirichlet equation \\(\\mathbf S\\mathbf q_e=(\\tfrac12\\mathbf I-\\mathbf H)\\mathbf u\\). At 20/80/320/1280 panels the relative error is \\(2.13\\times10^{-1}\\), \\(6.48\\times10^{-2}\\), \\(1.74\\times10^{-2}\\),
\\(4.47\\times10^{-3}\\), with observed order near 2. The pointwise exterior flux converges only near first order because of polyhedral-edge singularities; the integral added mass is the primary benchmark. This is a Laplace verification case, not free-surface, wave-body, or ship-hull validation. See [the three-dimensional verification report](docs/three_dimensional_bem.md).
The 3-D **velocity** field is available through `evaluate_surface_velocity`. The single-layer gradient kernel is integrable and uses the same fixed-order Duffy rule as the potential; the double-layer gradient is hypersingular and is evaluated from the closed-form gradient of each panel's signed solid angle, which is exact for constant panels. Against the exact exterior velocity of a translating sphere the relative L2 error is \\(1.02\\times10^{-1}\\), \\(2.76\\times10^{-2}\\), and \\(7.07\\times10^{-3}\\) at 80/320/1280 panels, with observed order 2.03 and 2.01. The reconstructed velocity **on** the surface is a different matter: the constant-panel trace makes the interior-limit gradient degrade to 53-65% error that does not vanish under refinement. That number is reported by the benchmark as a diagnostic; do not use the current discretization for hull-surface velocity or force integration.

CI runs the 2-D and 3-D benchmark families, executes both refinement studies, and uploads the resulting JSON evidence as a workflow artifact. This is a reproducibility gate, not a claim of general exterior-flow validation.

The additional analytical suite covers a cubic harmonic field on a disk, a quadratic field on an ellipse, a separated solution on a square, and a singular field at an L-shaped re-entrant corner. It reports boundary-flux errors, fixed-probe field errors, flux balance, conditioning, and observed convergence orders. See [the analytical verification report](docs/verification.md) for case definitions, measured results, and limitations. CI also archives this suite's JSON evidence.

Install the optional plotting dependency and generate unit-disk field cloudplots (BEM potential, analytical \\(u=x\\), and absolute error) over the interior region \\(r\\le0.9\\), avoiding the near-boundary region where midpoint field evaluation is less accurate:

```bash
python -m pip install -e ".[visualization]"
python -m tensorbem.benchmarks.unit_disk --panels 64 --plot unit_disk_comparison.png
python -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128 --plot analytical_convergence.svg --output artifacts/analytical-study.json
python -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3 --plot artifacts/three-dimensional-convergence.svg --output artifacts/three-dimensional-study.json
```

## Project layout

- `src/tensorbem/geometry.py`: 2-D panel geometry and validation
- `src/tensorbem/kernels.py`: 2-D Laplace kernels and exact straight-panel integrals
- `src/tensorbem/solver.py`: 2-D collocation assembly, solve, and interior field evaluation
- `src/tensorbem/geometry3d.py`: closed triangular surface meshes, icosphere generation, and quality diagnostics
- `src/tensorbem/kernels3d.py`: 3-D Laplace kernels, solid angles, Duffy and self-panel quadrature
- `src/tensorbem/solver3d.py`: 3-D collocation assembly, interior/exterior Dirichlet solves, and field evaluation
- `src/tensorbem/benchmarks/unit_disk.py`: manufactured-solution benchmark
- `src/tensorbem/benchmarks/cylinder_added_mass.py`: translating-cylinder added-mass benchmark
- `src/tensorbem/benchmarks/refinement_study.py`: reproducible 2-D multi-resolution JSON report
- `src/tensorbem/benchmarks/analytical_suite.py`: four additional 2-D analytical cases and convergence plots
- `src/tensorbem/benchmarks/sphere_added_mass.py`: translating-sphere added-mass benchmark
- `src/tensorbem/benchmarks/sphere_velocity.py`: translating-sphere velocity-field benchmark
- `src/tensorbem/benchmarks/three_dimensional_study.py`: 3-D verification cases, refinement study, and plots
- `tests/`: unit and numerical regression tests
- `docs/verification.md`: 2-D analytical verification cases, equations, numerical results, and limitations
- `docs/marine_bem_benchmarks.md`: marine BEM benchmark survey and numerical report
- `docs/three_dimensional_bem.md`: 3-D formulation, quadrature, verification cases, results, and limitations

## Numerical conventions

The 2-D fundamental solution is \\(G(x,y)=-\\frac{1}{2\\pi}\\log(|x-y|/2)\\), satisfying \\(-\\nabla^2G=\\delta\\). In 2-D it is defined up to an additive constant; the reference length of 2 avoids the unit-capacity degeneracy of the single-layer matrix on a unit circle. Panels are ordered counter-clockwise with outward normals. The baseline uses constant boundary elements and midpoint collocation; both single- and double-layer straight-panel integrals are analytical, including off-diagonal and interior-field evaluation terms. This is a correctness-oriented baseline, not a high-order production method.

The 3-D fundamental solution is \\(G(x,y)=1/(4\\pi|x-y|)\\), satisfying \\(-\\nabla^2G=\\delta\\). Closed triangular meshes are ordered consistently with outward normals, and collocation is at triangle centroids where \\(c=1/2\\). Double-layer panel integrals are evaluated analytically from signed solid angles; single-layer self-panels use a polar integral about the centroid and other panels use an 8x8 Duffy-transformed Gauss rule. The interior Dirichlet equation is \\(\\mathbf S\\mathbf q=(\\tfrac12\\mathbf I+\\mathbf H)\\mathbf u\\); the exterior equation is \\(\\mathbf S\\mathbf q_e=(\\tfrac12\\mathbf I-\\mathbf H)\\mathbf u\\), where \\(\\mathbf q_e\\) uses the fluid-domain normal. The dense direct solve is intended for verification meshes, not large models. Field evaluation requires points to be at least \\(0.05h_{\\max}\\) from the surface; near-boundary near-singular quadrature is not implemented.

## Roadmap

1. **Done:** reproducible Laplace benchmarks and convergence/error studies for 2-D polygons and 3-D triangles.
2. **In progress:** improve geometry, quadrature, and conditioning diagnostics. Exact 2-D panel integrals, analytic 3-D double-layer solid angles, self-panel polar integration, mesh-quality metrics, and conditioning/residual reporting are in place; graded meshes, higher-order/curved panels, and near-singular field quadrature remain.
3. **Partially done:** 3-D constant triangular surface elements, interior/exterior Dirichlet formulations, and the sphere added-mass benchmark. Free-surface conditions, wave radiation/diffraction, ship-hull geometry, and fast solvers remain.
4. **Future:** expose a stable task/result contract for TensorPlatform.

## License

Apache-2.0.
