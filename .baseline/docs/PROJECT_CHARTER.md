# TensorBEM — boundary-element solver charter

**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission
Provide boundary-integral and boundary-element numerical methods as independently usable, quantitatively verified capabilities.

## Owns
- Boundary discretization, kernels, singular/quadrature treatment, system assembly and supported BEM formulations.
- Solver-local geometry/boundary conditions, diagnostics, conditioning and convergence studies.
- Reference cases for formulations explicitly implemented in this repository.

## Does not own
- General research workflow orchestration or generic platform infrastructure.
- Claiming a method is supported merely because a class, interface, or placeholder exists.
- Treating two implementations as independent validation if they share assumptions or defects.

## Integration
Expose clear formulation, dimensionality, boundary conditions, units, discretization and output contracts. TensorPhysics composes BEM with other solvers and owns scientific workflow acceptance; TensorPlatform provides domain-neutral execution and provenance services.

## Quality gates
Use analytical solutions, mesh refinement, conditioning/error diagnostics, conservation or reciprocity checks where applicable, and independent reference comparisons. State exact formulations and limitations.

## Agent rules
Read this charter and README; explain why a change belongs here; avoid private cross-repository imports; add reproducible tests and benchmarks; report exact commands, failures, and untested paths; distinguish implementation from numerical verification and physical validation.

## Maturity
This charter defines ownership; benchmark status must be supported by reproducible evidence.
