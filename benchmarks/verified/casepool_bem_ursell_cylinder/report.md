# Case Pool Report: BEM-URSELL-CYL-RAD

## Case Description

- **Case code**: BEM-URSELL-CYL-RAD
- **Case pool ID**: 201
- **Mission ID**: 17530
- **Body**: Submerged circular cylinder in deep water
- **Radius**: 1.0 m
- **Submergence**: 2.0 m (a/f = 0.5)
- **Density**: 1025.0 kg/m³
- **Gravity**: 9.80665 m/s²
- **Reference**: Ursell (1949) multipole solution, closed-form infinite system
- **Acceptance**: relative error < 5% vs Ursell reference

## Reference Provenance

- **Citation**: Ursell, F. (1949) On the heaving motion of a circular cylinder on the surface of a fluid. Quart. J. Mech. Appl. Math. 2:228-244.
- **Method**: Multipole expansion reducing the body condition to an infinite linear system M·a = b, solved by truncation.
- **Implementation**: `tensorbem.analytic.ursell_cylinder_coefficients`
- **Truncation**: 80 terms, verified converged to < 1e-6 relative delta at 80→100 terms
- **Independence**: The reference is a pure NumPy evaluation of the analytical system, sharing no code with the BEM solver kernel.

## Verdict

**PASS**

| Gate | Pass | Measured | Threshold |
|---|---|---|---|
| G1_case_error_below_5pct | PASS | 0.01814219149922565 | 0.05 |
| G2_monotonic_convergence | PASS | — | — |
| G3_near_far_identity | PASS | 0.005351595698266376 | 0.03 |
| G4_heave_sway_symmetry | PASS | 5.781613292909732e-16 | 1e-08 |
| G5_reference_truncation_converged | PASS | — | — |

## Results: Case Mesh (128 panels)

| ka | ω | A (panel) | A (ref) | ε_A | B (panel) | B (ref) | ε_B | near/far |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 1.4005 | 3991.72 | 3946.88 | 1.1362e-02 | 1324.10 | 1300.50 | 1.8142e-02 | 1.00534 |
| 0.4 | 1.9806 | 3146.16 | 3117.08 | 9.3293e-03 | 2743.32 | 2701.54 | 1.5466e-02 | 1.00534 |
| 0.6 | 2.4257 | 2566.83 | 2545.17 | 8.5130e-03 | 2754.47 | 2718.15 | 1.3361e-02 | 1.00534 |
| 0.8 | 2.8009 | 2339.60 | 2319.70 | 8.5799e-03 | 2216.92 | 2190.12 | 1.2239e-02 | 1.00535 |
| 1.0 | 3.1316 | 2295.72 | 2275.57 | 8.8546e-03 | 1608.96 | 1590.39 | 1.1675e-02 | 1.00535 |

## Convergence Study

| ka | panels | ε_A | ε_B |
|---|---|---|---|
| 0.2 | 32 (coarse) | 4.0920e-02 | 6.2897e-02 |
| 0.2 | 128 (case) | 1.1362e-02 | 1.8142e-02 |
| 0.2 | 256 (fine) | 5.7719e-03 | 9.2604e-03 |
| 0.4 | 32 (coarse) | 3.4341e-02 | 5.2883e-02 |
| 0.4 | 128 (case) | 9.3293e-03 | 1.5466e-02 |
| 0.4 | 256 (fine) | 4.7261e-03 | 7.9117e-03 |
| 0.6 | 32 (coarse) | 3.2318e-02 | 4.5470e-02 |
| 0.6 | 128 (case) | 8.5130e-03 | 1.3361e-02 |
| 0.6 | 256 (fine) | 4.2930e-03 | 6.8412e-03 |
| 0.8 | 32 (coarse) | 3.2957e-02 | 4.1925e-02 |
| 0.8 | 128 (case) | 8.5799e-03 | 1.2239e-02 |
| 0.8 | 256 (fine) | 4.3185e-03 | 6.2626e-03 |
| 1.0 | 32 (coarse) | 3.3925e-02 | 4.0519e-02 |
| 1.0 | 128 (case) | 8.8546e-03 | 1.1675e-02 |
| 1.0 | 256 (fine) | 4.4580e-03 | 5.9641e-03 |

## Environment

- Python: 3.11.6
- NumPy: 2.4.6
- Host: 36cb7828dbd5
- Machine: loongarch64

