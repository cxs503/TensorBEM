# CASE-BEM-HULME-HEMISPHERE — Haskind reciprocity cross-check

**Repository**: TensorBEM · **Case**: `BEM-HULME-HEMISPHERE` (floating hemisphere, radius $a=1$ m, deep water, beam seas) · **Artifact**: `benchmarks/verified/casepool_bem_hulme_hemisphere/haskind_verification_results.json` · **Generated**: 2026-11-16T03:47:57Z

This page is rendered by `make_haskind_doc.py` from the run artifact; no digit on it is hand-copied.

## 1. What is being verified, and why it is not a tautology

The case page (`docs/bem_hulme_hemisphere_case.md`) certifies the excitation forces $F_x, F_z$ against Hulme's semi-analytical wave-free multipole series. That comparison decides whether the *numbers* are right; it does not decide whether the *diffraction solve* is right, because a wrong diffraction RHS and a wrong force integral can conspire to land on the reference value.

The Haskind–Newman relation removes the diffraction solve from the problem entirely: the excitation force follows from the **radiation** potentials alone,

$$F_j = -\mathrm{i}\omega\rho\int_{\Gamma_B}\left(\varphi_I\, n_j - \varphi_j \frac{\partial\varphi_I}{\partial n}\right)\mathrm{d}S,$$

with $\varphi_I$ the incident Airy potential and $\varphi_j$ the $j$-th radiation potential. Two implementations are present and both are exercised here:

| route | formula | what it shares with the diffraction solve |
|---|---|---|
| `boundary_haskind` (near field) | the integral above, on the body panels | the discretised free-surface operator and the panel quadrature |
| `kochin_haskind` (far field) | $\rho g A\, K_j(\beta+\pi)$, $K$ the Kochin function on the body surface | the radiation potential only — **no** $n_j$ weighted integral, so it is the most independent of the three |

So the boundary route is a *reciprocity* check on the assembled system, and the Kochin route is a *far-field* check on the same radiation potentials. Neither is independent of `tensorbem` (see §6), and that limit is stated rather than hidden.

### Predeclared gate

* **H1** — on the band mesh (256 panels, subdivisions 3), $|F_j^{\mathrm{Haskind}} - F_j^{\mathrm{diff}}| / |F_j^{\mathrm{diff}}| < 3\%$ for surge and heave at every $ka$ of the declared band (0.2, 0.5, 0.8, 1.0, 1.5, 2.0).
* **H2** — the coarse-mesh ladder must show the same discrepancy *shrinking* under refinement; an agreement that degrades with refinement would mean the two routes agree for the wrong reason.

## 2. Method and tool chain

One cell = one mesh $\times$ one frequency, with the diffraction and radiation problems sharing a single assembled operator (`assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")`), so the comparison is not polluted by two different discretisations of the same physics:

1. `hemisphere_hull_mesh(subdivisions=3, radius=1)` → 256 panels; the case geometry gate ($|\nabla/\nabla_{exact}-1|<2\%$, $|A_{wp}/\pi a^2-1|<2\%$) is inherited from the case driver.
2. `solve_radiation_3d(modes=(0, 2))` → surge and heave radiation potentials.
3. `solve_diffraction_3d(...)` → $F_x$ = `excitation_force[0]`, $F_z$ = `excitation_force[2]` (the 6-DOF vector is surge/sway/heave/roll/pitch/yaw; index 2 is heave, not 1).
4. `haskind_excitation_force_3d(...)` → both Haskind routes.
5. All three forces are normalised by $\rho g A a^2$, the same normalisation the case driver and the Hulme reference use.

Tool chain: TensorBEM `src/tensorbem` at the mission branch, Python 3.11.6, numpy 2.4.6, OMP/OpenBLAS threads 4/4. Runs are submitted to the cluster scheduler (8 cores, single command).

## 3. Results — band cells, 256 panels

### 3.1 Three-route force magnitudes (dimensionless, $\rho g A a^2$)

| $ka$ | surge diff. | surge Haskind | surge Kochin | surge Hulme | heave diff. | heave Haskind | heave Kochin | heave Hulme |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.57243 | 0.57244 | 0.57252 | 0.58438 | 2.40909 | 2.40908 | 2.40904 | 2.41864 |
| 0.5 | 1.25857 | 1.25863 | 1.25911 | 1.28578 | 1.68485 | 1.68490 | 1.68465 | 1.68539 |
| 0.8 | 1.63203 | 1.63220 | 1.63324 | 1.66682 | 1.23604 | 1.23646 | 1.23596 | 1.23204 |
| 1 | 1.68646 | 1.68663 | 1.68807 | 1.72087 | 1.02558 | 1.02640 | 1.02581 | 1.02007 |
| 1.5 | 1.47136 | 1.47119 | 1.47344 | 1.49719 | 0.67649 | 0.67756 | 0.67730 | 0.66974 |
| 2 | 1.17900 | 1.17888 | 1.18172 | 1.19762 | 0.47349 | 0.47045 | 0.47112 | 0.46462 |

### 3.2 Relative discrepancies against the direct diffraction solve

| $ka$ | Haskind $F_x$ | Haskind $F_z$ | Kochin $F_x$ | Kochin $F_z$ | diff. vs Hulme $F_x$ | diff. vs Hulme $F_z$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | +0.001% | +0.000% | +0.015% | +0.002% | +2.045% | +0.395% |
| 0.5 | +0.005% | +0.008% | +0.043% | +0.018% | +2.116% | +0.032% |
| 0.8 | +0.014% | +0.038% | +0.074% | +0.051% | +2.088% | +0.324% |
| 1 | +0.021% | +0.081% | +0.096% | +0.076% | +2.000% | +0.540% |
| 1.5 | +0.033% | +0.355% | +0.144% | +0.140% | +1.725% | +1.009% |
| 2 | +0.014% | +1.250% | +0.271% | +0.828% | +1.555% | +1.911% |

### 3.3 Solver health of the same cells

| $ka$ | panels | operator cond. | radiation residual | diffraction residual |
|---:|---:|---:|---:|---:|
| 0.2 | 256 | 2.386 | 1.30e-04 | 1.32e-04 |
| 0.5 | 256 | 2.701 | 1.40e-04 | 1.38e-04 |
| 0.8 | 256 | 2.819 | 1.75e-04 | 1.66e-04 |
| 1 | 256 | 2.783 | 2.18e-04 | 2.08e-04 |
| 1.5 | 256 | 2.329 | 4.53e-04 | 5.58e-04 |
| 2 | 256 | 2.352 | 1.20e-03 | 2.12e-03 |

**Band verdict**: `PASS` — worst Haskind discrepancy +1.250% (gate 3%), worst Kochin discrepancy +0.828%.

## 4. Results — coarse-mesh ladder (gate H2)

| $ka$ | panels | Haskind $F_x$ | Haskind $F_z$ | Kochin $F_x$ | Kochin $F_z$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | +0.052% | +0.252% | +0.363% | +0.296% |
| 1 | 256 | +0.021% | +0.081% | +0.096% | +0.076% |
| 2 | 64 | +0.043% | +2.953% | +1.110% | +1.265% |
| 2 | 256 | +0.014% | +1.250% | +0.271% | +0.828% |

**Ladder verdict**: `PASS`.

## 5. Figures

* `benchmarks/verified/casepool_bem_hulme_hemisphere/fig_haskind_magnitude_comparison.png` — diffraction / Haskind (boundary) / Haskind (Kochin) / Hulme magnitudes against $ka$, surge and heave.
* `benchmarks/verified/casepool_bem_hulme_hemisphere/fig_haskind_error_comparison.png` — relative discrepancy of both Haskind routes against the direct diffraction solve, with the H1 gate line.
* `benchmarks/verified/casepool_bem_hulme_hemisphere/fig_haskind_three_way.png` — the three-way comparison with the BEM-vs-Hulme error on the right axis.

## 6. Honest limits

1. **Not an independent code.** All three routes live in `tensorbem`; the boundary route shares the assembled operator with the diffraction solve, and both Haskind routes share the radiation potentials. A defect in the free-surface kernel (`tensorbem.free_surface3d.wave_function_3d`) is invisible to this page. The independent axes remain the capytaine cross-check (`docs/case_bem_hulme_independent_axis.md`) and the Hulme series re-derivation.
2. **Reciprocity, not solvability.** Agreement of the boundary route certifies that the discretised reciprocity relation $\int\varphi^{scat}n_j\,\mathrm{d}S = -\int\varphi_j\,\partial_n\varphi_I\,\mathrm{d}S$ holds on this mesh — it does not certify that either potential is the physical one.
3. **Band stops at $ka=2$.** The gate is applied on the declared band; the $ka>2$ behaviour of this case is documented separately (`docs/hulme_band_extension.md`) and is not re-opened here.
4. **Single wave heading.** Only beam seas ($\beta = 0$) are exercised, which is the heading of the case definition; heading symmetry is covered by `tests/test_hulme_heading_symmetry.py`.

## 7. Reproduce

```
PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/haskind_verification.py \
    --out benchmarks/verified/casepool_bem_hulme_hemisphere --ladder
python benchmarks/verified/casepool_bem_hulme_hemisphere/make_haskind_doc.py
pytest tests/test_hulme_haskind_verification.py
```
