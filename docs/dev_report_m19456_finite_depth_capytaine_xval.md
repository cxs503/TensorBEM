# Finite-depth cross-validation: TensorBEM vs capytaine

**Case** `benchmarks/verified/bem_finite_depth_capytaine_xval` · **repository** TensorBEM
**Anchor** `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` · **mission** m19456

## 1. Case introduction

### Physical background

Linear potential flow in water of constant depth $h$ with a flat impermeable bottom at $z = -h$ and a free surface at $z = 0$, time factor $\exp(-i\omega t)$. The vertical eigenproblem of the strip has one propagating mode $\phi_0 = \cosh(k_0(z+h))$ with $k_0\tanh(k_0 h) = \nu = \omega^2/g$, and a countable set of evanescent modes.

### Geometry

A fully submerged sphere of radius $a = 1$ m, centred at $z = -2$ m (bottom of sphere at $z = -3$ m). The sphere is closed and does not pierce the free surface or the sea bed, so both TensorBEM and capytaine can treat it without a waterplane lid or CHIEF points.

| Parameter | Value |
|---|---|
| Geometry | Submerged sphere |
| Radius | 1.0 m |
| Centre | $(0, 0, -2)$ m |
| Bottom of sphere | $z = -3$ m |
| Density | 1025.0 kg/m³ |
| Gravity | 9.80665 m/s² |

### Wave matrix

| Parameter | Values |
|---|---|
| $ka$ (dimensionless wavenumber) | 0.3, 0.6, 1.0, 1.5, 2.0 |
| $h/R$ (depth ratio) | 3.0, 5.0, 8.0, 12.0, 20.0, $\infty$ |
| TensorBEM mesh | icosphere subdivisions 2, 3, 4 (320, 1280, 5120 panels) |
| capytaine mesh ladder | (8,16), (12,24), (16,32), (24,48) ≈ 256, 576, 1024, 2304 faces |

### Why this matters

The existing `benchmarks/verified/bem_finite_depth/` verifies the 2-D and 3-D finite-depth solvers against analytic identities (energy conservation, Kramers-Kronig, deep-water asymptotics) but has **no third-party code** in the loop. The existing capytaine cross-checks (`casepool_bem_hulme_hemisphere`, `capytaine_crosscheck`) are all **deep-water** (infinite depth). This cross-validation closes that gap by running the same finite-depth radiation and diffraction problems in both TensorBEM and capytaine.

## 2. Method

### TensorBEM side

- **Kernel**: John (1950) eigenseries Green function (`tensorbem.finite_depth3d`)
- **Solver**: Direct collocation BEM (`tensorbem.finite_depth_solver3d`)
- **Singularity split**: Rankine $1/(4\pi r)$ integrated in closed form, smooth remainder quadratured with 6-point Dunavant rule
- **Mode truncation**: Automatic, targeting $\exp(-10)$ tail at nearest target/quadrature separation
- **DOFs**: Surge, sway, heave (modes 0, 1, 2)

### capytaine side

- **Green function**: Delhommeau (default in capytaine 3.x)
- **Solver**: Indirect BEM (default method)
- **Mesh**: `capytaine.mesh_sphere()` with varying resolution
- **DOFs**: Surge, sway, heave

### Comparison protocol

For each $(ka, h/R)$ pair:
1. Run TensorBEM at finest mesh (subdivisions 4)
2. Run capytaine at finest mesh (24, 48)
3. Compare added mass $A_{33}$ and damping $B_{33}$ for heave
4. Compute relative error: $|A_{33}^{\text{TBEM}} - A_{33}^{\text{cpt}}| / |A_{33}^{\text{cpt}}|$

### Pre-registered gates

| Gate | Description | Threshold |
|---|---|---|
| G1 | TensorBEM-fine vs capytaine-fine $A_{33}$ error | < 10% |
| G2 | TensorBEM-fine vs capytaine-fine $B_{33}$ error | < 15% |
| G3 | capytaine self-convergence (finest vs second) | < 5% |
| G4 | TensorBEM self-convergence (sub 4 vs sub 3) | < 5% |
| G5 | Deep-water limit ($h/R = 20$): TBEM vs cpt $A_{33}$ | < 5% |

## 3. Computation status

The cross-validation computation has been submitted to the cluster (job `cj2147`). The results will be written to `benchmarks/verified/bem_finite_depth_capytaine_xval/xval_results.json`.

### Expected result structure

```json
{
  "case": "bem_finite_depth_capytaine_xval",
  "geometry": {"type": "submerged_sphere", "radius_m": 1.0, ...},
  "parameters": {"ka_grid": [...], "depth_ratios": [...], ...},
  "tbem_study": {"rows": [...], "seconds": ...},
  "cpt_study": {"provenance": {...}, "levels": [...], "seconds": ...},
  "comparison": {
    "comparison": [...],
    "gates": {
      "G1_tbem_cpt_am_heave": {"passed": true, "max_error": ..., "threshold": 0.10},
      ...
    },
    "all_gates_passed": true
  }
}
```

## 4. Anticipated results and error analysis

### Expected behaviour

1. **Deep-water limit** ($h/R = 20$): Both solvers should agree within 5% since the finite-depth correction is small and both solvers converge to the same deep-water solution.

2. **Shallow depth** ($h/R = 3$): Larger discrepancies expected due to:
   - Higher evanescent mode count needed in TensorBEM
   - Different panel discretisations (icosphere vs lat-lon sphere)
   - Different Green function implementations (eigenseries vs Delhommeau)

3. **Damping vs added mass**: Damping is more sensitive to phase errors in the Green function, so G2 (15% threshold) is looser than G1 (10% threshold).

### Sources of discrepancy

| Source | Expected impact |
|---|---|
| Panel discretisation difference | 1-5% |
| Green function implementation | < 1% (both solve the same BVP) |
| Mode truncation (TensorBEM) | < 2% (automatic truncation at $\exp(-10)$) |
| Quadrature rule difference | < 1% |
| Numerical precision | < 0.1% |

## 5. Notes and limitations

### What this cross-validation does NOT cover

1. **Surface-piercing bodies**: Both solvers handle submerged bodies; surface-piercing hulls in finite depth require CHIEF/lid treatment which is not yet implemented in TensorBEM's 3D finite-depth solver.

2. **Diffraction excitation forces**: The current benchmark focuses on radiation coefficients. Diffraction cross-validation is planned as a follow-up.

3. **Experimental validation**: This is a code-to-code comparison, not a comparison with experimental data. The engineering (dimensional) magnitudes remain to be validated against tank tests.

### Honest limitations

- The shallowest station $h/R = 3$ places the sphere's bottom exactly at the seabed ($z = -3$). This is a boundary case where the body touches the bottom. For $h/R > 3$, the sphere is strictly above the seabed.
- capytaine's Delhommeau Green function and TensorBEM's eigenseries kernel are different representations of the same boundary-value problem. Agreement between them validates the physics, but disagreement could be due to either implementation.

## 6. Reproduction

```bash
# Run the full cross-validation (needs capytaine installed)
cd /root/tp-agent/work/m19456
PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py

# Run tests (reads frozen results)
PYTHONPATH=src python -m pytest tests/test_bem_finite_depth_capytaine_xval.py -v
```

## 7. Error判定

| Quantity | Benchmark | Error | < 3%? |
|---|---|---|---|
| $A_{33}$ (TBEM vs cpt) | G1 gate | TBD (threshold 10%) | N/A (gate is 10%) |
| $B_{33}$ (TBEM vs cpt) | G2 gate | TBD (threshold 15%) | N/A (gate is 15%) |
| Self-convergence (TBEM) | G4 gate | TBD (threshold 5%) | Pending |
| Self-convergence (cpt) | G3 gate | TBD (threshold 5%) | Pending |
| Deep-water limit | G5 gate | TBD (threshold 5%) | Pending |

**Note**: The gates for this cross-validation are set at 10-15% because this is a code-to-code comparison with different discretisations, not a comparison with an analytic solution. The existing `bem_finite_depth` benchmark already verifies the solver against analytic identities to machine precision; this cross-validation adds the third-party code axis.

## 8. Figures (pending computation)

Figures will be generated after the computation completes:
- Added mass comparison: TBEM vs capytaine across depth ladder
- Damping comparison: TBEM vs capytaine across depth ladder
- Self-convergence plots for both solvers
- Deep-water limit convergence

## 9. Next steps

1. **Diffraction cross-validation**: Add excitation force comparison at each depth
2. **Surface-piercing case**: Once CHIEF/lid treatment is implemented for finite depth
3. **Experimental validation**: Compare with tank test data for a submerged sphere
