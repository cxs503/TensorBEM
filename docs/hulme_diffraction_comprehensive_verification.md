# Hulme (1982) Floating Hemisphere 3D Diffraction: Comprehensive Verification Report

**Anchor:** CASE-BEM-HULME-DIFF-VERIF (TensorBEM case pool id 177)
**Mission:** m18257
**Date:** 2026-11-13

## 1. Case Introduction

**Physical problem:** Regular-wave excitation of a floating hemisphere of radius a = 1 m
in deep water. The hemisphere is fixed, and the incident wave has unit amplitude,
head-on incidence (β = 0°). The excitation force is non-dimensionalised by ρgAa².

**Reference:** Hulme, A. (1982). The wave forces acting on a floating hemisphere
undergoing forced periodic oscillations. *J. Fluid Mech.*, 121:443-463.

The reference solution is computed via Hulme's wave-free multipole expansion as
implemented in `tensorbem.analytic.hulme_hemisphere_radiation`, with term ladder
[48, 96, 144] ensuring truncation error < 10⁻⁵.

## 2. Method

Three independent routes are compared:

1. **Reference (Hulme multipole):** Semi-analytical spherical-harmonic expansion,
   with excitation forces derived from the origin Kochin amplitude via Haskind reciprocity.
2. **TensorBEM BEM:** `solve_diffraction_3d` — constant-strength panel method with
   Kelvin (Havelock) free-surface Green's function and CHIEF interior point suppression.
3. **Capytaine (third-party):** Foreign Rankine-panel BEM, scattered force only,
   completed by exact Froude-Krylov quadrature.

### Verification Gates

| Gate | Criterion | Target | Measured | Verdict |
|---|---|---|---|---|
| G1: Complex relative error | \|F-F_ref\|/\|F_ref\| | 5% | 0.54% | PASS |
| G2: Phase difference | degrees | 3° | 0.05° | PASS |
| G3: Mesh convergence | observed order | ≥0.8 | 1.88 | PASS |
| G4: Independent route | capytaine+FK vs ref | 5% | 4.30% | PASS |
| G5: Long-wave limit | ka→0 extrapolation | 1% | 0.008% | PASS |
| G6: Haskind reciprocity | radiation vs diffraction | 2% | 0.83% | PASS |

## 3. Parameters

| Parameter | Value |
|---|---|
| Density ρ | 1025 kg/m³ |
| Gravity g | 9.80665 m/s² |
| Radius a | 1.0 m |
| Wave amplitude A | 1.0 m |
| Wave angle β | 0° (head seas) |
| ka range (base) | 0.2 – 2.0 |
| ka range (extension) | 2.5 – 3.0 |
| Panel counts | 64, 256, 1024 |
| Reference terms | 48, 96, 144 |

## 4. Results: Reference vs BEM (Base Case, 1024 panels)

| ka | \|Fx\|_ref | \|Fx\|_BEM | err_surge | \|Fz\|_ref | \|Fz\|_BEM | err_heave |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 5874.1 | 5843.8 | +0.52% | 24311.7 | 24287.6 | +0.10% |
| 0.50 | 12924.4 | 12855.2 | +0.54% | 16941.2 | 16939.8 | +0.02% |
| 0.80 | 16754.6 | 16665.9 | +0.53% | 12384.3 | 12394.2 | +0.09% |
| 1.00 | 17297.8 | 17210.0 | +0.51% | 10253.5 | 10267.2 | +0.14% |
| 1.50 | 15049.5 | 14983.5 | +0.45% | 6732.1 | 6748.7 | +0.25% |
| 2.00 | 12038.2 | 11990.7 | +0.40% | 4670.2 | 4693.0 | +0.49% |

## 5. Band Extension (ka = 2.5, 3.0)

| ka | \|Fx\|_ref | \|Fx\|_BEM | err_surge | \|Fz\|_ref | \|Fz\|_BEM | err_heave |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 9398.5 | 9361.2 | +0.40% | 3358.7 | 3382.1 | +0.70% |
| 3.0 | 7234.1 | 7208.6 | +0.35% | 2456.3 | 2489.7 | +1.36% |

## 6. Convergence Analysis

Observed convergence orders (error ~ h^p, h ~ N⁻¹/²):

| ka | p_surge (64→256) | p_surge (256→1024) | p_heave (64→256) | p_heave (256→1024) |
|---:|---:|---:|---:|---:|
| 0.2 | 1.94 | 1.99 | 2.02 | 1.99 |
| 0.5 | 1.93 | 1.98 | 2.16 | 1.98 |
| 0.8 | 1.92 | 1.98 | 1.88 | 2.00 |
| 1.0 | 1.92 | 1.98 | 1.94 | 2.01 |
| 1.5 | 1.92 | 1.97 | 2.02 | 2.04 |
| 2.0 | 1.93 | 1.98 | 1.90 | 1.97 |

**Expected order:** p ≈ 2.0 in h (constant-strength panels).
The observed orders are consistent with O(h²) convergence in potential,
translating to O(N⁻¹) in panel count for force quantities.

## 7. Error Assessment

### Base case (ka ∈ [0.2, 2.0])

- **Worst surge error:** 0.54% (at ka=0.5)
- **Worst heave error:** 0.49% (at ka=2.0)
- **All cases pass <5% tolerance:** ✓

### Band extension (ka ∈ [2.5, 3.0])

- **Worst surge error:** 0.40%
- **Worst heave error:** 1.36%
- **All cases pass <5% tolerance:** ✓

## 8. Conclusions

1. **TensorBEM BEM agrees with Hulme (1982) reference to <1% across ka ∈ [0.2, 2.0]**,
   well within the 5% ledger tolerance (9× margin).
2. **Convergence is monotonic with observed order p ≈ 1.9–2.0** (in h), consistent
   with constant-strength panel theory.
3. **Band extension to ka = 3.0 confirms agreement holds at higher frequencies**,
   though errors increase slightly as wavelength approaches body size.
4. **All six verification gates PASS**, including independent route agreement
   (capytaine + exact FK), Haskind reciprocity, and long-wave limit.

**Overall verdict: PASS** — CASE-BEM-HULME-DIFF-VERIF is fully verified.

## 9. Artifacts

- `benchmarks/verified/casepool_bem_hulme_diff_verif/results.json` — Base case results (6 ka, 3 mesh levels)
- `benchmarks/verified/casepool_bem_hulme_diff_verif/band_extension.json` — Band extension (ka 2.5, 3.0)
- `benchmarks/verified/hulme_diffraction_3d/results_reduced.json` — Reduced verification results
- `benchmarks/verified/hulme_diffraction_3d/fig_hulme_comprehensive_magnitude.png` — Force magnitude comparison
- `benchmarks/verified/hulme_diffraction_3d/fig_hulme_comprehensive_error.png` — Relative error plot
- `benchmarks/verified/hulme_diffraction_3d/fig_hulme_comprehensive_convergence.png` — Convergence plot
- `tests/test_casepool_hulme_diff_verif.py` — 10 regression tests (all PASS)
- `tests/test_hulme_diffraction_verification.py` — 5 direct comparison tests (all PASS)
