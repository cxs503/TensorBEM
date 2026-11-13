# MacCamy-Fuchs Vertical Cylinder Wave Diffraction — Verified Anchor

## 1. Case introduction

**Physical problem**: Linear Airy wave diffraction by a bottom-mounted,
surface-piercing vertical circular cylinder of radius *a* in water depth *h*.
The three-dimensional problem separates exactly in the vertical eigenmode,
reducing to a 2-D exterior Helmholtz boundary-value problem on the cylinder
cross-section.

**Analytic reference**: MacCamy & Fuchs (1954) closed-form solution in terms
of Bessel and Hankel functions:

    F_x = -i ρ g A a² √(g/k) · 2π J₁'(ka) - i Y₁'(ka) / (J₁'(ka) - i Y₁'(ka))
         × tanh(kh) / (k² h)    (horizontal force)

where k is the wavenumber from the dispersion relation ω² = gk tanh(kh).

**Parameters**:
| Parameter | Value |
|---|---|
| Cylinder radius *a* | 1.0 m |
| Water depth *h* | 3.0 m (*h/a* = 3.0) |
| Wave number *ka* | 1.0 (refinement), 0.1–3.0 (sweep) |
| Fluid density ρ | 1025 kg/m³ |
| Gravity *g* | 9.80665 m/s² |
| Wave amplitude *A* | 1.0 m |

## 2. Method

**Solver**: `tensorbem.helmholtz.solve_vertical_cylinder_diffraction` — 2-D
exterior Helmholtz BEM with constant panels on a regular-polygon cross-section,
augmented by Schenck's (1968) CHIEF interior null-field constraints at the
first axisymmetric (J₀) and first dipole (J₁) irregular frequencies.

**Mesh**: Regular polygon with N = 16, 32, 64, 128 flat panels on the cylinder
circumference. Panel size h = πa/N (approximately).

**Verification strategy** (three independent aspects):
1. **h-convergence** at ka=1.0: force, moment, surface potential vs panel count
2. **Regime sweep** ka=0.1–3.0: effective inertia coefficient C_M(ka) from
   Morison long-wave limit (C_M→2) to short-wave scattering
3. **Irregular frequency suppression**: CHIEF augmentation at J₀(ka)≈2.4048
   and J₁(ka)≈3.8317 zeros

## 3. Results

### 3.1 Mesh refinement (ka = 1.0, h/a = 3.0)

| Panels | h_max/a | Force error | Surface φ error | κ |
|---|---|---|---|---|
| 16 | 0.3902 | 8.714e-03 | 1.195e-02 | 3.02 |
| 32 | 0.1960 | 2.140e-03 | 2.939e-03 | 2.67 |
| 64 | 0.0981 | 5.305e-04 | 7.294e-04 | 2.51 |
| 128 | 0.0491 | 1.321e-04 | 1.817e-04 | 2.43 |

**Observed convergence orders** (force error):
[None, 2.039841212136512, 2.015729072433639, 2.006930874004945]

→ Clean O(h²) convergence confirmed; last observed order = 2.007.

### 3.2 Frequency regime sweep (64 panels)

| ka | C_M (BEM) | C_M (exact) | Force error | κ |
|---|---|---|---|---|
| 0.1 | 2.018387 | 2.019181 | 3.932e-04 | 2.15 |
| 0.5 | 2.005270 | 2.005632 | 1.805e-04 | 2.36 |
| 1.0 | 1.371912 | 1.371616 | 5.305e-04 | 2.51 |
| 1.5 | 0.842300 | 0.842042 | 1.126e-03 | 2.04 |
| 2.0 | 0.561009 | 0.560834 | 1.632e-03 | 2.38 |
| 3.0 | 0.307836 | 0.307734 | 2.578e-03 | 2.31 |

→ Maximum force error across the sweep: 2.578e-03 at ka=3.0.

### 3.3 Irregular frequency suppression

| ka | Resonance | κ (no CHIEF) | κ (CHIEF) | Force err (no CHIEF) | Force err (CHIEF) |
|---|---|---|---|---|---|
| 2.4068 | j01_axisymmetric_resonance | 1002.5 | 2.31 | 2.031e-03 | 2.020e-03 |
| 3.8348 | j11_dipole_resonance | 665.0 | 3.34 | 1.035e-01 | 2.807e-03 |

→ At the J₁ dipole resonance (ka≈3.83), the unaugmented BIE condition number
reaches 665 with 10.3% force error;
CHIEF augmentation restores κ to 3.34 with 2.807e-03 error.

## 4. Figures

![MacCamy-Fuchs convergence](figures/maccamy_fuchs_convergence.png)

Three-panel figure: (a) h-convergence of horizontal force with O(h²) reference;
(b) effective inertia coefficient C_M(ka) from Morison limit to short-wave;
(c) irregular frequency suppression by CHIEF at J₀ and J₁ resonances.

## 5. Notes and limitations

1. **2-D Helmholtz only**: This benchmark verifies the depth-uniform cylinder
   path (exact vertical separation + 2-D Helmholtz BEM). It does NOT exercise
   the full 3-D Kelvin Green function solver — that is covered by
   `floating_hemisphere` and `submerged_sphere_wave` benchmarks.
2. **Flat panels**: The regular polygon approximates the circular cross-section
   with flat panels; the geometric error is O(h²) and dominates the force error
   at coarse meshes.
3. **CHIEF points**: Only 2 CHIEF interior points are used (one for J₀, one for
   J₁ resonance). More irregular frequencies would require more CHIEF points.
4. **Single depth**: h/a = 3.0 is representative but not exhaustive; finite-depth
   effects become stronger at smaller h/a.

## 6. Error assessment vs analytic reference

| Quantity | BEM value | Exact value | Relative error | < 3%? |
|---|---|---|---|---|
| F_x at ka=1.0, 128 panels | 43101.9390 N | 43099.6637 N | 1.321e-04 | ✅ YES |
| Surface φ L2 at 128 panels | — | — | 1.817e-04 | ✅ YES |
| C_M at ka=0.1 (Morison limit) | 2.018387 | 2.0 | 1.839e-02 | ✅ YES |
| Max force err across ka=0.1–3.0 | — | — | 2.578e-03 | ✅ YES |

**All key quantities are well within the 3% acceptance threshold.**
The force error at the finest mesh is 0.013%, four orders of magnitude below
the 3% limit.

## Acceptance gates

| G1_force_convergence_128 | Horizontal force relative error at 128 panels (ka= | 1.3207e-04 | 0.0005 | ✅ PASS |
| G2_observed_order_force | Observed convergence order for force (last refinem | 2.0069e+00 | 1.9 | ✅ PASS |
| G3_surface_potential_128 | Surface potential relative L2 error at 128 panels | 1.8171e-04 | 0.0002 | ✅ PASS |
| G4_frequency_sweep_max_error | Max force error across ka=0.1..3.0 sweep (64 panel | 2.5779e-03 | 0.005 | ✅ PASS |
| G5_chief_condition_number_reduction | CHIEF kappa reduction factor at J1 irregular frequ | 1.9895e+02 | 100.0 | ✅ PASS |
| G6_morison_long_wave_limit | Effective inertia coefficient CM at ka=0.1 approac | 1.8387e-02 | 0.02 | ✅ PASS |

**Overall**: ✅ ALL GATES PASSED

## 7. Re-verification (m16170, reopen of closed PR #683)

The original PR #683 carrying these deliverables was closed by the owner on
2026-11-05; the same content reached `main` via merge #645 (m14713). To close
the loop, the anchor was re-verified against the current `main` on 2026-11-12:

- **Base**: `main` @ `7f8a746` (post-#691).
- **Method**: re-ran `build_maccamy_fuchs_anchor.py` end-to-end (4-level h-refinement,
  6-point frequency sweep, 2-resonance CHIEF study) and diffed the regenerated
  artifacts against the committed ones.
- **Result**: `results.json` and `gates.json` are **byte-identical** to the
  committed artifacts (sha256 below); all 6 acceptance gates PASS with the same
  measured values as §5.
- **Pinned hashes** (machine-checked by
  `tests/test_bem_maccamy_fuchs_verified.py::TestReverificationRecord`):
  - `results.json`: `2a436541be85b62f67c46ecba84c2ef3747b2ffcd860362dfdb303783ff8980b`
  - `gates.json`: `ca65d37753c1c3c06521befd29b440d9541568f9ac6a709e7dcd4a7ac0a0d52b`
- **Conclusion**: the anchor is stable on the current `main`; the deliverables
  of the closed PR #683 are present and reproducible. No numeric drift since
  merge #645.


