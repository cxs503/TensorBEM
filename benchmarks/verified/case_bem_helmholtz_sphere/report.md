# CASE-BEM-HELMHOLTZ-SPHERE: Rigid Sphere Acoustic Scattering Verification Report

## 1. Problem Description and Reference

**Physical problem**: Acoustic scattering of a unit-amplitude plane wave by a rigid sphere of radius a = 1.0 m in sea water (c = 1500 m/s, rho = 1025 kg/m^3).

**Reference solution**: Mie series analytical solution

    sigma = (4*pi/k^2) * sum_n (2n+1) |A_n|^2
    A_n = -j_n'(ka) / h_n^{(1)'}(ka)

**Reference source**: Bowman, Senior & Uslenghi (1969), Electromagnetic and Acoustic Scattering by Simple Shapes; Johnson (1993), The Helmholtz Equation, Table 3.1, p.47.

**Tolerance**: < 6% relative error on scattering cross-section.

## 2. Method and Implementation

**Solver**: TensorBEM acoustic3d module -- 3D unbaffled exterior Helmholtz acoustic BEM.

**Key settings**:
- Kernel: G_k(r) = exp(ikr)/(4*pi*r), time factor exp(-i*omega*t)
- Double layer: Laplace part exact (signed solid angle), Helmholtz remainder O(k^2) quadrature
- Single layer: self-panel analytic integration (Duffy transform for near-singular)
- CHIEF points: 5 interior null-field points to remove irregular frequencies
- Mesh: icosphere subdivisions: sub=2 -> 320 panels, sub=3 -> 1280 panels
- Far-field integration: N_theta=64, N_phi=64 spherical Gaussian quadrature

## 3. Parameter Table

| Parameter | Value | Unit |
|-----------|-------|------|
| Sphere radius a | 1.0 | m |
| Sound speed c | 1500 | m/s |
| Density rho | 1025 | kg/m^3 |
| Incident direction | (1, 0, 0) | -- |
| ka range | 0.25 -- 5.0 | -- |
| Frequency range | 59.7 -- 1193.7 | Hz |
| Mesh resolutions | 320, 1280 panels | -- |
| Tolerance | 6% | -- |

Dimensionless numbers:

| ka | f [Hz] | lambda [m] | ka/2pi | Regime |
|-----|--------|------------|--------|--------|
| 0.25 | 59.7 | 25.1 | 0.040 | Rayleigh scattering |
| 0.50 | 119.4 | 12.6 | 0.080 | Transition |
| 1.00 | 238.7 | 6.28 | 0.159 | Resonance onset |
| 2.00 | 477.5 | 3.14 | 0.318 | Resonance |
| 3.00 | 716.2 | 2.09 | 0.477 | Optical onset |
| 4.00 | 954.9 | 1.57 | 0.637 | Optical |
| 5.00 | 1193.7 | 1.26 | 0.796 | Optical |

## 4. Comparison Table (ref vs got + error)

| ka | Panels | sigma_BEM [m^2] | sigma_Mie [m^2] | Error [%] | Verdict |
|-----|--------|-----------------|-----------------|-----------|---------|
| 0.25 | 320 | 0.008454 | 0.009005 | 6.119 | FAIL |
| 0.50 | 320 | 0.114832 | 0.121716 | 5.656 | PASS |
| 1.00 | 320 | 0.971037 | 1.010427 | 3.898 | PASS |
| 2.00 | 320 | 2.408908 | 2.455155 | 1.884 | PASS |
| 3.00 | 320 | 3.190582 | 3.246837 | 1.733 | PASS |
| 4.00 | 320 | 3.682558 | 3.748197 | 1.751 | PASS |
| 5.00 | 320 | 4.051102 | 4.094637 | 1.063 | PASS |
| 0.25 | 1280 | 0.008864 | 0.009005 | 1.572 | PASS |
| 0.50 | 1280 | 0.119952 | 0.121716 | 1.449 | PASS |
| 1.00 | 1280 | 1.000427 | 1.010427 | 0.990 | PASS |
| 2.00 | 1280 | 2.443080 | 2.455155 | 0.492 | PASS |
| 3.00 | 1280 | 3.224896 | 3.246837 | 0.676 | PASS |
| 4.00 | 1280 | 3.727362 | 3.748197 | 0.556 | PASS |
| 5.00 | 1280 | 4.082951 | 4.094637 | 0.285 | PASS |

**Summary**: 13 PASS, 1 FAIL / 14 total cases
**Max relative error**: 6.119%

## 5. Convergence and Parameter Sensitivity

### 5.1 Mesh Convergence

| ka | 320 panels error [%] | 1280 panels error [%] | Convergence ratio |
|-----|---------------------|----------------------|-------------------|
| 0.25 | 6.119 | 1.572 | 3.9x |
| 0.50 | 5.656 | 1.449 | 3.9x |
| 1.00 | 3.898 | 0.990 | 3.9x |
| 2.00 | 1.884 | 0.492 | 3.8x |
| 3.00 | 1.733 | 0.676 | 2.6x |
| 4.00 | 1.751 | 0.556 | 3.2x |
| 5.00 | 1.063 | 0.285 | 3.7x |

**Observations**:
- All ka values: 1280-panel mesh error < 2%, well within 6% tolerance
- 320-panel mesh: only ka=0.25 exceeds tolerance (6.12%), all others < 6%
- Convergence ratio ~3.7-3.9x, indicating geometric discretization error dominates

### 5.2 ka Sensitivity

- **Low ka (Rayleigh regime)**: sigma ~ (7/9)*pi*a^2*(ka)^4, BEM error dominated by flat-panel approximation of sphere geometry
- **Mid ka (resonance)**: error drops to 1-2%, BEM performs best
- **High ka (optical)**: error continues to decrease < 1%, but requires more panels to resolve shorter wavelengths

### 5.3 Far-field Directivity

- ka=0.5: Near-dipole pattern, high fore-aft symmetry
- ka=1.0: Emerging fore-aft asymmetry
- ka=3.0: Complex multi-lobe structure, enhanced forward scattering

## 6. Failure Modes and Artifacts

### 6.1 Failed Case

| ka | Panels | Error [%] | Failure reason |
|-----|--------|-----------|----------------|
| 0.25 | 320 | 6.12 | Low ka + coarse mesh: flat-panel approximation of sphere geometry dominates |

**Note**: This is a known geometric discretization effect. At ka=0.25, wavelength ~25m >> sphere diameter, scattering cross-section is very small (sigma ~ 0.009 m^2), and the 320-panel sphere approximation error (panel normal deviation) becomes the dominant error source. With 1280 panels, error drops to 1.57%, passing tolerance.

### 6.2 Ecosystem Friction

- **No new friction**: This case fully reuses existing TensorBEM acoustic3d module capability
- **Compute resource**: 1280-panel single-ka solve ~12-19s, far-field integration ~0.5s
- **Cluster queue**: Submitted to cluster but queue was busy (15+ jobs ahead), fell back to local run

### 6.3 Artifacts

| File | Description |
|------|-------------|
| results.json | Complete numerical results (14 cases) |
| fig_scattering_cross_section.png | BEM vs Mie scattering cross-section plot |
| fig_convergence.png | Mesh convergence plot |
| fig_far_field.png | Far-field directivity patterns |
| fig_error_vs_ka.png | Error vs ka relationship |
| run_study.py | Reproducible run script |
| compile_results.py | Results compilation and plotting |

### 6.4 Reproduction Command

======================================================================
CASE-BEM-HELMHOLTZ-SPHERE: Rigid Sphere Acoustic Scattering
======================================================================
Radius: 1.0 m
Tolerance: 6%
ka values: [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
Mesh refinements: subdivisions [2, 3]

  Running ka=0.25, sub=2... sigma_bem=0.008454, sigma_mie=0.009005, err=6.12% [FAIL]
  Running ka=0.50, sub=2... sigma_bem=0.114832, sigma_mie=0.121716, err=5.66% [PASS]
  Running ka=1.00, sub=2... sigma_bem=0.971037, sigma_mie=1.010427, err=3.90% [PASS]
  Running ka=2.00, sub=2... sigma_bem=2.408908, sigma_mie=2.455155, err=1.88% [PASS]
  Running ka=3.00, sub=2... sigma_bem=3.190582, sigma_mie=3.246837, err=1.73% [PASS]
  Running ka=4.00, sub=2... sigma_bem=3.682558, sigma_mie=3.748197, err=1.75% [PASS]
  Running ka=5.00, sub=2... sigma_bem=4.051102, sigma_mie=4.094637, err=1.06% [PASS]
  Running ka=0.25, sub=3... sigma_bem=0.008864, sigma_mie=0.009005, err=1.57% [PASS]
  Running ka=0.50, sub=3... sigma_bem=0.119952, sigma_mie=0.121716, err=1.45% [PASS]
  Running ka=1.00, sub=3... sigma_bem=1.000427, sigma_mie=1.010427, err=0.99% [PASS]
  Running ka=2.00, sub=3... sigma_bem=2.443080, sigma_mie=2.455155, err=0.49% [PASS]
  Running ka=3.00, sub=3... sigma_bem=3.224896, sigma_mie=3.246837, err=0.68% [PASS]
  Running ka=4.00, sub=3... sigma_bem=3.727362, sigma_mie=3.748197, err=0.56% [PASS]
  Running ka=5.00, sub=3... sigma_bem=4.082951, sigma_mie=4.094637, err=0.29% [PASS]

Results saved to /root/tp-agent/work/m19973/benchmarks/verified/case_bem_helmholtz_sphere/results.json

Summary: 13 PASS, 1 FAIL out of 14 cases
Max relative error: 6.119%
  Saved /root/tp-agent/work/m19973/benchmarks/verified/case_bem_helmholtz_sphere/fig_scattering_cross_section.png
  Saved /root/tp-agent/work/m19973/benchmarks/verified/case_bem_helmholtz_sphere/fig_convergence.png
  Saved /root/tp-agent/work/m19973/benchmarks/verified/case_bem_helmholtz_sphere/fig_far_field.png
Figures generated.
