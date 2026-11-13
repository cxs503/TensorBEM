# CASE-BEM-WIGLEY3-WAVE-RES --- Wigley III wave resistance vs Michell thin-ship theory

## 1. Case and reference

Parabolic Wigley III hull `y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)` with `L = 3.0 m`, `B = 0.3 m`, `T = 0.1875 m` (`L/B = 10.000`, `B/T = 1.600`), `rho = 1025.0 kg/m^3`, `g = 9.80665 m/s^2`, calm water, no sinkage/trim (thin-ship steady Neumann-Kelvin linearisation).

Reference: Michell (1898) far-field integral in the wave-angle form `Rw = (4 rho g^2/(pi U^2)) int_0^{pi/2} sec^3(theta) |A(theta)|^2 dtheta`, with the Kochin amplitude `A` of the centreplane sheet evaluated in closed form for this hull.  Form citations: Michell (1898) Phil. Mag. 45(5) 106-123; Tuck (1989) J. Austral. Math. Soc. Ser. B 30, 426-445.

**Reference-table status: WITHHELD.** A tabulated `Cw(Fr)` with page citation, stated `L/B/T` and stated `Cw` reference area is not available offline in this work tree (network egress blocked); it was not reconstructed from memory.  The case therefore verifies code against the published *theory*, which is a code-verification claim, not an experiment-validation claim.

### Geometry gate

| quantity | closed form | derived | rel error |
|---|---|---|---|
| L_over_B | 10 | 10 | 0.000e+00 |
| B_over_T | 1.6 | 1.6 | 1.388e-16 |
| Cb | 0.444444444444 | 0.444444444444 | 1.249e-16 |
| displaced_volume | 0.075 | 0.075 | 0.000e+00 |

Wetted area `S_w = 1.339116 m^2`; `L*T = 0.562500 m^2`.

## 2. Method and implementation

* reference: `Michell (1898) far field in the wave-angle form, closed-form Wigley amplitude, graded composite Gauss in theta (tensorbem.thin_ship_michell.wave_resistance_theta_form)`, levels [75, 150, 300, 600, 1200]
* route_kochin: `tensorbem.wigley.michell_wave_resistance_canonical` (production `n_quad = 256`)
* route_panels: `tensorbem.thin_ship_michell.wigley_panels + wave_resistance_from_amplitude`, densities [(20, 10), (40, 20), (80, 40), (160, 80)]
* repro: `PYTHONPATH=src python3 -m tensorbem.benchmarks.case_wigley3_wave_resistance --output benchmarks/verified/case_bem_wigley3_wave_res`
* environment: python 3.11.6, numpy 2.4.6, git 96f19f7ced7ba57e3e47c634615274d5811b36fd

## 3. Parameters

| Fr | U [m/s] | k0 = g/U^2 [1/m] | k0*T | U^2/(gL) |
|---|---|---|---|---|
| 0.20 | 1.084803 | 8.3333 | 1.5625 | 0.0400 |
| 0.25 | 1.356004 | 5.3333 | 1.0000 | 0.0625 |
| 0.30 | 1.627205 | 3.7037 | 0.6944 | 0.0900 |
| 0.35 | 1.898406 | 2.7211 | 0.5102 | 0.1225 |
| 0.40 | 2.169606 | 2.0833 | 0.3906 | 0.1600 |
| 0.45 | 2.440807 | 1.6461 | 0.3086 | 0.2025 |

## 4. Reference vs got

| Fr | Rw ref [N] | Rw kochin [N] | err | Rw panels(160x80) [N] | err | <5% |
|---|---|---|---|---|---|---|
| 0.20 | 7.168236e-01 | 7.165603e-01 | 0.0367 % | 7.167596e-01 | 0.0089 % | PASS |
| 0.25 | 1.342588e+00 | 1.342420e+00 | 0.0126 % | 1.342819e+00 | 0.0172 % | PASS |
| 0.30 | 3.891780e+00 | 3.891664e+00 | 0.0030 % | 3.891858e+00 | 0.0020 % | PASS |
| 0.35 | 3.086580e+00 | 3.086493e+00 | 0.0028 % | 3.086534e+00 | 0.0015 % | PASS |
| 0.40 | 8.832036e+00 | 8.831973e+00 | 0.0007 % | 8.832016e+00 | 0.0002 % | PASS |
| 0.45 | 1.698526e+01 | 1.698521e+01 | 0.0003 % | 1.698524e+01 | 0.0001 % | PASS |

Non-dimensional (two reference areas, both stated):

| Fr | Cw ref (L*T) | Cw kochin (L*T) | Cw ref (S_w) | Cw kochin (S_w) |
|---|---|---|---|---|
| 0.20 | 2.112973e-03 | 2.112197e-03 | 8.875613e-04 | 8.872353e-04 |
| 0.25 | 2.532821e-03 | 2.532503e-03 | 1.063920e-03 | 1.063786e-03 |
| 0.30 | 5.098560e-03 | 5.098406e-03 | 2.141667e-03 | 2.141603e-03 |
| 0.35 | 2.970866e-03 | 2.970783e-03 | 1.247922e-03 | 1.247887e-03 |
| 0.40 | 6.508524e-03 | 6.508478e-03 | 2.733927e-03 | 2.733908e-03 |
| 0.45 | 9.889833e-03 | 9.889803e-03 | 4.154257e-03 | 4.154245e-03 |

## 5. Convergence and sensitivity

| Fr | ref GCI | panel observed order | panel GCI | truncation drift |
|---|---|---|---|---|
| 0.20 | 1.545e-06 | 3.97 | 1.884e-05 | 8.027e-07 |
| 0.25 | 5.247e-06 | 7.05 | 3.901e-08 | 1.589e-06 |
| 0.30 | 5.198e-06 | n/a (non-monotone) | 8.987e-06 | 1.865e-06 |
| 0.35 | 6.614e-06 | n/a (non-monotone) | 1.204e-05 | 5.332e-06 |
| 0.40 | 8.545e-06 | 1.69 | 2.992e-06 | 3.754e-06 |
| 0.45 | 2.171e-06 | 1.86 | 1.791e-06 | 4.191e-06 |

Truncation probe (report-only): cutting the angular integral at `sec(theta) = 8` instead of 30 changes the reference by up to 9.702e-04 (relative), i.e. the tail beyond `lambda = 8` still carries that much of `Rw`; the `lambda = 30` truncation used by the production and panel routes is inside the gate.

Non-monotone panel triples (value signs crossing) are flagged in `panels_gci.monotone`; there the reported GCI is the fine-grid difference (an upper bound), not a fitted order.

Panel-route error vs reference (rel error, by density):

| Fr | 200 panels | 800 panels | 3200 panels | 12800 panels |
|---|---|---|---|---|
| 0.20 | 4.969e-02 | 3.622e-03 | 1.328e-04 | 8.916e-05 |
| 0.25 | 1.220e-02 | 7.190e-04 | 1.762e-04 | 1.721e-04 |
| 0.30 | 2.044e-03 | 8.523e-05 | 1.265e-05 | 1.984e-05 |
| 0.35 | 1.218e-03 | 2.435e-05 | 2.448e-05 | 1.485e-05 |
| 0.40 | 1.684e-04 | 2.466e-05 | 7.560e-06 | 2.248e-06 |
| 0.45 | 6.731e-06 | 1.870e-05 | 5.100e-06 | 1.346e-06 |

## 6. Failure modes, limits, friction

* **Legacy default is a defect, still shipped.** `wigley.michell_wave_resistance` (`docs/michell_unit_arbitration.md`) samples `k = alpha = k0*lam` and returns `kg s^2 m^-1`, low by a factor 10.3-24.4 over this grid. The registered artifact `benchmarks/verified/bem_wigley_wave_resistance/results.json` and several drivers still read it; this case deliberately does not.
* **Thick-ship panel BEM has no usable wave-resistance number yet** (`docs/m10003/REPORT.md`: near-field pressure integration noise-dominated, far-field prefactor uncalibrated, honest FAIL).  The discretised route used here is the *centreplane thin-ship sheet*, not the free-surface panel solve.
* **Sides convention**: the canonical Kochin `|H|^2` carries both hull sides (factor 4 on the one-side amplitude); the reference theta form uses the same convention, so the comparison is internally consistent, but an absolute `Cw` quoted elsewhere with a one-side amplitude would differ by that factor.
* **Absolute scale and sides convention are WITHHELD** (`docs/michell_unit_arbitration.md` section 6): both routes integrate the both-sides amplitude, so the factor-4 convention cancels in the comparison; a one-side convention would divide `Rw` by 4 in *both* columns, leaving this case's `<5%` verdict unchanged but every quoted absolute `Cw` different.
* Fr >= 0.45 is outside the regime where thin-ship theory is credible for `B/L = 0.1`; the case reports it but only the Fr = 0.20-0.40 band is claimed as the verified envelope.

### Criteria

| criterion | measured | limit | status |
|---|---|---|---|
| geometry_gate | 1.388e-16 | 1.000e-12 | PASS |
| kochin_vs_reference_resistance | 3.673e-04 | 5.000e-02 | PASS |
| panels_vs_reference_resistance | 1.721e-04 | 5.000e-02 | PASS |
| panels_800_vs_reference_resistance | 3.622e-03 | 5.000e-02 | PASS |
| reference_quadrature_gci | 8.545e-06 | 1.000e-03 | PASS |
| reference_truncation_drift | 5.332e-06 | 1.000e-03 | PASS |
| panels_gci | 1.884e-05 | 5.000e-02 | PASS |
| reference_truncation_probe_lam8 | 9.702e-04 | report-only | PASS |

**All passed: True**
