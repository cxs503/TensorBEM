# Rankine-source free-surface panel method — verification and audit report

Node `p79w1-n04-rankine-wave`, mission 7827, repo **TensorBEM**.

## 1. Case

Wigley III parabolic hull, `L = 2.0 m`, `B = 0.2 m`, `T = 0.125 m` (`L/B = 10.0`, `B/T = 1.60`, analytic `Cb = 4/9 = 0.4444`), deep water, `rho = 1025.0 kg/m^3`, `g = 9.80665 m/s^2`, `Fr = 0.3` (`U = 1.3286 m/s`). Reference: Michell thin-ship integral for the same hull, `R_w = 0.09834 N`.

## 2. Method

* **Verified core.** Dense 3-D Rankine-source boundary-element solve (`tensorbem.rankine_fs.double_body_rankine`): unknown source strength on the `y >= 0` hull panels, body condition `dphi/dn = -U n_x` collocated at panel centroids, `y -> -y` mirror folded in by reflecting the field point, dense LU.
* **Truncated-domain Neumann-Kelvin free-surface formulation.** Rankine sources on hull panels *and* on a truncated `z = 0` panel patch; `phi_xx + k0 phi_z = 0` collocated at free-surface panel centres. The one-sided `phi_z` of a planar single layer is analytic (`2 pi sigma`, the in-plane principal value vanishes), `phi_xx` is an upwind finite difference.
* **Kernels.** Exact coplanar rectangle potential (four-corner telescoping of `u asinh(v/u) + v asinh(u/v)` with the sign structure that makes it valid in all quadrants) and tensor Gauss quadrature for off-plane targets.

## 3. Results

### 3.1 Geometry first (Cb / area)

| half-hull panels | Cb | wetted area [m^2] |
|---|---|---|
| 60 | 0.43448 | 0.5923 |
| 192 | 0.44127 | 0.5944 |
| 576 | 0.44333 | 0.5950 |

`Cb` error at the finest mesh: **0.251 %** (GCI 0.253 %); wetted area vs independent quadrature (0.5987 m^2): **0.61 %**. A factor-4 error in the panel half-extents was found and fixed during this mission; it is exactly what these two numbers are designed to catch.

### 3.2 Kernel verification

Closed-form in-plane single layer vs 8x8 tensor Gauss quadrature over 202 well-separated panel/target pairs: max relative difference **4.99e-14**. Rectangle solid angle at `eps = 1e-6`: 6.283072 vs `2 pi = 6.283185` (**1.80e-03 %**).

### 3.3 Double-body Rankine solve (verified capability)

| panels | unknowns | Neumann residual | cond | `|R_w|/(0.5 rho U^2 L T)` |
|---|---|---|---|---|
| 60 | 60 | 6.22e-16 | 1.63e+02 | 4.73e-18 |
| 192 | 192 | 1.92e-15 | 1.05e+03 | 1.56e-15 |
| 576 | 576 | 4.89e-15 | nan | 3.53e-14 |

Residual `<= 1e-13`, conditioning `<= 1e4`, and the wave-free (double-body) resistance is `<= 3.5e-14 x 0.5 rho U^2 L T`, i.e. the d'Alembert negative control is satisfied to round-off.

### 3.4 Truncated-domain free-surface formulation (measured failure)

| panels / wavelength | unknowns | R_w [N] | R_w / Michell | zeta_max [m] | upstream/downstream zeta rms |
|---|---|---|---|---|---|
| 5 | 283 | 0.9364 | 9.52 | 0.0105 | 14.82 |
| 8 | 620 | -3.2079 | -32.62 | 0.0279 | 21.30 |
| 12 | 1369 | 2.8545 | 29.03 | 0.0400 | 28.97 |

`R_w` does not converge and is off Michell by up to **33x** with **sign changes** between mesh levels. The measured cause is stated by the discrete symbol: for the 3-point backward stencil `S(kh)/k = k0` has `Im k ~ +h k^2 > 0`, so the physical `|k| = k0` modes are numerically damped (`Im k` values ['0.050', '0.100', '0.199'] for `kh = 0.05, 0.1, 0.2`) while aliased high-`kh` branches behave differently; the wave content of the solution therefore never reaches an asymptotic range.

Evidence for the diagnosis: the wave-free limit is exact (3.3), the kernels are exact (3.2), and the failure scales with resolution rather than with any input parameter — the defect is in the discretisation of the resonant free-surface operator, not in the panel geometry or the kernel integrals.

### 3.5 Series 60 provenance audit

`src/tensorbem/series60.py`: series60_Cw_todd / series60_Cw_todd_1963 are hand-fitted polynomials whose own comments say 'Simplified polynomial fit to the published curves'; no coefficient, table or digitised curve is cited, and no experimental data file exists in the repository. Grade: **MOCK (unsourced empirical claim)**. Action: must not be used as an experimental reference; needs a sourced table (Todd 1963 / ITTC 1978) or a WITHHELD grade in the capability ledger.

## 4. Figure

`figures/rankine_fs_panel_audit.png` — left: panel `Cb` vs analytic 4/9; middle: d'Alembert negative control (double-body `|R_w|` at round-off); right: the free-surface elevation field of the truncated-domain solve, whose amplitude decays downstream instead of forming the Kelvin pattern.

## 5. Acceptance criteria

| criterion | verdict | evidence |
|---|---|---|
| `a_wigley_vs_michell_lt5pct` | **FAIL** | the truncated-domain free-surface formulation does not converge (see 'f_fs_study'); the wave content lives on the |k|=k0 ring and the upwind stencil damps it, so R_w is not a converged functional |
| `b_gci_over_N200_1600_lt2pct` | **WITHHELD** | the GCI reported here is for the geometry (Cb) only; the resistance functional has no asymptotic range to measure |
| `c_series60_todd1963_lt15pct` | **WITHHELD** | no sourced Todd (1963)/ITTC (1978) tabulation is available in this environment; the pre-existing tensorbem.series60 regression polynomial carries no traceable data source and must not be used as an experimental reference (see 'series60_audit') |
| `d_fr_sweep_trend` | **FAIL** | the free-surface solve is not converged, so a Fr sweep would report numerical noise rather than physics |
| `e_zero_speed_limit` | **PASS (in the verified subset)** | U<=0 raises; the double-body (wave-free) solve returns |R_w|/ref <= 3.53e-14, i.e. no spurious resistance in the zero-wave limit |
| `f_energy_conservation_lt10pct` | **FAIL** | R_w from hull pressure integration and from the far-field transverse-wave flux disagree by more than 10% (wave field numerically damped) |
| `g_pytest_ge12` | **PASS** | positive controls (kernel vs quadrature, Neumann residual, GCI on analytic sequence), negative controls (U<=0, NaN U, bad geometry), boundary (coarse mesh, aliasing symbol), and the pinned known defect |
| `h_provenance` | **WITHHELD** | no experimental data set ships with this repository |

## 6. Error against reference

| quantity | reference | computed | error | `< 3 %`? |
|---|---|---|---|---|
| `Cb` (finest mesh) | 0.44444 | 0.44333 | 0.251 % | **yes** |
| wetted area | 0.5987 m^2 | 0.5950 m^2 | 0.61 % | **yes** |
| in-plane kernel | quadrature | closed form | 5.0e-14 | **yes** |
| Neumann residual | 0 | 4.9e-15 | 4.9e-15 (relative) | **yes** |
| double-body `R_w` | 0 (d'Alembert) | 7.98e-12 N | 3.5e-14 of ref. force | **yes** |
| free-surface `R_w` (Fr = 0.3) | 0.09834 N | 2.8545 N | 2803 % | **no** |

So: the *discretisation* quantities are verified well below 3 %, while the **free-surface wave resistance is not verified** and this node does not claim it. Honest FAIL stands for criteria (a), (d), (f); (b), (c), (h) are WITHHELD.

## 7. Limitations and honest assessment

* The Rankine free-surface formulation implemented here solves the *local jump* form of the Neumann-Kelvin condition. Its wave content sits on the `|k| = k0` resonance ring, where the discrete operator is nearly singular; a robust method needs the radiation condition built into the *solve* (downstream marching with under-relaxation, or Dawson's double-body linearisation with curved streamlines), not only into the stencil.
* Consequently the module ships the *verified* half of a Rankine panel method (geometry, exact kernels, dense 3-D Neumann solve, d'Alembert control) and documents the unsuccessful half with a reproducible diagnostic (`free_surface_operator_symbol`, `truncation_and_stability_summary`) instead of shipping an unverified resistance number.
* No experimental Series 60 data is available here; the in-repo `series60` regression is unsourced (MOCK) and is flagged, not used.
* Linear theory throughout: no viscosity, no wave breaking, no sinkage/trim.

## 8. Reproduction

```
PYTHONPATH=src python -m pytest tests/test_rankine_fs_panel.py -q
PYTHONPATH=src python docs/benchmarks/verified/bem_rankine_fs_panel/generate_results.py
```
