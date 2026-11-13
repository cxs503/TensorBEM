# Independent-code verification of the Hulme floating-hemisphere anchor

*Anchor **CASE-BEM-HULME-HEMISPHERE** (case-pool entry BEM-HULME-HEMISPHERE, floating hemisphere, 3-D radiation and diffraction).  Every number below is rendered from `benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_excitation.json` and the anchor's own `results.json` by `independent_axis.render_document`; `tests/test_casepool_hulme_independent_axis.py` re-renders the document and fails when code, artifacts and prose disagree.*

**Verdict: PASS** -- the reference agrees with an independent BEM code to 3.36 % at worst over all six case quantities, and the two excitation-force components (previously unverified by any independent code) to 2.33 % at worst.

## 1. Why a second axis was needed

The anchor scores the panel solution against Hulme (1982)'s wave-free multipole series (`tensorbem.analytic.hulme_hemisphere_radiation`).  Both routes draw on `tensorbem.free_surface3d.wave_function_3d`: the BEM assembly from it takes the free-surface Green function, the analytic reference from it takes the incident-wave and Kochin machinery.  A defect inside that kernel would move both numbers the same way and leave the gate green.  Hulme's printed tables are behind a paywall and no digit of them is transcribed in this repository, so the only available adjudicator is a *third* code.

The existing cross-code study (`docs/capytaine_crosscheck.md`) covers **radiation only** at three frequencies, and its third axis for the hemisphere is again the analytic series.  Hulme's paper is about *wave forces*: the excitation components $|F_x|$ and $|F_z|$ -- two of the anchor's six scored quantities -- had no independent check at all until this study.

**Trap recorded for the next reader.**  In capytaine >= 3.0, `DiffractionResult.forces` holds the **diffraction force only**; the Froude-Krylov part must be added separately (`capytaine.bem.airy_waves.froude_krylov_force`).  Comparing the result object against a *total* exciting force silently drops the FK term: it is most of the heave force of this body and it turned the surge force into a 3.7x under-estimate during this study.  The producer adds the two parts explicitly and the artifact keeps them apart.

## 2. Method and provenance

| item | value |
|---|---|
| independent code | capytaine 3.1.dev0 (installed build without git metadata), loongarch64, Python 3.11.6 |
| body | surface-piercing hemisphere, $a$ = 1.0 m, flat face on the free surface |
| fluid | $\rho$ = 1025.0 kg/m³, $g$ = 9.80665 m/s², infinite depth |
| band $ka$ (= $\nu a$) | 0.2, 0.5, 0.8, 1, 1.5, 2 |
| capytaine reference mesh | 1600 panels (`mesh_sphere(radius=1).immersed_part()`, resolution (40, 80)) |
| anchor case mesh | 1024 panels (`hemisphere_hull_mesh(subdivisions=4)`) |
| mesh ladder (panels) | 256 -> 576 -> 1024 -> 1600 |
| geometry check | $V_{mesh}/\nabla - 1$ = -0.257 % ($\nabla$ = 2.094395 m³) |
| excitation convention | total force = diffraction part + Froude-Krylov part |
| anchor tolerance | 5 % per cell (pre-registered, unchanged) |
| independent-code band | same 5 %, applied to capytaine as denominator |

Normalisations are the anchor's: $A/(\rho\nabla)$, $B/(\rho\omega\nabla)$ with $\nabla = 2\pi a^3/3 = 2.094395$ m³ (exact, not the mesh volume), and $|F|/(\rho g A a^2)$ with wave amplitude $A$ = 1 m; time convention $e^{-i\omega t}$.

## 3. Three-way results

| $ka$ | quantity | capytaine | anchor reference | anchor solver | capytaine vs reference | solver vs capytaine |
|---:|---|---:|---:|---:|---:|---:|
| 0.2 | A11 | 0.56353 | 0.55153 | 0.54936 | +2.18% | -2.52% |
| 0.2 | B11 | 0.00838 | 0.00815 | 0.00807 | +2.75% | -3.61% |
| 0.2 | A33 | 0.80273 | 0.79383 | 0.79260 | +1.12% | -1.26% |
| 0.2 | B33 | 0.28157 | 0.27930 | 0.27873 | +0.81% | -1.01% |
| 0.2 | Fx | 0.58772 | 0.58438 | 0.58136 | +0.57% | -1.08% |
| 0.2 | Fz | 2.41274 | 2.41864 | 2.41624 | -0.24% | +0.14% |
| 0.5 | A11 | 0.65765 | 0.64390 | 0.64091 | +2.14% | -2.55% |
| 0.5 | B11 | 0.10141 | 0.09867 | 0.09770 | +2.78% | -3.67% |
| 0.5 | A33 | 0.59306 | 0.58608 | 0.58544 | +1.19% | -1.28% |
| 0.5 | B33 | 0.34058 | 0.33906 | 0.33896 | +0.45% | -0.47% |
| 0.5 | Fx | 1.29314 | 1.28578 | 1.27889 | +0.57% | -1.10% |
| 0.5 | Fz | 1.67835 | 1.68539 | 1.68525 | -0.42% | +0.41% |
| 0.8 | A11 | 0.65428 | 0.64212 | 0.63939 | +1.90% | -2.28% |
| 0.8 | B11 | 0.27184 | 0.26531 | 0.26276 | +2.46% | -3.34% |
| 0.8 | A33 | 0.47623 | 0.46979 | 0.46916 | +1.37% | -1.48% |
| 0.8 | B33 | 0.29027 | 0.28990 | 0.29036 | +0.13% | +0.03% |
| 0.8 | Fx | 1.67376 | 1.66682 | 1.65800 | +0.42% | -0.94% |
| 0.8 | Fz | 1.22499 | 1.23204 | 1.23304 | -0.57% | +0.66% |
| 1 | A11 | 0.58356 | 0.57395 | 0.57206 | +1.67% | -1.97% |
| 1 | B11 | 0.36097 | 0.35349 | 0.35030 | +2.12% | -2.96% |
| 1 | A33 | 0.43483 | 0.42844 | 0.42765 | +1.49% | -1.65% |
| 1 | B33 | 0.24815 | 0.24840 | 0.24915 | -0.10% | +0.40% |
| 1 | Fx | 1.72493 | 1.72087 | 1.71213 | +0.24% | -0.74% |
| 1 | Fz | 1.01308 | 1.02007 | 1.02143 | -0.69% | +0.82% |
| 1.5 | A11 | 0.37391 | 0.36832 | 0.36829 | +1.52% | -1.50% |
| 1.5 | B11 | 0.40669 | 0.40135 | 0.39843 | +1.33% | -2.03% |
| 1.5 | A33 | 0.39555 | 0.38905 | 0.38771 | +1.67% | -1.98% |
| 1.5 | B33 | 0.15907 | 0.16062 | 0.16174 | -0.97% | +1.68% |
| 1.5 | Fx | 1.49450 | 1.49719 | 1.49063 | -0.18% | -0.26% |
| 1.5 | Fz | 0.66226 | 0.66974 | 0.67139 | -1.12% | +1.38% |
| 2 | A11 | 0.25424 | 0.24928 | 0.24985 | +1.99% | -1.73% |
| 2 | B11 | 0.34538 | 0.34241 | 0.34043 | +0.87% | -1.43% |
| 2 | A33 | 0.39468 | 0.38838 | 0.38634 | +1.62% | -2.11% |
| 2 | B33 | 0.09960 | 0.10307 | 0.10425 | -3.36% | +4.67% |
| 2 | Fx | 1.19223 | 1.19762 | 1.19289 | -0.45% | +0.05% |
| 2 | Fz | 0.45377 | 0.46462 | 0.46688 | -2.33% | +2.89% |

Phases (degrees, $e^{-i\omega t}$):

| $ka$ | quantity | capytaine | anchor reference | anchor solver | capytaine vs reference (°) |
|---:|---|---:|---:|---:|---:|
| 0.2 | surge force phase | -89.708 | -89.714 | -89.715 | +0.006 |
| 0.2 | heave force phase | -2.824 | -2.793 | -2.790 | -0.031 |
| 0.5 | surge force phase | -86.937 | -87.004 | -87.019 | +0.067 |
| 0.5 | heave force phase | -12.767 | -12.639 | -12.627 | -0.128 |
| 0.8 | surge force phase | -82.851 | -82.999 | -83.038 | +0.148 |
| 0.8 | heave force phase | -25.434 | -25.185 | -25.163 | -0.248 |
| 1 | surge force phase | -81.593 | -81.757 | -81.806 | +0.163 |
| 1 | heave force phase | -34.629 | -34.289 | -34.258 | -0.340 |
| 1.5 | surge force phase | -87.792 | -87.842 | -87.887 | +0.050 |
| 1.5 | heave force phase | -59.181 | -58.538 | -58.509 | -0.643 |
| 2 | surge force phase | -104.062 | -103.921 | -103.915 | -0.141 |
| 2 | heave force phase | -85.416 | -84.126 | -84.129 | -1.290 |

## 4. Gate table

| gate | criterion | measured | pass |
|---|---|---:|:--:|
| `I1_excitation_independent_band` | |capytaine/reference - 1| < 5 % for every excitation-force cell (kx ka) -- the quantities no independent check covered | 2.33 % (Fz at ka = 2) | yes |
| `I2_all_quantities_independent_band` | |capytaine/reference - 1| < 5 % for all six quantities and every ka | 3.36 % (B33 at ka = 2) | yes |
| `I3_capytaine_mesh_converged` | worst of the last ladder step (1024 -> 1600 panels) below 1 %, so the frozen capytaine values are not themselves discretisation noise | 0.67 % (1024->1600 panels) | yes |
| `I4_geometry_precondition` | |V_mesh/nabla - 1| < 2 % for the capytaine reference mesh | 0.26 % | yes |

### Post-hoc observation (not a gate)

Measured **after** the run and therefore reported, not scored: the strictest reading of the anchor's own band -- solver against the independent code -- reaches 4.67 % (B33 at ka = 2).  No cell reaches the band.

The tightest cell is B33 at ka = 2, where the reference and the solver agree with each other (+1.15 %) and the independent code sits below both -- the signature the shared-kernel hypothesis predicts, so the anchor's own gate could not have excluded it.  The level sensitivity table of section 5 settles the question in the other direction: the same comparison at 1024 capytaine panels -- the level that matches the anchor's own case mesh -- would have given 5.38 %, i.e. just outside the band, while the frozen 1600-panel reference gives 4.67 %.  The margin at the top of the band is therefore set by the independent code's own discretisation, and the honest statement is that the anchor sits *at* the 5 % edge there, not that it clears it with room.

## 5. Convergence and level sensitivity

| ladder step (panels) | A11 | B11 | A33 | B33 | Fx | Fz | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256->576 | 1.78 % | 0.96 % | 0.78 % | 1.12 % | 0.85 % | 1.81 % | 1.81 % |
| 576->1024 | 0.88 % | 0.74 % | 0.49 % | 0.94 % | 0.34 % | 0.96 % | 0.96 % |
| 1024->1600 | 0.52 % | 0.53 % | 0.34 % | 0.67 % | 0.17 % | 0.58 % | 0.67 % |

The worst drift decays monotonically (1.81 % -> 0.96 % -> 0.67 % over the ladder), so the frozen reference is usable at the 1 % level.  Because the deviations themselves are of the same order, the same comparison is repeated at every level:

| capytaine panels | worst capytaine vs reference | worst solver vs capytaine |
|---:|---:|---:|
| 256 | 5.98 % | 7.58 % |
| 576 | 4.91 % | 6.37 % |
| 1024 | 4.01 % | 5.38 % |
| 1600 | 3.36 % | 4.67 % |

## 6. Error judgement against the platform standard

| quantity | worst capytaine vs reference | worst phase difference |
|---|---:|---:|
| A11 | +2.18% (ka = 0.2) | - |
| B11 | +2.78% (ka = 0.5) | - |
| A33 | +1.67% (ka = 1.5) | - |
| B33 | -3.36% (ka = 2) | - |
| Fx | +0.57% (ka = 0.5) | - |
| Fz | -2.33% (ka = 2) | - |
| force phase | - | -1.290° (heave force phase, ka = 2) |

**Is the error below 3 %?**  Not uniformly.  35 of 36 cells agree with the independent code to better than 3 %; the outliers are the 1 cell(s) at ka = 2 (worst 3.36 %, heave damping), and both excitation-force components stay inside 3 % at every ka.  Against the platform's own acceptance standard this anchor therefore holds its pre-registered < 5 % band on an independent axis, but it does **not** hold a 3 % band at the top of the frequency range.

## 7. Limitations

* Both codes solve the same linearised potential-flow problem, so agreement cannot speak to viscosity, wave breaking or nonlinearity; the axis only checks the numerical realisation of the model.
* capytaine's mesh is clipped at $z = 0$, so its wetted surface is 0.26 % smaller in volume than the exact hemisphere and its waterline is a polygon inscribed in the circle, whereas the anchor's mesh puts every waterline vertex exactly on $z = 0$.  Part of the residual spread at the top of the band is a meshing difference, not a model difference.
* The independent code is frozen in an artifact, not re-run in CI: a capytaine free of the FK/diffraction split would have to be re-frozen by the producer script before the numbers here could be trusted again.
* Only the deep-water, zero-speed case is covered; the anchor itself is a deep-water case.

## 8. What changed in the anchor's standing

1. The two excitation-force components -- the actual subject of Hulme (1982) -- now carry an independent check (previously: shared-kernel reference only).
2. The anchor's inference `solver within 1.15 % of the reference` is confirmed as *internal consistency*, and the independent axis puts the worst cell at 4.67 %; the 5 % band is corroborated (worst 3.36 % for the reference itself) but is marginal at ka = 2 for heave damping.
3. Recommendation for the anchor's write-up: quote the independent spread, not only the shared-kernel figure, whenever the excitation forces or the top of the band are discussed.
