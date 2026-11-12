# Whose error is it? Three-axis attribution of the TensorBEM / capytaine cross-code deviation

*Every number below is rendered from `benchmarks/verified/capytaine_attribution/results.json` by `tensorbem.benchmarks.capytaine_attribution.render_document`; `tests/test_capytaine_attribution.py` fails if code, artefacts and prose disagree.*

**Verdict.** The cross-code FAIL stands unchanged (8.75 % worst deviation from capytaine against a 5 % band, `benchmarks/verified/capytaine_crosscheck/results.json`). Against the semi-analytical axis, TensorBEM's worst cell is 1.13 % on 1024 panels (**PASS**, band 5 %) and capytaine's worst is 3.66 % on 1296 panels (**PASS**). 12 of 12 cells are attributed `BOTH_WITHIN_BAND`: the cross-code gaps are two in-band one-sided errors pointing in opposite directions, not a solver defect. Measured on the 1024-panel TensorBEM mesh the same cross-code metric gives 4.97 % (band 5 %), i.e. the recorded FAIL belongs to the 256-panel configuration it was measured on, and refinement -- not a code change -- is what closes it.

## 1. Case and the three axes

| item | value |
|---|---|
| body | surface-piercing hemisphere, R = 1.0 m, displaced volume 2/3*pi*R^3 = 2.0944 m^3 |
| fluid | rho = 1025.0 kg/m^3, g = 9.80665 m/s^2, infinite depth |
| frequencies | nu = 0.5, nu = 1.0, nu = 2.0 (nu = omega^2/g, m^-1), surge and heave |
| axis 1 (semi-analytical) | Hulme (1982) multipole solution, `n_terms` = 96 / 144 / 192; truncation residual estimated from the series tail |
| axis 2 | TensorBEM `solve_radiation_3d`, 64 / 256 / 1024 panels |
| axis 3 | capytaine 3.1.dev0 (commit `c0516db5b0df`), frozen reference levels 36 / 144 / 400 / 784 / 1296 panels |

## 2. Method

1. **Analytic axis, graded.** `analytic_axis` evaluates Hulme's series at 96, 144, 192 terms, keeps the largest, and publishes the geometric-tail residual `d2*r/(1-r)` with `r = d2/d1`; worst relative residual over all 12 cells is 1.24e-05. The truth axis is therefore at least 8e+04 times sharper than the deviations it judges.
2. **Attribution rule fixed before the run** (module docstring): `BOTH_WITHIN_BAND` if `max(|TensorBEM-analytic|, |capytaine-analytic|) <= band`, else the code with the larger one-sided error is blamed; equal one-sided errors above the band resolve to `TENSORBEM` (the conservative choice for our own code). The band is inherited verbatim from the cross-code study and is not re-fitted.
3. **Both codes refined toward the same truth.** TensorBEM re-solved on 64 / 256 / 1024 panels; capytaine contributes all 5 levels frozen in `capytaine_ref.json`. The per-level worst residual against the analytic axis is the discriminating measurement: a solver defect does not shrink under refinement, a mesh resolution limit does.
4. **The FAIL is read, not re-derived.** `cross_code` is copied from the cross-check artifact so that this study can never quietly overturn it.

## 3. Results

### 3.1 Per-cell three-axis table (each code on its finest mesh)

| nu | dof | quantity | Hulme | TensorBEM | capytaine | TB vs ana | capy vs ana | TB vs capy | attribution |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 0.5 | surge | added mass | 1382.29 | 1375.88 | 1414.78 | -0.46% | +2.35% | -2.75% | BOTH_WITHIN_BAND |
| 0.5 | surge | damping | 469.04 | 464.41 | 483.26 | -0.99% | +3.03% | -3.90% | BOTH_WITHIN_BAND |
| 0.5 | heave | added mass | 1258.18 | 1256.81 | 1274.64 | -0.11% | +1.31% | -1.40% | BOTH_WITHIN_BAND |
| 0.5 | heave | damping | 1611.75 | 1611.29 | 1619.61 | -0.03% | +0.49% | -0.51% | BOTH_WITHIN_BAND |
| 1 | surge | added mass | 1232.13 | 1228.08 | 1254.89 | -0.33% | +1.85% | -2.14% | BOTH_WITHIN_BAND |
| 1 | surge | damping | 2376.39 | 2354.95 | 2431.08 | -0.90% | +2.30% | -3.13% | BOTH_WITHIN_BAND |
| 1 | heave | added mass | 919.74 | 918.07 | 934.83 | -0.18% | +1.64% | -1.79% | BOTH_WITHIN_BAND |
| 1 | heave | damping | 1669.92 | 1674.94 | 1668.21 | +0.30% | -0.10% | +0.40% | BOTH_WITHIN_BAND |
| 2 | surge | added mass | 535.14 | 536.36 | 547.04 | +0.23% | +2.22% | -1.95% | BOTH_WITHIN_BAND |
| 2 | surge | damping | 3255.40 | 3236.59 | 3286.11 | -0.58% | +0.94% | -1.51% | BOTH_WITHIN_BAND |
| 2 | heave | added mass | 833.75 | 829.35 | 848.40 | -0.53% | +1.76% | -2.25% | BOTH_WITHIN_BAND |
| 2 | heave | damping | 979.85 | 990.95 | 944.00 | +1.13% | -3.66% | +4.97% | BOTH_WITHIN_BAND |

### 3.2 Residual against the analytic axis vs mesh

| code | panels | worst \|code - Hulme\| |
|---|---:|---:|
| TensorBEM | 64 | 17.63 % |
| TensorBEM | 256 | 4.54 % |
| TensorBEM | 1024 | 1.13 % |
| capytaine | 36 | 13.16 % |
| capytaine | 144 | 7.08 % |
| capytaine | 400 | 5.45 % |
| capytaine | 784 | 4.43 % |
| capytaine | 1296 | 3.66 % |

TensorBEM's apparent order from the two finest levels is p = 1.00 (constant-strength panels: O(h), so 1.0 is the expected value).

### 3.3 The cells the recorded cross-code study fails on

The recorded failures (frozen in the cross-check artefact, evaluated here on the finest TensorBEM mesh):

| cell | TB vs ana | capy vs ana | TB vs capy | attribution |
|---|---:|---:|---:|---|
| nu=0.5 surge damping | -0.99% | +3.03% | -3.90% | BOTH_WITHIN_BAND |
| nu=0.5 heave damping | -0.03% | +0.49% | -0.51% | BOTH_WITHIN_BAND |
| nu=1 surge damping | -0.90% | +2.30% | -3.13% | BOTH_WITHIN_BAND |
| nu=2 heave damping | +1.13% | -3.66% | +4.97% | BOTH_WITHIN_BAND |

4 of the 4 recorded failure cells were matched against the analytic axis. In each of them the two one-sided errors have **opposite sign** and each is inside the band, so their difference can exceed it: that is the arithmetic of a two-code comparison rather than a defect of either code. The residual that survives on the finest meshes is dominated by capytaine (3.66 %) rather than by TensorBEM (1.13 %).

At 1024 panels the cross-code metric itself falls to 4.97 %, inside the 5 % band -- the FAIL is a resolution limit of the mesh it was measured on. It is nevertheless **not** rewritten: what was recorded stays recorded, and the refinement evidence is published beside it.

### 3.4 Mesh geometry of the hemisphere

| code | panels | volume / (2/3 pi R^3) - 1 |
|---|---:|---:|
| TensorBEM level 2 | 64 | -8.858% |
| TensorBEM level 3 | 256 | -2.320% |
| TensorBEM level 4 | 1024 | -0.587% |

The coarsest mesh is off by 8.86% in displaced volume, which is why the geometry is graded before any hydrodynamic number from that level is used; the 256-panel level used by the cross-code study is off by 2.32% and the 1024-panel level by 0.59%.

## 4. Figures

* `docs/fig_capytaine_attribution_residual_vs_mesh.png` -- left: worst residual vs analytic for both codes as the mesh is refined, with the band and the analytic truncation residual; right: per-cell bars of the two one-sided errors against the cross-code difference that produced the FAIL.
* `docs/fig_capytaine_attribution_cells.png` -- residual against the semi-analytic truth, including the analytic axis's own truncation floor.

## 5. Caveats

* The analytic axis exists only for the hemisphere. The floating-sphere case of the cross-check has no analytic value, so no attribution is claimed for it here -- the cross-code FAIL on that cell is left unresolved rather than explained away.
* Hulme's series converges slowly (the tail ratio is reported per cell in `results.json`); the residual bound assumes a geometric tail, which the three term counts support but do not prove. This is stated as an assumption, not a theorem.
* Attribution says *which* code deviates from the analytic truth in a given cell; it does not certify either code for geometries outside the hemisphere, nor does it address sharp-corner bodies (the barge negative control of the cross-code study remains unresolved).
* TensorBEM is refined to 1024 panels here; extrapolating the observed first-order rate suggests the residual would fall below 0.5 % at ~2500 panels, which is not run (cost), so that statement is a trend, not a measurement.

## 6. Error verdict

* cross-code (unchanged): worst 8.75 % -> **FAIL** at band 5 %.
* TensorBEM vs analytic: worst 1.13 % -> **PASS** at band 5 %.
* capytaine vs analytic: worst 3.66 % -> **PASS** at band 5 %.
* cells blamed on TensorBEM: 0; on the reference: 0; inconclusive: 0.

**Is the key quantity within 3 %?** Yes -- worst cell 1.13 % on 1024 panels. The cross-code number, which is the one the FAIL quotes, is 8.75 % and is *not* a TensorBEM error; both statements are reported.

