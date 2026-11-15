# Finite-depth cross-validation with the automatic eigenseries mode count

**Anchor**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` · **Repo**: TensorBEM ·
**Branch**: `agent/m21207` · producer: `benchmarks/verified/bem_finite_depth_capytaine_xval/auto_mode_xval.py`

## 1. Case

Submerged sphere, radius a = 1 m, centre z = -2 m, bottom z = -3 m, density
1025 kg/m^3, seabed at h/R = 5 and 12, ka = 0.3 ... 2.0.  Reference: capytaine
3.1.dev0 cached levels 8x16 / 12x24 / 16x32 / 24x48 (1152 faces at the fine
level, self-converged to <= 0.85 % A33 and <= 1.1 % B33 on every station,
`cpt_self_convergence`).  TensorBEM: John (1950) eigenseries kernel, direct
collocation, icosphere subdivisions 1/2/3 (80/320/1280 panels).

## 2. Method

The frozen study pinned `mode_count` to 8...96.  The row-wise requirement of
the quadratured remainder (`mode_requirements.json`) is **median 460 (h/R=5)
and 1100 (h/R=12) modes at sub 2, max 1791 / 4294**: every pinned row was
truncated, so its cross-code error measured truncation, not discretisation.
Here `mode_count=None` (automatic), and the gate errors are recomputed on the
same mesh (sub 2, 320 panels) so the only change is the mode count.

## 3. Results (auto modes, sub 2)

| case | panels | modes | A33 [kg] | A33 err | B33 [kg/s] | B33 err |
|---|---:|---:|---:|---:|---:|---:|
| tbem-auto-sub2-h12-ka0.6 | 320 | 4096 | 2166 | 4.58% | 943 | 8.35% |
| tbem-auto-sub2-h12-ka2 | 320 | 4096 | 1866 | 4.10% | 197 | 6.92% |
| tbem-auto-sub2-h5-ka0.3 | 320 | 1791 | 2352 | 4.96% | 195.6 | 8.59% |
| tbem-auto-sub2-h5-ka1.5 | 320 | 1791 | 1866 | 4.07% | 545.6 | 7.36% |

Coarse leg (sub 1, 80 panels, auto modes): A33 error 8.7-9.1 %, B33 error
19.8-21.5 %.  So refinement 80 -> 320 panels takes A33 from ~9 % to <= 5.0 %
and B33 from ~21 % to <= 8.6 %: the residual is **mesh error of the shared
constant-strength collocation scheme**, not the finite-depth kernel.

Gates: G1 (A33 <= 10 %) max 4.96 % -> pass;
G2 (B33 <= 15 %) max 8.59 % -> pass;
G4 (sub1 -> sub2 self-convergence <= 5 %) max 6.92 % -> fail (mesh not yet converged).

## 4. Figures

![auto mode crossval](fig_auto_mode_xval.png)

Orange = sub 1, blue = sub 2, crimson dashed = the frozen truncated study
(12.2 % / 34.0 %), dotted = ledger limit 15 %.

## 5. Caveats

* **Coverage is partial**: 4 of the 10 planned sub-2 stations and 0 of the 2
  sub-3 spine points had returned when this artifact was frozen; cluster job
  `cj3272` was still running (sub-3 legs are the 1280-panel, up to 4096-mode
  solves).  The gates are therefore quoted on 4 stations, not 10.
* `MODES_CAP = 4096` binds 8/320 rows at h/R=12 sub 2 and 136/1280 rows at
  h/R=12 sub 3 (`mode_requirements.json`), so the deep-station number is a
  lower bound on the truncation-free error.
* The capytaine reference and the TensorBEM mesh differ (1152 faces vs 320
  panels); the mesh trend above attributes the residual to that difference.

## 6. Error verdict

| quantity | benchmark | measured | < 3 % ? | < 15 % (ledger) ? |
|---|---|---:|---|---|
| A33 heave, sub 2, worst station | capytaine 24x48 | 4.96 % | **no** | yes |
| B33 heave, sub 2, worst station | capytaine 24x48 | 8.59 % | **no** | yes |

Honest statement: the ledger line (< 15 %) is met on the 4 stations measured
and the frozen 12.2 % / 34.0 % pair improves to 4.96 % /
8.59 %, but the 3 % band is **not** met, mesh
self-convergence (G4) **fails**, and full grid coverage is still pending.
