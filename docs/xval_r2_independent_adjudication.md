# Round-2 capytaine cross-validation - independent re-adjudication

Anchor: `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`; acceptance criterion: the owner 15 % band (`0.15`).

The module under audit is `tensorbem.benchmarks.capytaine_crosscheck_r2`
(artifact `results.json`, document `docs/capytaine_crosscheck_r2.md`).
Its own verdicts are **FAIL** on three of four axes; a previous progress
note read the same artifact as a closed cross-validation. This note
re-derives the numbers instead of quoting either side.

## 1. Reference provenance: is the referee real and repeatable?

`scripts/verify_capytaine_reference_r2.py` rebuilds each recorded mesh
from the artifact (asserting the recorded face count), re-solves all
six radiating dofs on the recorded frequency grid, and compares the
full 6x6 matrices with the stored reference.

| quantity | value |
|---|---:|
| capytaine version (recorded / installed) | 3.1.dev0 / 3.1.dev0 |
| version match | True |
| recorded git commit | None |
| pre-registered tolerance (relative, per entry) | 1e-08 |
| worst reproduction deviation | 5.343e-16 |
| verdict | PASS |

| case | re-meshed faces | recorded faces | worst deviation |
|---|---:|---:|---:|
| hemisphere_r1 | 784 | 784 | 5.34e-16 |
| hemisphere_r1_cz-0.5 | 784 | 784 | 5.34e-16 |
| sphere_r1_z-0.5 | 1120 | 1120 | 2.34e-16 |
| sphere_r1_z-0.5_c0 | 1120 | 1120 | 2.34e-16 |

So the stored reference **is** capytaine output: it reproduces to
machine precision from the recorded settings, and the mesh
specification in the artifact rebuilds to the recorded face count.
`metadata.git_commit` is `null` (the producer had no capytaine source
tree), which is the one provenance field the re-solve replaces: the
version string plus a bit-for-bit re-run is a stronger statement than
a commit hash of an unrecorded tree.

## 2. What was re-derived, and from what

Every entry was recomputed from the two **primary** artifacts
(`capytaine_ref_r2.json` referee matrices and `solves/<case>.json`
TensorBEM matrices), not from `results.json`'s digested entry list:

| case | TensorBEM faces | capytaine faces | entries |
|---|---:|---:|---:|
| hemisphere_r1 | 1024 | 784 | 216 |
| sphere_r1_z-0.5 | 1026 | 1120 | 216 |
| hemisphere_r1_cz-0.5 | 1024 | 784 | 216 |
| sphere_r1_z-0.5_c0 | 1026 | 1120 | 216 |

Policy (bands, null floor, invariant ratio, which dofs are exact zeros)
is echoed from the artifact / anchor module, deliberately not re-invented:
`{'tolerance_diagonal': 0.05, 'tolerance_offdiagonal': 0.1, 'null_floor': 0.01, 'invariant_ratio': 0.01}`.

## 3. Reproduction of the recorded statistics

| case | counts | worst active (recorded) | worst active (recomputed) | delta |
|---|---|---|---|---:|
| hemisphere_r1 | match | 5.818683% | 5.818683% | 0.0e+00 |
| sphere_r1_z-0.5 | match | 6.523207% | 6.523207% | 0.0e+00 |
| hemisphere_r1_cz-0.5 | match | 5.818683% | 5.818683% | 0.0e+00 |
| sphere_r1_z-0.5_c0 | match | 6.705166% | 6.705166% | 0.0e+00 |

The recomputed statistics equal the recorded ones, so the module's
verdicts are the arithmetic of the raw matrices, not a transcription
error - the disagreement is about the *criterion*, not about the numbers.

## 4. Ledger adjudication (15 % band)

| quantity | value |
|---|---:|
| physically scored (active) entries | 142 |
| null/non-null conflicts | 0 |
| active entries over the module self band (5 % / 10 %) | 8 |
| active entries over the **ledger 15 %** band | 0 |
| worst active relative deviation | 6.7052% |

The entries that miss the module's own bands (all of them inside the
owner 15 % band), for traceability:

| case | nu | quantity | entry | deviation | self band |
|---|---:|---|---|---:|---:|
| hemisphere_r1 | 2 | damping | A[heave,heave] | +5.82% | 5% |
| sphere_r1_z-0.5 | 2 | added_mass | A[surge,surge] | -6.49% | 5% |
| sphere_r1_z-0.5 | 2 | added_mass | A[sway,sway] | -6.52% | 5% |
| hemisphere_r1_cz-0.5 | 2 | damping | A[heave,heave] | +5.82% | 5% |
| sphere_r1_z-0.5_c0 | 2 | added_mass | A[surge,surge] | -6.49% | 5% |
| sphere_r1_z-0.5_c0 | 2 | added_mass | A[sway,sway] | -6.52% | 5% |
| sphere_r1_z-0.5_c0 | 2 | added_mass | A[roll,roll] | -6.71% | 5% |
| sphere_r1_z-0.5_c0 | 2 | added_mass | A[pitch,pitch] | -6.54% | 5% |

## 5. Attribution of the `invariant` failures (exact zeros)

| quantity | value |
|---|---:|
| exact-zero violations | 8 |
| largest TensorBEM zero-defect (fraction of `rho L**k`) | 1.3149% |
| largest capytaine zero-defect | 1.5175% |

* `sphere_r1_z-0.5`: 8 violation(s), maximum set by `capytaine`, TensorBEM/capytaine
  defect ratio 0.87

Exact-zero residual under mesh refinement (recorded degeneracy ladder):

| case | decay ratio per level step | finest residual ratio |
|---|---|---:|
| hemisphere_r1 | 3.47, 3.84 | 1.10e-03 |
| sphere_r1_z-0.5 | 3.36 | 2.81e-03 |

Panels quadruple per ladder step, so a residual decaying like `h**2`
falls by ~4 per step: the measured 3.4-3.8 is that behaviour. The exact
zero is not an identity on a *faceted* sphere - `(r - c) x n` vanishes
pointwise only on the smooth surface, and the panel centroid sits off the
normal through the centre - so a percent-level residual at the finest
round-2 mesh is the expected discretisation defect, shared by both codes.

## 6. Figures

`docs/fig_xval_r2_audit_heatmaps.png` - left: |stored - re-solved| per
entry of the referee (~1e-16 everywhere); right: TensorBEM vs capytaine
per entry, red boxes marking the exact-zero (invariant) entries.
`docs/fig_xval_r2_audit_convergence.png` - left: worst physically scored
deviation vs panel count with the 5 % and 15 % bands; right: exact-zero
residual vs refinement against an `h**2` guide.

## 7. Verdict and what it does not say

* `ledger_15pct`: **PASS**
* `module_self_band`: **FAIL**
* `invariant_class`: **FAIL**
* `reproduction`: **PASS**

Reading: the anchor's acceptance criterion is met - 142 physically
scored entries, none above 15 %, worst 6.71 %, zero null/non-null
conflicts over 864 classified entries. The module's stricter self bands
(5 % diagonal, 1 % of `rho L**k` on exact zeros) are **not** met, and
that is reported here rather than reframed: the 5 % failures are
discretisation differences on near-degenerate entries (both codes
converging to the same value from different meshes), and every 1 %
exact-zero violation is set by the *referee*, whose defect is 1.15x
TensorBEM's. Neither class is a defect of this solver, but neither is
evidence that TensorBEM is right to better than a few percent.

Limits of this note:

1. The bands and the exact-zero dof sets are declared inputs, not
   re-derived; the note verifies the arithmetic under them, not the
   choice of them.
2. Only capytaine's **default** solver settings were compared; a
   settings sweep (quadrature order, lid mesh) is not included, so the
   reference's own discretisation error is bounded only indirectly (by
   the h**2 decay and by its agreement with the analytic deep-
   submergence limit recorded in `results.json`).
3. Both bodies with the rotation centre at the sphere centre have an
   exactly zero rotational block, so the rotational terms are scored
   only on the two off-centre cases; the sphere's rotational block is
   never scored as physics here.
4. The 15 % band is coarse; passing it is not a 3 % statement.

