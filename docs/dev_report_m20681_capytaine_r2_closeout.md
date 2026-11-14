# m20681 — XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE: the round-2 FAIL was a scoring defect, not a physics result

Anchor: `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE` (acceptance: cross-code deviation **< 15 %**).
Repo: TensorBEM. Branch: `agent/m20681`. Compute: job **cj2863** (node n315, 4 cores, 14.1 min,
succeeded) — the module's own producer `scripts/run_r2_study.py --jobs 4`.

## 1. What was wrong with the record that was already merged

`docs/audit_m19943_xval_bem_r2_crossval.md` graded the first submission L2 with
`gate_state = reopened`: the TensorBEM side had never been run. A later ticket produced
`benchmarks/verified/capytaine_crosscheck_r2/results.json`, and the anchor was left with the
verdict it recorded:

| recorded on `main` before this ticket | value |
|---|---|
| `comparison.verdict` | **FAIL** |
| `comparison.failures` | **137** entries |
| `comparison.worst_active_relative` | **98.60** (= 9859 %) |
| `test_capytaine_crosscheck_r2.py` | **1 red** (`test_deep_submergence_is_the_analytic_axis`) |
| `convergence` table | diagonal columns up to `1.3e+30` % |

Two independent defects, both measurable:

1. **Degenerate null normaliser.** `compare()` decided "is this entry a structural null" with
   `sqrt(|M_ii M_jj|)`. On `hemisphere_r1` / `sphere_r1_z-0.5` the rotation centre is the centre
   of curvature, so `(r - centre) x n = 0` pointwise and the whole rotational block has an exact
   target of zero — including its **diagonal**. The normaliser therefore collapsed onto the very
   residual under test (`scale = |M_roll,roll| = 0.0958`), the null test `|M_ij| <= 1 % scale`
   could never fire, and 137 floor-vs-floor ratios were scored as physics. The module even
   carried `invariant_null_dofs()` and `invariant_scoring()` for exactly this — but `compare()`
   never consulted them, and the headline `verdict` came from `compare()`.
2. **Mesh mismatch.** The comparison ran TensorBEM at **256 / 270** panels and compared it with
   capytaine's **finest** meshes (**784 / 1120**), so the "cross-code gap" mixed code
   disagreement with a 3–4× panel-density mismatch; the declared finest levels of round 1
   (1024 / 1026) were skipped for "compute budget".

## 2. What was changed (`src/tensorbem/benchmarks/capytaine_crosscheck_r2.py`)

* `compare()` classifies by **declared invariants**: an entry touching a null dof of
  `invariant_null_dofs(case)` is class `invariant`, bounded by `INVARIANT_RATIO * rho L^k`
  (`INVARIANT_RATIO = 1 %`, unchanged, pre-registered), and is **never** divided by a reference
  value that is itself a zero. Bands (`5 %` diagonal / `10 %` coupling) are untouched.
* `reference_scale()` — when `sqrt(|M_ii M_jj|)` is itself below the body-scale floor
  (`NULL_FLOOR * rho L^k`, the off-centre cases whose rotational block is real but small), the
  body scale is used and the fall-back is flagged per entry (`scale_degenerate`), instead of
  silently turning a floor into an O(1) deviation.
* `convergence()` and `render_figures()` apply the same exclusion (they had the same defect:
  the ladder printed `1e+30 %`).
* `deep_submergence()`: `worst_rotation_ratio` is now the worst over **all** depths, not only the
  deepest one (that was the red test: depth 3 gives a larger floor than depth 8; both are below
  the 1 % bound, so the verdict is unchanged — the *reported* number was wrong).
* `HEMISPHERE_LEVELS = (2, 3, 4)`, `SPHERE_LEVELS = (2, 3)` → the codes now meet at comparable
  density (1024 / 1026 panels vs 784 / 1120).
* New producer `scripts/run_r2_study.py`: per-case solves in `--jobs` processes with a cache,
  so `--assemble-only` re-renders the document and the figures without re-solving.
* The old headline number is **kept in the artifact** (`worst_relative_any_entry`), so the FAIL
  that was reported is not overwritten; it is now labelled for what it is.

## 3. Result (cj2863, finest levels)

| case | TB panels | capytaine panels | active | null | invariant | conflict | worst active | worst invariant ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hemisphere_r1 | 1024 | 784 | 18 | 36 | 162 | 0 | 5.82 % | 0.42 | FAIL |
| hemisphere_r1_cz-0.5 | 1024 | 784 | 54 | 96 | 66 | 0 | 5.82 % | 8.7e-04 | FAIL |
| sphere_r1_z-0.5 | 1026 | 1120 | 17 | 37 | 162 | 0 | 6.52 % | **1.52** | FAIL |
| sphere_r1_z-0.5_c0 | 1026 | 1120 | 53 | 97 | 66 | 0 | 6.71 % | 1.7e-02 | FAIL |

Failures went **137 → 21**, and every one of the 21 is attributable:

* **band exceedances (5 items)** — `hemisphere_r1(r1_cz-0.5) nu=2 damping heave,heave +5.82 %`
  (band 5 %), `sphere_r1_z-0.5 nu=2 added_mass surge,sway −6.49 / −6.52 %` (band 5 %);
* **exact-zero bound (16 items)** — `sphere_r1_z-0.5` damping `surge-pitch` / `sway-roll`
  residual 13.8–15.6 vs bound `10.25 = 1 % * rho L^4`, i.e. the residual of an exact zero is
  1.3–1.5 % of the body scale, 1.3–1.5x the pre-registered bound. capytaine itself reports
  `-1.3` on the same entries, so the difference is a floor, not a formulation error: the
  degeneracy audit shows the residual decaying monotonically (hemisphere 1.47e-2 → 4.22e-3 →
  1.10e-3 over 64/256/1024 panels).

Refinement ladder, worst **diagonal** deviation vs the capytaine reference:

| case | coarsest | middle | finest |
|---|---:|---:|---:|
| hemisphere_r1 | 9.81 % (64) | 4.63 % (256) | **3.30 % (1024)** |
| hemisphere_r1_cz-0.5 | 9.81 % (64) | 4.63 % (256) | **3.30 % (1024)** |
| sphere_r1_z-0.5 | 7.25 % (270) | — | **6.52 % (1026)** |
| sphere_r1_z-0.5_c0 | 8.90 % (270) | — | **6.71 % (1026)** |

Other axes: deep submergence **PASS** (`A_surge` vs Lamb −0.602 % at 1280 panels; `A_roll` floor
2.7e-05 of `A_surge`), degeneracy audit **PASS**, rotation-centre transfer **PASS**
(1.65e-16), symmetry **FAIL for `sphere_r1_z-0.5`** (8.41 % `A_surge` vs `A_sway`, i.e. the
UV-sphere mesh is anisotropic at this density — an honest, still-open item).

## 4. Verdict, honestly

`comparison.verdict` is still **FAIL**, and it is left as FAIL:

* against the **pre-registered** bands (5 % diagonal, 10 % coupling, 1 % `rho L^4` exact-zero
  bound) five diagonal entries and sixteen exact-zero entries are out of band. The bands were
  declared before the numbers and are **not** relaxed here;
* against the **owner acceptance standard** (< 15 %): the worst genuinely-active deviation is
  **6.71 %**, i.e. inside the standard; the residual floor of the exact zeros is 1.5 % of
  `rho L^4`, and the coarsest-mesh numbers were the ones that used to look like 9859 %.

So the substantive claim of this round can now be stated falsifiably:
**TensorBEM's full 6×6 radiation matrix agrees with capytaine to ≤ 6.71 % (worst active entry,
finest compared meshes), ≤ 3.30 % on the diagonal of the surface-piercing hemisphere; the
rotational block of the centre-on-curvature bodies is not scoreable by relative deviation at all
and stays at the discretisation floor (≤ 1.52 % of `rho L^4`, decaying under refinement).**

## 5. Limitations

* Same panels, different meshers: 1024 (TensorBEM UV sphere) vs 784 (capytaine panels) — the gap
  is an upper bound on code-to-code disagreement, not an equality test.
* `hemisphere_r1` and `sphere_r1_z-0.5` cannot falsify the rotational block by construction; that
  is what the two off-centre cases are for, and their rotational entries are scored.
* The 1 % `rho L^4` exact-zero bound is exceeded at every compared density on the sphere; the
  bound was calibrated on the coarsest ladder level, which this round shows was optimistic. A
  future round should register a **mesh-dependent** bound *before* running, not re-label this one.
* `sphere_r1_z-0.5` surge/sway anisotropy 8.41 % remains unexplained (mesh anisotropy most
  likely) and is the next verification target.

## 6. Tests

`python -m pytest tests/test_capytaine_crosscheck_r2.py -q` → **17 passed**. The suite now binds
(a) the classes partition every entry including `invariant`, (b) no declared null is ever scored
by relative deviation, (c) `compare` and the independent recomputation `invariant_scoring` agree
entry for entry, (d) every invariant bound equals `INVARIANT_RATIO * rho L^k`, (e) the deep floor
is the worst over all depths.