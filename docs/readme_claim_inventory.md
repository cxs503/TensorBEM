# README capability-claim inventory: an executable gate for "what this package can do"

**Task** m3203 (verifier seat, self-started) · **branch** `agent/m3203` · **repository** TensorBEM
**Date** 2026-11-04 · **Artifacts** `outputs/verif_m3203/readme_claim_inventory.json`,
`outputs/verif_m3203/fig_claim_inventory.png`, `outputs/verif_m3203/build_report.py`
**Code** `src/tensorbem/benchmarks/doc_inventory.py` (gate), `tests/test_doc_inventory.py` (19 tests)

## 1. What was wrong

`README.md` is the entry point that humans *and agents* read before choosing work. Its three
capability sections (milestones, roadmap, numerical conventions) make 19 decidable statements:
16 "this is shipped" and 3 "this is still a gap". On 2026-11-04 three of them contradicted the
tree, and two of the contradictions were *inside the same file*:

| id | claim row | location | sentence found 2026-11-04 | resolution |
|---|---|---|---|---|
| D1 | `near_singular_field` | README.md (Numerical conventions) | “near-boundary near-singular quadrature is not implemented” | replaced by a description of the two near-surface entry points and their honest limits |
| D2 | `near_singular_field` | README.md (roadmap item 5) | “near-singular field quadrature remain” | removed from the remaining-gap list; the shipped rule is named |
| D3 | `finite_depth3d_kernel` | README.md (roadmap item 6) | “**Next, needs new physics:** finite-depth 3-D John eigenseries Green function” | gap narrowed to the BEM solver layer; the verified kernel is named |

D3 is the expensive one. The README announced the 3-D finite-depth John eigenseries kernel as
*"now implemented and verified"* in the milestone text and, 24 lines later, listed that same
Green function under *"Next, needs new physics"*. Anyone — human or agent — reading only the
roadmap would re-derive a kernel that already exists with 26 test gates behind it. Nothing
executable objected, because a sentence is not a test.

## 2. What was built

`tensorbem.benchmarks.doc_inventory` is a curated table of the 19 claims. Each row carries the
evidence that decides it:

* `modules` — importable module paths (a shipped claim must import; *an absence claim's module must not exist*);
* `test_files` — files that must exist **and contain `def test_`**, so "shipped" means "guarded";
* `docs` — report files that must exist;
* `probes` — numerical proofs, so a shipped claim carries a measured number and not only a filename;
* `phrase` (absence claims) — the gap sentence must still be in the README;
* `stale_phrases` — sentence fragments that must **not** reappear once a claim is shipped.

`python -m tensorbem.benchmarks.doc_inventory --strict` prints strict JSON and exits non-zero on
any drift; `--markdown` renders the table used below. The three defects are now regression pins:
re-introducing any of those sentences turns the gate red (three parametrised negative controls in
the test suite prove it).

Negative controls in `tests/test_doc_inventory.py`: re-introduced stale sentence, missing test
file, unimportable module, closed gap behind an absence claim, dropped absence sentence, and a
probe that cannot meet its limit — six independent ways for the gate to fail, all exercised.

## 3. Results

**Gate状态** — before: 17/19 PASS, 2 rows DRIFT
(`near_singular_field, finite_depth3d_kernel`), 4 stale sentences hit.
After the README correction: **19/19 PASS**, drifted
`[]`.

| claim | probe (measured quantity) | measured | limit | verdict |
|---|---|---|---|---|
| `near_singular_field` | `near_singular_clearance_1e3` | 1.886e-14 | < 1e-10 | PASS |
| `finite_depth3d_kernel` | `deep_water_limit_k0h77` | 2.908e-07 | < 1e-05 | PASS |
| `finite_depth3d_kernel` | `free_surface_residual` | 2.430e-12 | < 1e-10 | PASS |
| `declared_number_registry` | `registry_size` | 2.200e+01 | > 2e+01 | PASS |

The two finite-depth probes reproduce the numbers quoted in `docs/finite_depth3d.md`
(2.908e-7 at `k0*h = 76.8`, 2.430e-12 at `h = 40 m`) to better than 1e-4 relative from an
independent re-run; the near-singular probe reproduces the clearance-independent quadrature of
`docs/near_surface_field.md`.

### Inventory

| # | claim | status | verdict | evidence |
|---:|---|---|---|---|
| 1 | `exterior_neumann_2d` | shipped | PASS | module:tensorbem.solver; tests:tests/test_exterior_neumann_added_mass.py(9); tests:tests/test_exterior_solver.py(12) |
| 2 | `rigid_body_added_mass_3d` | shipped | PASS | module:tensorbem.solver3d; tests:tests/test_exterior_neumann_added_mass.py(9) |
| 3 | `ursell_2d_radiation` | shipped | PASS | module:tensorbem.free_surface_solver; module:tensorbem.benchmarks.ursell_cylinder; tests:tests/test_ursell_benchmark.py(15); tests:tests/test_ursell_reference_pinning.py(6); doc:docs/marine_bem_benchmarks.md |
| 4 | `dean_diffraction_haskind` | shipped | PASS | module:tensorbem.benchmarks.dean_diffraction; tests:tests/test_dean_diffraction.py(6) |
| 5 | `helmholtz_maccamy_fuchs` | shipped | PASS | module:tensorbem.helmholtz; module:tensorbem.benchmarks.maccamy_fuchs; tests:tests/test_helmholtz_maccamy_fuchs.py(6); tests:tests/test_doc_numbers_maccamy_fuchs.py(5) |
| 6 | `kelvin_kernel_3d` | shipped | PASS | module:tensorbem.free_surface3d; tests:tests/test_free_surface3d.py(4); tests:tests/test_free_surface_kernel.py(10) |
| 7 | `solver3d_6dof` | shipped | PASS | module:tensorbem.free_surface_solver3d; tests:tests/test_floating_hemisphere_3d.py(5); tests:tests/test_submerged_sphere_wave_3d.py(4) |
| 8 | `near_singular_field` | shipped | PASS | module:tensorbem.near_singular; module:tensorbem.solver3d; tests:tests/test_near_singular.py(10); tests:tests/test_near_surface_error_budget.py(8); doc:docs/near_surface_field.md; doc:docs/near_surface_error_budget.md; probe:near_singular_clearance_1e3=1.886e-14 < 1.0e-10 |
| 9 | `finite_depth3d_kernel` | shipped | PASS | module:tensorbem.finite_depth3d; tests:tests/test_finite_depth3d.py(16); doc:docs/finite_depth3d.md; probe:deep_water_limit_k0h77=2.908e-07 < 1.0e-05; probe:free_surface_residual=2.430e-12 < 1.0e-10 |
| 10 | `finite_depth_2d` | shipped | PASS | module:tensorbem.finite_depth; module:tensorbem.finite_depth_solver; tests:tests/test_finite_depth.py(14); tests:tests/test_finite_depth_solver.py(9); tests:tests/test_finite_depth_energy_convergence.py(8) |
| 11 | `symmetry3d` | shipped | PASS | module:tensorbem.symmetry3d; tests:tests/test_symmetry3d.py(14); doc:docs/symmetry3d.md |
| 12 | `hydrostatics_restoring` | shipped | PASS | module:tensorbem.hydrostatic; module:tensorbem.benchmarks.hydrostatic_benchmarks; tests:tests/test_hydrostatic_verification.py(15); tests:tests/test_waterplane_multi_loop.py(5); doc:docs/hydrostatic_verification.md |
| 13 | `rao_6dof` | shipped | PASS | module:tensorbem.rao; tests:tests/test_rao_verification.py(24); tests:tests/test_hydrostatic_rao.py(14) |
| 14 | `lifting_kutta_2d` | shipped | PASS | module:tensorbem.lifting; tests:tests/test_lifting_flow.py(12); tests:tests/test_lifting_foil.py(12); doc:docs/lifting_kutta_closure.md |
| 15 | `causality_kkr` | shipped | PASS | module:tensorbem.causality; tests:tests/test_causality.py(7) |
| 16 | `declared_number_registry` | shipped | PASS | module:tensorbem.benchmarks.declared_numbers; tests:tests/test_declared_numbers.py(15); probe:registry_size=2.200e+01 > 2.0e+01 |
| 17 | `finite_depth3d_bem_solver` | absent | PASS | stated-gap:3-D finite-depth *boundary element solver*; absent-module:tensorbem.finite_depth3d_solver |
| 18 | `forward_speed_rankine` | absent | PASS | stated-gap:Forward-speed Rankine-source wave resistance; absent-module:tensorbem.rankine; absent-module:tensorbem.wave_resistance |
| 19 | `lifting_3d` | absent | PASS | stated-gap:three-dimensional lifting wakes; absent-module:tensorbem.lifting3d |

## 4. Figures

![claim inventory](fig_claim_inventory.png)

*Left:* the four executable probes, measured error against its acceptance limit on a log axis —
every probe sits two to nine orders of magnitude below its limit, so the shipped claims are not
merely present, they meet the documented accuracy. *Right:* how the 19 claims decompose by status
and by evidence type, with the before/after pass count in the title.

## 5. Caveats

* The table is **curated, not scraped.** It covers every capability statement in the milestone,
  roadmap and numerical-convention sections whose truth value is decidable by evidence; it does
  not parse arbitrary prose, and a sentence no row mentions is unchecked. The row count is
  reported so that "what the README says" versus "what is pinned" stays visible.
* "Test file exists and contains tests" is a *presence* gate, not a coverage gate. The numerical
  content of those tests is guarded by the repo's own suites (e.g. `test_declared_numbers.py`,
  `test_finite_depth3d.py`), which this module deliberately does not duplicate.
* The absence claims are only as good as the module names they name(`tensorbem.finite_depth3d_solver`,
  `tensorbem.rankine`, `tensorbem.lifting3d`). A future implementation under a different name would
  close the gap without tripping the row — the reviewer must still edit the README, which is the
  point: the gate forces the sentence and the tree to move together.
* The probe comparison for the deep-water limit is only meaningful at *fixed physical geometry*.
  At `k0*R >> 1` the source and its mirror image nearly cancel, so the same comparison at rescaled
  points is cancellation-limited (measured 2e-1 relative deviation for the same two kernels when
  the geometry is rescaled at fixed `k0*h`); that is a property of the comparison, not of either
  kernel.

## 6. Error judgement

| key quantity | reference | measured | relative error | < 3%? |
|---|---|---|---|---|
| deep-water limit of the finite-depth kernel | Kelvin kernel `free_surface3d` (independent) | 2.908e-7 | vs documented 2.908e-7: < 1e-4 | ✅ |
| free-surface residual of the finite-depth kernel | `∂G/∂z = νG` on `z=0` | 2.430e-12 | vs documented 2.430e-12: < 1e-4 | ✅ |
| near-singular quadrature at 1e-3 panel sizes | converged instance of the same rule | 1.886e-14 | machine level | ✅ |
| README claims vs tree | 19 curated statements | 19 PASS / 0 DRIFT | 0 misstated claims after the fix (3 before) | ✅ |

No claim labelled "verified" here is a self-comparison: the deep-water probe uses a different
kernel module, the free-surface probe uses the boundary condition itself, and the near-singular
probe uses a converged instance of the *same* rule with different parameters (which measures
quadrature error only).

## 7. Reproduce

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.doc_inventory --strict
PYTHONPATH=src python -m tensorbem.benchmarks.doc_inventory --markdown
PYTHONPATH=src python -m pytest tests/test_doc_inventory.py -q
PYTHONPATH=src python outputs/verif_m3203/build_report.py    # regenerates §3-§4 and the figure
```
