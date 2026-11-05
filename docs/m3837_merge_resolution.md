# PR #133 merge resolution — `agent/m3245` → `main` (TensorBEM)

Mission m3837. Merge base `4f5b73e`; sides `origin/main` `ecba798`, `origin/agent/m3245` `c3efb75`.
GitHub reported `mergeable=false`; the local three-way merge conflicted in two files.

## 1. What the two sides had built

Both lineages independently made tables 10–12 of `docs/marine_bem_benchmarks.md`
(Hulme floating hemisphere radiation/diffraction, interior-waterline CHIEF
regularisation, Wang/Linton submerged sphere) executable registry rows, with
different registration vocabularies:

| | `origin/main` | `agent/m3245` |
|---|---|---|
| registration | one literal row per documented cell | rows generated from the study dict |
| quantity vocabulary | `sub<k>_<q>`, `ka2.58_unregularized_<q>` | `panels<N>.<q>`, `ka2.58.unregularized.<q>` |
| tiers | every cell `fast`/`standard` | published cells `heavy` + one `standard` coarse guard row per family |
| sphere group name | `submerged_sphere` | `submerged_sphere_wave` |
| closure residuals | pinned at the measured value | registered as upper bounds |

Neither side is a superset, so an overlay in either direction drops cells.

## 2. Per-file resolution

| file | conflict | resolution |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | imports, `DECLARED_NUMBERS`, `_run_group` | semantic union (+473 lines): both registries kept, one dispatch |
| `tests/test_declared_numbers.py` | family set, both suites | both suites kept, each scoped to its own vocabulary |
| `docs/marine_bem_benchmarks.md` | auto-merged | verified: main's re-derived numbers **and** the PR #133 paragraph present |
| `docs/free_surface3d_registry.md`, `dev_report_m3245_*.md`, `docs/m3245/**` | branch additions | kept verbatim |

`declared_numbers.py`: `DECLARED_NUMBERS` = 56 generated rows + 62 per-cell
literal rows = 140 entries, 140 unique keys. `_run_group` runs **one** study and
flattens it into both vocabularies (`_study_rows` and `_flatten_levels` /
`_flatten_reference`); `submerged_sphere` and `submerged_sphere_wave` share one
dispatch, `hulme_chief` returns `_chief_rows` ∪ `_flatten_chief_rows`;
`radius`/`density`/`gravity` are passed explicitly.

`tests/test_declared_numbers.py`: family set lists both sphere names;
`entry.declared != 0.0` kept from main (table 11 quotes the negative B33');
count-style assertions scoped to their own vocabulary through
`_TABLE_DRIVEN_ROWS = _free_surface_3d_rows()`; two new guards pin the union
(`test_the_merge_kept_both_table_10_12_registration_vocabularies`,
`test_the_two_sphere_family_names_share_one_dispatch`).

## 3. Nothing was lost (machine-checked)

`artifacts/m3837/audit_union.py` re-imports both side versions from the conflict
stages plus the merged registry, maps each row onto `(family, panel level,
quantity)` and compares (`registry_union_audit.json`, `registry_union_table.md`):

| quantity | main | branch | merged |
|---|---|---|---|
| registry entries | 84 | 78 | 140 |
| table 10–12 cells | 56 | 41 | 58 |
| shared cells | — | — | 39 (35 identical, 4 pin-vs-bound) |
| cells only on one side | 17 | 2 | all kept |
| **cells lost** | **0** | **0** | — |

## 4. Test evidence (single-threaded BLAS; host at load ≈1000)

| run | result |
|---|---|
| `origin/main` baseline, `pytest tests/test_declared_numbers.py -q` | 14 passed, 3 skipped, 96.35 s |
| merged tree, same command | **17 passed, 4 skipped, 100.65 s** |
| merged tree, `TENSORBEM_HEAVY_TESTS=1 -k "3d_free_surface or merge_kept or sphere_family"` | 4 passed, 15.71 s (incl. `test_documented_3d_free_surface_tables_are_reproduced`) |

No `main` test regressed; the three extra passes are the branch's coarse-level
guard and the two union guards.

## 5. Registry recomputation

`artifacts/m3837/make_figures.py`: 71 table 10–12 rows in the default tiers
recomputed, verdict `PASS`, worst relative deviation 0.379 %
(`hulme_hemisphere_64_surge_damping_error`), 71/71 below 3 %. Figure:
`artifacts/m3837/declared_vs_measured.png`; numbers `union_metrics.json`.

## 6. Caveats

* Nearly the same cells are registered twice, on purpose, to keep both lineages'
  coverage and tests; a maintainer can retire one vocabulary together with its
  tests. Extra cost: one coarse study per family in `standard`, one `heavy`
  study when the heavy tier is enabled.
* `submerged_sphere` / `submerged_sphere_wave` remain two names for one
  benchmark; consolidating them would invalidate one side's registered keys.
* No solver code, benchmark module or documented number was changed.
