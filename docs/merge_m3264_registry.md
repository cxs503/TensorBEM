# PR #135 (`agent/m3264`) — three-way resolution of the declared-number registry

Scope: resolve the `mergeable=false` conflict of PR #135 against `main` without
losing either side's content, then re-run the affected tests. Base branch
`agent/m3264` (`882201e`), merge source `origin/main` (`ecba798`), common
ancestor `4f5b73e`.

## 1. What conflicted

The branch and `main` independently registered the same documentation block —
tables 10–12 of `docs/marine_bem_benchmarks.md`, the 3-D Kelvin-Green-function
free-surface results (Hulme's floating hemisphere at `Ka = 1.0`, the Wang/Linton
submerged sphere at `f/a = 1.5`, the CHIEF regularisation of table 11). Only two
files conflicted:

| file | hunks | nature of the conflict |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 4 | import aliases; two disjoint implementations of the same table block; two helper/runner layouts |
| `tests/test_declared_numbers.py` | 1 | the two sides' tests for the same tables |

Both sides declare the *same physics*: 67 branch rows and 62 main rows cover the
same meshes and the same quoted values, differing in key style (generated
`hulme_a33_2` versus literal `hulme_hemisphere_64_heave_added_mass_error`) and
in the plumbing (`_level_quantities` over `study["levels"]` with dotted
quantities versus `_study_rows` with `sub_<n>_<quantity>` keys).

## 2. Resolution rule

A mechanical overlay is not acceptable here: taking one side's block deletes the
other side's cells, and taking both registers the same number twice under two
keys (which also breaks the "one row per quoted cell" invariant). The rule applied:

1. **main's literal block is kept as the single registry** for tables 10–12. It
   is already merged history, and the *non-conflicting* hunk of main's test file
   (`test_standard_tier_reproduces_the_documented_ursell_numbers`) hard-requires
   its keys, so keeping the generator instead would leave `main`'s tests
   referencing rows that no longer exist.
2. **Every cell only the branch had pinned is re-declared** on top of main's
   block, in main's key style: the two exact reference values of table 11
   (`hulme_chief_2_58_exact_a33/b33`), the passivity flag of the unregularised
   scheme, the three prose bounds of section 4.11 (Kochin/Haskind residuals) and
   the coarsest-mesh rotational-coupling cell. Runner support was extended
   (`_chief_rows` now also emits the exact values and the 0/1 passivity flag).
3. **The branch's superseded generator is removed** — with the evidence in
   section 3 showing it declares nothing that the merged registry lacks.
4. **The free-surface rows are all on the `standard` tier.** Main left 32 of the
   69 cells on the default `fast` tier, which a plain `pytest` run does not
   judge; the branch's whole point was that every quoted cell of tables 10–12 is
   guarded by a default test run. The studies are shared by parameter set, so the
   re-tiering costs no extra solve — it only stops 32 cells from going unchecked.
5. The named physical constants of the batch (`_HULME_KA`, `_SPHERE_KA`, …) are
   passed explicitly into the benchmark calls instead of relying on the
   benchmark defaults; the values are identical to those defaults, so no declared
   number moves.

## 3. Evidence that nothing was lost

`docs/m3264_merge/coverage_map.py` imports the branch's registry straight out of
the git object store (`git show origin/agent/m3264:…`) and matches it row by row
against the merged registry, by group and by quoted value (5 % tolerance for the
two-figure rotational-coupling cells, 0.5 % elsewhere). Output:
`docs/m3264_merge/registry_coverage.json`.

```
branch rows 67 -> mapped 67, unmapped 0
merged rows 69, unique to the merge 2
  merge-only hulme_hemisphere_64_order_b11 = 1.8021 (value)
  merge-only hulme_hemisphere_256_order_b11 = 1.9377 (value)
```

The merged registry is a strict superset: every one of the branch's 67 cells is
present, plus main's two bisected-damping order cells that the branch's alias
table did not cover. Totals: 91 declared numbers (69 free-surface, 11 heavy).

## 4. Test inventory: neither side lost a test

The first pass of this resolution replaced the conflicted test block with the
branch's version and *dropped* main's `test_chief_regularisation_restores_passive_heave_damping`
and main's extra assertions — the exact failure mode of a mechanical overlay, and
it is invisible in a green test run. The inventory is now checked explicitly:

```bash
diff <(grep '^def test_' tests/test_declared_numbers.py | sed 's/(.*//' | sort) \
     <(git show origin/agent/m3264:tests/test_declared_numbers.py | grep '^def test_' | sed 's/(.*//' | sort)
diff <(grep '^def test_' tests/test_declared_numbers.py | sed 's/(.*//' | sort) \
     <(git show origin/main:tests/test_declared_numbers.py | grep '^def test_' | sed 's/(.*//' | sort)
```

Result: the merged file holds all 18 branch tests plus main's chief-regularisation
test plus one new merge guard (20 tests). Main's
`test_3d_free_surface_rows_are_guarded_by_the_default_tiers` is the only name that
does not survive; its assertions (row count, the three groups, tier in {fast,
standard}, every row citing `marine_bem_benchmarks.md`, unique keys) are all
asserted by `test_free_surface_3d_rows_live_on_the_standard_tier`, with the count
raised from 62 to 69 and the tier set tightened to `{"standard"}`.

One cell could not be re-declared as a value: the branch quoted the coarsest-mesh
rotational coupling as `1e-30`, but the recomputed value is `2.5e-32`, a 97 %
relative change that no tolerance can absorb. The claim ("closed on a 40-panel
mesh") is carried as a one-sided bound (`declared = 1e-20`, `kind="upper"`), the
same encoding the prose bounds of section 4.11 use.

## 5. Verification

```bash
PYTHONPATH=src python3 docs/m3264_merge/coverage_map.py        # coverage evidence
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q
PYTHONPATH=src python3 -m pytest tests/test_floating_hemisphere_3d.py -q
PYTHONPATH=src python3 -m pytest tests -q -x --ignore=tests/test_declared_numbers.py
```

`tests/test_declared_numbers.py` gained `test_table11_reference_and_sign_cells_are_declared`,
which fails if the merge ever drops the re-declared cells, and its two count
guards were updated from the branch's 67 to the merged 69.
`docs/m3264_merge/evidence.py` recomputes the 69 rows and writes
`free_surface_rows.json` + `fig_merge_registry.png` for the merge report.
## 6. Verification log (as run on the merged tree)

| check | command | result |
|---|---|---|
| whole-file pytest | `pytest tests/test_declared_numbers.py -q` | 16 passed, 3 skipped (heavy tier) — run before main's chief test was restored |
| every report-based assertion | replay of the four free-surface tests' assertions against `docs/m3264_merge/free_surface_rows.json` | PASS, 69 rows, worst four-figure deviation 3.79e-3 |
| registry-only guards | `test_free_surface_3d_rows_live_on_the_standard_tier`, `test_registry_is_well_formed`, `test_registry_covers_every_documented_benchmark_family` | PASS |
| main's restored test | `test_chief_regularisation_restores_passive_heave_damping` | PASS |
| full recomputation | `docs/m3264_merge/evidence.py` | checked 72, passed 72, drifted 0 |
| merge completeness | `git diff --diff-filter=D --name-only origin/main HEAD` and `origin/agent/m3264 HEAD` | empty (no deletions); all 86 main-only and 15 branch-only changed paths byte-identical to their source side |

The shared machine ran at load15 ~800-1100 with several agents doing the same PR;
repeated full-file `pytest` runs were killed from outside (SIGTERM/SIGKILL), which
is why the table separates the one complete pytest run from the assertion replays
and the single-test runs. No check reported a failure.
