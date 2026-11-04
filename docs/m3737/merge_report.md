# Merge report — agent/m3181 → main (TensorBEM PR #128, mission m3737)

Status: resolved locally, audited line-by-line, branch ref updated. `git push` is
left to the platform: the executor has no egress (`git fetch` from
`github.com/cxs503/TensorBEM` times out at 120 s), so the new tip is published as
`refs/heads/agent/m3181`.

| item | SHA / value |
|---|---|
| merge base | `4f5b73e73c4e731582f2b050328ec5fcf62b4a86` |
| main tip merged in | `b420b2ee1e98276f341a10eafe3fda2d80807756` (Merge #126 / m3150) |
| PR branch tip merged in | `55db3976dc283e9522334f2ab7309fc655f3ae27` (`origin/agent/m3181`) |
| union merge commit | `79b527c` |
| reconciliation commit | `c6615ec` |
| third-parent merge (see §6) | `0a6e10a` |
| agent/m3181 **old** SHA (remote-tracking) | `55db3976dc283e9522334f2ab7309fc655f3ae27` |
| agent/m3181 old local SHA (unpublished, stale) | `71c870ca87b80196c22a877d17c8a57c5ab52d00` → kept as `salvage/m3686-merge-attempt` |
| agent/m3181 **new** SHA | `0a6e10affbd0f1e20c440c44ff00becc95a355c7` |

`origin/main` and `origin/agent/m3181` are both ancestors of the new tip, so
`merge_governor`'s next cycle should see `mergeable = true`.

## 1. What conflicted

`git merge-tree --write-tree origin/main origin/agent/m3181` reports **two**
conflicted files plus one file that both sides touched but git merged silently:

| file | git | semantics |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 3 conflicts | both branches implement the same three 3-D free-surface studies with different group names, aliases, row builders — and one **colliding function name** |
| `tests/test_declared_numbers.py` | 2 conflicts | group-coverage set + a comment rewritten on both sides |
| `docs/marine_bem_benchmarks.md` | auto-merged | **silent semantic conflict**: m3150 re-quoted four cells that m3181 had quoted under another convention |

Both branches register the same physics, neither is a subset of the other:

| | main (m3150, PR #126) | agent/m3181 (PR #128) |
|---|---|---|
| aliases | `hemisphere_benchmark`, `sphere_benchmark` | `hulme_benchmark`, `wang_linton_benchmark` |
| registry groups | `hulme_hemisphere` 31, `hulme_chief` 6, `submerged_sphere` 25 | `hulme3d` 28, `hulme3d_chief` 9, `wang_linton3d` 26 |
| row construction | hand-written `DeclaredNumber` per cell inside the tuple literal | table-driven `_refinement_rows()` / `_free_surface_3d_rows()` |
| measurement | `_study_rows()` + `_chief_rows(rows)` | inline flattening in `_run_group`, `_chief_rows()` |
| table 10 `B11'` order | measured `h_mean` ratio → 1.8021 / 1.9377 | `log2(e0/e1)` → 1.62 / 1.88 |

## 2. Per-file resolution

### 2.1 `src/tensorbem/benchmarks/declared_numbers.py`

1. **imports** — keep both alias sets (4 aliases over 2 modules), with a comment
   naming which group family uses which pair.
2. **`DECLARED_NUMBERS` tail** — main's literal already closes the tuple with
   `)`; m3181 turns that same parenthesis into `) + _free_surface_3d_rows()`.
   Result: main's hand-written rows **plus** m3181's table-driven rows.
   Verified after import: **147 rows, 147 unique keys**, both families present
   (31/6/25 and 28/9/26).
3. **`_run_group()`** — all six group branches kept.
4. **`_chief_rows` collision** — both sides define it with incompatible
   signatures: m3181's `_chief_rows()` is a registry-row builder called at
   **import time** by `_free_surface_3d_rows()`; main's `_chief_rows(rows)` is
   the measurement flattener defined *later* in the file. Git merged both
   definitions without a marker, so the later one shadows the former and the
   merged module raises `TypeError` on import. Fix: m3181's builder renamed
   `_free_surface_3d_chief_rows()` (call site updated, original docstring
   sentence kept verbatim and extended with the reason). This is precisely the
   silent-loss class the mission warned about; it was only visible because the
   union was audited line by line.

### 2.2 `tests/test_declared_numbers.py`

* group-coverage set: **union** of both sides' groups (`hulme3d`,
  `hulme3d_chief`, `hulme_hemisphere`, `hulme_chief`, `submerged_sphere`,
  `wang_linton3d` + pre-existing ones).
* the "declared may be non-positive" comment: both sides rewrote it to say the
  same thing; one merged sentence carries both meanings (originals in
  `merge_union_allowlist.json`).
* **new test** `test_three_dimensional_b11_order_conventions_are_reconciled`:
  pins that both conventions' registrations survive, that they are not copies of
  each other, that `order_h = order_log2 · ln2 / ln(h_ratio)` reproduces main's
  values from the documented `h0/h1 = 1.87`, that the two-significant-digit
  rotational couplings sit inside their declared 2.5e-2 rounding band of the
  four-significant-digit rows, and that the document keeps both tables *and* the
  reconciliation block.

### 2.3 `docs/marine_bem_benchmarks.md` (git auto-merged; resolved semantically)

Main's m3150 correction paragraph (§4.10) and m3181's §4.12 are both kept, but
they stated different values for the same four cells:

| cell | m3181 (§4.12) | m3150 / main (§4.10–4.11) | cause |
|---|---:|---:|---|
| table 10 `B11'` order 16→64 | 1.62 | 1.8021 | order normalised by `log2` (ideal h halving) vs measured `h_mean` ratio |
| table 10 `B11'` order 64→256 | 1.88 | 1.9377 | same |
| table 12 rotational coupling, 80 panels | 2.2e-4 | 2.1510e-4 | 2 significant digits (2.5e-2 band) vs 4 significant digits |
| table 12 rotational coupling, 320 panels | 2.6e-5 | 2.6481e-5 | same |

Resolution: a generated (not typed) reconciliation block appended to §4.12 —
both values, both conventions, back-solved edge-length ratios 1.8647 / 1.9591
(increasing towards 2 as refinement requires, matching the m3150 note `h0/h1=1.87`),
the identity check, rotation-coupling deviations **+2.28 % / −1.82 %** (inside the
declared 2.5e-2 band) and a pointer to the new test. §4.12's conclusion is
rewritten from "the document needs no number change" to "**59 of 63 cells need no
change; the other 4 were re-quoted by m3150 under another convention**". No
number from either side was deleted; only the unconditional conclusion is
corrected (original sentence quoted above and in the allowlist).

## 3. No-silent-loss audit

`docs/m3737/verify_merge_union.py --base 4f5b73e --merged HEAD` →
`docs/m3737/artifacts/merge_union_audit.json`:

* **65 single-side files**: merged blob byte-identical to the branch that edited
  it — **0 mismatches**; 23 binary files compared by blob identity — 0 mismatches.
* **3 both-sides files**: every line either side added is present in the merged
  blob except **9** lines, each listed in `docs/m3737/merge_union_allowlist.json`
  with its reason (2 identifier-rename lines, 6 comment-rewrite lines, 1
  conclusion rewrite).
* **`clean_union: true`, 0 unexplained losses, audit exit code 0.**

## 4. Test evidence

| run | result |
|---|---|
| `tests/test_declared_numbers.py` (default tiers), union before reconciliation | **15 passed, 5 skipped** (heavy-gated), 236 s |
| `tests/test_declared_numbers.py -k "reconciled or well_formed or covers_every"` | **3 passed** |
| `tests/test_declared_numbers.py + tests/test_doc_claims.py + tests/test_benchmark_cli_contract.py` | log: `runs/pytest_merge_core.log` |
| registry import check | 147 rows / 147 unique keys / no duplicate top-level definitions (AST scan) |

Logs are kept in `runs/` and summarised in
`docs/m3737/artifacts/merge_report_evidence.json`.

## 5. Honest limitations

* The heavy tier (`TENSORBEM_HEAVY_TESTS=1`, ≈2 × 100 s of dense 3-D solves) was
  not rerun for this report; the claim is limited to the default tiers plus the
  import-time/row-count checks. The merge changes which rows exist, not any
  measurement code path.
* `git push` is not attempted (no egress). The tip is left on
  `refs/heads/agent/m3181` for the platform to publish; `merge_governor`
  re-verifies next cycle.
* The stale local tip `71c870c` (unpublished earlier attempt, `docs/m3686/**`)
  was merged in as a third parent so its artifacts are not orphaned (§6). Its
  version of the two conflicted files was superseded by this audited resolution:
  it carries 4 registry keys that exist in **neither** branch tip
  (`hulme_chief_2_58_exact_a33`, `hulme_chief_2_58_exact_b33`,
  `hulme_chief_2_58_chief_point_count`, `submerged_sphere_20_rotational_coupling`),
  i.e. that attempt's own additions — dropping them loses nothing from main or
  from PR #128.

## 6. Third-parent merge (`71c870c`)

The local ref `refs/heads/agent/m3181` pointed at `71c870c`, an **unpublished**
earlier merge of the same two branches (mission m3686,
`docs/m3686/merge_report.md`). It is not an ancestor of `origin/agent/m3181`, so
PR #128 never saw it. It was merged as an extra parent (`0a6e10a`) to avoid
orphaning its artifacts; its copies of the two conflicted files lost to this
resolution for the reason in §5, everything else (`docs/m3686/**`, figures,
logs) is preserved unchanged. The original ref is also kept as
`refs/heads/salvage/m3686-merge-attempt`.

## 7. Reproduction

```bash
git worktree add /tmp/wt-m3737 origin/main && cd /tmp/wt-m3737
git merge --no-ff origin/agent/m3181            # 2 conflicted files
python3 docs/m3737/resolve_conflicts.py         # per-file union + de-collision
python3 docs/m3737/reconcile_docs.py            # §4.12 convention alignment
python3 docs/m3737/add_reconciliation_test.py   # the pinning test
python3 docs/m3737/verify_merge_union.py --base 4f5b73e \
    --allowlist docs/m3737/merge_union_allowlist.json \
    --json docs/m3737/artifacts/merge_union_audit.json
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q
```

Figures: this is a merge/repository-hygiene mission — no field quantities are
computed, so no contour/curve figure is produced (the relevant numeric evidence
is the registry row census, the audit counters and the four-cell comparison table
above, all programmatically generated by the scripts in `docs/m3737/`).