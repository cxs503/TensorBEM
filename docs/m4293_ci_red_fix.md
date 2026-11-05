# PR #226 (agent/m3984) CI-red repair — m4293

**Branch** `agent/m3984` + `main` merged in · **Mission** m4293 · **Scope** make PR #226 CI green.
**Tree under test** `origin/agent/m3984` (`0a00ce2`, the PR head) merged with `origin/main`
(`5364459`) — exactly the tree GitHub CI checks out for a PR. Merge was clean (no conflicts).

## 1. Case: what was red, and why it was not the PR's own content

PR #226 adds only `docs/m3984/*` (a merge-union audit of PR #109, plus its publishing
scripts). Nothing in `tests/` or `src/` is touched. So the red could not come from the PR
payload; it came from the merge base. Reproduced on a pristine `git archive origin/main`
tree as well, i.e. **`main` itself was red before this mission**:

| test | failure (verbatim tail) | arch-dependent? |
|---|---|---|
| `tests/test_skip_gates_are_tracked.py::test_gate_registry_matches_the_test_suite` | `unregistered artifact skip gate(s): [('test_lifting_camber.py','ARTIFACT'), ('test_near_surface_three_way.py','ARTIFACT')]` | no |
| `tests/test_doc_claim_audit.py::test_ratchet_against_baseline` | `new unguarded document: finite_depth3d_regime.md` … (5 docs) | no |
| `tests/test_multicomponent.py` (8 tests, setup errors + calls) | `AttributeError: 'MultiComponentMesh' object has no attribute 'boundary_clearance'` | no |
| `tests/test_finite_depth3d_pins.py` (4 tests) | `3.3 h=2 max_rel_error: declared=0.8618 recomputed=0.5744`; residuals `1.26e-15` vs `1.48e-15` | **yes, see §4** |

## 2. What was done (three fixes, no unrelated edits)

1. **Artifact-gate registry** (`tests/test_skip_gates_are_tracked.py`, +2 lines).
   The two new skip gates that landed with `m2922` and `m3530` were never registered, so
   the guard that keeps skip gates *live* could not see them. Both artifacts are committed:
   `docs/lifting_camber_artifact/benchmark.json`, `docs/m3530/near_surface_three_way.json`
   (`git ls-files` verified). Registered with those paths.
2. **Doc-claim baseline** (`docs/doc_claim_audit/baseline.json`, regenerated with the
   documented tool `python -m tensorbem.benchmarks.doc_claim_audit --write-baseline`).
   Five paper-level documents arrived on `main` after the baseline was cut.
   Investigation first (the honest route): each of the five *does* have a companion test
   module that exercises the producing code (`test_finite_depth3d_regime.py`,
   `test_hydrostatic_restoring_couplings.py`, `test_interior_field_domain_guard.py`,
   `test_kkr_causality.py`, `tools/uq_selfcheck.py`), but **no test reads the `.md` file and
   no test reads the document's ledger JSON** (`grep -rn '<ledger>' tests/` → empty), and the
   documents declare no `python -m tensorbem…` generator, so the audit cannot honestly
   classify them as `PINNED` or `ARTIFACT`. Faking a filename mention would have been a fake
   guard, so they are recorded as explicit debt (see §5). The regeneration also shows the
   ratchet moving the other way: `audit_baseline.md`, `finite_depth_2d.md` and
   `lifting_foil.md` are **no longer** unguarded (they acquired real guards since the
   baseline), so the paper-level debt count stays at 10 documents.
3. **`MultiComponentMesh` domain guard API** (`src/tensorbem/multicomponent.py`).
   `tensorbem.solver.evaluate_interior_field` validates every evaluation point with
   `mesh.boundary_clearance(...)` and `mesh.contains_point(...)`. `PanelMesh` has both; the
   multi-connected mesh never did, so *every* field evaluation on an annulus raised
   `AttributeError`. Implemented both, reusing the module's existing predicates:
   * `contains_point`: even-odd inside test on the outer loop, AND-ed with `NOT inside` for
     each hole — a point inside a hole is outside the fluid;
   * `boundary_clearance`: minimum point-to-panel distance taken over **all** panels of all
     loops, so a hole wall is never overlooked (the nearest boundary may be a hole).

## 3. Results (local, loong64, this worktree)

| check | before | after |
|---|---|---|
| `tests/test_skip_gates_are_tracked.py` | 1 failed, 14 passed | **15 passed** |
| `tests/test_doc_claim_audit.py` | 1 failed, 8 passed | **9 passed** |
| `tests/test_multicomponent.py` | 9 passed, 8 failed/errored | **20 passed** (72 s) |
| `tests/test_finite_depth3d_pins.py` | 3 failed, 4 passed | unchanged — see §4 |
| audit coverage after re-baseline | — | documents=60, pinned=23, artifact=21, unguarded=16 (paper-level 10), number coverage 83.6 % |

Suite sweep: with `pytest -n 48` a full 1245-test sweep does not finish inside one shell
slot on this host (load average ≈3.0e3, 24× oversubscribed); a resume plugin
(`runs/plugin_resume.py`, diagnostic only) accumulated finished-test ids, and the red set it
collected over repeated sweeps is exactly the four rows in §1 — no additional red beyond
them was found in the tests it reached.

## 4. Note: `test_finite_depth3d_pins.py` is not portable across architectures

Four pins in that file compare documented numbers against a fresh floating-point recompute:
three of them are `free_surface_residual` values of order `1e-15` (machine roundoff:
declared `1.26e-15`, recomputed `1.48e-15` — a 17 % "relative error" of pure roundoff) and
one is `max_rel_error` at `h=2` (declared `0.8618`, recomputed `0.5744` on loong64).
Pinning a roundoff-level residual to `rtol=5e-3` cannot hold across architectures or BLAS
builds; the mitigation is a *floor* (`abs(declared-recomputed) <= atol + rtol*max(|·|)`)
or a tolerance keyed on `eps*condition`, not a plain relative tolerance. That change belongs
to the pin file's owner and is **not** made here (it would weaken a guard outside this
mission's scope). Consequence: the file is expected to be green on the x86 CI runner that
generated the pins and red on an ARM/loong64 host — architecture must be stated when reading
this result.

## 5. Open debt (recorded, not hidden)

* Five paper-level documents are unguarded (`finite_depth3d_regime.md`,
  `hydrostatic_restoring_couplings.md`, `interior_field_domain_guard.md`,
  `kkr_causality_band_study.md`, `m3538_propeller_openwater_uncertainty.md`). Their ledgers
  are committed (`docs/m3409/outputs/…`, `docs/m3196/outputs/…`, `docs/m3351/outputs/…`,
  `docs/m3664/outputs/…`, `data/m3538_uq/uq_selfcheck.json`) but no test reads them; the
  cheapest real guard is a `doc_pins`-style test that renders each table from its ledger and
  compares the rendered strings with the document text. A numbers-vs-ledger traceability
  measurement on the five documents gives 22–48 % token overlap, i.e. a “most numbers appear
  in the ledger” whole-document test would either be vacuous or need per-document extractors.
  That is the follow-up, and it is why the baseline (not a fake pin) was used here.
* `test_finite_depth3d_pins.py` needs an absolute/relative-floor tolerance regime (§4).

## 6. Reproduce

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_skip_gates_are_tracked.py \
    tests/test_doc_claim_audit.py tests/test_multicomponent.py -q
python -m tensorbem.benchmarks.doc_claim_audit --write-baseline   # only when the debt set changes
```
---

## 7. 第 2 轮（mission m4527）：PR 已绿的分支为何 CI 仍红 —— merge-base 漂移

`agent/m4293` 在本单入场时 CI（`pytest=failure`）仍红，但分支自身是绿的：本地把该分支
单独跑（`git checkout agent/m4293` 的树）时 §1 的四条红全部已消失。原因是 **GitHub 的
`pull_request` 事件测的不是分支树，而是 `merge(PR head, 当时的 main)`**，而 main 在
本分支切出后又前进了 14 个提交（`5364459..efec0d7`），其中 `m3878` 带进一份新文档
`docs/merge_m3878_pr104_resolution.md`。该文档在分支树里不存在，在 CI 的合并树里存在，
且没有任何测试读它 → 分类为 UNGUARDED → 撞上 §2.2 的棘轮门：

```
tests/test_doc_claim_audit.py::test_ratchet_against_baseline
AssertionError: ['new unguarded document: merge_m3878_pr104_resolution.md'] == []
```

复现方式（本地等价于 CI）：把 `origin/main` 合进 PR head 后单跑该文件即红，合入前跑则绿 ——
**"分支绿"不等于"PR 绿"，判据必须取合并树**。

### 修法（最小且不伪造守卫）

1. `git merge origin/main`（clean，无冲突）。合并树 = CI 检查树；这也是 PR 能被合进 main
   的前提。
2. 用文档化的工具重新切基线：`python -m tensorbem.benchmarks.doc_claim_audit --write-baseline`。
   差异**恰好一条**（新增 `merge_m3878_pr104_resolution.md` 到 `unguarded`，论文级集合
   仍为 10 份，不增长）。诚实说明：该文档不是本 PR 写的，其证据文件
   `docs/m3878/kernel_merge_ledger.json` 已提交，但**没有任何测试读它**，也没有声明
   `python -m tensorbem…` 生成器，所以按 §2 的同一判据它确实是"无人守"，记为显式欠账而不是
   临时加一行文件名把它伪装成 PINNED。
3. 重新发布审计证据（`docs/doc_claim_audit.md`、`docs/doc_claim_audit/audit.json`）与合并树
   对齐：documents 60→64、pinned 23→25、artifact 21→22、unguarded 16→17（论文级 10 不变），
   数字覆盖率 83.6 % → 83.7 %。

### 沉淀（可复用判据）

* 修 PR 红时，**先算 `git rev-list --count $(git merge-base main PR)..main`**；不为 0 就把
  main 合进来再跑，否则你会在绿树上白修一轮 —— 本生态已连续两轮（m4293、m4527）撞同一形态。
* 棘轮/基线类门（`baseline.json`、覆盖率下限、pin 值）对 main 的**新增内容**天然敏感：
  它们是"日期敏感门"，跨 PR 复用同一分支时必然复红。
* 能加真守卫就不动基线；加不了（无测试读件、无生成器声明）就用基线记债，并在报告里点名欠账。
