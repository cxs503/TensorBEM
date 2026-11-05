# PR #186（agent/m3744）CI 分诊与复现证据（m4283，第 1 轮）

**结论（honest）**：本轮**未完成修复**。已用 git 事实证明 PR #186 相对其基线 `origin/main` 只增加 1 个二进制文件
（`docs/m3744/figures/fig_merge_composition.png`），**不含任何代码/测试改动**；因此该 PR 的 `pytest=failure`
不可能由 PR 内容引入。在本地对「PR head ⊕ origin/main」合并树做复现时，失败同样出现在**纯 `origin/main` 基线**上，
指向「红由基线继承 + 复现环境（无 CI 前置证据步骤）差异」而非分支漂移。受本轮预算限制，失败清单未能在中断前完整落地，
故未提交任何代码级修复（不猜测式改测试）。

---

## 1. 事实面（可复算，命令与输出逐字）

```
$ git fetch origin agent/m3744 ; git rev-parse FETCH_HEAD
96c18480e027a0d3ae733b75b804fb7cc08a45df
$ git rev-parse origin/main
536445997b31e52982eb4d0055ec9280843463fe
$ git merge-base origin/main 96c1848
536445997b31e52982eb4d0055ec9280843463fe      # = origin/main 自身，即分支已并入当前 main
$ git diff --stat origin/main 96c1848
 docs/m3744/figures/fig_merge_composition.png | Bin 0 -> 55216 bytes
 1 file changed, 0 insertions(+), 0 deletions(-)
```

要点：PR head 已经是 `origin/main` 的**后代**（其 parent#2 即 `53644599`）。合并树与 `origin/main` 的差异仅一张 PNG，
**零代码、零测试、零文档（md）差异**。任何由 "分支改动" 导致的 pytest 红在本 PR 上不成立——
红要么来自基线 `main@5364459`，要么来自 PR 记录陈旧（早期 head 含 `docs/m3744/*.md` 与 `runs/*.log`，后被提交门滤除重写为 96c1848）。

## 2. 复现方法与工具链

* 工作树：`git worktree add ... origin/agent/m3744` → 合并 `origin/main`（**无冲突**，`git merge --no-edit` 干净）；
  基线对照树：`git worktree add ... origin/main`。
* 依赖：本地 `numpy/scipy/matplotlib 3.10.8` 齐备（CI 装 `.[dev,visualization]`，与本地等价）。
* 命令（两树同参，`--dist loadfile` 避免同文件并发夹具冲突）：
  `PYTHONPATH=$PWD/src python3 -u -m pytest -q -n 12 --dist loadfile -p no:cacheprovider -rf --tb=line`
* 日志：`runs/ci_repro_pr186.log`（合并树）、`runs/pytest_x16_main.log`（基线树）。

## 3. 计算结果（进度条即程序化计数，无手抄）

| 树 | 观测到的红/错标记 | 位置（进度百分比） | 备注 |
|---|---|---|---|
| PR head ⊕ main | ≈10 `F` + ≈5 `E` | 17%、23%、46%、69%、86% | 跑到 98% 后卡在 Ursell 多极慢块，被中断；`-rf` 汇总未产出 |
| `origin/main` 基线 | ≈5 `F` + 5 `E` | 40%、46%、63% | 同样在长尾处被中断 |

两树**同参同环境**都出现红，且合并树与基线树代码逐字节相同（§1），因此红是**基线属性**，
不是 PR #186 引入的漂移。`E` 簇（4–5 个连续 error）与 `artifacts/` 缺失（CI 在 pytest 前有 13 个 benchmark 步骤生成 `artifacts/*.json`，
本地裸检出没有）高度吻合，属**复现条件差异**，需先按 `.github/workflows/tests.yml` 的步序补跑证据生成步骤才算 CI 等价。

已单独验证**全绿**的门禁族（48 passed，31.9 s，与红无关）：
`tests/test_doc_claims.py tests/test_ci_workflow_contract.py tests/test_capability_evidence.py tests/test_doc_inventory.py`。

## 4. 结果图

本单为 CI 分诊（无新物理量、无新数值解），未生成场量云图/收敛曲线；以 §1 的命令-输出对照与 §3 的进度条计数代替，
满足「纯后端/文档任务无图须写明理由」的豁免条件。

## 5. 注意点（踩过的坑）

1. **并行跑出的 `E` 不等于代码红**：`artifacts/` 是 gitignore 且由 CI 前置步骤生成；裸检出上依赖它的夹具会成簇报错。
2. **`-n` + 无 `--dist loadfile`** 会让同文件用例争抢同一 artifact/缓存文件，制造假红；本仓历史（`docs/ci_red_triage_m3314.md`）已记录过同类假红。
3. **`pkill -f pytest` 会误杀自己**：`sh -c` 命令行本身包含匹配串，`pkill -f "<pytest 参数>"` 会连当前 shell 一起 SIGTERM（本轮两次长跑因此中断）。
4. **xdist 在 SIGINT/被杀时不打印 `-rf` 汇总**：中断即丢结果，取失败清单必须让控制器自然收尾，或改 `-p no:xdist` 串行分段跑。
5. 尾部 Ursell 多极用例单块 10 min 量级，串行全量约 40 min，短预算下应**先按文件分片**（每片 ≤100 s）拿清单。

## 6. 误差判定（是否 <3%）

本单无数值结果，**不适用**：无新物理量输出，故无对比基准，也无「是否 <3%」的结论；
唯一量化指标是「失败计数/位置」，见 §3，属**二值判定 + 计数**，不是精度量。

## 7. 未完成部分与下一步（可直接执行的判据）

1. 先做 CI 等价树：干净 `git archive` 检出 → 依次跑 `tests.yml` 的 13 个 benchmark 步骤 → 再 `pytest`（串行，与 CI 一致）。
2. 若仍红，用**分片**取清单：`for f in tests/*.py; timeout 100 pytest -q $f` 逐文件记 PASS/FAIL，得到精确失败集。
3. 拿到清单后按 `docs/ci_red_triage_m3314.md` 的处置口径分类（真实漂移 / 环境 / 断言过期），只改被证明有漂移的文件。
4. 修好后推回 `agent/m3744`，由 pr-gate 复核 CI 绿并合并。