# m23272 · XVAL-BEM-R2 退化审计文档：从手抄叙述改为工件算术（并逐字节 pin）

**仓库** TensorBEM · **分支** `agent/m23272` · **锚点** `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`
**前置** PR #1089（m20476 matured mission deliverables，owner 2026-11-05 裁定关闭；其内容已由
Merge #923 进入 main，现由重开单管到底）· **本文对应交付** `docs/audit_xval_r2_degeneracy.md`

## 0. 为什么要重做这一单：PR 的内容在 main 里已经**漂移**

重开单的文件清单描述的是 256 面板时代的 R2 退化审计。核对当前 main：

| 文件 | main 现状 | 结论 |
|---|---|---|
| `src/tensorbem/benchmarks/capytaine_crosscheck_r2.py` | 1770 行，含 `degeneracy_audit` / `invariant_scoring` / `centre_transfer_invariant` | 在（Merge #923） |
| `benchmarks/verified/capytaine_crosscheck_r2/{results,capytaine_ref_r2}.json` | 在，且已被后续单细化到 1024/1026 面板 | 在且已更新 |
| `docs/capytaine_crosscheck_r2.md` | 440 行，**有逐字节 pin**（`test_document_is_the_arithmetic_of_the_artifact`） | 在且受保护 |
| `docs/audit_xval_r2_degeneracy.md` | 在，但**无任何 pin**，数字仍停在 256 面板的旧算例 | **漂移（本单修的缺陷）** |
| `tests/test_capytaine_crosscheck_r2.py` | 25 项，全过 | 在 |

也就是说：功能与测试齐备，但审计文档描述的是**已不存在的算例**。它不是"少一个文件"，
而是仓库里一份**不可信证据**：读者会引用 `9.38% / 87.93% / 137 条失败`，而工件现在是
`5.82% / 6.52% / 6.71% / 8 条失败`。这比"PR 未合并"更糟——它把错误结论留在了正典里。

## 1. 缺陷的具体形态（可机检）

旧 `docs/audit_xval_r2_degeneracy.md` 与 `benchmarks/verified/capytaine_crosscheck_r2/results.json`
的逐项对账（均由脚本重算，非目测）：

| 文档声明 | 工件实际 | 状态 |
|---|---|---|
| `hemisphere_r1` 最差计分偏差 9.38%（256 面板） | 5.82%（1024 面板，`nu=2 damping A[heave,heave]`） | 失效 |
| `sphere_r1_z-0.5` 最差计分偏差 87.93% | 6.52%（1026 面板） | 失效 |
| `sphere_r1_z-0.5_c0` 87.93% | 6.71% | 失效 |
| "半球的 3.47x/3.84x 说明仍在收敛前段" | 3.47/3.84 正确 | 有效 |
| "恒等式残差 1.7e-16" | 1.10e-16（`centre_transfer.worst_residual`） | 失效（数值层面） |
| "第一轮复现 1.85e-12%" | 1.78e-14 相对 = 1.8e-12 %（工件量纲为相对值） | 未标量纲 |
| "精确零界限越界 4 / 12 格" | 0 / 0（工件 `invariant_violated = 0`） | 失效 |

根因：文档是**手写 + 手抄数字**，而工件在后续单里被细化（256→1024 面板）。文档没有
pin，所以漂移不会被任何测试拦住；对照之下 `docs/capytaine_crosscheck_r2.md` 因为
`render_document` + 逐字节 pin 一直是对的。

## 2. 改法（最小、可验证）

1. **文档变成工件的算术**：新增 `render_degeneracy_document(study)`，从 `results.json` 的
   `comparison / degeneracy / convergence / invariant_scoring / centre_transfer /
   ledger_acceptance / round1_reproduction` 生成 `docs/audit_xval_r2_degeneracy.md` 的
   **全部数字**（六节结构与主文档对齐：案例介绍 / 方法 / 计算 / 图 / 注意点 / 误差判定）。
   散文固定、数字注入，零手抄。
2. **落盘路径纳入 `write_outputs`**：新增 `AUDIT_DOC_PATH`，`write_outputs()` 现在同时写
   `results.json`、`docs/capytaine_crosscheck_r2.md`、`docs/audit_xval_r2_degeneracy.md` 与三张图，
   返回键增加 `audit_document`；因此任何一次重跑研究都会顺手把审计叙述刷新，不可能再留旧账。
3. **逐字节 pin + 算术 pin**：新增 `tests/test_capytaine_r2_audit_doc_pins.py`（9 项）：
   * 文档 == `render_degeneracy_document(study)`；
   * 六节标题齐备；
   * 每个算例的最差计分偏差（`%.2f%%`）与**每一条**失败项逐字出现在文档中；
   * 第 6 节物理行的百分比必须等于工件里的某个 `worst_physics_relative`（拦住"引用了
     已作废数字"这一类漂移）；
   * 第 3.3 节细化阶梯逐行等于 `convergence`，且阶梯确实单调（文档论点"离散差而非实现错"
     的证据链）；
   * 不变式轴（衰减比、最细残差）与 `degeneracy` / `centre_transfer` 逐项一致；
   * 内部带宽 FAIL 与 owner 15% 验收 PASS **分别**成文，不得互相替代；
   * 网格不对等的面板数被写出；
   * `write_outputs` 在临时目录里确实落盘审计文档且内容 == 重渲染。
4. **主文档加一条指针**（`render_document` 第 7 条注意点），使转动块的判定依据可从
   `docs/capytaine_crosscheck_r2.md` 找到——这是原来缺失的可发现性。主文档重新渲染后
   只多了这一条（`git diff --stat` = `+1`），没有引入其它漂移。
5. **一张审计图**：`benchmarks/verified/make_m23272_figure.py` →
   `docs/fig_xval_r2_degeneracy_ladder_m23272.png`，从工件画"最差计分偏差 vs TensorBEM 面板数"
   的阶梯 + 三条带宽线，是第 3.3 节论点的可视化。

改动文件（6 个，单功能点，无跨模块耦合）：

| 文件 | 性质 |
|---|---|
| `src/tensorbem/benchmarks/capytaine_crosscheck_r2.py` | +`AUDIT_DOC_PATH`、+`render_degeneracy_document`、`write_outputs` 多写一份、主文档加第 7 条注意点 |
| `docs/audit_xval_r2_degeneracy.md` | 由工件重新生成（数字全部刷新） |
| `docs/capytaine_crosscheck_r2.md` | 重新渲染（仅 +1 行指针） |
| `tests/test_capytaine_r2_audit_doc_pins.py` | 新增 9 项 pin |
| `benchmarks/verified/make_m23272_figure.py` | 新增，审计图生成 |
| `docs/fig_xval_r2_degeneracy_ladder_m23272.png` | 新增图 |

> 热文件声明：`src/tensorbem/benchmarks/capytaine_crosscheck_r2.py` 是 R2 域的核心模块，
> 其它 R2 任务可能同时改它。本单只在文件尾部**新增**两个函数/常量并扩写 `write_outputs`，
> 未改动 `solve_matrix` / `compare` / `degeneracy_audit` / `invariant_scoring` 任何算法行；
> `render_document` 只加了第 7 条注意点的文本。

## 3. 验证（本机跑了什么）

```
$ python -m pytest tests/test_capytaine_crosscheck_r2.py tests/test_capytaine_r2_invariant_residual.py \
      tests/test_capytaine_r2_audit_doc_pins.py tests/test_capytaine_crosscheck.py -q
56 passed, 1 warning in 8.53s

$ python -m pytest tests/test_capytaine_same_surface_crossval.py tests/test_xval_r2_reference_reproduction.py \
      tests/test_benchmark_cli_contract.py -q
42 passed, 1 warning in 9.23s

$ ruff check src/tensorbem/benchmarks/capytaine_crosscheck_r2.py tests/test_capytaine_r2_audit_doc_pins.py
All checks passed!

$ python -c "...assert doc == render_degeneracy_document(study) and main doc == render_document(study)"
docs in sync
```

已知与本单无关的既有失败：`tests/test_gauss_legendre_fallback.py::
test_quadrature_quantities_are_unchanged_by_the_fallback` 在 **clean main 上同样失败**
（`git stash` 后复跑确认），不是本单引入。

## 4. 改动后的诚实结论（不粉饰）

* 不变量轴 **PASS**：退化残差单调衰减（半球 3.47x / 3.84x，最细 1.1e-03·ρL^k；球 3.36x，
  2.8e-03·ρL^k）、转动中心变换恒等式残差 1.1e-16、深潜 Lamb 极限 0.60%。
* 可判定物理项 **FAIL**（内部带宽 5%/10%）：最差 6.71%（`sphere_r1_z-0.5_c0`，`nu=2`
  附加质量 `A[roll,roll]`），失败 8 项，全部集中在 `nu=2` 对角项。
* **满足 owner 的 15% 验收口径**（`ledger_acceptance.satisfied = True`，6.71%）。两个口径
  分账，不互相代表。
* 是否 <3%：**否**（四个算例全部否）。
* 剩余偏差的主因是两侧网格不对等（TensorBEM 1024/1026 vs capytaine 784/1120）且阶梯单调
  收缩（23.08% → 9.38% → 5.82%）；**等面板数重判仍未做**，这是本锚点下一步的欠账，本单
  不冒充已解决。

## 5. 局限

* 本单没有重跑任何求解（工件是既有的、由 m21265/m22034 产出的）；pin 是**工件算术**层面
  的一致性，不是物理正确性证明。
* 图只画既有 `convergence` 阶梯；没有新增网格层级。
* `docs/audit_xval_r2_degeneracy.md` 里的中文散文仍是固定模板，随算例增删需要同步改
  renderer（这是刻意的取舍：宁可手写散文 + 机器注入数字，也不让散文变成模板噪音）。