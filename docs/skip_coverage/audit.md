# 静默死覆盖审计：5 条被 git-ignore 吃掉的断言重新执行（TensorBEM）

**任务**：m3726（paper 席自找活，model 钉扎 deepseek-v4-flash） · **分支**：`agent/m3726`
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3726`，基线 `b420b2e`）
**工件**：`docs/skip_coverage/scan.py`（审计器）、`docs/skip_coverage/report.json`（现状）、
`docs/skip_coverage/report_before.json`（修复前快照）、`tests/test_skip_coverage.py`（机器门）、
`tests/test_finite_depth_energy_convergence.py`、`tests/test_ursell_reference_pinning.py`（两处守卫修复）

EDIT_WHITELIST: docs/skip_coverage/, tests/

---

## 1. 案例介绍（为什么做这件事）

TensorBEM 的验证体系靠「文档数字 → 注册表 → 测试」三层闭环（`tensorbem.benchmarks.declared_numbers`、
`tensorbem.doc_claims`），m3314 已经修过 CI 的**顺序病**（证据生成必须在 `pytest` 之前，否则测试看到的是空树）。
那条修复只解决了一半：**顺序对了，但路径可能根本不在检出里**。

本仓的 `.gitignore` 忽略 `artifacts/` 与 `outputs/`。而多个测试用
`pytest.mark.skipif(not ARTIFACT.exists(), ...)` 守卫自己的物理断言，`ARTIFACT` 恰好指向
`outputs/...`。于是干净检出（=CI、=新同事、=三个月后的自己）里这个文件永远不存在，守卫永远触发：

> 这不是 skip，是**静默删除**：套件全绿、报告照旧写「已覆盖」，而断言一次都没跑。

这不是假想。上一轮 m3059（heavy 层验证欠账结清）在「注意点」里断言
「三维自由液面已由 m2609 等测试钉桩」——**钉桩的正是这类守卫测试**，而 CI 上看不到它们跑过。

### 审计对象（仓内自证）

| 项 | 值 |
|---|---|
| 被审计的测试模块 | `tests/test_*.py` 全量（AST 解析，不执行求解器） |
| 守卫形态 | 模块级 JSON 路径常量 + `pytest.skip`/`skipif` |
| 干净检出判据 | `git ls-files` 是否跟踪该路径；否则看 `.github/workflows/*.yml` 是否有步骤写它 |
| 机器 | nproc=128，load15=122（重载窗口）→ 本单只做静态审计 + 分钟级复算，不起大算力 |
| 平台报告 | `/tensor/reports` id **3869**（含修复前后对比图 `fig_skip_coverage.png`） |

## 2. 怎么做的

1. **写审计器**（`docs/skip_coverage/scan.py`）：用 `ast` 取出每个测试模块里指向 `.json` 的模块级路径常量，
   再取该模块 `skip`/`skipif` 引用的名字与字面原因，两者相交即为「守卫工件」；对每个工件按
   `VERSIONED`（git 跟踪）/ `CI`（workflow 步骤写出）/ `UNTRACKED`（都不是）分类。
   模块内若**没有任何**可见工件，则记为 **dark**（断言在 CI 上必然不执行）。
2. **留修复前快照**：`git stash` 掉本单的两处测试改动后跑一次审计 → `report_before.json`（可复算，见 §3.1）。
3. **定位真因**（不是「重算一遍」）：两条 `UNTRACKED` 路径的**同一份 JSON 早已有版本化归档**——
   `docs/m2498/outputs/finite_depth_energy_identity/convergence.json`（m2498）与
   `docs/m2232/outputs/ursell_reproducibility_study.json`（m2232）。也就是说数据一直在仓库里，
   只是测试去开那个被忽略的 `outputs/` 影子路径。**修复代价 = 0 算力**，不需要新增 CI 步骤。
4. **改守卫语义**：两个模块各加一个 `_artifact()` / `_study()`：先试 `outputs/`（本地刚重算的新值），
   再退到版本化归档，两者都不存在则 `AssertionError` 并打印再生命令。
   **缺失从 skip 变成 fail**——这才是「覆盖」该有的语义。
5. **装机器门**（`tests/test_skip_coverage.py`）：① dark 模块必须为空；② 任何 `UNTRACKED` 守卫工件必须
   在 `KNOWN_UNTRACKED` 里带着理由被显式复核；③ 审计报告 `report.json` 必须等于一次新鲜扫描（防腐化）；
   ④ 已审计模块清单仍在。负控见 §4。
6. **证明修复真有效**：在**没有** `outputs/` 副本的树上（=干净检出等价条件）跑这两个模块。

## 3. 计算结果

### 3.1 修复前后（程序化产出，零手抄）

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 被守卫的工件数 | 6 | 4 |
| 其中 `UNTRACKED`（干净检出拿不到） | **2** | **0** |
| dark 模块（断言在 CI 上必不可见） | **2** | **0** |
| 有守卫的模块总数 | 6 | 4 |
| 因守卫而不在 CI 执行的测试函数 | **5** | **0** |

修复前的两条 `UNTRACKED` 路径与成因：

| 工件 | 守卫模块 | 受影响测试 | 版本化归档（早已在库） |
|---|---|---|---|
| `outputs/finite_depth_energy_identity/convergence.json` | `test_finite_depth_energy_convergence.py` | `test_report_tables_match_the_artifact_verbatim`、`test_committed_artifact_is_reproducible`、`test_deep_water_control_shows_the_same_offset` | `docs/m2498/outputs/finite_depth_energy_identity/convergence.json` |
| `outputs/ursell_reproducibility_study.json` | `test_ursell_reference_pinning.py` | `test_study_artifact_matches_the_current_code`、`test_recorded_convergence_is_first_order` | `docs/m2232/outputs/ursell_reproducibility_study.json` |

修复后剩下的 4 条守卫全部 `VERSIONED`（`docs/finite_depth_envelope/benchmark.json`、
`docs/m2952/outputs/finite_depth_refinement/benchmark.json`、`docs/mixed_boundary/outputs/study.json`、
`docs/m3383/near_surface_band_study.json`），干净检出可读。

### 3.2 重新执行的断言与其实测结果

干净检出等价条件（`outputs/` 已删除）下：

| 模块 | 修复前 | 修复后 | 耗时 |
|---|---|---|---|
| `test_ursell_reference_pinning.py` | 4 passed, **2 skipped** | **6 passed** | 8.6 s |
| `test_finite_depth_energy_convergence.py` | 5 passed, **3 skipped** | **8 passed** | 75.9 s |
| `test_skip_coverage.py`（新） | — | **4 passed** | 0.5 s |

复活的断言本身（这就是被删除的物理内容，现在真的跑了）：

| 断言 | 判据 | 实测 |
|---|---|---|
| 归档研究 vs 当前代码（Ursell 64 面元） | `rel=1e-6` | 一致（`A=2587.7627`、`B=2787.2368`） |
| 记录的一阶收敛趋势 | 相邻误差比 `1.7–2.3` | 64→128→256 误差 `1.674e-2 → 8.513e-3 → 4.293e-3`（比 1.966、1.983） |
| 近/远场阻尼比收敛到 1 | 最细级 `|ratio-1| < 5e-3` | `1.0105 → 1.0053 → 1.0027` |
| 64 面元解 vs 归档收敛研究 | `rel=1e-9`（5 个量） | 全部一致（有限水深求解器自 m2498 未漂移） |
| 文档表 vs 归档 JSON 逐字一致 | 由 `build_tables.py` 生成的块 | 一致（无手抄） |
| 深水对照的 1% 能量偏移 | `1.04e-2`，`rel=0.15` | 一致 |

**关键判据（3% 标准）**：Ursell 64 面元相对解析多极子解 `1.674e-2`（附加质量）/`2.542e-2`（阻尼），
均在 3% 内；这正是 `test_accuracy_meets_the_acceptance_standard` 断言的量，此前它与归档比对一起
在 CI 上不完整执行。

## 4. 注意点

- **负控已做**：临时塞入一个指向 `outputs/does_not_exist/study.json` 的守卫模块后，机器门 3 条失败
  （dark 判定 + 未复核路径 + 报告不一致），删除后恢复绿。门不是摆设。
- **成本零增量**：没有新增 CI 步骤，没有重算任何基准。修复只是把测试指向仓库里已有的归档。
  这也是本条最值得记的教训：**死覆盖往往是路径问题，不是算力问题**——先审计再重算。
- **`outputs/` 优先、归档兜底的先后顺序**是有意的：本地重算后测试比的是新值，干净检出比的是归档值；
  两者都不相容时才会红（例如求解器漂移而归档未更新——这正是应该红的场景）。
- **未做的事**：没有把 `declared_numbers` 注册表扩展到「docs/ 归档 vs 现算」的统一视图（那需要跑
  standard/heavy 层，本机 load15=122，按容量纪律不做）；没有审计其他 Tensor* 仓的同类静默 skip
  （`data_search` 未做，属跨仓动作，建议各自席位按本单 `scan.py` 复刻一份）；没有重跑全量套件
  （m3314 记录基线 270 passed；本单只改动 2 个测试模块 + 新增 1 个，二者均已单独跑绿）。
- **适用范围**：审计器只识别「模块级 JSON 路径常量 + skip 守卫」这一种形态；`pytest.importorskip`
  （matplotlib/scipy，属可选依赖，不是覆盖漏洞）与 `TENSORBEM_HEAVY_TESTS` 环境门（成本分级，
  文档已声明）不在判定范围内，属正常设计。

## 5. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| Ursell 64 面元附加质量 | Ursell 多极子 2545.1672 | `1.674e-2`（1.67 %） | **是** |
| Ursell 64 面元阻尼 | Ursell 多极子 2718.1540 | `2.542e-2`（2.54 %） | **是** |
| 归档 vs 现算（Ursell 64 面元，`rel=1e-6` 门） | 归档 JSON | `< 1.0e-6` | 是 |
| 归档 vs 现算（有限水深 64 面元，`rel=1e-9` 门） | `docs/m2498` 归档 | `< 1.0e-9` | 是 |
| 审计判决 | `git ls-files` / workflow 文本（确定性判据） | 0（无近似） | 是 |

**结论**：本单交付的不是新数值，而是**覆盖自身的可信度**——修复前 5 条断言在 CI 上从未执行，
修复后全部执行且全部通过；两条归档与现算的偏差分别在 `1e-6`、`1e-9` 判据内，即归档没有腐化。