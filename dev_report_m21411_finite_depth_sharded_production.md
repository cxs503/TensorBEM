# m21411 — 有限水深交叉验证：为什么 12 格网格一直没跑出结论（生产器自身坏了）+ 分片生产/合并工具

**锚点**：`XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（capytaine 的 finite_depth_crossval_plan 做法）
**仓**：TensorBEM · **分支**：`agent/m21411` · **日期**：2026-11-15
**上游**：`finite_depth_crossval_plan.py`（预登记计划，12 格）、`capytaine_finite_depth_crossval.py`（消费端）、
`scripts/capytaine_finite_depth_ref.py`（capytaine 参考生产器）

---

## 1. 案例介绍

有限水深自由面 Green 函数（John 1950 本征级数）是三维有限水深 BEM 的核函数。TensorBEM 走
「本征级数 + 直接配点」，capytaine 走「波数积分 + 色散求根」，两条独立数值路线在同一几何/参数下
的一致性，是有限水深能力最硬的证据。预登记网格：完全淹没球 $a=1$ m、球心 $z=-1.5$ m，
$h/a\in\{3,5,10,20\}\times k_0a\in\{0.5,1.0,2.0\}$ = 12 格，每格 $\nu$ 由 $k_0\tanh k_0h=\nu$ 反解，
两侧各跑面板梯级后比较 Richardson 外推值；预登记带 A 3 % / B 5 %，owner 不可接受线 15 %。

**本单要回答的问题**：计划已合并多轮（m20134/m21322），为什么仍只有 1/12 格有判定？
本单先查「跑不动」的原因，再补跑所需的最小工具。

## 2. 怎么做的（只读核对 → 缺陷定位 → 工具补全）

1. 只读核对冻结工件：`benchmarks/verified/capytaine_finite_depth/{capytaine_ref,tensor_ladder,results}.json`
   —— 参考与梯级都只覆盖 `h_over_a=3.0|ka=0.50`（1/12 格），梯级只有 20/80 面板两级。
2. 在现网代码上按计划跑全网格梯级入口（`--ladder-only`），**第一次就崩**：

   ```
   File ".../capytaine_finite_depth_crossval.py", line 196, in _store
       f"seconds={record['seconds']}",
   KeyError: 'seconds'
   ```

   根因：`deep_water_values()`（深水负对照，计划预登记的「必须看得见水深」门禁）返回的字典里没有
   `seconds` 键，而 `build_ladder._store()` 无条件打印 `record['seconds']`；`_store` 的写缓存动作在打印
   **之后**，异常抛出即丢记录。于是一旦梯级要记录负对照，生产器就崩，**整个 12 格网格在现网代码上根本
   产不出来**——冻结工件停在 1 格不是算力问题，是生产器缺陷。
3. 修复：`deep_water_values()` 增加 `seconds`（顺带 `mode_count`，深水求解器无该字段时取 0），
   `_store` 的进度打印改成 `record.get(...)`（打印不得因为字段缺失毁掉已算好的解）。
4. 补分片生产/合并工具（该网格单核约 20 min 量级，必须能分片递交、断点续跑、且**不能悄悄变小**）：
   * `scripts/capytaine_finite_depth_ref.py`：新增 `--shard/--shards`（分配 = `index % shards`，可复现）
     与 `--merge`；合并**拒绝**缺格/重复格，缺一格就退出且不落盘。
   * `capytaine_finite_depth_crossval.py`：新增 `merge_ladders()`（合并分片梯级缓存；同一格同一级两次
     计算不一致即报错，缺级即报错）、`--merge-caches`、`--ladder-only`、`--cell-ids`、`--shard/--shards`、
     `--no-extra-level`；`--ladder-only` 用显式 `--cell-ids` 时不再受参考覆盖过滤，梯级与参考互为独立计算，
     一侧不必等另一侧进度。
   * `execution_status()`：`results.json` 里原来的 `PARTIAL: ...` 是**手写的散文**（`run_study` 根本不写
     这个字段，重跑一次就丢掉，`_partial_run()` 随即返回 False，门禁会被当成完整研究去判）。现在由梯级
     实际覆盖**算出**：多少格 / 缺哪些级 / 确认级是否缺席。
5. 重算并刷新冻结梯级（客户机短算，< 1 min）：锁定单元格 `h_over_a=3.0|ka=0.50` 的 20/80 面板两级，
   在 m21150（逐行模态需求，`e77f2615`）之后的代码上重算，负对照取 80 面板。

## 3. 计算结果

### 3.1 冻结梯级刷新前后（同格，同网格）

| 量 | 面板 | 旧冻结（m21150 之前） | 现网活算 | 相对漂移 |
|---|---:|---:|---:|---:|
| $A_{33}$ | 20 | 1928.821218 | **1928.758993** | $-3.2\times10^{-5}$ |
| $A_{33}$ | 80 | 2617.261302 | **2617.3250** | $+2.4\times10^{-5}$ |

漂移与 m21322 报告的核函数改后失配同一量级：冻结数据早于核函数修正。刷新后
`tests/test_capytaine_finite_depth_crossval.py::test_frozen_ladder_reproduces_under_a_live_recompute`
（活算复现冻结值，rel 1e-9）由**红转绿**。

### 3.2 两级梯级下的原始偏差（$h/a=3$、$k_0a=0.5$、$k_0h=1.5$，**未外推**）

| 量 | TensorBEM 80 面板 | capytaine（staircase，取最细级） | 相对偏差 |
|---|---:|---:|---:|
| $A_{11}$ | 2276.484 | 2461.593 | −7.52 % |
| $A_{33}$ | 2617.325 | 2999.917 | −12.75 % |
| $B_{11}$ | 616.037 | 797.406 | −22.74 % |
| $B_{33}$ | 564.211 | 740.899 | −23.85 % |

**这些数字不是判定**：两级量不出阶（`tensorbem_ladder_in_asymptotic_range = false`），
`execution_status = "PARTIAL: 1 of 12 planned cells"`。m21322 已证明补上 320 面板后同一格外推偏差收缩到
−0.03 %/−0.03 %/+1.7 %/+3.4 %——本单的冻结数字只是把「同一格的未外推原始偏差」与核函数修正对齐，
不重复那条结论，也不移动任何预登记门禁。

## 4. 注意点 / 局限

* **未完成的部分（如实）**：12 格 × 0/1/2 面板梯级与 12 格 capytaine 参考（288/1152/4608）已按分片
  递交集群（`cj3426` 参考 12 分片、`cj3430` 梯级 12 分片，stage_dir=本工作树），递交后 25 min 仍排队
  （队列 59 个在飞），本单预算内未取回，故撤销递交；全网格判定仍欠账。工具链已就位，重跑命令：
  ```
  python3 scripts/capytaine_finite_depth_ref.py --shard {i} --shards 12 --out shard_out/cp_ref_{i}.json
  python3 scripts/capytaine_finite_depth_ref.py --merge shard_out/cp_ref_*.json --out benchmarks/verified/capytaine_finite_depth/capytaine_ref.json
  python3 -m tensorbem.benchmarks.capytaine_finite_depth_crossval --ladder-only --levels 0,1,2 --no-extra-level --shard {i} --shards 12 --cache shard_out/ladder_{i}.json
  python3 -m tensorbem.benchmarks.capytaine_finite_depth_crossval --merge-caches shard_out/ladder_*.json --cache benchmarks/verified/capytaine_finite_depth/tensor_ladder.json
  python3 -m tensorbem.benchmarks.capytaine_finite_depth_crossval --write-doc
  ```
* 1280 面板确认级（`EXTRA_LEVEL_CELLS` 4 格）本单未跑：深水格模态数随 $k_0h$ 增长，1280 面板单格
  预估小时级，属集群长作业。
* 负对照缺陷的教训：计划门禁所需的记录一旦没有 `seconds` 字段，生产器就在写缓存前崩——「门禁实现了但
  产物永远产不出来」这类失效不会被任何数值测试发现，只能由真正跑一遍发现。

## 5. 误差判定

| 关键量 | 本单结论 | 是否 < 3 % |
|---|---|---|
| 冻结梯级 vs 现网活算（$A_{33}$，20 面板） | 相对漂移 $3.2\times10^{-5}$ | **是** |
| $A_{33}$ vs capytaine（未外推，80 面板） | −12.75 % | **否**（且不是判定：梯级未进渐近区） |
| $B_{33}$ vs capytaine（未外推，80 面板） | −23.85 % | **否**（同上） |
| 全网格外推判定 | 未取回（集群排队超预算） | **未判定（unknown）** |

honest-FAIL：本单不宣称任何精度已达标；唯一新达标的量是「冻结数据与现网代码一致（$10^{-5}$ 级）」。