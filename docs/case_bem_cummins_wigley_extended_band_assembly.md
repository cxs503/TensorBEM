# CASE-BEM-CUMMINS-WIGLEY — 扩展带装配链路的缺陷修复（C8/C9 前置设施）

**Mission**: m22587 ｜ **Anchor**: `CASE-BEM-CUMMINS-WIGLEY` ｜ **仓**: TensorBEM
**对象**: `scripts/assemble_cummins_extended.py`、`render_markdown`（`src/tensorbem/benchmarks/case_bem_cummins_wigley.py`）

## 1. 背景：这条链路为什么重要

算例 12 rad/s 频带上 surge / sway / yaw 的阻尼尾部指数 0.28–1.62 未进入代数衰减，
截断预算无界，因此这三个自由度的 C3a 残差只是**经验一致**而非带限数据蕴含
（`docs/case_bem_cummins_wigley_extended_probe.md` §2、round 9 审计 §2.2）。
唯一出路是把频带扩到 20 rad/s，并在满足 C2 的 60×16（3840 面板）网格上求解：
`scripts/case_cummins_wigley_extended_band.py`（55 个频点分片）+
`scripts/case_cummins_wigley_extended_anchor.py`（44×14 / 60×16 镜像锚点梯）。
两者产出落盘后，由 `scripts/assemble_cummins_extended.py` 装配成
`sweep_extended.npz` 并把 `C8` / `C3a_extended` / `C9` 挂到已发布判据上。

本单核对这条**装配端点**时发现它不可用——闸门基础设施（round 9 交付）从未在真数据上跑通，
因为负责把它接上发布物的那一步本身是坏的。

## 2. 发现的缺陷（全部为静默失效，不抛错）

| # | 缺陷 | 后果 | 严重度 |
| --- | --- | --- | --- |
| D1 | 装配时读 `data.get("residual", 0.0)`，而求解器写出的键是 `solver_relative_residual` | 残差全为 0.0 → **C0 把未收敛的 sweep 判为收敛**（C0 判据是 `max residual ≤ 1%`），后续全部闸门建立在污染数据上。仓内已有 m20021 事故先例（`C0_DEPENDENT_GATES` 注释记录的正是这一课） | 高（会伪证） |
| D2 | `update_case_result()` 调用 `gates_from_artifact(<extended sweep>)`，不带 `extended_path` | 被重判的是**扩展 sweep 自己**：`C8` 需要「已提交 12 rad/s 频带 vs 扩展频带」两条曲线，自比自没有意义；`C9` 的 `band_at_12` 也会变成 20 | 高（判据空转） |
| D3 | 同一处用 `existing["gates"] = result["gates"]` 整体覆盖 | 覆盖后只留下 replay 默认闸门集，已发布的 C5b/C5c/C10 等被静默删除，判据总数与 `CASE.md` 对不上 | 高（口径漂移） |
| D4 | 装配不校验网格一致性、不按频率排序（依赖 `sorted(glob)`） | 分片来自不同网格时无 mesh identity（锚点梯按 mesh identity 选档），文件名词序 ≠ 频率序 | 中 |
| D5 | `render_markdown` 完全没有 C8/C9/C3a_extended 的**数值**呈现 | 即使闸门挂上，`CASE.md`「## 2 Gates」只有 PASS/FAIL 一行；扩展带存在的意义是「哪个自由度动了、动多少」，这一点在文档里看不见 | 中（可读到结论、读不到证据） |

## 3. 本单改动

1. `scripts/assemble_cummins_extended.py` 重写装配段与重判段：
   * 逐点读取 `solver_relative_residual`（缺键即 `KeyError`，不再默认 0.0）；可选带
     `system exit` 而非静默；
   * 校验所有分片同一 `(n_sections, n_levels)` 与 `panel_count`，否则拒绝装配；
   * 按 `omega` 排序并要求严格递增；
   * 重判对象改为**已提交的 `sweep_medium.npz`**，`extended_path=` 显式指向扩展 sweep，
     因此 `C8/C3a_extended/C9` 由模块自身逻辑挂载（不在脚本里重写判据）；
   * 只替换 `gates / verdict / extended` 三个字段，`hull / settings / sweeps` 等描述求解的
     元数据原样保留，并记录 `gate_replay` 出处块（mission、时间、两个工件名、锚点工件是否在）；
   * `CASE.md` 由 `render_markdown(JSON)` 重新生成——文档始终是 JSON 的函数。
2. `render_markdown` 增加 `## 4b. Extended band 0.2 .. 20 rad/s (C8 / C9 / C3a_extended)`：
   C8 共享带最差偏差、逐自由度 `p@12 / p@20 / 预算可判性 / 预算相对量 / 可判样本数 / 锚定残差`
   表、C9 状态与「可判自由度 12 → 20」，以及 C3a_extended 最差自由度与锚点来源。
   **无工件时该节不出现**，因此已提交的 `CASE.md` 逐字节不变（见测试）。

## 4. 验证（本机实测）

```
PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py -q
# 73 passed, 1 warning in 109.05s
```

新增 4 项测试（`tests/test_case_bem_cummins_wigley.py` 末节）：

| 测试 | 钉住的不是「能跑」而是不变量 |
| --- | --- |
| `test_assembler_reads_the_solver_residual_under_its_own_name` | 分片写 `residual=1.25` → 装配后 `sweep.solver_relative_residual == 1.25` 且 `solver_validity_gate` **FAIL**；写 `2.4e-3` → PASS。直接封住 D1 |
| `test_assembler_orders_nodes_by_frequency_not_by_file_name` | 乱序写盘 → 装配结果 `[12, 15, 20]` |
| `test_assembler_refuses_nodes_from_two_meshes` | 60×16 与 44×14 混合 → `SystemExit`（不允许没有 mesh identity 的 sweep） |
| `test_render_markdown_carries_the_extended_band_numbers` | 合成扩展带挂载后，逐自由度行 `| <dof> | p@12 |` 必须出现；已发布的 `case_result.json`（无扩展工件）渲染后**不得**出现 4b 节 |

## 5. 未完成：扩展带工件仍未落盘（honest）

本单把装配端点修通了，但 `sweep_extended.npz` / `anchor_ladder_extended.json` 仍不存在：
本单递交的两个集群作业在收单时**仍在排队**，未产出任何分片文件。

| 作业 | 命令 | 规模 | 状态（收单时） |
| --- | --- | --- | --- |
| `cj4241` | `case_cummins_wigley_extended_band.py --shard {i} --shards 55` | 55 分片 × 60×16（3840 面板），覆盖 0.2–20 rad/s | queued, age 14.5 min |
| `cj4242` | `case_cummins_wigley_extended_anchor.py --shard {i} --shards 2` | 2 档镜像锚点（44×14、60×16→7680 未知量） | queued, age ~7 min |

成本是可核算的（`extended_probe_cost.json` 实测拟合 `t ~ N^1.93`）：3840 面板单频点
≈ 193 s，55 点 ≈ **3 CPU·h**；锚点 60×16 档 ≈ 12 min、44×14 档 ≈ 20 s。
因此这一步不是「算法没做完」，而是**排队到期即出数**：

```
PYTHONPATH=src python3 scripts/assemble_cummins_extended.py --mission <下一单号>
```

该命令会自动装配（缺分片则报缺失数）、重判并重写 `case_result.json` + `CASE.md`；
随后 `python -m tensorbem.benchmarks.case_bem_cummins_wigley --replay benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz`
给出同一判据集（`gates_from_artifact` 会自动挂载目录内的 `sweep_extended.npz`）。

**因此本单不宣称 C8/C9 已判**：它们的状态仍是 `not_run`，`CASE.md` 仍是 16/16，
与已提交工件一致。任何「扩展带已通过」的表述在工件落盘前都不成立。

## 6. 下一步建议（给下次自找活）

1. `cj4241`/`cj4242` 到期后 `--assemble`，看 `C9` 的 `newly_decidable` 是否真包含
   surge/sway/yaw（探针在 20 rad/s 给出 coarse/medium 分歧，所以**允许**是空列表——
   那将是关于 3840 面板网格的必要负面结果）；随后 C3a 的余量（yaw 8.85% vs 10%）才有资格收紧。
2. 若 `C9` 显示 surge/sway 仍未进入代数衰减，下一步不是继续抬高网格，而是查
   `tensorbem.causality.local_tail_exponent` 的窗口在 18–20 rad/s 上只剩两三个样本
   （55 点/0.2–20 rad/s ⇒ 18.7 点/decade，末端窗口太短），属于**估计器**问题而非物理问题。