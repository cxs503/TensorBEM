# m21591 — 有限水深交叉验证：冻结工件审计 + 两处发布链缺陷修复（全网格判定未完成）

**锚点**：`XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` · **仓**：TensorBEM · **分支**：`agent/m21591`
**结论先说**：本单**没有**产出全网格判定；冻结工件仍是 `PARTIAL: 1 of 12 planned cells`。实质产出 = 审计 + 两个真缺陷修复 + 可复用工具 + 算力阻塞的完整证据链。

## 1. 审计发现（只读复核）

| 项 | 现状 | 影响 |
|---|---|---|
| `results.json` | `execution_status = "PARTIAL: 1 of 12 planned cells"`，TensorBEM 梯级仅 level 0/1 | 6 条门禁在测试里是 `xfail`，给不出判定 |
| `capytaine_ref.json` | 只覆盖 1 格（h/a=3, k0a=0.5），3 分辨率 + legacy | 参考侧缺 11 格 |
| 发布页逐格表 | 只列 A33/B33 | 漏掉纵荡 A11/B11，而上一轮唯一边缘 FAIL（A11 +3.40 %）正是纵荡量 |
| 发布页溯源 | 无生产者脚本/版本/生成时间 | `test_published_page_quotes_the_frozen_numbers` 要求页面含 `scripts/capytaine_finite_depth_ref.py`，该串只在模块 docstring 里 → **研究一旦 COMPLETE 这条门禁必红**，现被 `_partial_run` 的 xfail 掩盖 |
| 布尔门禁渲染 | `reference_is_finite_depth` 实测列 `n/a`、判决列 PASS | 页面对没印出的数下判决 |
| 参考梯级渐近性 | `reference_ladder_in_asymptotic_range` = 0.2412 < 0.3（FAIL） | 部分量的 capytaine 梯级未进渐近区，参考不确定度大于被测侧 |

## 2. 本单改动（最小、可验证）

1. `src/tensorbem/benchmarks/capytaine_finite_depth_crossval.py`
   * `render_document` 增**参考溯源 + 执行状态**行（生产者脚本 / capytaine 版本 / GF 方法 / 分辨率表 / 生成时间 / `execution_status`），全部取自 `results.json`；
   * 逐格表补 **A11/B11** 四列 → 覆盖全部 4 个判定量；
   * 布尔门禁渲染 `yes/no`；
   * 新增 `_order_coverage()`，局限节报「多少条目的参考梯级量出了阶」（本次 1/4）。
2. `scripts/fig_xval_bem_finite_depth_grid.py`（新，可复用）：读 `results.json` → 4 量偏差热图 + 偏差随 `k0_h` 走向（点线=预登记带，空心点=参考梯级非渐近）。
3. `post_report_m21591.py`（新）：从冻结工件程序化生成报告并内嵌图。
4. `docs/capytaine_finite_depth_crossval.md` 用新渲染器重出（数字未变，仍对应现冻结工件）。

## 3. 算力阻塞（诚实记录）

| 尝试 | 结果 |
|---|---|
| 集群 `cj3552`：capytaine 12 格 + TensorBEM 12 格 × 3 级 + 出版页（8 核，`--jobs 6`） | 排队 27 分钟未起跑（在飞 50+），为避免重复占核已 kill |
| 集群 `cj3575`：同命令重提 | 排队 50 分钟仍未起跑（在飞 64），本单结束时仍在队列 |
| 本机全长直跑 / 后台分离运行 | 被 owner 2026-11-13 计算调度令硬拦（本机仅准 <8 核、<10 min）；且本席 shell 硬上限 120 s |
| capytaine 第 4 级 (96,192)=18432 面板单格探针 | 单次调用 >115 s 被收割、未落盘 → 本机不可达，须集群 |

**因此**：锚点核心欠账（12 格 × 三级梯级 → 全网格外推判定）本单**未完成**；不得宣称「全网格 <15 %」或任何门禁转 PASS。

补齐只需一条命令（集群，约 25–30 分钟）：

```
PYTHONPATH=src python scripts/capytaine_finite_depth_ref.py --out benchmarks/verified/capytaine_finite_depth/capytaine_ref.json
PYTHONPATH=src python -m tensorbem.benchmarks.capytaine_finite_depth_crossval --levels 0,1,2 --no-extra-level --jobs 6 --write-doc
```

跑完后 `execution_status` 变 `COMPLETE`、6 条 xfail 自动转真断言（无需改测试）。

## 4. 现有数字（唯一已执行格，引自冻结工件，非本单新算）

| 量 | TensorBEM（两级梯级，未外推） | capytaine | 相对偏差 |
|---|---:|---:|---:|
| A33 垂荡附加质量 | 2617.32 | 2999.92 | -12.75 % |
| B33 垂荡阻尼 | 564.21 | 740.90 | -23.85 % |
| A11 纵荡附加质量 | 2276.48 | 2461.59 | -7.52 % |
| B11 纵荡阻尼 | 616.04 | 797.41 | -22.74 % |

m21322 已在同一格补第三级并外推，证明这些两位数的差是离散化（收缩到 −0.03 %…+3.40 %）。

## 5. 验证

`PYTHONPATH=src python -m pytest tests/test_capytaine_finite_depth_crossval.py -q` → 11 passed, 6 xfailed（xfail 由 `execution_status` 决定，如实反映执行程度）。

## 6. 给下一次自找活的线索

1. `cj3575` 若已终态，`fetch` 工件即可直接替换冻结工件（无需重算）。
2. 之后跑 `scripts/fig_xval_bem_finite_depth_grid.py` 出全网格图，`post_report_m21591.py` 出报告。
3. 若全网格后 `reference_ladder_in_asymptotic_range` 仍 FAIL → capytaine 参考分辨率不足：把 `plan.CAPYTAINE_RESOLUTIONS` 扩到含 (96,192) 并在集群重出参考。
4. 未做的确认梯级：`EXTRA_LEVEL_CELLS`（4 格 × 1280 面板，估约 1 h/格）。
