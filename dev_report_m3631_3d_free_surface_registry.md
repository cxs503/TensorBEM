# TensorBEM 验证欠账结清：三维自由液面（Hulme/Wang-Linton）声明数字纳入注册表

**任务**：m3631（generalist 席，自找活） · **分支**：`agent/m3631` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3631`，基线 `407a64f`）
**工件**：`outputs/verif_m3631/{standard_tier.json, fig_3d_free_surface_convergence.png,
fig_3d_free_surface_deviation.png, patch_fh.py, patch_registry.py, fix_conventions.py,
patch_tests_doc.py, make_figures.py}`

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py,
src/tensorbem/benchmarks/floating_hemisphere.py, tests/test_declared_numbers.py,
docs/marine_bem_benchmarks.md, outputs/verif_m3631/, dev_report_m3631_3d_free_surface_registry.md

---

## 1. 案例介绍（为什么做这件事）

2026-11-04 的 heavy 层结清单（m3059，`dev_report_m3059_heavy_verification_debt.md` §4）
明确留下一条未结清欠账：

> 三维自由液面（Hulme/Wang-Linton）虽已由 m2609 等测试钉桩，但**仍未纳入注册表**
> （成本更高，建议后续单独立项）。

`docs/marine_bem_benchmarks.md` §4.10–§4.11 的三张表（表 10 Hulme 漂浮半球收敛、
表 11 免盖板 CHIEF 对比、表 12 Wang/Linton 潜没球收敛）加相邻正文共 **66 个数字**
此前没有任何可执行判据守护：内核/网格/求积一旦改动，这些数字会像 Ursell `7.7e-3`
那样悄悄腐化，只能等下一次审计发现。本单把该家族整体登记进
`tensorbem.benchmarks.declared_numbers` 注册表并挂上回归测试。

**算例参数**

| 项 | Hulme 漂浮半球（表 10/11） | Wang/Linton 潜没球（表 12） |
|---|---|---|
| 几何 | 半径 1 m 半球，穿水开口网格 | 半径 1 m 球，球心 (0,0,−1.5) |
| 频率 | Ka=1.0（CHIEF 行 Ka=2.58） | Ka=0.8 |
| 离散 | 16/64/256 面元（CHIEF 行 256） | 20/80/320 面元 |
| 真值 | Hulme (1982) 48 项无波势 + Kochin 振幅 | 零数值求积 Thorne 多极子闭式 |
| 判据量 | 5 量相对误差 + 条件数 + 观测阶 + Kochin/Haskind 残差 | 4 量相对误差 + 转动耦合比 + 条件数 + 观测阶 |

## 2. 怎么做的

1. **成本重估**：m3059 把该家族推迟的理由是「成本更高」——指的是含频率扫描的
   *完整* 套件。但三张表引用的只是收敛研究，实测总耗时 **14.7 s**
   （Hulme 3.2 s + CHIEF 6.6 s + 球体 4.9 s），故归入 **standard 层**，
   默认 `pytest` 与 CLI 即覆盖，无需 heavy 门。
2. **注册表扩展**（`declared_numbers.py`，+66 条）：新增 `hulme_hemisphere`
   （31 条）、`hulme_chief`（9 条）、`submerged_sphere`（26 条）三组；
   runner 复用 `run_refinement_study` / `run_irregular_frequency_study`，
   同参数条目共享一次求解（注册表自带缓存）。
3. **最小增量补丁**（`floating_hemisphere.py`）：表 10 的 B11' 观测阶列
   研究模块原本未追踪，补 `err_b11` 序列与 `observed_order_b11` 字段
   （纯增量，不动既有字段）。
4. **口径归因**（本单唯一实质发现）：表 10 的 B11' 阶（1.62/1.88）来自
   m2609 研究的**面元数口径**（h∝N^(−1/2)），其余四列是 **h_mean 口径**；
   16→64 层 h_mean 比 1.87 而非 2.0，两口径相差 11%。注册表按各自口径
   分别钉桩（`observed_order_b11_sqrtN`），文档补口径说明（m3280 先例）。
5. **回归钉桩**（`test_declared_numbers.py`）：覆盖面判据扩到三组新组；
   新增 standard 层测试判 66 条全 PASS，并断言 CHIEF 物理声明
   （未正则化必现负阻尼、CHIEF 后条件数下降）。
6. **文档**：§4.10 补 B11' 阶口径说明。

## 3. 计算结果

`python -m tensorbem.benchmarks.declared_numbers --tiers standard --strict`
→ **70/70 PASS**（既有 4 条 + 新增 66 条），零手抄，全部由
`outputs/verif_m3631/standard_tier.json` 程序化注入。

### 3.1 表 10（Hulme，Ka=1.0）—— 31 条全 PASS

| N | 条件数 | A33' 误差 | B33' 误差 | A11' 误差 | B11' 误差 | Fz 误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3.893 | 1.2013e-1 | 1.2190e-1 | 1.8024e-1 | 4.0241e-1 | 6.6716e-2 |
| 64 | 3.052 | 2.9736e-2 | 4.5023e-2 | 5.0508e-2 | 1.3049e-1 | 2.2094e-2 |
| 256 | 2.783 | 7.4054e-3 | 1.1812e-2 | 1.3038e-2 | 3.5275e-2 | 5.6679e-3 |

文档值 1.20e-1/1.22e-1/1.80e-1/4.02e-1/6.67e-2 等逐格吻合（最大相对偏差
3.8e-3，来自 B11' 阶的 √N 口径舍入）。观测阶 256 行：2.06/1.98/2.01/1.88/2.02
（B11' 按 √N 口径 1.8877 vs 声明 1.88）。正文残差：Kochin 近/远场 5.4478e-4
（声明 5.45e-4）、Haskind 纵荡 2.0892e-4（2.09e-4）、垂荡 8.0854e-4（8.09e-4）。

### 3.2 表 11（CHIEF，Ka=2.58，N=256）—— 9 条全 PASS

| 方案 | 条件数 | A33'（精确 0.4007） | B33'（精确 0.0631） |
|---|---:|---:|---:|
| 未增广 BIE | 145.66 | 0.3481 | −0.1042（非物理负阻尼，复现） |
| 5 点内部 CHIEF | 4.24 | 0.3919 | +0.0687（严格耗散，复现） |

### 3.3 表 12（Wang/Linton，f/a=1.5，Ka=0.8）—— 26 条全 PASS

| N | 条件数 | A33' 误差 | A11' 误差 | Fx/Fz 误差 | 转动耦合比 |
|---:|---:|---:|---:|---:|---:|
| 20 | 2.407 | 2.5497e-1 | 2.6799e-1 | 3.6029e-1 / 3.6977e-1 | 2.492e-32 |
| 80 | 2.259 | 7.2988e-2 | 8.0164e-2 | 1.1594e-1 / 1.1935e-1 | 2.151e-4 |
| 320 | 2.188 | 1.8509e-2 | 2.0766e-2 | 3.1000e-2 / 3.1972e-2 | 2.648e-5 |

观测阶 320 行 2.06/1.93/2.03/1.98 与文档一致（偏差 ≤2.4e-3）。

## 4. 注意点（坑 / 局限）

1. **口径混用是真实发现，不是粉饰**：表 10 同一行内 B11' 阶与其他四列阶
   口径不同（√N vs h_mean），相差 11%。已按文档原值（√N 口径）钉桩并在
   文档显式说明；若未来统一口径，需同时改 m2609 研究输出与文档。
2. **耦合比容差**：2.2e-4/2.6e-5 只印 2 位有效数字，默认 1.5% 容差比印刷
   精度还紧，按注册表「容差来自印刷精度」原则放宽到 2.5%。
3. **未做**：未改任何求解器实现；1024 面元 B11' 专项（m3245 建议）仍是
   heavy 层欠账（本机 load15≥100 不满足容量纪律）；有限水深三维 John 核
   仍是 Roadmap 功能缺口（建议派单）。
4. **平台**：结论在 Python 3.11.6 / NumPy 2.4.6 / loongarch64 /
   OPENBLAS_NUM_THREADS=1 下复算；基准是确定性的，容差远高于 BLAS 噪声。

## 5. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| 66 条注册项 vs 文档声明 | 文档印刷值 | 最坏 2.23e-2（耦合比，2 位有效数字舍入）；其余 ≤3.9e-3 | ✅ |
| 256 面元 A33'/B33'/A11'/B11'/Fz | Hulme (1982) 闭式 | 7.41e-3 / 1.18e-2 / 1.30e-2 / 3.53e-2 / 5.67e-3 | ✅（离散误差，二阶收敛） |
| 320 面元 A33'/A11'/Fx/Fz | Thorne 多极子闭式 | 1.85e-2 / 2.08e-2 / 3.10e-2 / 3.20e-2 | ✅ |
| CHIEF 正则化 | 物理耗散性 B33>0 | −0.1042 → +0.0687，条件数 145.66 → 4.24 | ✅ |

结论：三维自由液面家族 66 个声明数字全部在印刷精度内复现，**无 DRIFT**；
欠账结清，文档与代码从此同口径，漂移会在 CI 变红而不是变成脚注。
