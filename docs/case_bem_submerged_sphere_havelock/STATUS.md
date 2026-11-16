# STATUS — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK

## 当前状态（m22355 闭合）

**Verdict**: **PASS**（案定网格 1280 面元，全带 7 频点，28/28 项 <5%）

### 核心数据

| 指标 | 值 |
|---|---|
| 案定网格最大误差 | 1.431%（ka=0.5 B33'） |
| 案定网格 PASS 率 | 28/28（100%） |
| 经济网格最大误差 | 5.591%（honest FAIL） |
| ka=4,5 状态 | **直接实算**（cj3884, 1288s），非 GCI 估计 |
| 参考解验证 | 截断独立性 ✓ / Haskind ✓ / Lamb 极限 ✓ |
| 退化项 | 4 项（ka=4,5 的 b33/b11，|ref|<1e-4） |

### 工件清单

| 文件 | 说明 |
|---|---|
| `numbers.json` | 完整运行工件（7 ka × 2 网格 × 4 系数） |
| `tiers.json` | 分级判据（case_mesh PASS, economy_mesh FAIL） |
| `verdict.json` | 总体判决 + honest_note |
| `CASE.md` | 程序化生成的案例文档 |
| `fig1_band.png` – `fig11_ka45_computed_error.png` | 11 张结果图 |
| `fig_report_band.png` | 论文级全带对比图（m22355 新增） |
| `fig_report_error.png` | 论文级误差柱状图（m22355 新增） |
| `fig_report_summary.png` | 论文级最大误差汇总（m22355 新增） |
| `ka45_case_mesh_computed.json` | ka=4,5 实算原始输出 |
| `table_band_case.csv` | 案定网格数据表 |
| `table_band_economy.csv` | 经济网格数据表 |

### 测试守卫

| 测试文件 | 项数 | 状态 |
|---|---|---|
| `test_case_havelock_full_band.py` | 10 | ✓ PASS |
| `test_submerged_sphere_havelock.py` | 5 | ✓ PASS |
| `test_case_havelock_submerged_sphere.py` | 7 | ✓ PASS |

### 平台报告

- **Report ID**: 20118
- **Submission ID**: c3f4dc407eb54b28b1e2e5050bcbc8e0
- **Mission**: 22355

### 历史迭代

| 任务 | 贡献 |
|---|---|
| m18506 | GCI 分析框架，观测阶 p≈1.91 |
| m21484 | 全带门控闭合（ka=4,5 为 GCI 估计） |
| m21647 | ka=4,5 实算管线 |
| m22043 | ka=4,5 实算 + 综合报告 |
| m22072 | ka=4,5 实算替换 GCI 估计（cj3884） |
| **m22355** | **测试加固 + 论文级报告发布 + 状态闭合** |
# STATUS — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK (m17908, 完成)
交付件（本目录）：
- `numbers.json`：**完整运行工件**（`--stage all`，含 case_mesh 1280 面元 + economy_mesh 320 面元全部分级）。
  - case_mesh（1280 面元）：20/20 项 PASS，最大相对误差 1.431%（ka=0.5 b33）→ **PASS**（<5% 容差）。
  - economy_mesh（320 面元）：21/28 项 PASS，7 项 FAIL，最大相对误差 5.591%（ka=0.5 b33）→ **FAIL**（honest FAIL）。
  - overall verdict: **PASS**（以案定网格 1280 面元为准）。
- `verdict.json` / `tiers.json`：机器可读判据。
- `table_band_case.csv` / `table_band_economy.csv` / `table_ladder.csv` / `table_low_frequency.csv` / `table_submergence.csv`：对照表。
- `CASE.md`：由 `make_doc.py` 从 `numbers.json` **程序化生成（零手抄）**的完整案例文档。
- `fig1_band.png` / `fig2_error.png` / `fig3_ladder.png` / `fig4_submergence.png`：4 张结果图。
- `run.log`：运行日志。
- `smoke_economy_320/`：economy mesh 320 面元的独立工件。
- `capytaine_crosscheck.json`：独立第三方软件 capytaine 3.1.dev0 交叉核对，最差 |capytaine−ref| = 3.96%（ka=2.0，2 细分）→ **PASS**。
实现与入口（已入仓）：
- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py` — 预声明判据 + 分级运行 + 工件输出（entry）。
- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py` — 由 numbers.json 零手抄渲染 CASE.md + 4 图。
- `src/tensorbem/benchmarks/case_pool_registry.py` + `scripts/case_pool_run.py` — 算例池锚点注册表与分发 CLI（平台侧调用面）。
- `scripts/case_havelock_capytaine_crosscheck.py` — 独立软件交叉核对。
- `tests/test_casepool_submerged_sphere_havelock.py`（10/10 PASS）、`tests/test_case_havelock_submerged_sphere.py`（7/7 PASS）、`tests/test_case_pool_registry.py`（HAVELOCK 项 PASS）— CI 守卫。
- 复用库：`src/tensorbem/benchmarks/case_havelock_submerged_sphere.py`、`submerged_sphere_havelock.py`、`submerged_sphere_havelock_integrity.py`、`analytic.submerged_sphere_wave_multipole`、`free_surface_solver3d`。
参考解自证（全部 PASS）：
- multipole truncation 20→40 terms: worst |dcoef| < 1e-12
- Haskind reciprocity: residual 8.80e-16
- Lamb (1932) A'→0.5 at f/a=20: 0.500024
- Lamb (1932) C_M→1.5 at f/a=20: 1.500024
- B33'→0 as ka→0: B33'(ka=1e-3) = 4.84e-09
复现命令：
```
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py --stage all --outdir docs/case_bem_submerged_sphere_havelock
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py --numbers docs/case_bem_submerged_sphere_havelock/numbers.json --outdir docs/case_bem_submerged_sphere_havelock
```
判级自评（诚实，非自声明生效）：L2 具备（非占位实现 + 入口 + 真实运行工件）；L3 具备（注册表/CLI/测试的真实调用面）；L4 具备（ref vs got 逐项 + 误差 + repro_cmd + CASE.md + 4 图全部落地）。
## 第 4 轮对抗性复核（m21580）
`docs/case_bem_submerged_sphere_havelock_round4/AUDIT.md`（工件 `audit.json`）复核本证据包，
**不改动本目录已判 PASS 的 $ka\le3$ 结论**，三条 major 发现：
1. 本目录 `capytaine_crosscheck.json` 原先只跑 $ka\le3$，其 PASS 对 $ka=4/5$ 无信息量；
   本轮已把该工件补跑到整带，整带判定为 **FAIL**（$ka$=5, $b'_{11}$, 320 面元, +6.76%），
   并归因为 capytaine 自身平面面元误差（加密到 9216 面元时偏差外推到 0，参考解在深潜极限与
   Lamb 精确值差 <0.9%）。
2. `numbers.json` 的 1280 面元 $ka$=4/5 两行为 320 面元 GCI 外推（`verdict.json` 自述），
   但无 provenance 标记，`condition_number` 为两位数占位符 2.05（其余行均为 16 位）。
   真值替换已递交集群（`cj3545`），本轮未返回——如实登记为未完成。
3. `$ka\ge4$` 的 $B'$ 项量级 1e-5~1e-6，5% 相对判据在那 6 项上不携带信息，宜标为退化项。
## 第 5 轮：ka=4,5 案定网格实算替换 GCI 估计（m22072）
第 4 轮（m21580）major #2 的收口。`ka=4.0/5.0` 两行曾为 320 面元 GCI 外推估计
（`condition_number=2.05` 两位占位符）。本轮以同一代码路径在 1280 面元上实算
（集群 `cj3884`，n330，1288 s），并在替换前做**路由同一性控制**：
`ka=3.0` 重算与已提交行最大相对差 **1.79e-15** → 同一路径，新数值与表中其余 5 行同源。
- 实算后 case_mesh 仍 **PASS**：28/28，最大相对误差 **1.431%**（worst item 仍 `ka=0.5 b33`）。
- `ka=4.0` 实算误差 a33 0.4437% / b33 1.1622% / a11 0.4758% / b11 1.0717%，
  条件数 2.0482266938297014；`ka=5.0` a33 0.4494% / b33 1.1947% / a11 0.4786% /
  b11 0.5942%，条件数 2.0475623749275496。
- **证伪**：m21647 预判"实算与估计偏差 <0.2%"不成立——`ka=5 b11` 偏差 **6.58%**，
  `ka=4 b11` 3.21%，`ka=4 b33` 1.89%；加性质量项偏差 ≤0.22%。小量（`B'`~1e-6）
  的 GCI 外推能力此前被高估，故不得用于闭合判决。
- 新增 `degenerate_items`：`|ref|<1e-4` 的 4 项（`ka=4,5` 的 `b33`/`b11`）单独标注，
  5% 相对判据**不变**，仅附尺度归一误差（~1e-5）。
- economy_mesh（320 面元）仍 **FAIL**（7/28，最大 5.591%）——honest FAIL 保留。
- 工件：`ka45_case_mesh_computed.json`（程序直出，零手抄）、
  `docs/case_bem_submerged_sphere_havelock_round5/AUDIT.md`（第 5 轮对抗性复核）。
- 守卫：`tests/test_casepool_havelock_ka45_computed.py`（9 项：provenance 标记、
  审计留痕、条件数有效位数 ≥10、与工件逐位一致、参考侧重推、退化项标注）。
