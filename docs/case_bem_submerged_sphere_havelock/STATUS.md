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
