# TensorBEM Hulme 半球带扩展单元交付物钉死（PR#1195 重开·管到底第3轮·m25365）

**Task**: m25365（重开 TensorBEM#1195，管到底第3轮）· **Branch**: `agent/m25365` · **Repo**: TensorBEM · **Date**: 2026-11-19

**结论**: PR#1195 的 m21845 交付物（casepool README +19 行、`band_current_code/cells/` 9 个单元 JSON、cells 目录）**自 #1026 起已全部在 main**（commit 52641fdb，2026-11-15）。本轮不重复实现，补上缺的那块证据面——**一个纯 JSON 一致性测试，把 9 个单元交付物钉死**：清单 1:1、schema 完整、每个 rel_error 从文件自身 got/ref 重推、幅相从复数分量重推、与 `results.json` 跨工件逐位一致、E1/E3/E5 判决从单元文件单独重放并与记录判决一致。10 项断言全过（0.25 s，零求解）。

## 1. 案例介绍

**物理背景**：Hulme (1983) 浮体半球基准——半径 a = 1 m 的半球，平底面位于平均自由面，无限水深，受单位幅值正航波激励，做 surge（11）与 heave（33）自由振荡。六个计分量：附加质量 `A11`、`A33`，阻尼 `B11`、`B33`，兴波力幅值 `|Fx|`、`|Fz|` 及其相位。case-pool 条目 `BEM-HULME-HEMISPHERE`（id 138），每格容差 5 %（开案以来未变）。

**几何/参数**（本 PR 涉及的 `band_current_code` 带扩展研究，round 2，ka 2.2 → 4.0）：

| 参数 | 值 |
|---|---|
| 网格 | 1024 常曲率面板（subdivisions 4）；粗档 256 面板（subdivisions 3） |
| 求解 | 3-D 频域 BEM（`tensorbem.free_surface3d`），CHIEF 开/关成对 |
| 参考 | 无波多极子级数（Hulme 半解析法重推，144→192 项漂移 7.42e-05 @ ka=4，134.7× 低于 5 % 门） |
| 环境 | python 3.11.6 / numpy 2.4.6（控制面）；loongarch64 |

**本 PR 的对象**：`band_current_code/cells/` 下 9 个单元 JSON（m21845 成熟交付物）：

| kind | ka | panels | chief | 文件 |
|---|---|---|---|---|
| extension | 2.2 / 2.58 / 3.0 / 3.5 / 4.0 | 1024 | on | `extension_ka{2.2,2.58,3,3.5,4}_n1024_chief_on.json` |
| chief_off | 2.58 / 3.0 | 1024 | off | `chief_off_ka{2.58,3}_n1024_chief_off.json` |
| ladder_coarse | 3.0 / 4.0 | 256 | on | `ladder_coarse_ka{3,4}_n256_chief_on.json` |

## 2. 怎么做的

**PR#1195 交付物对账**（owner 2026-11-05 关闭原 PR，功能要点是否已在 main）：

| 文件 | PR#1195 声明 | main 现状 | 判定 |
|---|---|---|---|
| `casepool_bem_hulme_hemisphere/README.md` | +19/-0 | 已入 main（Merge #1026，52641fdb） | ✓ |
| `cells/chief_off_ka2.58_n1024_chief_off.json` | +51/-0 | 已入 main（同上） | ✓ |
| `cells/chief_off_ka3_n1024_chief_off.json` | +51/-0 | 已入 main（同上） | ✓ |
| `cells/extension_ka2.2_n1024_chief_on.json` | +51/-0 | 已入 main（同上） | ✓ |
| `band_current_code/cells/`（目录） | 新增 | 已入 main（9 个单元 + 9 个 spec） | ✓ |

五条全部在 main，所以本轮**不重复实现**，只补证据面（新增 1 个测试 + 1 个图脚本 + 1 张图 + 本报告 + 1 个发布脚本）。测试面新增内容全部落在 `tests/test_hulme_band_current_cells.py`（新文件），**未触碰任何 `src/` 核心模块**；GitHub 当前 0 个 open PR，无热文件重叠。

**测试设计**（`tests/test_hulme_band_current_cells.py`，六层，全部从已提交数字重推，任何一位数字被手改都会被抓）：

1. **inventory**：`cells/` 与 `specs/` 9:9 一一对应；PR#1195 点名的 3 个文件在列且内部 `name` 字段与文件名一致；
2. **schema**：每个单元带全量记录（六量 + reference 块 + rel_error 块 + 求解诊断 + 几何输入）；`kind`/`chief`/`ka`/`nu`/`panels`/`subdivisions` 互洽；
3. **内部一致性**：`rel_error_X == (X − ref_X)/ref_X`（1e-12）；`|Fx| == hypot(Fx_real, Fx_imag)`（1e-12）；相位与 delta 相位从 `atan2` 重推并包卷到 ±180°（1e-9°）；
4. **几何前置**：1024 面板单元满足 2 % 体积/水线面前置（E2 口径）；256 面板粗档在 E2 范围外，只给 5 % 界防几何回归藏进粗档；
5. **跨工件**：每个单元文件与 `results.json` 同名行在六量 + 条件数上 1e-9 相对一致（今日逐位相同，max |Δ| = 0.0）；
6. **判决重放**：E1/E3/E5 从单元文件单独重算，复现记录判决（E1 FAIL @ ka=4 A11；E3 PASS；E5 PASS 且 CHIEF-off 对照在门外）。

**图**：`band_current_code/make_fig_m25365.py`（纯读已提交工件，零求解）→ `fig_m25365_band_cells.png`：(a) 场量云图——committed 场数据（ka=1.0，256 面板）上 `|φ_surge|` 的半球面分布；(b) 比较曲线——1024 面板扩展带六量相对误差 vs ka，5 % 门与 15 % 硬限同轴，标出 ka=4 A11 网格受限失败点；(c) 不规则频率处 CHIEF 开/关对照（带最大误差 + 条件数）。

## 3. 计算结果（程序化注入，零手抄；源 `band_current_code/cells/*.json`）

### 3.1 扩展带（1024 面板），六量相对误差

| ka | A11 | B11 | A33 | B33 | Fx | Fz | band-max |
|---|---|---|---|---|---|---|---|
| 2.2 | +0.27% | −0.52% | −0.63% | +1.35% | −0.40% | +0.73% | 1.35% |
| 2.58 | +0.23% | −0.42% | −0.47% | +2.25% | −0.45% | +1.05% | 2.25% |
| 3.0 | −0.03% | −0.30% | −0.31% | +2.31% | −0.57% | +1.40% | 2.31% |
| 3.5 | −0.98% | −0.15% | −0.42% | +2.78% | −0.60% | +2.18% | 2.78% |
| 4.0 | **+8.94%** | −1.16% | −0.47% | +3.25% | −1.98% | +1.88% | **8.94%** |

### 3.2 CHIEF 对照对（不规则频率邻域）

| ka | chief | band-max err | cond | A33 | B33 |
|---|---|---|---|---|---|
| 2.58 | on | 2.25% | 8.29 | 0.39886 | 0.06450 |
| 2.58 | off（对照） | 6.12% | 65.32 | 0.41959 | 0.06259 |
| 3.0 | on | 2.31% | 2.73 | 0.40980 | 0.04620 |
| 3.0 | off（对照） | 2.24% | 2.82 | 0.40990 | 0.04617 |

### 3.3 网格梯（256 → 1024，加密不恶化）

| ka | 256 band-max | 1024 band-max | 改善 |
|---|---|---|---|
| 3.0 | 9.73% | 2.31% | 4.2× |
| 4.0 | 34.49% | 8.94% | 3.9× |

## 4. 结果图

`fig_m25365_band_cells.png`（`band_current_code/make_fig_m25365.py` 生成，纯读已提交工件）：

- 图 (a)：场量云图——committed 场数据（ka = 1.0，256 面板）上 `|φ_surge|` 的半球面分布（3-D 散点，viridis）；
- 图 (b)：比较曲线——1024 面板扩展带六量相对误差 vs ka，5 % case-pool 门与 15 % 硬限同轴，ka = 4 A11 失败点标注；
- 图 (c)：CHIEF 开/关对照——ka = 2.58 / 3.0 的带最大误差与条件数（2.58 处 CHIEF-off 对照 6.12 % 在门外、cond 65.3，CHIEF-on 2.25 % 在门内、cond 8.3）。

## 5. 注意点

- **参考与求解器共享内核**：参考级数与求解器共用 `tensorbem.free_surface3d.wave_function_3d`，一致性本身抓不到该内核的缺陷；独立轴是 capytaine 第三方交叉（case 级已入库）与 ka→0 刚盖偶模级数。这是 case README 已登记的诚实局限，本 PR 不改变它。
- **ka = 4 是网格受限，不是代码回归**：E1 在 ka = 4 A11 处 FAIL（8.94 % > 5 %），256→1024 改善 3.9× 说明误差随加密下降；带边裁定（`evidence/band_edge_resolution.json`）把信用止于 ka = 3.5，ka = 4 如实记为 mesh-limited。本 PR 的测试把这一判决**钉死**：谁把 ka = 4 的 A11 手改成 <5 % 来「修绿」，E1 重放测试立刻红。
- **CHIEF-off 单元是故意留在门外的对照**：chief_off_ka2.58 的 6.12 % 不是缺陷，是 E5 的对照臂；测试钉的是「对照在门外、CHIEF-on 在门内、对照条件数更大」这一结构，而不是要求对照也过门。
- **256 面板粗档不在 E2 几何前置范围内**（vol_err 2.32 % > 2 %），测试对其只给 5 % 界，防止几何回归藏进粗档，也不误伤粗档。
- **本 PR 零求解**：所有断言从已提交数字重推；不重跑 1024 面板（单格 34–111 s），避免报告脚本变成分钟级作业。

## 6. 误差判定（逐项，基准 = 重推多极子级数）

| 关键量（1024 面板） | 实测 | 基准 | 是否 <3% |
|---|---|---|---|
| ka=2.2 六量 | 0.23–1.35 % | 重推级数 | **是** |
| ka=2.58 六量 | 0.23–2.25 % | 重推级数 | **是** |
| ka=3.0 六量 | 0.03–2.31 % | 重推级数 | **是** |
| ka=3.5 六量 | 0.15–2.78 % | 重推级数 | **是** |
| ka=4.0 A11 | 8.94 % | 重推级数 | **否**（网格受限，如实记录；5 % 门同破） |
| ka=4.0 B33 | 3.25 % | 重推级数 | **否**（同上） |
| ka=4.0 其余四量 | 0.47–1.98 % | 重推级数 | **是** |
| chief_off ka=2.58（对照） | 6.12 %（Fz） | 重推级数 | **否**（对照臂，预期在门外） |
| chief_off ka=3.0（对照） | 2.24 % | 重推级数 | **是** |
| 256 面板 ka=3 / ka=4 | 9.73 % / 34.49 % | 重推级数 | **否**（粗档，预期） |

**显式回答**：扩展带 ka ≤ 3.5 的全部 24 个量（4 ka × 6 量）**全部 <3 %**；ka = 4 有 2 个量（A11 8.94 %、B33 3.25 %）≥3 %，属网格受限的诚实 FAIL，与已入库的带边裁定一致，不粉饰。

## 7. 测试复验（本 PR）

| 范围 | 命令 | 结果 |
|---|---|---|
| 本 PR 新增 | `python -m pytest tests/test_hulme_band_current_cells.py -q` | **10 passed**，0.25 s |
| 同案回归（band 家族） | `python -m pytest tests/test_hulme_band_edge_analysis.py tests/test_hulme_band_edge_resolution.py tests/test_hulme_np246_contradiction_resolution.py -q` | **31 passed, 1 skipped**（skip 为预存条件：failing artifact 无 git commit 记录），1.4 s |

**热文件声明**：本 PR 不触碰 `src/` 任何核心模块；新增文件为 `tests/test_hulme_band_current_cells.py`、`band_current_code/make_fig_m25365.py`、`band_current_code/fig_m25365_band_cells.png`、本报告与发布脚本。GitHub 当前 0 个 open PR，无重叠。
