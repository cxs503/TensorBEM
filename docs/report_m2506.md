# TensorBEM hydrostatic 验证套件 ImportError 欠账修复 —— 补报（mission 2506）

**报告性质**：补报单（scenario_key=`report-backfill`）。原单（mission 2506，repo=TensorBEM）已成功交付但未发布报告，且**原单工件目录未留档**。本报告为诚实 meta-报告：

- 原单声明以平台嵌入文本为准，该文本在根因句首 "**`te**…" 处**截断**，全文不可得；
- 所有原单当时运行的数字一律标注 **未留档**，本报告不编造；
- 本报告补充的数字全部来自 ①仓库已提交工件（程序化 `json.load` 提取）②本补单 2026-11-26 在工作树 `/root/tp-agent/work/m24910`（分支 `agent/m24910` @ `04cacde2`）的**复跑**（pytest + 基准 CLI，JSON 程序化提取，零手抄）。

---

## 1. 案例介绍

**物理背景**：`tensorbem.hydrostatic` 由湿表面网格计算排水体积 ∇、浮心 KB、水线面面积 A_wp 与二阶矩 I_xx/I_yy，以及 6×6 回复矩阵 C。C 是频域 RAO 方程 `[−ω²(M+A)+iωB+C]ξ=F` 中**唯一的回复项**，其误差直接传入 GM、横摇/纵摇回复与全部 RAO 结果——因此需要能把**实现误差**与**离散（faceting）误差**分离的验证工件。

**验证工况与闭式基准（参数表）**：

| 工况 | 几何/参数 | 关键闭式量（基准值，程序化提取） |
|---|---|---|
| box（矩形驳船，复跑） | L×B×T = 10×4×2 m | ∇=LBT=80.0 m³，A_wp=LB=40.0 m²，I_xx=LB³/12=53.333 m⁴，I_yy=BL³/12=333.333 m⁴，KB=T/2=1.0 m，BM_T=B²/(12T)=0.6667 m |
| spheroid（半椭球，复跑） | a,b,c = 5,2,2 m | ∇=2πabc/3=41.888 m³，A_wp=πab=31.416 m²，I_xx=πab³/4=31.416 m⁴，I_yy=πba³/4=196.350 m⁴，KB=5c/8=1.25 m，BM_T=3b²/(8c)=0.75 m |
| barge（仓库工件） | 10×4×1 m，nx×ny 细分 | ∇=40.0 m³，A_wp=40.0 m²，I_xx=53.333，I_yy=333.333，KB=0.5，BM_T=1.333，BM_L=8.333 |
| hemisphere（仓库工件） | R=1 m，细分阶梯 16→4096 面元 | ∇=2πR³/3=2.0944，A_wp=πR²=3.1416，I=πR⁴/8=0.7854，KB=5R/8=0.625，BM=3R/8=0.375 |
| elliptic_cylinder（仓库工件） | a=5, b=2, T=1 m，水线 16→128 段 | A_ellipse=πab=31.4159 m² |
| hemisphere_restoring（仓库工件） | R=1，KG=0.6 m（1024 面元） | C_heave=ρgA_wp=31578.71，C_roll=ρg∇GM=8420.99，GM=KM−KG=0.4 |

驳船全部面元为平面且位于解析面上 → 离散静水力必须**精确到舍入误差**，非零误差即实现缺陷；曲面工况度量 O(h²) faceting 误差与曲水线积分路径。

## 2. 怎么做的

**原单做了什么（嵌入原单声明，截断处如实保留）**：

> "## 做了什么 修复了 TensorBEM 中 hydrostatic 验证套件的 ImportError 欠账（m2228 遗留）。 **根因**：`te…"

**根因重构（以下为仓库证据链推断，非原单原文）**：

1. `tests/test_hydrostatic_verification.py` 在**收集期**即执行 `from tensorbem.benchmarks.hydrostatic_benchmarks import …` 与 `from tensorbem.geometry3d import box_hull_mesh, hemisphere_hull_mesh, spheroid_hull_mesh`——被引用模块/符号在 main 缺失时 pytest 收集阶段直接 ImportError（与根因句以 "`te…" 开头相符，即测试文件引用缺失）；
2. m2228 交付自动入库 commit `51e8625d`（2026-11-03，提交信息自证 "**5 个文件被门禁滤除**"）实际只落下 3 个文件：dev report（87 行）、`src/tensorbem/benchmarks/hydrostatic_benchmarks.py`（543 行）、`tests/test_hydrostatic_verification.py`（223 行）；
3. PR #54 `fix/m2228-salvage`（2026-11-03 合入，标题 "Restore m2228 hydrostatic verification implementation (**fix main CI collection failure**)"）恰好恢复被滤除的 5 个文件：`geometry3d.py` 网格生成器、`__init__.py` 导出、README、docs 报告与收敛图。

即：m2228 交付物被平台门禁部分滤除 → 套件在 main 收集失败（ImportError）→ 该欠账由本单（m2506）修复。**原单当时的具体 diff 与运行工件：未留档。**

**本补单复跑验证（2026-11-26，程序化）**：

1. `PYTHONPATH=src python3 -m pytest tests/test_hydrostatic_verification.py -q` → **26 passed, 1 warning, 1.12 s**（收集期无 ImportError——欠账确已修复的现势证据）；
2. `PYTHONPATH=src python3 -m tensorbem.benchmarks.hydrostatic_benchmarks --hull box --json /tmp/box_fresh.json`、`--hull spheroid --json /tmp/sph_fresh.json` → 表 1（直接读 JSON）。

**工具链**：Python 3 / numpy / pytest；仓库 TensorBEM，分支 `agent/m24910` @ `04cacde2`。

## 3. 计算结果

**表 1　本补单复跑（2026-11-26，agent/m24910 @ 04cacde2，CLI JSON 程序化提取）**

| 工况 | 量 | 闭式基准 | 最细档相对误差 |
|---|---|---|---|
| box 10×4×2 | ∇、Cb、A_wp、I_xx、I_yy、KB、BM_T、BM_L（全部 8 量） | 见 §1 | **全部 0.0**（观测阶 NaN：零误差无阶可估） |
| spheroid 5,2,2 | ∇（=Cb） | 41.888 m³ | 5.870e-3 |
| spheroid | A_wp | 31.416 m² | 1.606e-3 |
| spheroid | I_xx / I_yy | 31.416 / 196.350 m⁴ | 3.208e-3 |
| spheroid | KB | 1.25 m | 1.252e-3 |
| spheroid | BM_T / BM_L | 0.75 m | 2.677e-3 |
| spheroid | 观测收敛阶（∇/A_wp/I_xx/KB/BM_T） | 理论 O(h²)=2 | 1.983–1.998 |

**表 2　仓库已提交工件 `hydrostatic_verification_report.json`（根目录，api-sync 链现势工件；原单当时运行数字未留档）**

| 工况 | 量 | 相对误差 | 观测阶 |
|---|---|---|---|
| barge 10×4×1，最细档 5×3 段（62 面元） | ∇ / A_wp / I / C_heave / C_roll / C_pitch | 0 ～ 3.55e-16（工况 max_error=6.66e-16） | —（精确） |
| hemisphere R=1，4096 面元 | ∇ | 1.472e-3 | 1.9957 |
| hemisphere | A_wp | 4.015e-4 | 1.9995 |
| hemisphere | I_transverse | 8.029e-4 | 1.9984 |
| hemisphere | KB | 3.135e-4 | 1.9982 |
| hemisphere，Richardson 外推 | ∇ / A_wp / I_xx | **0.0** | — |
| hemisphere，最粗档 16 面元（对照） | ∇ | 2.975e-1 | — |
| elliptic_cylinder，水线 128 段（384 面元） | A_wp vs 椭圆 | 4.015e-4 | 1.9995 |
| elliptic_cylinder | I_transverse vs 多边形精确值 | 5.66e-16 | — |
| hemisphere_restoring（KG=0.6，1024 面元） | C_heave / C_roll | 1.606e-3 / 1.429e-3 | — |
| hemisphere_restoring | GM / ∇ / KB / BM | 4.467e-3 / 5.870e-3 / 1.252e-3 / 2.677e-3 | — |

**原单（m2506）当时的运行数字：未留档。**

## 4. 结果图

无图。理由：①原单工件目录未留档，原始图缺失；②本单为补报 meta-报告，按补报单规范 `figures=[]`，以 markdown 表格承载程序化提取的数字证据；③仓库内已有可查图件：`docs/figures/hydrostatic_convergence.png`（PR #54 恢复的 m2228 收敛图）及生成器 `src/tensorbem/benchmarks/hydrostatic_figures.py`（根目录验证 JSON 的生成源），读者可一键再生成。

## 5. 注意点

- **诚实性边界**：原单声明截断、工件未留档；根因部分为仓库证据链推断（已逐条标注证据来源）；数字全部来自仓库已提交工件与本补单复跑，无手抄、无编造。
- 粗网格误差大是 **faceting 离散误差**而非实现误差：hemisphere 16 面元 ∇ 误差 29.7% → 4096 面元 0.147%，阶≈2；驳船（全平面面元）才是实现回归测试，其任何非零误差都指向实现缺陷。
- Wigley 抛物线船体无法用 quad-sheet 方案网格化（艏艉柱 y≡0 处龙骨-艏柱角点产生零体积 sliver、镜像绕向相反，`WettedHullMesh` 正确拒绝为非流形）——**细长船型静水力仍未验证**（m2228 已声明的 scope 限制，本单未解决）。
- 回复矩阵 C 线性化于零横倾/零纵倾，大角度稳性 out of scope；验证工况水线均前后/左右对称，非对称水线质心路径（多体、附体贯穿）未测。
- **门禁滤除教训**：交付自动入库可能滤文件（m2228 被滤 5 个），测试在收集期即 ImportError、CI 直接红；后续交付应在 api-sync 后自查文件清单与 CI 收集日志。
- 复跑环境提示（沿 m2228 报告）：满载主机上全套件运行极慢，建议按模块定向回归（本单 26 测仅 1.12 s）。

## 6. 误差判定（关键量 vs 基准，逐项，显式回答是否 <3%）

| # | 工况.量（来源） | 相对误差 | <3%？ |
|---|---|---|---|
| 1 | box 全部 8 量（复跑） | 0.0 | ✅ 是 |
| 2 | spheroid ∇（复跑） | 0.587% | ✅ 是 |
| 3 | spheroid A_wp（复跑） | 0.161% | ✅ 是 |
| 4 | spheroid I_xx / I_yy（复跑） | 0.321% | ✅ 是 |
| 5 | spheroid KB（复跑） | 0.125% | ✅ 是 |
| 6 | spheroid BM_T / BM_L（复跑） | 0.268% | ✅ 是 |
| 7 | barge 全部量（仓库工件） | ≤6.66e-16（=6.66e-14 %） | ✅ 是 |
| 8 | hemisphere 4096 面元 ∇ / A_wp / I / KB（仓库工件） | 0.147% / 0.040% / 0.080% / 0.031% | ✅ 是 |
| 9 | hemisphere Richardson ∇ / A_wp / I_xx（仓库工件） | 0.0 | ✅ 是 |
| 10 | elliptic A_wp vs 椭圆（仓库工件） | 0.040% | ✅ 是 |
| 11 | hemisphere_restoring C_heave / C_roll / GM / ∇（仓库工件） | 0.161% / 0.143% / 0.447% / 0.587% | ✅ 是 |

**显式回答：全部关键量 vs 闭式基准误差均 <3%（最大 0.587%，spheroid 排水体积）。无基准的量：无——所有验证量均有闭式解。** 观测收敛阶 1.98–2.00 与曲面一阶几何离散的 O(h²) 理论一致；粗档（16 面元 29.7%）为离散误差，已由收敛阶梯显式分离，不计入实现误差判定。

---

*补报完成时间：2026-11-26；复跑环境：工作树 /root/tp-agent/work/m24910（agent/m24910 @ 04cacde2）；本报告未修改任何业务代码。*
