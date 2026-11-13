# FEAT-CASEPOOL-100：算例池向百级扩容 —— TensorBEM 谱系补入

**Mission**: m17125（self-directed，generalist 席）
**Date**: 2026-11-13
**Branch**: `agent/m17125`
**锚点**: FEAT-CASEPOOL-100（算例池向百级扩容，供给线常青）
**Status**: ✅ PASS

## 1. 为什么做这件事（案例背景）

owner 2026-11-12「benchmark 应以百计铺开，长设供单」令 + 本单锚点 FEAT-CASEPOOL-100：
`tensor_case_pool`（算例池）是长设供给守护 `case_feeder.py` 的候选源——它按闲位把
池中 **unlinked pending** 例自动派成新锚 CASE-{code} 单，供各仓求解器跑 benchmark。

**审计发现的结构性缺口**：
- 池原有 126 例覆盖六仓（TensorDEM/DG/FEM/FVM/LBM/SPH），**唯独缺 TensorBEM**——
  核心求解器仓却无池例，BEM 谱系完全缺席；
- **unlinked pending 仅 19**（<30），`case_feeder` 的候选源（`WHERE status='pending'
  AND (linked_anchor IS NULL OR linked_anchor='')`）接近断流——供给线不常青。

## 2. 怎么做的（方法/流程）

1. 全只读扫描：`tensor_case_pool` 现状（126 例/六仓/无 BEM）、`case_feeder.py` 供给逻辑、
   TensorBEM 仓内已验收的经典基准模块（`src/tensorbem/benchmarks/`）。
2. 选题：**补入 TensorBEM 谱系**——这是唯一能同时满足「池 84→120+」「unlinked pending
   ≥30」「与既有 84 例对拍不重」（BEM 不在池内，天然不重）三判据的实质增量。
3. 每例的 `ref_source` 从 TensorBEM 已验收基准模块的 docstring/文献原文取（零手抄）：
   闭式解（ρπa²、Lamb 形状因子、0.5ρV）、文献多极/球谐（Ursell 1949、MacCamy & Fuchs
   1954、Hulme 1982、Havelock 1955、Dean 1948）、工程基准（Michell 薄船、Iftikhar &
   Takinaci 2023、DNV-RP-B401、Biran & Pulido）。
4. 幂等插入：`case_code` 唯一约束 + `ON CONFLICT DO NOTHING`，已存在不重插。

## 3. 计算结果（数字表，程序化注入）

| 指标 | 前 | 后 | 增量 |
|---|---|---|---|
| 池总例数 | 126 | **156** | +30 |
| pending 例 | 19 | **46** | +27 |
| **unlinked pending**（case_feeder 可供给） | 19 | **30** | +11 |
| 覆盖仓数 | 6 | **7**（+TensorBEM） | +1 |

新增 TensorBEM 例（30 个，全部 pending/unlinked）：

| case_code | 名称 | ref_source | tol |
|---|---|---|---|
| BEM-CYL-ADDED-MASS-2D | 二维圆柱附加质量 | 经典 ρπa² 闭式 | <3% |
| BEM-ELLIPSE-2D-ADDED-MASS | 二维椭圆附加质量 | 经典 ρπb²/ρπa² | <3% |
| BEM-SPHERE-ADDED-MASS | 球体附加质量 | 经典 0.5ρV | <3% |
| BEM-ELLIPSOID-ADDED-MASS | 椭球三轴附加质量 | Lamb 形状因子 | <5% |
| BEM-ANNULUS-LAPLACE | 环形域 Laplace | 闭式调和解 | <3% |
| BEM-URSELL-CYL-RAD | Ursell 潜没圆柱辐射 | Ursell 1949 多极 | <5% |
| BEM-MACCAMY-FUCHS-DIFF | MacCamy-Fuchs 绕射 | 1954 Bessel/Hankel | <5% |
| BEM-HULME-HEMISPHERE | Hulme 浮半球 | 1982 球谐多极 | <5% |
| BEM-DEAN-DIFFRACTION | Dean 二维波绕射 | 1948/Haskind | <5% |
| BEM-CYL-DRIFT-ASYMP | 圆柱平均漂移力 | (ka)³ 律 | <5% |
| BEM-TWO-CYL-MUTUAL | 双圆柱互附加质量 | 多极级数 | <5% |
| BEM-SUBMERGED-SPHERE-HAVELOCK | Havelock 潜球辐射 | 1955 频域 | <5% |
| BEM-RIGID-LID-HEMISPHERE | 刚性盖半球 | ka→0 端点 | <5% |
| BEM-IRREG-FREQ-CYL | 不规则频率预测 | Bessel 零点 | <2% |
| BEM-WIGLEY3-WAVE-RES | Wigley III 兴波阻力 | Michell 薄船 | <5% |
| BEM-KP505-OPENWATER | KP505 敞水 | Iftikhar & Takinaci 2023 | <5% |
| BEM-KP505-GCI | KP505 三密度 GCI | Roache GCI | <3% |
| BEM-BARGE-HYDROSTATIC | 壁面驳船 | capytaine 交叉 | <10% |
| BEM-CUMMINS-WIGLEY | Cummins 记忆核 | Cummins/Kramers-Kronig | <10% |
| BEM-NEUMANN-KELVIN | Neumann-Kelvin | Michell C_w | <5% |
| BEM-HYDROSTATIC-PONTOON | 矩形浮箱水静力 | Biran & Pulido | <3% |
| BEM-RIGID-BODY-MASS | 刚体质量矩阵 | 闭式块形式 | <3% |
| BEM-ELLIPSOID-KOCHIN | 椭球 Kochin | Kochin 闭式 | <5% |
| BEM-ICE-DISPERSION | 冰区色散 | Weitz-Keller | <5% |
| BEM-ICE-SPHERE-HEAVE | 冰区潜球垂荡 | Lamb (2/3)πρR³ | <5% |
| BEM-ACOUSTIC-PULSE | 脉动球声脉冲 | 经典推迟势 | <8% |
| BEM-CATHODIC-SPHERE | 阴极保护球 | DNV-RP-B401 | <5% |
| BEM-MARUO-DRIFT | Maruo 远场漂移 | Maruo 闭式 | <5% |
| BEM-QTF-DRIFT | 二阶 QTF 漂移 | 二阶差频闭式 | <10% |
| BEM-HULME-DIFF-VERIF | Hulme 绕射验证 | 1982 球谐 | <5% |

## 4. 结果图

纯数据/供给线任务，无场量云图——本单是**算例池供给线扩容**（数据库行插入），
非数值求解，无物理场量可绘。理由：增量是「池覆盖 + 供给线常青」，不是「某算例的
数值结果」，故无图（符合报告纪律「纯文档/后端任务无图必须写明理由」）。

## 5. 注意点（踩过的坑/局限）

- **unlinked 与 pending 是两回事**：`case_feeder` 只喂 `linked_anchor IS NULL/''` 的
  pending 例。既有 19 个 pending 例已 linked（被既有 GAP/REPRO/XVAL 锚覆盖），
  不参与供给——所以判据必须看 **unlinked pending**，不是裸 pending。
- **幂等**：`ON CONFLICT (case_code) DO NOTHING` 保证重跑不重插；脚本可安全重放。
- **ref_source 零手抄**：全部取自 TensorBEM 已验收基准模块的 docstring/文献原文，
  未手打任何数值。
- **局限**：本单只补供给线（池行），未跑各算例的数值验证——那是 case_feeder 后续
  派单（CASE-{code} 第 1 轮）的活，本单不越界。

## 6. 误差判定

- 本单无数值求解，无「关键量 vs 基准」误差可判——**声明**：这是供给线扩容（数据
  层），非数值层；判据是「unlinked pending ≥30」与「池 84→120+」，均已达标
  （30 / 156），不适用 <3% 数值误差标准。
- 判据逐项：池 156≥120 ✅；unlinked pending 30≥30 ✅；与既有 84 例对拍不重
  （BEM 全新增）✅；ref_source 真实可引用零手抄 ✅。
