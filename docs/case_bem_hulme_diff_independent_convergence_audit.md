# 第三方路线收敛审计：CASE-BEM-HULME-DIFF-VERIF（Hulme 绕射验证·3D）

*审计对象：`benchmarks/verified/casepool_bem_hulme_diff_verif/`（台账 slug `CASE-BEM-HULME-DIFF-VERIF`，case-pool id 177）的门 `G4_independent_route`。*
*本文件的每个数字都由 `benchmarks/verified/casepool_bem_hulme_diff_verif/capytaine_convergence_audit.json` 渲染，该 JSON 由
`scripts/capytaine_hulme_diff_convergence_audit.py` 程序化产出；`tests/test_casepool_hulme_diff_convergence_audit.py` 反查代码、
工件与本文是否一致。*

**结论（诚实版）：** 门 G4 原先把「第三方代码在**未收敛**网格上的误差（4.30%）」当作「两个代码的差异」。审计发现其
Richardson 外推把**加密比取反**（`r = sqrt(N_coarse/N_fine) < 1`），于是 `r**p - 1 < 0`：修正项符号被翻转、量级被
`1/(1-r**p)` 放大，外推后的误差（7.48%）反而**大于**最细网格误差，12/12 个格子都是如此——这不是物理，是缺陷。
修正后同一组数据的收敛值差为 **2.51%**（假定 p=2）/ **0.37%**（实测阶），全部 <5% 台账容差。此外新跑了
两条加密阶梯（capytaine 经纬球 1024→1600→2304 面；**与 TensorBEM 同网格族**的正八面体半球 256→1024→4096 面），
观测阶 0.41–1.07（≈O(h)），最细网格最差 2.17%，外推 1.45%（p=2）/ 0.22%（实测阶）——残差是第三方代码自身的
一阶离散误差，不是两条路线的公式分歧。

## 1. 案例介绍（物理背景/几何/参数表）

与锚点一致，未改动任何约定：表面 piercing 半球，半径 $a=1$ m，平面端面与静水面齐平（$z=0$），曲面浸没，无限水深。

| 项 | 值 |
|---|---|
| 排水体积（解析） | $\nabla = 2\pi a^3/3 = 2.094395$ m³ |
| 水线面（解析） | $A_{wp} = \pi a^2 = 3.141593$ m² |
| 密度 / 重力 | $\rho = 1025.0$ kg/m³，$g = 9.80665$ m/s² |
| 入射波幅 | $A = 1$ m（波向 $+x$，$e^{-i\omega t}$） |
| 频段（锚点） | $ka \in \{0.2,0.5,0.8,1.0,1.5,2.0\}$ |
| 审计加密频段 | $ka \in \{0.5,1.0,2.0\}$ |
| 参考解 | `tensorbem.analytic.hulme_hemisphere_radiation`，多极项数 144、求积阶 288 |
| 度量 | 总激励力 $|F - F_{\rm Hulme}|/|F_{\rm Hulme}|$（刚性散射+精确 Froude-Krylov） |
| 台账容差 | 5%（逐格，预先登记，未调） |

**参考值来源的诚实说明：** Hulme (1982) 原文表格在付费墙后，本机无副本，仓库内没有任何一位数字是转抄的；
参考值由本仓多极级数重算（截断阶梯 48/96/144 收敛，验证裁决依赖「级数 ↔ Kelvin BIE ↔ Rankine BIE」三方
不共享离散化的三角锁定，而非原文数字）。台账验收语句「目标数值任务从 ref 原文查得」在本机条件下**无法满足**，
本审计不粉饰这一点；它把可做的部分做到位：第三方路线的收敛**实测**而非假设。

## 2. 怎么做的（方法/流程/工具链）

1. **复现闸**：在冻结分辨率 (32,64)=1024 面上重跑 capytaine，与 `capytaine_scattered.json` 对比 → 最差偏差
   2.2e-16（判 **REPRODUCED**）。第三方路线的工件确实是跑出来的。
2. **冻结阶梯再分析**：只读 `capytaine_scattered.json` 的 64/256/576/1024 面阶梯，用**正确**的加密比重算
   Richardson（$r = h_{\rm coarse}/h_{\rm fine} = \sqrt{N_{\rm fine}/N_{\rm coarse}} > 1$），并与「取反版」并列，
   量化缺陷幅度。
3. **加密阶梯实测**（新算）：capytaine 经纬球 (32,64)/(40,80)/(48,96) → 1024/1600/2304 面。
4. **同网格族对照**（新算）：把 `tensorbem.geometry3d.hemisphere_hull_mesh`（正八面体细分，subdivisions=3/4/5
   → 256/1024/4096 面）直接喂给 capytaine。两条路线吃**同一个多面体**，于是「公式分歧」与「capytaine 的
   经纬球是更差的多面体」被分开。
5. 每条频率的总力 = capytaine 散射力 + `run_case.exact_froude_krylov` 的精确入射力（capytaine 的
   `forces` 只含散射部分，这个约定由锚点的 ka=0.05 探针钉住）。

工具链：capytaine 3.1.dev0（loongarch64，Python 3.11），numpy；`src/` 内不导入 capytaine（由测试用 AST 守住）。
运行时间：全阶梯审计 ≈ 80 s（3 线程，nice 10）。

### 2.1 缺陷定位与修复

```python
# scripts/capytaine_hulme_diff_scattered.py（修复前）
n2, n1 = panels[-1], panels[-2]
r = math.sqrt(n1 / n2)                    # < 1  ← 加密比取反
return f2 + (f2 - f1) / (r**order - 1.0)  # 分母为负 → 修正项反向放大
# 修复后
r = math.sqrt(n2 / n1)                    # > 1：h_coarse / h_fine
```

修复后重新生成 `capytaine_scattered.json` 与 `results.json`（同一条 `run_case.py`，全部 6 道门仍 PASS，
锚点主门 G1 最差 **0.5361%** 不变）。审计工件里保留「取反版」重算，用于**量化**缺陷而不是只描述它。

## 3. 计算结果（数字表）

### 3.1 冻结阶梯 12 格：三种读法

阶梯 64/256/576/1024 面（capytaine 经纬球），体积 1.9633/2.0610/2.0795/2.0860 m³（解析 2.0944）。

| ka / dof | 64 | 256 | 576 | 1024（最细） | 修正 Richardson (p=2) | 取反版（原发表） |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 surge | 2.29% | 1.63% | 1.18% | 0.91% | 0.57% | 1.52% |
| 0.2 heave | 0.38% | 0.32% | 0.24% | 0.19% | 0.13% | 0.31% |
| 0.5 surge | 2.07% | 1.60% | 1.17% | 0.92% | 0.59% | 1.50% |
| 0.5 heave | 1.38% | 0.92% | 0.66% | 0.51% | 0.32% | 0.85% |
| 0.8 surge | 1.67% | 1.32% | 0.99% | 0.78% | 0.51% | 1.25% |
| 0.8 heave | 3.02% | 1.71% | 1.17% | 0.89% | 0.53% | 1.54% |
| 1.0 surge | 1.40% | 1.02% | 0.76% | 0.60% | 0.40% | 0.97% |
| 1.0 heave | 4.47% | 2.36% | 1.59% | 1.19% | 0.69% | 2.09% |
| 1.5 surge | 1.02% | 0.27% | 0.16% | 0.11% | 0.06% | 0.22% |
| 1.5 heave | 9.09% | 4.52% | 2.98% | 2.21% | 1.23% | 3.96% |
| 2.0 surge | 1.97% | 0.56% | 0.44% | 0.37% | 0.30% | 0.55% |
| 2.0 heave | 15.14% | 8.37% | 5.69% | **4.30%** | 2.51% | **7.48%** |

最差：最细网格 4.2975% → 修正外推 **2.5094%** → 取反版 7.4769%（原工件值，逐位复现）。

### 3.2 加密阶梯实测（新算）

| 族 | 面数 | 体积 [m³] | 最差误差（最细） | 修正外推 p=2 | 观测阶外推 | 单调 |
|---|---|---:|---:|---:|---:|---|
| capytaine 经纬球 | 1024 / 1600 / 2304 | 2.0860 / 2.0890 / 2.0907 | 2.881% | 1.584% | 0.113% | ✔ |
| 本仓正八面体半球（同网格族） | 256 / 1024 / 4096 | 2.0458 / 2.0821 / 2.0913 | 2.170% | 1.447% | 0.223% | ✔ |

观测阶（$e \sim h^p$，$h \propto N^{-1/2}$）：经纬球 0.73–1.00；同网格族 0.41–1.07。逐格数据（含
`ka=2 heave` 在 4096 面同网格族上为 2.17%，比 256 面的 8.73% 干净地掉了一个量级的一阶）见 JSON 的
`uv_family.cells` / `octa_family.cells`。

### 3.3 图

- `benchmarks/verified/casepool_bem_hulme_diff_verif/convergence_audit_ladders.png`：两族误差 vs 面数（双对数，
  附 $h^1$ 参考斜线），6 条曲线全部沿 ~1 斜率下降。
- `benchmarks/verified/casepool_bem_hulme_diff_verif/convergence_audit_richardson.png`：12 格三种读法柱状图 +
  5% 门线；「取反版」柱全部高于最细网格柱（缺陷签名）。

## 4. 注意点（坑/局限/适用范围）

1. **坑（本次的核心）**：Richardson 的加密比方向。`h ∝ N^{-1/2}` 时 $r = \sqrt{N_{\rm fine}/N_{\rm coarse}}$；
   取反不报错、不越界，只是静默地把结果推向极限的**反面**，而且看上去「合理」（数字仍然同量级）。
   仓库里 `src/tensorbem/` 的几处 Richardson（`drift_far_field3d.panel_richardson`、
   `drift_force.richardson_extrapolate`、`cylinder_drift.richardson_extrapolate`、
   `benchmarks/declared_numbers.richardson_limit`）都带 `ratio > 1` 的守卫——只有这个外部代码的产出脚本漏了。
   现在审计脚本与产出脚本都加了守卫（`tests/...::test_richardson_helper_converges_a_synthetic_family` 用合成
   $C h^p$ 家族守住）。
2. **局限（外部）**：Hulme (1982) 原文数字不可得，参考值是重算级数；本审计强化的是「第三方路线收敛到该级数」，
   不是「级数等于原文表格」。台账该条目的原文数字一项仍未闭合。
3. **局限（统计）**：观测阶由**误差模**在三级上估计，而 Richardson 作用在**复数值**上；当误差方向随加密
   摆动时，观测阶外推会偏乐观（本审计并列给出 p=2 的保守数，宣称一律取两者中较差者）。
4. **范围**：加密阶梯只覆盖 ka∈{0.5,1,2}（审计关注第三方路线自身的收敛，其误差随频率平滑）；辐射量只经过
   产出脚本的 sanity 门（ka=1 垂荡 $A_{33}$ 0.4362 vs 级数 0.4285，2.0%）。

## 5. 误差判定（显式回答「是否 <3%」）

| 关键量 | 值 | 基准 | 误差 | <3%？ |
|---|---|---:|---:|---|
| TensorBEM 总激励力（最坏格，1024 面） | 见 G1 | 多极级数 | 0.536% | **是** |
| 第三方路线最细网格（最坏格，1024 面） | 4.298% | 多极级数 | — | 否（未收敛的离散误差） |
| 第三方路线修正外推 p=2（最坏格） | 2.509% | 多极级数 | — | **是** |
| 第三方路线同网格族 4096 面（最坏格） | 2.170% | 多极级数 | — | **是** |
| 第三方路线同网格族外推 p=2（最坏格） | 1.447% | 多极级数 | — | **是** |
| 参考级数截断（surge/heave） | ≤2.2e-5 | 自收敛 | — | **是** |
| 冻结第三方工件复现 | 2.2e-16 | 冻结文件 | — | **是** |

即：**锚点主门（0.54%）与修正后的第三方门（2.51%）都 <3%**；唯一 >3% 的量（4.30%）是第三方代码在 1024 面
上**尚未收敛**的离散误差，被如实标注为「未收敛」，不计入一致性的宣称。

## 6. 复现

```bash
PYTHONPATH=src python scripts/capytaine_hulme_diff_convergence_audit.py          # ~80 s，写审计 JSON
python benchmarks/verified/casepool_bem_hulme_diff_verif/make_convergence_audit_figures.py
python benchmarks/verified/casepool_bem_hulme_diff_verif/run_case.py \
    --out benchmarks/verified/casepool_bem_hulme_diff_verif                      # 重生成 results.json/report.md
PYTHONPATH=src python -m pytest tests/test_casepool_hulme_diff_convergence_audit.py -q
```