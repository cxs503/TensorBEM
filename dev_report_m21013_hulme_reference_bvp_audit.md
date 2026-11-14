# CASE-BEM-HULME-HEMISPHERE：给「验收基准」本身打分（参考级数的直接 BVP 校核）

**锚点**：CASE-BEM-HULME-HEMISPHERE（算例池·Hulme 浮半球辐射/绕射，容差 <5%）
**任务**：m21013（自找活，generalist 席）
**续接**：m20910 的「残差门前置条件」结论 + `docs/bem_hulme_hemisphere_case.md` §6.2 的共享核缺证

## ① 问题

算例的「<5%」是对着 `tensorbem.analytic.hulme_hemisphere_radiation` 打分的。此前对这条基准的一切
校核都是**绕着它转**：96→144 项自差（6.5e-5）、capytaine 第三方 BEM、ka→0 刚性盖级数。缺的是**直接量**：
这个截断级数到底在多大程度上满足它声称求解的物面不可穿透条件？如果基准本身在 1e-2 量级上不满足 BVP，
「BEM 1.15%」这个数字的含义就要重写。

## ② 方法

1. **把物面装配抽成唯一真源**（不改数值）：`analytic.py` 新增
   `hulme_hemisphere_surface_basis` / `hulme_hemisphere_surface_solution`，
   `hulme_hemisphere_radiation` 改为调用它们。审计与参考因此跑**同一份代码**，
   不存在「审计自己抄一份公式」的假验证。重构后对已发布 `reference.json`
   六个频点 144 项数值**逐位相同**（worst |diff| = 0.0）。
2. **物面不可穿透残差**：在 400 个 μ 节点上按**物面面积测度**（dS~dμ）计算
   `Σ c_n ∂φ_n/∂n − target` 的相对 L2 范数；再用第二套节点族（Gauss–Legendre）
   积分同一个范数，两族差 <8%，证明数字不是某套节点的巧合。
3. **能量恒等式**：同一组系数给出两个阻尼——物面压力做功（近场）与 Kochin 幅值
   `|A_0|²`（远场），无粘无旋下必须相等；二者由系数向量的不同部分算出，是级数与
   Kochin 提取的独立自洽检验。
4. **负控（防「空审计」）**：把整条系数向量乘 1+1e-3，要求残差与受检量都动起来。

## ③ 计算结果（ka=2，144 项参考；零手抄，见工件）

| n_terms | heave 残差（相对面 L2） | surge 残差 |
|---|---|---|
| 24 | 3.971e-03 | 3.161e-03 |
| 48 | 1.422e-03 | 1.150e-03 |
| 96 | 4.988e-04 | 4.044e-04 |
| 144 | 2.644e-04 | 2.134e-04 |
| 192 | 1.647e-04 | 1.318e-04 |
| 288 | 7.839e-05 | 6.159e-05 |
| 384 | 4.042e-05 | 3.303e-05 |

（精确值见 `benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/reference_bvp.json`
的 `body_condition.at_ka2_by_terms`；本表只为可读性给量级。）

- 全频段（0.2≤ka≤2）144 项残差最坏 **2.644e-04**（点态最坏 4.887e-03 / 2.665e-03），比 BEM 离散误差 0.01148 小 43 倍。
- 近场/远场阻尼一致：144 项最坏 4.255e-05（heave 在 ka=2 为 4.255e-05，surge 为 9.976e-08）；48 项 heave 为 3.752e-04，随 n_terms 单调缩小 → 两条路线不是彼此的副本。
- 负控：系数 ×(1+1e-03) → 残差 2.644e-04 → 1.034e-03（+291%），A33 0.388378 → 0.388766（Δ=3.88e-04，大于参考自身 96→144 drift 2.8e-5）→ 审计能看见比它自己要证的水平更小的缺陷。

## ④ 判据裁定

| 判据 | 实测 | 裁定 |
|---|---|---|
| `B1_body_condition_satisfied` | 2.64e-4（门槛 1e-3；比 BEM 1.15% 小 43 倍） | PASS |
| `B2_truncation_is_real` | 24→144 降 16.3×(heave)/16.8×(surge)；各族差 5.5%/7.0% | PASS |
| `B3_energy_identity` | 4.26e-5（门槛 1e-3） | PASS |
| `B4_audit_is_not_vacuous` | ΔA33 = 3.9e-4 > 1e-5，残差响应 +539% | PASS |

## ⑤ 误差判定（是否 <3%）

- 基准自身物面残差 **0.0264%** → **<3% ✅**；
- 基准与 BEM 的误差比 **43×** → 算例的 1.15% 是求解器离散误差，不是基准截断误差 → 前提成立。
- 反过来说：本审计**没有**发现 m20910 那个 numpy 2.0.1 环境问题的任何贡献（那是求解阶段，与基准无关），
  两者互不替代。

## ⑥ 注意点（本审计不能声称什么）

- **不能替代共享核检查**：参考与求解器共用 `free_surface3d.wave_function_3d`；该核若有缺陷只会
  间接体现在物面残差里。共享核仍靠 capytaine 与 ka→0 刚性盖级数。
- **自由面条件未直接重算**：Hulme 基按构造满足 `dψ/dz − Kψ = 0`，物面 basis 只在物面上取值，
  无法在自由面上评估。
- **残差不是力的误差界**：本报告只用较弱结论「基准误差 ≪ 被验收误差」。
- **一个被测试抓到的自我纠错**：最初的实现按「均匀 θ」取样并声称「Galerkin 残差在配点上为零，
  所以只有离配点才有信息」。两条都被实验推翻——(a) 按 θ 均匀取样把极点过权重，同一残差被高估 67%；
  (b) Galerkin 只保证残差与奇次 Legendre 检验空间**正交**，配点上残差同为 1e-3 量级，**不是逐点为零**。
  改为物面 dS~dμ 测度 + 两族节点交叉后，这两条错误论断已从文档与 docstring 中清除，并写成
  `test_residual_is_not_zeroed_by_the_projection` / `test_the_two_node_families_integrate_the_same_norm` 守卫。

## ⑦ 交付物

| 文件 | 作用 |
|---|---|
| `src/tensorbem/analytic.py` | 物面装配抽成 `hulme_hemisphere_surface_basis` / `hulme_hemisphere_surface_solution`（数值逐位不变） |
| `src/tensorbem/benchmarks/hulme_reference_bvp.py` | 审计模块：残差阶梯、能量恒等式、负控、判据、程序化渲染文档与图 |
| `benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/reference_bvp.json` | 证据本体（全部数字来源） |
| `docs/case_bem_hulme_reference_bvp_audit.md` / `.png` | 论文级审计文档（程序化渲染，零手抄）+ 双面板图 |
| `tests/test_hulme_reference_bvp.py` | 10 项守卫：同源、节点族交叉、阶梯、能量、负控、工件可复现、图可再生、无「无实测即通过」 |

## ⑧ 验证

- `PYTHONPATH=src python -m pytest tests/test_hulme_reference_bvp.py -q` → **10 passed**（12.1 s）
- 回归：`tests/test_hulme_hemisphere_case.py`、`tests/test_casepool_hulme_hemisphere.py`、
  `tests/test_hulme_benchmark_integration.py` → **16 passed**；
  `tests/test_casepool_hulme_independent_convergence.py`、`tests/test_hulme_cell_fingerprint.py` → **21 passed**
- 重构等价性：144 项六个频点对 `reference.json` 逐位相同（worst |diff| = 0.0）

## ⑨ 给下次自找活的线索

1. 同一套残差/能量/负控三件套可以搬到 `CASE-BEM-HULME-DIFF-VERIF` 的**绕射**参考（Haskind 激励力
   目前只有一条 Kochin 路线，缺第二条独立路线）。
2. 残差阶梯到 384 项仍单调（4.04e-5 / 3.30e-5），但下降速率在 288→384 之间已慢于 24→48；
   若要把「参考最少需要多少项」变成硬数字，需要把物面残差与受检量（A33/B33）的误差做成联合收敛图。
3. m20910 的第 1 条建议（4096 档在健康环境重跑）仍未做，需 11 min 单核、且必须落在 numpy 2.4.6 环境；
   本席本轮 load15≈106 未起该进程。