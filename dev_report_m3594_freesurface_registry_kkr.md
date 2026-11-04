# 三维自由液面文档数字的判据化（Hulme 表 10 入注册表）与三维 Kramers–Kronig 因果门的否定结论

**任务**：m3594（platform 席自找活） · **分支**：`agent/m3594` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3594`，基线 `4f5b73e`）
**工件**：`outputs/m3594/declared_numbers_fast_standard.json`、`outputs/m3594/kkr3d_band_audit.json`、
`outputs/m3594/fig_freesurface_registry_and_kkr.png`、新模块 `src/tensorbem/benchmarks/kkr_floating_hemisphere.py`、
新测试 `tests/test_kkr_floating_hemisphere.py`

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py, src/tensorbem/benchmarks/kkr_floating_hemisphere.py, tests/test_declared_numbers.py, tests/test_kkr_floating_hemisphere.py, docs/marine_bem_benchmarks.md, outputs/m3594/, dev_report_m3594_freesurface_registry_kkr.md

---

## 1. 案例介绍（为什么做这件事）

### 1.1 选题：一个被上一单显式点名的验证欠账

m3059（2026-11-04，`dev_report_m3059_heavy_verification_debt.md` §4）结清 heavy 层后明确留下一条：

> 三维自由液面（Hulme/Wang-Linton）虽已由 m2609 等测试钉桩，但**仍未纳入注册表**（成本更高，建议后续单独立项）。

`docs/marine_bem_benchmarks.md` §4.10 表 10 是三维穿水浮体（Hulme 1982 漂浮半球，\\(Ka=1.0\\)，\\(a=1\\) m，\\(\\rho=1025\\)）的网格收敛主表：5 个量 × 3 个细分层级 = 15 个误差格、3 个条件数，再加上正文里的近场/远场 Kochin 阻尼差与两条 Haskind 互易残差，共 21 个数字。它们此前**没有任何测试读取**——`grep` 表 10 的数值（`3.893`、`2.97e-2`、`5.45e-4` 等）在 `tests/` 中零命中，测试只以间接门（`far_field_damping_rel_diff_heave < 0.002`、二阶收敛）近似覆盖。这正是本仓已经发生过三次的同型事故模式（Ursell 7.7e-3 → 8.51e-3、MFS 1.9e-4 → 7.6e-4、Wigley 排水体积 \\(2LBT/3\\) → \\(4LBT/9\\)）：**文档数字没有执行者**。

### 1.2 几何与参数

| 项 | 设置 |
|---|---|
| 物体 | 漂浮半球，半径 \\(a=1\\) m，水线面截断（`WettedHullMesh`），\\(L=2a\\) |
| 控制方程 | 三维深水自由液面势流：\\(\\nabla^2\\phi=0\\)，\\(\\partial\\phi/\\partial z=\\nu\\phi\\) on \\(z=0\\) |
| 核 | Kelvin/Delhommeau 四分区兴波核 + 解析立体角双层 + Duffy 单层 |
| 离散层级 | subdivisions 1/2/3 → 16 / 64 / 256 面元 |
| 真值 | Hulme (1982) 波无势（wave-free potentials）48 项半解析解 |
| 物性 | \\(\\rho=1025\\ \\mathrm{kg/m^3}\\)、\\(g=9.80665\\ \\mathrm{m/s^2}\\)、\\(Ka=1.0\\) |
| 物理量 | \\(A'_{33},B'_{33},A'_{11},B'_{11}\\) 相对误差、\\(F_z\\) 相对误差、条件数、近/远场阻尼差、Haskind 互易残差 |

## 2. 怎么做的

1. **注册表新增组** `floating_hemisphere`：`declared_numbers._run_group` 新增分支，调用既有
   `floating_hemisphere.run_refinement_study(subdivisions=(s,), ka=1.0)`（未改求解器一行），把
   `levels[0]` 的 5 个误差字段 + 条件数 + 3 个恒等式残差摊平为 `dict[str, float]`；注册表原有的
   `(group, parameters)` 缓存保证**一个层级只解一次**，21 条注册项共享 3 次求解。
2. **21 条注册项**（`DECLARED_NUMBERS`，source 逐条指向表 10 或 §4.10 正文），容差按**印出的有效位**取：
   三位有效数字 → \\(5\\times10^{-3}\\)（最坏印出舍入 \\(0.005/1.20=4.2\\times10^{-3}\\)），四位条件数 → \\(5\\times10^{-4}\\)。
   16/64 面元层归 `fast`（0.2 s / 0.5 s），256 面元层归 `standard`（9.5 s，含辐射+绕射+远场+Haskind）。
3. **复算**：`python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard --strict --output outputs/m3594/...`
   → 33 条（原 12 + 新 21）全 PASS。
4. **回归钉桩**：`tests/test_declared_numbers.py` 新增 `test_table_10_free_surface_rows_are_reproduced`
   （standard 层 9 条，含三条恒等式行），并把"覆盖面"判据加入 `floating_hemisphere` 组。
5. **附带探索（负结果，如实报告）**：仿照二维 `benchmarks/kkr_causality.py`，新写
   `benchmarks/kkr_floating_hemisphere.py`，用同一因果变换（`tensorbem.causality.kkr_check`）审计
   三维面板扫频，并设负控（阻尼 ×1.02/×1.10/×0.90/×1.30）。结论见 §3.2 —— **不能当门用**。
6. **工具链**：Python 3.11.6 / NumPy / matplotlib（Agg）；全程 `OPENBLAS_NUM_THREADS=1` 单核，
   本机 load15≈130 的重载窗口下未起并行重算力（单核 <15 s 级）。

## 3. 计算结果（程序化，零手抄）

### 3.1 表 10 与 §4.10 正文 21 个数字全部逐位成立（`declared_numbers_fast_standard.json`）

| 量 | 16 面元（声/实） | 64 面元（声/实） | 256 面元（声/实） |
|---|---|---|---|
| \\(A'_{33}\\) 相对误差 | 1.20e-1 / 1.20128e-1 | 2.97e-2 / 2.97358e-2 | 7.41e-3 / 7.40490e-3 |
| \\(B'_{33}\\) 相对误差 | 1.22e-1 / 1.21900e-1 | 4.50e-2 / 4.50228e-2 | 1.18e-2 / 1.18118e-2 |
| \\(A'_{11}\\) 相对误差 | 1.80e-1 / 1.80237e-1 | 5.05e-2 / 5.05084e-2 | 1.30e-2 / 1.30376e-2 |
| \\(B'_{11}\\) 相对误差 | 4.02e-1 / 4.02410e-1 | 1.30e-1 / 1.30492e-1 | 3.53e-2 / 3.52753e-2 |
| \\(F_z\\) 相对误差 | 6.67e-2 / 6.67161e-2 | 2.21e-2 / 2.20939e-2 | 5.67e-3 / 5.66825e-3 |
| 条件数 | 3.893 / 3.89255 | 3.052 / 3.05155 | 2.783 / 2.78288 |

正文恒等式（256 面元）：近/远场 Kochin 阻尼差 5.45e-4 / **5.4492e-4**；Haskind 互易纵荡 2.09e-4 / **2.0889e-4**、垂荡 8.09e-4 / **8.0910e-4**。

- 汇总：33 条注册项 **33 PASS / 0 DRIFT**，新增 21 条最大相对偏差 \\(3.787\\times10^{-3}\\)（纯印出舍入，远低于 \\(5\\times10^{-3}\\) 容差）；
  全表最坏偏差 \\(1.10\\times10^{-1}\\) 来自既有的 MFS 行（其自身容差内）。
- 判决含义：表 10 与 §4.10 正文**不需要改写**，本次是把"读者被告知的数字"升级为"CI 每轮重算的数字"。

### 3.2 负结果：三维 Kramers–Kronig 残差地板 O(10⁻¹)，不具分辨力（`kkr3d_band_audit.json`）

`Ka\\in[1,4]`，12 个等比频点，`subdivisions=1,2`（16/64 面元）：

| 模态 | 层级 | 参考解 KKR 残差（地板） | 面板解 KKR 残差 | 面板/地板比 | 阻尼人为 ×1.10 后残差比 |
|---|---|---:|---:|---:|---:|
| 纵荡 | 64 面元 | 0.4346 | 0.4186 | 0.96 | 1.09 |
| 垂荡 | 64 面元 | 0.7327 | 1.0463 | 1.43 | 0.94 |

- 关键点：**参考解自己是 0.43 / 0.73**——因果变换在半解析 Hulme 解上就报出几十个百分点残差，说明残差不是面板解的属性。
- 根因：关系 (1) 需要 \\(A_\\infty\\)。二维 Ursell 有闭式高频极限（`ursell_high_frequency_limit`，`kkr_causality` 正是靠它拿到可用地板）；三维附加质量没有闭式 \\(A_\\infty\\)，只能靠 \\(A_\\infty+c/\\omega^2\\) 尾部拟合，本算例拟合出的 \\(A_\\infty\\) 在扫频/加宽频带时从 251 漂到 1054 再漂到 −305（非物理），残差被它主导。
- 分辨力检验：阻尼整体放大 10% 这个"明显非因果"的扰动只把残差抬到 1.09 倍（垂荡甚至 0.94 倍，方向都不对），与地板同量级 → **该指标不能作为门**。
- 同时，面板解不劣于参考解（比值 0.96/1.43，O(1)），即没有证据表明三维自由液面核在注入非因果性。

### 3.3 图

`outputs/m3594/fig_freesurface_registry_and_kkr.png`（左）21 条新注册项的实测相对偏差（对数轴）与 5e-3 DRIFT 容差线，无一条越线；
（右）KKR 残差随 \\(Ka\\)：参考解地板（虚线）与 16/64 面元面板解（实线）同量级，直观显示"门被地板吃掉"。

## 4. 注意点（坑 / 局限）

1. **容差必须按印出位数取**：表 10 印三位有效数字，若沿用注册表默认 1.5% 容差，则 \\(3.8\\times10^{-3}\\) 级的真实漂移会被吞掉；本次取 5e-3，既能吸收印出舍入又能抓住 1% 级代码漂移。
2. **成本分层**：256 面元层单次 9.5 s（辐射+绕射+远场+Haskind 一次组装复用），是全注册表最贵的 `standard` 项；16/64 层 0.2/0.5 s。默认 `pytest` 因此增加约 10 s，若需更轻可整体降到 `heavy`。
3. **KKR 模块不是"解不出来"，是"没有 \\(A_\\infty\\) 就没有意义"**：模块保留是因为它把这条判据的适用边界量化了；要用三维因果门，先得有 \\(A_\\infty\\) 的闭式解或高精度极限（例如对半球用 φ=0 自由面条件的独立无界流解，或 \\(Ka\\to\\infty\\) 外推 + 极点消去的专门研究）。这是后续单独立项的方向，本单不做。
4. **扫频误差的大头在高端**：`kkr3d_band_audit.json` 里 64 面元在 \\(Ka\\) 高端的点误差可达 \\(A_{33}\\) 84%、\\(B_{33}\\) 122%（短波欠分辨），这是离散欠账而非因果问题，勿与 §3.1 的 \\(Ka=1.0\\) 单点误差（2.97e-2 / 4.50e-2）混淆。
5. **未做**：未改任何求解器实现；未把 Wang/Linton 潜没球体（§4.11）纳入注册表（其多极子真值为零求积闭式，代价与 256 面元同量级，可作下一单）；未跑重载窗口下的全量套件（只跑直接相关的两个测试文件）。

## 5. 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---:|---|
| 21 条新注册项 vs 文档声明（表 10 + §4.10 正文） | 文档印出值 | 最大相对偏差 3.787e-3（0.38%，纯印出舍入） | ✅ |
| 256 面元 \\(A'_{33}\\) / \\(B'_{33}\\) / \\(A'_{11}\\) / \\(B'_{11}\\) / \\(F_z\\) vs Hulme (1982) | 半解析真值 | 0.74% / 1.18% / 1.30% / 3.53% / 0.567% | 前三与 \\(F_z\\) ✅；\\(B'_{11}\\) ❌（3.53%，离散误差，1024 面元 0.90% 已由 m2609 另证） |
| 近/远场 Kochin 阻尼恒等式（256 面元） | 自身远场积分 | 5.45e-4（0.05%） | ✅ |
| Haskind 互易残差（256 面元，纵荡/垂荡） | 自身边界积分 | 2.09e-4 / 8.09e-4（0.02% / 0.08%） | ✅ |
| 三维 KKR 面板/参考残差比 | 参考解地板 | 0.96（纵荡）/ 1.43（垂荡） | 不适用（该量不是与真值的比较，而是分辨力检验；正因地板 43%–73% 才判"不可用"） |

结论：本单的可判定量全部远优于 3% 门；唯一未达 3% 的是既有物理量 \\(B'_{11}\\)（256 面元 3.53%），属已归因的离散误差，如实保留、未粉饰。

## 6. 验证记录

- `pytest tests/test_declared_numbers.py` → **12 passed, 3 skipped**（heavy 门未触发）；其中新增 `test_table_10_free_surface_rows_are_reproduced` 8.4 s 通过。
- `pytest tests/test_kkr_floating_hemisphere.py` → **4 passed，6.6 s**。
- `python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard --strict` → 退出码 0，33 PASS。
- AST 检查通过；未改求解器任何文件（`git status` 只含本单登记的白名单路径）。

## 7. 给下次自找活的线索

1. **§4.11 Wang/Linton 潜没球体入注册表**：零求积多极子真值，20/80/320 面元三档，判据现成（Haskind 1e-16 级自洽、深潜极限 \\(A'=0.5\\)）。
2. **三维 \\(A_\\infty\\) 的独立求法**：若要恢复三维因果门，需一条不依赖面板解的 \\(A_\\infty\\)（高频极限外推或 φ=0 自由面条件的无界流辅助解）——这是本单否定的直接后继。
3. 高 \\(Ka\\) 端 64/256 面元的短波欠分辨（\\(B_{33}\\) 122%）提示：三维兴波核仅 3 点二阶求积，能否上更高阶求积做一次带内误差预算（参照 m3383 的 near-surface band 打法）。