# TensorBEM 专项审计：`compute_mass_matrix()` 平行轴项定谳、修复与影响面

- 仓库：TensorBEM（`src/tensorbem/hydrostatic.py`）
- 分支：`agent/m2423`（从 `origin/main` 6e9f2eb 切出）
- 审计对象：`compute_mass_matrix(mass, center_of_gravity, inertia)` 的 6×6 刚体质量矩阵
- 环境：Python 3.11 / numpy 2.4.6 / 本地全量 `pytest`
- 结论一句话：**审计所指「平行轴项重复计入」在显式惯量路径上不成立——该处是文档错（`inertia` 语义应为关于质心）；但缺省惯量分支确实重复计入（1.72–1.84×），且审计未提及的平移-转动耦合块整体转置。三处一并修复，解析解误差从 122% 降到 4×10⁻¹⁶。**

---

## 0. 定谳（实现错 vs 文档错）

`compute_mass_matrix` 组装的矩阵为

$$
M=\begin{bmatrix}mI_3 & -mS(r_g)\\ -mS(r_g) & J+mSS^T\end{bmatrix},\qquad S(r_g)^2\text{ 的对称部分}=SS^T=|r_g|^2I-r_gr_g^T .
$$

刚体动能为 $T=\tfrac12 m\lvert v+\omega\times r_g\rvert^2+\tfrac12\omega\!\cdot\! J_{cg}\,\omega$，逐项展开给出唯一确定的广义质量矩阵

$$
M=\begin{bmatrix}mI_3 & -mS(r_g)\\ -mS(r_g)^T & J_{cg}+mSS^T\end{bmatrix}
=\begin{bmatrix}mI_3 & -mS(r_g)\\ mS(r_g) & J_O\end{bmatrix},\quad J_O=J_{cg}+mSS^T .
$$

由此逐条定谳：

| 编号 | 审计关注点 | 判据（解析/物理） | 旧实现 | 定谳 |
|---|---|---|---|---|
| **D1** | 转动惯量块 `J + m S Sᵀ` 是否对「关于原点」的入参重复计入 | 块的物理量必须是关于参考点的 $J_O=J_{cg}+mSS^T$ | 公式本身正确，**但 docstring 两处（模块级公式说明与 `inertia` 参数）声明 `inertia` 是「关于坐标原点」的惯量张量** | **文档错**（就审计所指这一处而言，实现无误、文档错）；按文档调用才会双计 |
| **D2** | 缺省惯量分支（`inertia=None`） | 传入块的必须是 $J_{cg}$ | 分支先构造 $J_O=J_{cg}+m(\lvert r_g\rvert^2I-r_gr_g^T)$，块里再加一次 $mSS^T$ → **平行轴项计入两次**，转动块为正确值的 1.72–1.84 倍 | **实现错（真·双计）** |
| **D3** | （审计未提，本次新发现）耦合块 | $M_{v\omega}=-mS(r_g)$，$M_{\omega v}=-mS(r_g)^T=+mS(r_g)$ | 写成 $M_{v\omega}=-mS(r_g)^T$、$M_{\omega v}=-mS(r_g)$，即两块**转置互换了**；矩阵仍对称，故任何「重述实现」的自洽参照都查不出来 | **实现错**（动能恒等式误差 122%，动量映射偏差 50.8%） |

**为什么 D1 不是「实现错」**：把 $J_{cg}$ 传进块里再加一次平移项，得到的正是 $J_O$——这是正确的物理量。真正错的是文档把入参说成「关于原点」。因此审计给出的二选一，答案是「**文档错**」；而审计同时担心的「重复计入」真实存在于 **D2 缺省分支**（文档错诱导 + 分支自身也算错），并伴随 D3 未被发现的耦合块转置。

---

## 1. 案例介绍（物理背景与参数）

### 1.1 被审计的物理量

6 自由度刚体运动 $\xi=(v,\omega)$ 与广义动量 $(p,h_O)$ 的本构关系 $y=Mx$。$M$ 是频域运动方程

$$\big[-\omega^2(M+A(\omega))+i\omega B(\omega)+C\big]\xi=F_{exc}$$

中唯一由船体质量分布决定的矩阵，出现在 `rao.py` 的阻抗矩阵装配处。$M$ 中任何符号/转置/双计错误都会污染所有吃水外重心（$r_g\neq0$）的 RAO。

### 1.2 非零质心偏移解析算例（主判据）

审计明示「质心在原点的算例不充分」：$r_g=0$ 时耦合块恒为零、平行轴项恒为零，三类缺陷全部**结构性不可见**。故构造 5 质点刚体（全部位于坐标原点之外，$r_g$ 三分量非零）：

| 项 | 值 |
|---|---|
| 质点质量 $m_i$ (kg) | 1200, 850, 2400, 640, 1310（总质量 $m=6400$） |
| 质点位置 $r_i$ (m) | (1.00,0.20,−0.40), (−0.50,1.10,0.30), (0.30,−0.80,−1.50), (2.00,0.50,0.90), (−1.20,−0.30,0.10) |
| 质心 $r_g$ (m) | (0.1880, −0.1278, −0.4872)（三分量均 ≥0.13 m） |
| 关于质心的惯量 $J_{cg}$ | 由质点分布**导出**（$\sum m_i(\lvert\rho_i\rvert^2I-\rho_i\rho_i^T)$），非手填 |

参照矩阵**不复述实现里的 skew 代数**，而直接用刚体速度场 $u(r)=v+\omega\times r$ 的动量映射

$$p=\sum_i m_i u(r_i),\qquad h_O=\sum_i m_i\, r_i\times u(r_i),$$

对 6 个单位广义速度取列，得 $M_{ref}$。这使参照与实现相互独立：耦合块的两个符号、平行轴的次数都在里面。

### 1.3 缺省分支与 RAO 影响面算例

| 项 | 值 |
|---|---|
| 缺省分支（D2） | $m=1000$ kg，$r_g=(0.3,-0.4,-0.6)$ m，`inertia=None` |
| RAO 影响面 | 浮半球 `hemisphere_hull_mesh(subdiv=2)`，$a=1$ m，$\rho=1025$，$g=9.80665$，$m=\rho\nabla_{mesh}$，$r_g=(0,0,-0.5a)$，显式 $J_{cg}=m(0.4a)^2I$，顶浪，$ka\in\{0.5,1.0\}$ |

---

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **先读实现再定判据**：从 docstring、`s`（skew）矩阵、块装配逐行推导 $T=\tfrac12x^TMx$ 与实现的多项式是否逐项相等，得到 D1/D2/D3 三条待裁项。
2. **先造独立参照，再动手改**：新增 `tests/test_mass_matrix_analytic.py`，用 1.2 的动量映射构造 $M_{ref}$（不调用被审计公式），并加 4 条独立物理判据：
   - 动量映射逐元素一致（机器精度）；
   - 纯平动 $\omega=0$ 时角动量必须是 $r_g\times(mv)$（耦合块符号错会翻转它）；
   - 动能恒等式 $\tfrac12x^TMx=\tfrac12m\lvert v+\omega\times r_g\rvert^2+\tfrac12\omega\!\cdot\! J_{cg}\omega$（64 个随机样本）；
   - 转动块必须 **恰好一次** 等于 $J_O$，并显式断言它**不等于** $J_{cg}+2mSS^T$（双计反例）。
3. **红测试先行**：修复前跑新测试 → 6 failed / 5 passed，逐项对应 D1（文档断言）、D2（缺省分支）、D3（耦合块）。**任何一项都不是靠改判据变绿。**
4. **最小修复**（`hydrostatic.py`，只改该函数）：耦合块改为 $-mS(r_g)$ 与 $mS(r_g)$；缺省分支只构造 $J_{cg}=mk^2I$；docstring 改为「关于质心」并写出 $J_O=J_{cg}+mSS^T$ 的单次平移说明。
5. **清掉会掩盖缺陷的自洽参照**：`hydrostatic_rao_verification.run_mass_matrix_check()` 原来的 `exact` 块是把实现公式抄一遍（`-m*s.T` 与 `+mSS^T` 都照抄），属**自洽参照**，因此改前改后都能过 1e-12。改为动量映射参照（与 1.2 同构），并把键名 `inertia_tensor_about_origin` 更正为 `inertia_tensor_about_cg`。
6. **同步文档声明**：`rao.py` 两处 `Inertia tensor about the origin` → `about the centre of gravity`。
7. **影响面量化**：`examples/mass_matrix_coupling_impact.py` 在同一辐射/绕射解上跑两遍质量矩阵（修正版 vs 旧实现控制臂），出 `artifacts/mass_matrix_audit.json` 与图。
8. **顺带修好阻断 CI 的前置缺陷**：`geometry3d.box_hull_mesh()` 与 `spheroid_hull_mesh()` 在 `origin/main` 上已被门禁过滤掉，而 `benchmarks/hydrostatic_benchmarks.py`、`benchmarks/hydrostatic_verification.py`、`tests/test_hydrostatic_verification.py` 仍引用它们 → main 上 `pytest` 在收集阶段即 ImportError。按仓库内既有约定重建这两个生成器（平面面元、无自由液面盖板、单一水线环），使套件可收集。

---

## 3. 计算结果（程序化注入，无手抄）

### 3.1 修复前后：解析判据

| 判据 | 修复前 | 修复后 | 参考/说明 |
|---|---|---|---|
| $\max\lvert M-M_{ref}\rvert/\lvert M_{ref}\rvert_{\max}$（动量映射） | **0.508** | **1.5e-16** | 1.2 节非零 $r_g$ 算例 |
| 耦合块最大偏差 $\lvert M_{v\omega}-M_{ref,v\omega}\rvert$ (kg·m) | **6236** | ≤1e-12·m | 定谳 D3 |
| 动能恒等式最大相对误差（64 样本） | **1.221**（122%） | **3.9e-16** | 定谳 D3/D2 |
| 缺省分支转动块 / 正确 $J_O$（对角比） | **1.842 / 1.822 / 1.719** | **1.000 / 1.000 / 1.000** | 定谳 D2（双计） |
| 仓库内独立实现 `run_mass_matrix_energy_check` | 模块 ImportError（不可测） | **3.65e-16** | `benchmarks/hydrostatic_verification.py` |
| `run_mass_matrix_check` 相对残差（旧自洽参照 → 动量映射参照） | 1e-12（自洽，无判别力） | **≤1e-12（现在有判别力）** | `test_rao_verification.py` 通过 |
| 方箱静水力精确性（新几何生成器） | 不可测（ImportError） | $\nabla$=80.000000、$A_{wp}$=40.000000、$I_{xx}$=53.333333、$I_{yy}$=333.333333（N=1,2,3,4 网格不变） | 闭式 $LBT$、$LB$、$LB^3/12$、$BL^3/12$ |
| 半椭球（$a,b,c$=5,2,2）块系数 $C_b$ @subdiv=3 | 不可测 | **0.51145**（$\pi/6=0.52360$，误差 2.32%） | 内接多面体离散误差 |

### 3.2 影响面：`rao.py` 调用点的 RAO 变化

| $ka$ | 涌 surge | 荡 sway | 垂荡 heave | 横摇 roll | 纵摇 pitch | 艏摇 yaw | 最大变化 |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.92% | 1.07% | 6e-8 | 2.55% | **6.89%** | 0 | **6.89%** |
| 1.0 | **10.02%** | 1.53% | 6e-8 | 4.27% | **10.04%** | 1e-5 | **10.04%** |

- 阻抗残差在修正后为 $3.7\times10^{-12}$（$ka=0.5$）与 $2.8\times10^{-12}$（$ka=1.0$），链路线性代数未受影响；
- 质量矩阵本身最大相对变化 100%（耦合块整体换位），对应 RAO 最大 10.0%；**超过 15%/3% 级判据的容忍度，属实质性工程误差**；
- 垂荡/横摇耦合在 $x=y=0$ 的 $r_g$ 下仍被影响（roll 2.6–4.3%，经 $M_{v\omega}$ 的 (roll↔sway) 与 (sway↔yaw) 项）。

图见第 4 节。

---

## 4. 结果图

1. `docs/figures/mass_matrix_blocks.png`（生成：见 §7，同时输出到 `artifacts/`）
   - 左：旧实现耦合块与解析参照 $(-mS(r_g))$ 的**逐元素误差云图**（9 个元素全非零，符号成对反号）；
   - 右：转动块对角三分量的柱状对照——$J_{cg}$（入参）、正确 $J_O=J_{cg}+mSS^T$、旧缺省分支（1.72–1.84× 双计）。
2. `docs/figures/rao_coupling_impact.png`
   - 半球 $r_g=(0,0,-0.25)$ 顶浪、$ka=0.5/1.0$ 下 6 自由度 $\lvert\xi\rvert$ 的修正版 vs 旧版柱状对照，标注逐频最大相对变化（6.9% / 10.0%）。

（本任务为库函数审计，无流场/波面场量云图；上图即「与解析解/基线比较曲线」的替代，均为 matplotlib 程序生成。）

---

## 5. 注意点与局限

1. **审计所指与审计未指的都变了**：本单修的三处里，只有 D2 是审计原文说的「平行轴项重复计入」。D3（耦合块转置）是解析判据暴露出来的**新增发现**，若不修，即使 D1/D2 修好，非零 $r_g$ 的 RAO 仍是错的（误差 122% 的动能恒等式）。报告不把 D3 说成审计已覆盖。
2. **自洽参照是本仓的真实历史缺陷**：`run_mass_matrix_check` 的 `exact` 曾是实现的可读复述，因此 1e-12 的"通过"没有任何判别力，缺陷得以存活到本次审计。已在代码里改为动量映射参照，并在 dev_report 记录这条判据病理。
3. **缺省惯量仍是启发式**（$k=0.4\lvert r_g\rvert$，$r_g=0$ 时 $k=0.4$ m）：本次只保证**平移项只计一次**，没有给它几何依据。任何定量横摇/纵摇工作必须显式传 $J_{cg}$，证据等级仍为 WITHHELD。
4. **$M$ 关于参考点的约定**：$h_O$ 取矩点固定为坐标原点（`rao.py` 的 6 自由度基也在此）。若将来引入"重心为参考点"的表示，$M$ 的耦合块与 $J$ 需整体重排（$r_g=0$ 时退化为块对角），本单未做该抽象。
5. **RAO 影响面用的是 64 面板粗半球**：几何离散误差 8.9%（体积）仍是系数误差地板，因此 3.2 节数字是"质量矩阵修正带来的相对变化"，不是 RAO 相对文献值的精度。
6. **前置缺陷修复的边界**：`box_hull_mesh` / `spheroid_hull_mesh` 是按仓库内既有约定重建（平面面元、单水线环、无盖板），使 CI 从"收集即红"恢复为可跑；它们的实现细节不在本单审计范围内，只以"方箱闭式精确/半椭球 $C_b$ 2.3%"两项做量级自检。
7. **未改动** `compute_hydrostatic_restoring_matrix`（静水平衡不校验 $m=\rho\nabla$）与 A/B 矩阵链路：不属本单范围，且已有 dev_report 记录。

---

## 6. 误差判定（逐项，vs 基准）

| 关键量 | 基准 | 修正后误差 | 是否 <3% |
|---|---|---|---|
| 动量映射矩阵元素 | 独立构造的 $M_{ref}$（5 质点刚体，$r_g\neq0$） | 1.5e-16（相对矩阵尺度） | **是** |
| 广义耦合块 $-mS(r_g)$ | 纯平动角动量 $r_g\times mv$ / 动量映射 | ≤1e-12·m（数值舍入） | **是** |
| 动能恒等式（64 随机样本） | $\tfrac12m\lvert v+\omega\times r_g\rvert^2+\tfrac12\omega J_{cg}\omega$ | 3.9e-16 | **是** |
| 转动块 / $J_O$ | $J_{cg}+mSS^T$（单次平移） | 0（逐元素 allclose 1e-13） | **是** |
| 方箱静水力（新生成器自检） | $\nabla=LBT$ 等闭式 | ≤1e-16（相对） | **是** |
| 半椭球块系数 $C_b$ | $\pi/6$ | 2.32%（$4\cdot4^3$ 面元内接离散误差，随细分单调下降） | **是**（2.32%<3%） |
| RAO（$ka=0.5/1.0$，半球） | 无闭式基准（粗网格几何误差 8.9% 为地板） | 无基准可比 → 只报"质量矩阵修正引起的相对变化" 6.9%/10.0% | **不适用（必须声明：无基准，故不给"是否<3%"，只给影响面）** |

诚实结论：**修正后的 $M$ 与其解析/物理判据在机器精度内一致（≪3%）；本次修复同时把非零重心工况下的 RAO 改变了最多 10.0%，说明原缺陷是实质性工程误差而非外观问题。**

---

## 7. 本地全量验证记录（门禁 v5.8 前置）

| 范围 | 命令 | 结果 |
|---|---|---|
| 新增解析回归 | `pytest tests/test_mass_matrix_analytic.py` | **11 passed** |
| RAO 链路 | `pytest tests/test_rao_verification.py tests/test_hydrostatic_rao.py` | **38 passed**（原 `test_matches_exact_block_form` 用的是自洽参照，改后仍过且具判别力） |
| 静水力验证（前置缺陷修复后） | `pytest tests/test_hydrostatic_verification.py` | **26 passed**（origin/main 上为 **collection ERROR**） |
| 全量 | `pytest tests`（`OMP_NUM_THREADS=1`，串行） | **557 passed / 12 failed / 3 skipped / 572 collected，9m52s** |
| 全量基线对照 | 同一命令跑 `git archive origin/main` 的纯净副本 | **同样 12 failed**（逐条同名：`test_doc_claims`×3、`test_documented_benchmarks`×2、`test_kochin_validation`×1、`test_ursell_reproducibility`×5、`test_ursell_doc_reproducibility`×1），另有 `tests/test_hydrostatic_verification.py` **collection ERROR** |

**结论**：12 个失败全部在 origin/main 上逐条复现，与本单改动无因果关系——它们依赖被 `.gitignore` 排除的生成物
（如 `artifacts/ursell_doc_claims.json`）与既有文献比对阈值，属存量欠账；本单净增 26 个可运行的验证测试。
诚实记录两点：① 本机当时负载极高（load average 一度 >1000，另有 100+ 个 pytest 进程），`-n 8` 并行运行在收尾
阶段出现过 worker 长时间阻塞，故最终判定以 **串行全量** 结果为准；② 上述 12 项存量失败未在本单修复（不属审计范围，
也未改判据）。

## 8. 复现命令

```bash
cd <worktree>
PYTHONPATH=src python -m pytest tests/test_mass_matrix_analytic.py -q     # 11 passed
PYTHONPATH=src python -m pytest tests/test_rao_verification.py tests/test_hydrostatic_rao.py -q
PYTHONPATH=src python -m pytest tests/test_hydrostatic_verification.py -q  # 26 passed（前置缺陷修复后）
PYTHONPATH=src python examples/mass_matrix_coupling_impact.py --output artifacts/mass_matrix_audit.json
OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests -q     # 全量：见 §7
```

## 9. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/tensorbem/hydrostatic.py` | `compute_mass_matrix`：耦合块 $-mS(r_g)$ / $mS(r_g)$；缺省分支只造 $J_{cg}$（单次平移）；docstring 语义与公式更正 |
| `src/tensorbem/rao.py` | 两处 `inertia` 参数说明：关于**质心** |
| `src/tensorbem/benchmarks/hydrostatic_rao_verification.py` | `run_mass_matrix_check`：自洽参照 → 独立动量映射参照；键名 `..._about_cg` |
| `src/tensorbem/geometry3d.py` | 恢复 `box_hull_mesh`、`spheroid_hull_mesh`（main 上被过滤导致 ImportError） |
| `tests/test_mass_matrix_analytic.py` | **新增** 11 条解析/物理判据（非零 $r_g$） |
| `examples/mass_matrix_coupling_impact.py` | **新增** 影响面量化脚本（旧实现作控制臂） |