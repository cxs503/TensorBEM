# m19943 — XVAL-BEM-R2: 6x6 辐射矩阵的 capytaine 交叉验证（工具链落地，算例未闭合）

锚点 `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`（acceptance <15%）。本单交付的是**第二轮交叉验证的工具链 +
独立侧实测参考**，以及两个真实缺陷的修复；TensorBEM 侧的 6x6 矩阵扫描**未跑完**（原因见 §4），
故本轮**不宣称**通过/不通过，只报独立侧事实。

## 1. 第一轮留下的缺口（为什么需要 R2）
第一轮（`docs/capytaine_crosscheck.md`）只比对了升沉/纵荡**对角项**。转动块（roll/pitch/yaw）、
平移-转动耦合项、几何强制的结构零项从未入库；且第一轮不需要固定**转动中心**——capytaine 3.1 的
`FloatingBody.rotation_center` 默认是 `nan`，只能经 `cpt.rigid_body_dofs(rotation_center=...)` 写进自由度。

## 2. 独立侧已完成的实测（capytaine 3.1.dev0，本机 live）
| 项 | 值 |
|---|---|
| 半球网格阶梯 | [64, 256, 576, 784] 面板（(n_theta,n_phi) 阶梯）|
| 浮球网格阶梯 | [192, 600, 1120] 面板 |
| 6 自由度整列 | `A[i,j]`/`B[i,j]` 全 36 项，转动中心显式写入并回读断言 |
| 第一轮冻结参考 live 重算 | 18 个格子，最大相对差 **0.0e+00**（逐位复现）|
| 深潜 8R 球 A_surge | 2209.8 kg vs Lamb 2146.75（+2.94%）|
| 深潜 8R 球 A_roll | 0.0021 kg·m^2 → |A_roll|/|A_surge| = 9.3e-07 |

## 3. 两个缺陷修复（可验证）
1. **主分支上已有的红门**：`tests/test_capytaine_crosscheck.py::test_capytaine_not_a_runtime_dependency`
   自 m17363 起是 FAIL —— 它用 `^\s*import capytaine` 把 `find_spec` 保护的**函数内惰性导入**
   （`laplace_sphere_xval.py`、`rigid_body_mass_orientation.py` 的正确可选依赖写法）也判为违规。
   修正为**模块级**导入才算运行时依赖（`^(?:import|from)\s+capytaine`），门恢复绿且仍能拦住真违规。
2. **本单生产脚本的列错位**：`_columns` 用 `enumerate(dofs)` 定位矩阵列，对**部分自由度**清单
   （深潜块只解 surge/roll；round-1 复现只解 heave/surge）把 roll 写进了第 2 列而非第 4 列，
   导致复现比对拿到 100% 假失配。改为 `DOF_ORDER.index(dof)` 后，第一轮 18 个格子**逐位复现**。

## 4. 未完成的部分（如实；状态已于 §7 更新）

> **2026-11-16 更新**：本节描述的是本单当时的状态。`results.json`、`docs/capytaine_crosscheck_r2.md`
> 及 `tests/test_capytaine_crosscheck_r2.py` 的非 skip 化已由后续 PR（m20681/m21175/m21265）落地，
> 当前 main 上该锚点已有完整 artifact 与全绿验收测试。本节的「未生成/相应 skip」不再是当前事实。

- TensorBEM 侧 6x6 扫描需 ~15 min 单核：本机 shell 上限 120 s，集群节点**未安装 capytaine**
  （cj2417 日志：`ModuleNotFoundError: No module named 'capytaine'`），因此生产脚本只能在控制机跑、
  而矩阵扫描只能在节点跑——两段无法在同一作业内衔接。已按纪律改派 mission（runner=native）。
- 因此 `docs/capytaine_crosscheck_r2.md` 与 `benchmarks/verified/capytaine_crosscheck_r2/results.json`
  未生成；`tests/test_capytaine_crosscheck_r2.py` 相应 skip（不伪绿）。

## 5. 被否定的假设（负面结果也入账）
本轮原打算用「深潜球 A_roll → (2/5)m_aR^2」作解析第三轴。**错**：球体绕自身几何中心转动时
`(r-centre) x n ≡ 0` 在物面上处处成立，激励为零，故 A_roll **精确为零**（任意频率、任意潜深，
与自由面条件无关）。实测 A_roll/|A_surge| = 9.3e-07，capytaine 同样为 0 → 该精确零反而成为
检验转动广义法向装配的**更强**解析轴（阈 1%）；平动仍用 Lamb (2/3)pi·rho·R^3 作极限。

## 6. 下一步（给下次自找活）
把 mission 跑出来的 `results.json` 落地即可闭环；随后建议新条目
`XVAL-BEM-R2-FINITE-DEPTH`：`solve_radiation_3d` 是 Rankine+lid、不含有限水深色散，
有限水深路线（`docs/BEM_FINITE_DEPTH_GREEN.md`）至今没有任何外部代码交叉验证。

## 7. 重开轮（m22034）的终局核算

### 7.1 本轮修掉的缺陷：收敛阶梯只统计一张矩阵
`convergence()` 原先只遍历 `added_mass`，`damping` 矩阵从不进阶梯。后果是可验证的：
`hemisphere_r1` 在 1024 面板上，附加质量侧对角最差 3.30%（落在 5% 带内），
而阻尼侧 `A[heave,heave]`（nu = 2 m^-1）偏差 **5.82%**——正是 `comparison` 逐项表里唯一报出的
失败项。于是「阶梯显示偏差在带内且随网格衰减」这句话在**被判失败的同一个算例**上成立，
却没有把失败项放进阶梯。修复后：

- `worst_diagonal` / `worst_coupling` = 两张矩阵的最大值，另附
  `worst_*_added_mass` / `worst_*_damping` 分列；
- 阶梯在最细网格上的值与 §3.3 逐项表的 active 项最大值**逐位相等**（新增测试钉死这条恒等式，
  旧实现会在阻尼侧差 5.82% vs 3.30%）；
- `results.json`（仅 convergence 块）、`docs/capytaine_crosscheck_r2.md`、图 2 由
  `scripts/capytaine_r2_recompute_convergence.py` 从冻结矩阵离线重算生成（不重解），
  `--check` 为防回归门；文档 §5 局限中「半球只跑到 256 面板」的过时描述同时按 artifact 重写。

### 7.2 新增证据：锚点密度上的「裁判」自己也没收敛
`scripts/capytaine_r2_referee_ladder.py` 只加密 capytaine（裁判）一侧、不重解 TensorBEM，
在 nu = 2 m^-1 上把裁判从 784/1120 面板推到 6400/8800 面板，预登记两条判据（阈值均 1%）：

| 判据 | 定义 | 结果 |
|---|---|---|
| C1 渐近 | 最细两层之间变化 | 最大 1.08% → **不满足** |
| C2 锚点密度 | 锚点自身网格 vs 最细网格 | 最大 4.69% → **不满足** |

关键量：锚点密度上超出 5% 带的 3 项，换成最细裁判后**全部回到带内**——
`hemisphere_r1 A[heave,heave]` 阻尼 +5.82% → **+3.02%**；
`sphere_r1_z-0.5 A[surge,surge]` −6.49% → **−2.14%**；`A[sway,sway]` −6.52% → **−2.18%**。
故锚点在 nu = 2 m^-1 的「physics FAIL」主要是**裁判网格未收敛**造成的，不是 TensorBEM 的实现缺陷；
锚点 15% 验收（最差 active 6.71%）与此并不冲突。

### 7.3 验证与残余欠账
- `pytest tests/test_capytaine_crosscheck_r2.py tests/test_capytaine_crosscheck.py`：37 passed（含新增 8 条：
  两矩阵阶梯恒等式、5.82% vs 3.30% 钉死、离线复算门、裁判阶梯算术/判据/文档一致/最粗层 live 复现）。
- 未做：TensorBEM 侧再加密一层（4098 面板，单核约 20–30 min）以做**双侧**加密判决实验；
  本轮的结论只到「裁判侧移动量与被测偏差同量级，故单方面归因不成立」，未宣称 TensorBEM 正确。
