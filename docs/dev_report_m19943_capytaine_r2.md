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

## 4. 未完成的部分（如实）
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
