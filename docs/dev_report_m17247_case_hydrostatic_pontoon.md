# 开发报告 m17247 — CASE-BEM-HYDROSTATIC-PONTOON（矩形浮箱水静力，TensorBEM）

> 任务：算例池（case pool）第 1 轮铺开，锚点 `BEM-HYDROSTATIC-PONTOON`，ref = Biran & Pulido /
> Rawson & Tupper 闭式，判据 **< 3 %**。
> 运行：`python3 scripts/case_hydrostatic_pontoon.py --out artifacts/case_hydrostatic_pontoon`
> 工件：`artifacts/case_hydrostatic_pontoon/{results.json, report.md, fig_*.png}`，
> 回归：`tests/test_case_hydrostatic_pontoon.py`（8 项，9.1 s 全绿）。

## 1. 问题 + 参考解出处

壁面（wall-sided）矩形浮箱 L = 50 m，B = 12 m，D = 5 m，吃水 T = 3 m，重心高 KG = 4 m（龙骨以上），
ρ = 1025 kg/m³，g = 9.80665 m/s²。求全部水静力量：∇、Δ、A_wp、Cb、Cwp、Cm、Cp、KB、LCB、LCF、TCF、
I_T、I_L、I_xy、BM_T、BM_L、KM_T、KM_L、GM_T、GM_L，以及复原矩阵对角 C33/C44/C55 与零耦合项。

参考解：Biran & López-Pulido, *Ship Hydrostatics and Stability*, 2nd ed.（Butterworth-Heinemann），
ch. 3–4（箱形船水静力：∇ = LBT、KB = T/2、BM = I/∇、KM = KB + BM）；Rawson & Tupper, *Basic Ship Theory*,
5th ed.，ch. 3–4（箱驳水静力）。横倾用壁面公式 GZ = sinφ·(GM + ½·BM·tan²φ)。

**引用诚实性（本单明确的未达标项）**：本沙箱内无网络、无该两书的印刷本或电子本，因此**未给出页码/表号**，
引用只到章级；所有参考数值由 `reference_values()` 在代码里从几何量求值（零手抄、无手打数字）。
同域生态内已有等价实现可交叉核对：`TensorPlatform/docs/verification_debt/stability_pontoon/refs.py::pontoon_analytic`
（同一公式集，已被稳性门使用）。

## 2. 方法与实现（复用既有能力，不另起炉灶）

两条**结构性独立**的 TensorBEM 既有路线跑同一箱体：

| 路线 | 入口 | 原理 |
|---|---|---|
| `waterplane` | `tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` | 追踪 z=0 水线环，2D Green 定理积分 A、S_x、S_y、I_xx、I_yy、I_xy，再由闭式水静力公式组装 C |
| `pressure` | `tensorbem.hydrostatic_pressure_route.richardson_restoring_matrix` | 闭合箱体裁剪到 z ≤ 0，积分 p = ρg(0−z) 得合力/合力矩，中心差分得 C，再对步长做 Richardson 外推 |

第三条独立校核（本单新增的小工具函数 `immersed_first_moment_z`）：由裁剪面按
∫z dV = ½∮z²n_z dS 直接得 ∇·z_B，给出与「水线面 Green 定理」和「四面体形心」都不同的 B 高度路线
（压力合成本身只定 x_B、y_B：竖直力给出 M_x = y_B·W，z_B 不可见——这正是生态里
`PressureRouteResult.center_of_buoyancy[2]` 恒为 0 的原因）。

第四路校核：横倾 0.5°–5° 时用压力积分直接算复原力矩（含重量项，力矩原点=静水面），除以 W 得 GZ，
与壁面闭式 GZ 逐角比对。

**几何先验门**（先验几何再比数）：由网格顶点量 L/B/T 包围盒（= 50/12/3，误差 0）、
Cb = ∇/(LBT) = 1、Cwp = A_wp/(LB) = 1；另核 divergence 路线体积与水线环面积互等（误差 0）。

## 3. 参数表（含无量纲数）

| 量 | 值 | 无量纲量 | 值 |
|---|---|---|---|
| L | 50 m | L/B | 4.166667 |
| B | 12 m | B/T | 4.000000 |
| D | 5 m | T/D | 0.600000 |
| T | 3 m | KG/T | 1.333333 |
| KG | 4 m | Cb = Cwp = Cm = Cp | 1.000000 |
| ρ | 1025 kg/m³ | GM_T/T | 0.500000 |
| g | 9.80665 m/s² | GM_L/L | 1.338889 |

主网格 = (n_L, n_B, n_T) = (4, 3, 3)，水线路线 108 个湿表面三角形，压力路线 132 个闭合三角形
（含 2 m 干舷，被 z=0 裁剪）。

## 4. 对照表（ref vs got，误差逐项）

完整表在 `artifacts/case_hydrostatic_pontoon/report.md` §3（22 项，全部 ✅）。摘要：

| 量 | 参考 | waterplane | 误差 | pressure | 误差 |
|---|---|---|---|---|---|
| ∇ | 1800 m³ | 1800 | 0 | 1800 | −1.3e−16 |
| A_wp | 600 m² | 600 | 0 | — | — |
| KB | 1.5 m | 1.5 | 0 | 1.5 | −3.0e−16 |
| I_T | 7200 m⁴ | 7200 | 0 | — | — |
| I_L | 125000 m⁴ | 125000 | 0 | — | — |
| BM_T | 4 m | 4 | 0 | — | — |
| BM_L | 69.444 m | 69.444 | 0 | — | — |
| GM_T | 1.5 m | 1.5 | 0 | — | — |
| C33 | 6.03109e6 N/m | 同 | 0 | 同 | −1.6e−12 |
| C44 | 2.71399e7 N·m/rad | 同 | 0 | 同 | −1.2e−12 |
| C55 | 1.21124e9 N·m/rad | 同 | 0 | 同 | −5.8e−13 |
| max|C_ij|off-diag/max|C| | 0 | 0 | 0 | 6.95e−14 | 6.95e−14 |

**最大相对误差 1.601e−12（≪ 3 %，判据 ✅ 22/22）**。

## 5. 收敛性 / 参数敏感性（四点）

1. **面板加密**：(2,2,2)→(16,12,12)（40→1728 面板）最大误差 1.47e−12 → 3.89e−11，**不随 h 下降**；
   水线路线逐位不变（体积与水线面矩是精确和），上升部分来自压力路线在更多面上的舍入累加。
   → 平面箱体上误差只可能来自实现 bug，不可能是离散误差（这正是该算例的诊断价值）。
2. **吃水扫描** T = 2.0…4.0 m：BM_T 与 B²/(12T) 逐点一致（误差 0），且中心差分 d∇/dT 与 A_wp = 600 m²
   一致到 5e−13——验证了体积-水线面这一对物理恒等式，而不只是两个孤立数字。
3. **g 取值**：g = 9.81 / 9.8 相对 9.80665 使 C 全部偏 ±3.4e−4 / −6.8e−4（几何量不变）；
   即「g 约定差」占 3 % 判据的 1 % 量级，必须声明约定（本单用 9.80665）。
4. **压力路线差分步长** 4e−3…5e−4：C44/C55 误差按 **实测二阶**（obs. order = 2.00）下降，
   Richardson 外推后主网格误差 1e−12；C33 对步长不敏感（零阶力精确）。
5. **横倾复原力臂**：0.5°–5° 五个角，压力积分 GZ 与壁面闭式 GZ 一致到 1e−15 相对
   （例：5° 时 0.1320678 m vs 0.1320678 m），GM·sinφ 在 5° 已偏 1.0 %——说明「小角公式」的
   偏离被同一条壁面公式精确捕获，不是矩阵线性化误差。

## 6. 失败模式与工件

- 已知失败模式：(a) 曲面板（如 spheroid）会暴露 O(h²) 剖面误差，本箱体不触发；
  (b) 压力路线步长过大时截断误差可淹没真实差（§5.4）；(c) KG > KM_T 则 GM < 0、复原矩阵变号，
  本算例 GM_T = 1.5 m > 0 不触发；(d) 力矩积分必须用二阶精确规则（m16662 定位过的 4 % 缺陷），
  本单用模块现成规则。
- 工件：`artifacts/case_hydrostatic_pontoon/{results.json, report.md, fig_rel_errors.png,
  fig_refinement.png, fig_draft_sensitivity.png, fig_heel_righting.png}`
  （`artifacts/` 在本仓被 .gitignore 忽略，故 `results.json` / `report.md` 另存一份入库副本：
  `docs/case_hydrostatic_pontoon/{results.json, report.md}`；png 由 repro_cmd 重新生成，
  平台报告 id 16242 已内嵌四图）；
  `results.json` 含环境（python/numpy/平台/git commit）、案例参数、全部参考值与实测值、三张敏感表、
  verdict、脚本 sha256。
- 复现：`python3 scripts/case_hydrostatic_pontoon.py --out artifacts/case_hydrostatic_pontoon`
  （≈11 s 单机；无需集群）。

## 7. 生态摩擦（如实记）

1. **knowledge_base 引擎离线**：`GET /tensor/knowledge/search` 返回
   `{"available": false, "reason": "engine_offline", "engine": {"base": "http://127.0.0.1:8021",
   "reachable": false}}`；`/knowledge/search` 返回 0 条（store_dir 为空）。
   → 开工三查的「knowledge_base 同域能力」这一查**未取到**，以仓内检索替代（`hydrostatic*.py`、
   `hydrogen_benchmarks`、`docs/*`）。
2. **negative_knowledge_ledger 查询 422**：`GET /tensor/negative-knowledge/search?keywords=TensorBEM&limit=5`
   → 422 Unprocessable Entity（参数名/必填项与文档不一致），未能查同域否定性知识。
3. **oss_solutions 端点 404**：`/tensor/oss_solutions/search`、`/tensor/knowledge_base/search` 均 404；
   实际路由是 `/tensor/knowledge/search` 与 `/knowledge/search`。
4. **算例池参数不可读**：平台无 case_pool 读接口，`psql` 无口令、本地 sqlite 无 case 表，
   因此 BEM-HYDROSTATIC-PONTOON 池行的几何参数无法取回——本单几何为**自定并写入报告**（L=50/B=12/T=3），
   若池行另有既定几何，属供给线断点，建议后续把 case_pool 的 `parameters` 走 API 暴露。
5. **页级引用不可达**：沙箱无网络（`openlibrary`/`googleapis` 连接失败，`example.com` 偶通），
   两本参考书均无法取页号 → 本单结论的引用止于章级（诚实标注，不编页码）。
6. **API 陷阱（代码层面，已在本单验证）**：`pressure_loads` 必须喂**已裁剪到 z ≤ 0 的湿表面**；
   直接喂闭合箱体，z > 0 的面带负压（p = ρg(0−z) < 0），复原力矩会翻符号并偏 1.22 倍
   （本单第一次实现即踩到，5° 时 GZ 变负）。另：`PressureRouteResult.center_of_buoyancy[2]`
   恒为 0，不能当 KB 用（竖直力对 z_B 不可见）。