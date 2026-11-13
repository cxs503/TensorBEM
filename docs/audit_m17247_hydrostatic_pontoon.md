# 交差审计 — m17247 / CASE-BEM-HYDROSTATIC-PONTOON（TensorBEM）

审计单：m17267（board-audit）。对象：m17247（已 succeeded，commit `7c4043a`，工作树 `/root/tp-agent/work/m17247`）。
本单是**核验单**，不重做算例；所有数字来自本审计自己的重跑/复算。

## 0. 一句话结论

**判级 L4（基准验证），无一等缺陷；6 项发现全部为登记/口径/披露类，无造假、无虚标。**
关键量 vs 基准：22 项判据量全部 < 3 %（实测最大 **1.6013383151465308e-12**），独立复算（本审计自写闭式，不调被审代码）30 个路线值最差 **1.601e-12**，与作者自述逐位一致。

## 1. 取证与复跑

| 项 | 命令 | 结果 |
|---|---|---|
| 一键复现 | `PYTHONPATH=src python3 scripts/case_hydrostatic_pontoon.py --out /tmp/m17267_repro` | `pass: true, n_quantities: 22, max_abs_rel_err: 1.6013383151465308e-12`，≈8 s 单机 |
| 工件一致性 | 重跑 `results.json` vs 提交版 `docs/case_hydrostatic_pontoon/results.json` 逐字段比对（跳过 `environment/repro_cmd/figures/sha`） | **numeric diffs |dx|>1e-12 = 0**（bit 级可复现） |
| 回归测试 | `pytest tests/test_case_hydrostatic_pontoon.py -q` | **8 passed**（8.60 s） |
| 报告六段 | `tensor_reports` id=**16242**（mission_id=17247，6718 B，4 张 PNG） | 六要素齐全（①案例/参数 + ②方法 + ③对照表 + ④4 图 + ⑤注意点 + ⑥"是否<3%"显式回答） |
| 基准裁判 | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-HYDROSTATIC-PONTOON'` | **0 行**（见 F1） |
| 实证行 | `SELECT count(*) FROM tensorbench_results WHERE mission_id=17247` | **0 行**（见 F6） |

## 2. 独立复算（本审计自写，不 import 被审 `reference_values`）

对 `results.json` 的 22 项逐项用一阶原理重算：∇=LBT、Δ=ρ∇/1000、A_wp=LB、KB=T/2、
I_T=LB³/12、I_L=BL³/12、BM_T=I_T/∇、GM=KB+BM−KG、C33=ρgA_wp、C44=ρg∇GM_T、C55=ρg∇GM_L。
**30 个 (量 × 路线) 值，最差相对误差 1.601e-12，超 1e-9 的失配 0 项**。
即：作者"ref"虽由同脚本闭式求值（自引用风险，见 F3），但数值与外部一阶原理完全一致。

## 3. 反硬编码/泛化探针（本审计新增，被审单未做）

用被审脚本的 `measure_waterplane_route` / `measure_pressure_route` 喂**新几何**（非算例默认值）：

| 输入 | ∇ (wp / pr / 闭式) | BM_T (wp / 闭式) | C44 (wp / 闭式) |
|---|---|---|---|
| L=50, T=3 | 1800.000000 / 1800.000000 / 1800.000000 | 4.000000 / 4.000000 | 27139903.8750 / 27139903.8750 |
| L=50, T=4 | 2400.000000 / 2400.000000 / 2400.000000 | 3.000000 / 3.000000 | 24124359.0000 / 24124359.0000 |
| L=75, T=3 | 2700.000000 / 2700.000000 / 2700.000000 | 4.000000 / 4.000000 | 40709855.8125 / 40709855.8125 |
| L=75, T=4 | 3600.000000 / 3600.000000 / 3600.000000 | 3.000000 / 3.000000 | 36186538.5000 / 36186538.5000 |

输出随几何正确变化且两条路线互相独立地命中闭式 → **不是回读常量**；底层
`compute_hydrostatic_restoring_matrix` / `richardson_restoring_matrix` 为仓内既有公共 API
（`src/tensorbem/__init__.py` 导出，34 个测试文件引用），本次提交未改 `src/`（commit stat 6 文件全在
`scripts/ docs/ tests/ post_report_*.py`）。

## 4. 发现（按严重度）

- **F1（medium，登记缺口）**：`tensor_benchmark_cases` 该锚点 **0 行** —— 池子里 5 个 BEM 锚点
  （MACCAMY-FUCHS / BARGE-HYDROSTATIC / SPHERE / ELLIPSOID / TWO-CYL）都有行，唯本锚点缺，
  故"基准判据=唯一裁判"在本单原本**无判据可裁**；m17247 的 succeeded 不依赖注册行。
  处理：本审计补注册 1 行（见 §5），容差按任务书 3 %，不放宽。
- **F2（medium，口径）**：**1.6e-12 不是"精度"，是"代数实现正确"**。壁面箱体 + 平面面板下，
  水线路线退化为矩形面积/惯性矩的精确求和，压力路线也只差 Richardson 外推；因此该数说明
  Green 定理积分与 C 矩阵组装无错，**不含离散热化误差信息**。旁证：加密阶梯 1.47e-12 → 1.60e-12 →
  5.01e-12 → 3.89e-11 随面板数**缓慢上升**（纯舍入），无收敛阶可言。报告 §5 已自陈
  "平面面板算例，只查实现、不查剖面离散" → **披露诚实**，但读者若读成"TensorBEM 水静力精度 1e-12"属误读。
- **F3（low，外锚）**：`external_anchor` 为**章级**引用（Biran & Lopez-Pulido ch.3-4；Rawson & Tupper ch.3-4），
  无页码、无原表数值抄录（沙箱无原书，作者已在报告与 `results.json.citation` 显式披露）。
  风险是 ref 由同一脚本的闭式函数求值（自引用）；本审计以自写复算（§2）+ 泛化探针（§3）消解，
  残余风险：未能与印刷手册表逐位对齐 —— 属**外部数据可达性**限制，不计作者过失。
- **F4（low，复现口径）**：`repro_cmd` 为仓相对路径
  `python3 scripts/case_hydrostatic_pontoon.py --out artifacts/case_hydrostatic_pontoon`，可执行（已重跑）；
  但需 `PYTHONPATH=src`（results.json 未写），直接照抄会 ImportError → 建议后续 repro_cmd 显式带 env。
- **F5（info，真陷阱）**：`hydrostatic_pressure_route.PressureRouteResult.center_of_buoyancy[2]` 恒为 0
  （竖直浮力对 z_B 不可见）——作者踩到并写了防误用说明；本审计确认该行为是 API 现状，记入复用知识。
- **F6（low，平台口径）**：`tensorbench_results` 本 mission **0 行** → 按
  `tools/maturity_gate/benchmark_source.py` 的新判据，本单在该数据源下会落 `CLAIMED_NOT_PROVEN`；
  本单 L4 依据是**审计重跑 + 注册行 + 闭式锚**，不是平台复算的 cluster 行（本地 8 s 计算无需集群）。

## 5. 落表

- `tensor_benchmark_cases`：补 `CASE-BEM-HYDROSTATIC-PONTOON` 一行（tolerance `<3%`，status=passed，
  metric 内注明审计号与实测值）。
- `tensor_maturity_verdicts`：`('TensorBEM', 'm17247 CASE-BEM-HYDROSTATIC-PONTOON', 17267, 4, <evidence JSON>, 'board-audit')`。

## 6. 判级理由（L4，非 L5）

- L1 ✅ 数学/单元验证：22 项对照闭式 + d∇/dT=A_wp 恒等式（≤5.3e-13）+ 步长二阶阶梯。
- L2 ✅ 真代码：`scripts/case_hydrostatic_pontoon.py`（864 行）输入几何→计算→JSON/图/表全链。
- L3 ✅（就能力面而言）：调用的是 `src/tensorbem` 公共 API（`__init__` 导出，34 个测试文件引用）；
  但**算例脚本本身无下游调用者**（grep 无引用）→ 作为"算例"看 L3 不适用，作为"能力"看满足。
- L4 ✅ 完整算例 + 逐项误差分析 + 一键复现（重跑 bit 一致）+ 回归测试 8/8 + 几何先验门（Cb=1、包围盒 0 误差）。
- L5 ❌：无实验/外部门级数据锚（ref 为章级闭式，且为自求值），且**本算例结构上无法暴露离散误差**（F2），
  不构成科研级精度证据。