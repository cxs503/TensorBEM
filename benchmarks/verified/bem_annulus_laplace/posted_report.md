# `CASE-BEM-ANNULUS-LAPLACE` — 环形域 Laplace（多连通）算例池条目

## ① 案例介绍

物理背景：环形域上的二维 Laplace 方程是势流（绕流/附加质量）、稳态导热、静电势的共同模型；真实几何对应月池、带内筒的浮体剖面、套管壁。域为 `b < r < a`，`a = 1.0`、`b = 0.5`（壁厚比 `b/a = 0.50`），边界由两条互不相连的闭曲线组成，**内环的域外法向指向孔洞内部**——单环面元数据结构根本无法装配的拓扑。

参考解（闭式调和函数，全部由程序从公式生成，零手抄）：

| 算例 | 闭式解 |
|---|---|
| `radial_log` | `u = 0.7 log r + 1.3` |
| `angular_cos` | `u = (r + 0.16/r) cos(theta)` |

**ref 出处**：Lamb H. *Hydrodynamics*, 6th ed., CUP, 1932（均匀流绕圆柱，与 `docs/classical_benchmarks.md` 参考文献 [1] 同版本）；Milne-Thomson L M. *Theoretical Hydrodynamics*, 5th ed., Macmillan, 1968（平移圆柱势，同文件 [3]）。**诚实缺口**：容器内两本书不可达，具体页码/条目号未核，登记为 `page_or_article_number = WITHHELD`，不编页码；补偿手段是把引用换成测量——`harmonicity_check` 用中心差分验证闭式解确实调和（见下表 gate）。

| 算例 | 采样点 | h | max abs Lap(u) | 相对残差 |
|---|---|---|---|---|
| radial_log | 64 | 1e-04 | 8.882e-08 | 6.961e-16 |
| angular_cos | 64 | 1e-04 | 1.332e-07 | 1.199e-15 |

## ② 怎么做的

复用仓内既有能力，不另起炉灶：`tensorbem.multiloop.MultiLoopMesh` / `annulus_mesh`（多环网格，孔洞法向翻转）→ `solve_laplace_multiloop`（常数面元、中点配点、面元解析积分的直接 BEM：`S q = (1/2 I + H) g`）→ `evaluate_multiloop_field`（Green 表示求内部场）→ 与 `tensorbem.benchmarks.annulus_laplace.CASES` 的闭式场函数逐点对照。本单新增的只是算例池外壳：**阈值先声明后计算**（写进 `results.json` 的 `acceptance` 后不再改）、参考值程序生成、两组敏感性、两个负控、机器可判 verdict、算例卡与图件。

工具链：Python 3.11 + numpy（解析面元积分）+ matplotlib（图件）；单机运行约 3 分钟，未递交集群（远低于 10 分钟门槛，且不计入控制面重计算）。

## ③ 参数表（无量纲数完整）

| 参数 | 值 |
|---|---|
| 外半径 `a` | 1.0 |
| 内半径 `b` | 0.5 |
| 壁厚比 `b/a` | 0.50 |
| `radial_log`: `A`, `B` | 0.7, 1.3 |
| `angular_cos`: `c`（故 `c/a`） | 0.4 |
| 核参考长度 `L`（故 `a/L`） | 2.0（0.5） |
| 探针半径 `r = b + f(a−b)` | `f ∈ {0.2, 0.5, 0.8}` → r/a = 0.6, 0.75, 0.9 |
| 每半径方位角数 | 16（合计 48 探针/算例） |
| 声明容差 | 3.0% |
| 声明最小拟合阶 | 1.5 |
| 广告面元预算 | 64/32 |

## ④ 计算结果（ref vs got，逐项误差）

### `radial_log` — `u = 0.7 log r + 1.3`

| 外/内面板 | N | cond(S) | ‖q−q_ref‖₂/‖q_ref‖₂ | ‖q−q_ref‖∞/‖q_ref‖∞ | max|u−u_ref|/max|u_ref| | 逐环通量误差 | 相对残差 |
|---|---|---|---|---|---|---|---|
| 16/8 | 24 | 1.761e+01 | 2.718e-02 | 3.127e-02 | 9.678e-03 | 1.739e-02 | 3.220e-16 |
| 32/16 | 48 | 3.308e+01 | 7.097e-03 | 8.472e-03 | 1.825e-03 | 4.436e-03 | 2.568e-16 |
| 64/32 | 96 | 6.524e+01 | 1.840e-03 | 2.224e-03 | 4.000e-04 | 1.140e-03 | 3.350e-16 |
| 128/64 | 192 | 1.300e+02 | 4.701e-04 | 5.706e-04 | 9.585e-05 | 2.903e-04 | 4.604e-16 |
| 256/128 | 384 | 2.598e+02 | 1.189e-04 | 1.446e-04 | 2.348e-05 | 7.330e-05 | 8.312e-16 |

观测阶（逐级 log2 比）：1.937, 1.947, 1.969, 1.984；拟合阶 **1.959**。

逐环通量（最细层）：

| 环 | 计算 ∮q ds | 参考 ∮q ds |
|---|---|---|
| 外环 | +4.398558e+00 | +4.398451e+00 |
| 内环 | -4.398477e+00 | -4.399113e+00 |

### `angular_cos` — `u = (r + 0.16/r) cos(theta)`

| 外/内面板 | N | cond(S) | ‖q−q_ref‖₂/‖q_ref‖₂ | ‖q−q_ref‖∞/‖q_ref‖∞ | max|u−u_ref|/max|u_ref| | 逐环通量误差 | 相对残差 |
|---|---|---|---|---|---|---|---|
| 16/8 | 24 | 1.761e+01 | 1.074e-01 | 1.389e-01 | 1.619e-02 | 2.821e-16 | 3.011e-16 |
| 32/16 | 48 | 3.308e+01 | 2.558e-02 | 3.496e-02 | 3.067e-03 | 1.899e-16 | 3.369e-16 |
| 64/32 | 96 | 6.524e+01 | 6.315e-03 | 8.742e-03 | 6.319e-04 | 2.090e-16 | 3.107e-16 |
| 128/64 | 192 | 1.300e+02 | 1.572e-03 | 2.184e-03 | 1.551e-04 | 3.835e-16 | 4.380e-16 |
| 256/128 | 384 | 2.598e+02 | 3.924e-04 | 5.455e-04 | 3.882e-05 | 5.060e-16 | 7.289e-16 |

观测阶（逐级 log2 比）：2.070, 2.018, 2.006, 2.002；拟合阶 **2.022**。

逐环通量（最细层）：

| 环 | 计算 ∮q ds | 参考 ∮q ds |
|---|---|---|
| 外环 | -1.894318e-15 | -6.938894e-17 |
| 内环 | +9.835882e-16 | +1.908196e-17 |

### 判定（阈值先声明，逐项打勾/叉）

| 算例 | 检查项 | 值 | 阈值 | 判定 |
|---|---|---|---|---|
| radial_log | q_relative_l2_error | 1.189e-04 | 0.03 | ✅ PASS |
| radial_log | q_relative_linf_error | 1.446e-04 | 0.03 | ✅ PASS |
| radial_log | field_relative_max_error | 2.348e-05 | 0.03 | ✅ PASS |
| radial_log | flux_relative_error | 7.330e-05 | 0.03 | ✅ PASS |
| radial_log | fitted_order_q_relative_l2 | 1.959e+00 | 1.5 | ✅ PASS |
| radial_log | harmonicity_max_abs_laplacian_relative | 6.961e-16 | 1e-06 | ✅ PASS |
| radial_log | budget_panels_meet_tolerance | 1.840e-03 | 0.03 | ✅ PASS |
| angular_cos | q_relative_l2_error | 3.924e-04 | 0.03 | ✅ PASS |
| angular_cos | q_relative_linf_error | 5.455e-04 | 0.03 | ✅ PASS |
| angular_cos | field_relative_max_error | 3.882e-05 | 0.03 | ✅ PASS |
| angular_cos | flux_relative_error | 5.060e-16 | 0.03 | ✅ PASS |
| angular_cos | fitted_order_q_relative_l2 | 2.022e+00 | 1.5 | ✅ PASS |
| angular_cos | harmonicity_max_abs_laplacian_relative | 1.199e-15 | 1e-06 | ✅ PASS |
| angular_cos | budget_panels_meet_tolerance | 6.315e-03 | 0.03 | ✅ PASS |
| radial_log | orientation_control_ratio | 9.020e+02 | 10.0 | ✅ PASS |
| angular_cos | orientation_control_ratio | 2.398e+02 | 10.0 | ✅ PASS |
| probe_guard | hole_probe_refused | True | True | ✅ PASS |

**总体判定：PASS**（容差 3.0%）。

## ⑤ 收敛性 / 参数敏感性（两组）

**敏感性 1：壁厚比 `b/a`（固定 64/32 面元，`radial_log`）**

| `b/a` | cond(S) | q 相对 L2 误差 | 场内相对最大误差 | 是否 <3% |
|---|---|---|---|---|
| 0.20 | 1.320e+02 | 2.683e-03 | 3.739e-04 | ✅ |
| 0.30 | 9.516e+01 | 2.435e-03 | 3.876e-04 | ✅ |
| 0.50 | 6.524e+01 | 1.840e-03 | 4.000e-04 | ✅ |
| 0.70 | 7.199e+01 | 1.361e-03 | 4.383e-04 | ✅ |
| 0.85 | 7.658e+01 | 2.889e-03 | 5.836e-04 | ✅ |
| 0.90 | 7.803e+01 | 5.010e-03 | 6.903e-04 | ✅ |
| 0.95 | 7.946e+01 | 1.103e-02 | 8.646e-04 | ✅ |

**敏感性 2：核参考长度 `L`（对数容量退化，固定 64/32 面元）**

| `L` | `a/L` | cond(S) | q 相对 L2 误差 | 相对残差 | 是否 <3% |
|---|---|---|---|---|---|
| 1.00 | 1.000 | 5.945e+02 | 1.220e-01 | 2.906e-16 | ❌ |
| 1.50 | 0.667 | 4.266e+01 | 1.864e-03 | 2.765e-16 | ✅ |
| 2.00 | 0.500 | 6.524e+01 | 1.840e-03 | 3.350e-16 | ✅ |
| 4.00 | 0.250 | 1.217e+02 | 1.829e-03 | 5.381e-16 | ✅ |
| 8.00 | 0.125 | 1.786e+02 | 1.826e-03 | 7.379e-16 | ✅ |

## ⑥ 注意点 / 失败模式 / 工件

**失败模式 1：内环法向未翻转。**

| 算例 | 面板 | 正确 q 误差 | 未翻转 q 误差 | 倍数 |
|---|---|---|---|---|
| radial_log | 64/32 | 1.840e-03 | 1.660e+00 | 902.0× |
| angular_cos | 64/32 | 6.315e-03 | 1.514e+00 | 239.8× |

关键陷阱：未翻转系统的相对残差为 3.924e-16（正确为 3.350e-16），**残差型验收完全抓不到方向错**——只能靠网格类型约束（`MultiLoopMesh.hole_loops`）。

**失败模式 2：孔洞内探针（本单发现的缺陷并已修）。** 
负控探针 `[0.1, 0.0]` 被拒绝：**True**；消息：`points must lie in the fluid domain: 1 target(s) fall inside a hole loop, where the Green representation evaluates the analytic continuation and not the field of the problem`。修复前 `evaluate_multiloop_field` 对孔洞内/外环外的目标点照算 Green 表示，返回「看起来合理」的解析延拓值；`tensorbem.multicomponent.evaluate_fluid_field` 早有同类守卫，多环路径漏了。本单补 `_point_in_polygon` + `domain_membership`（`tensorbem/multiloop.py`）并加测试。

**适用范围**：64/32 面元下壁厚比扫到 `b/a` 最薄一档仍达标（全部达标）；条件数最差出现在 `L = 1.00`（`a/L = 1.000`，cond = 5.945e+02，q 误差 1.220e-01），说明环形域继承了自由空间核的对数容量悬崖，默认 `L = 2`（a/L = 0.5）在安全侧但不远。

**工件与复现**：

- `benchmarks/verified/bem_annulus_laplace/results.json`（机器记录，schema `tensorbem.case_annulus_laplace/1`）
- `benchmarks/verified/bem_annulus_laplace/case.json`（算例卡：参数/预算/复现命令）
- `benchmarks/verified/bem_annulus_laplace/report.md`（由 results.json 渲染）
- `benchmarks/verified/bem_annulus_laplace/figures/`（收敛、场+误差、两组敏感性，四张图）
- `docs/case_annulus_laplace.md`（索引与判断记录）

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_annulus_laplace \
    --outdir benchmarks/verified/bem_annulus_laplace --check
PYTHONPATH=src python -m pytest tests/test_case_annulus_laplace.py tests/test_multiloop_annulus.py -q   # 24 passed
```

## ⑦ 误差判定（是否 <3%，逐项显式回答）

- `radial_log` @ 256/128：边界 q 相对 L2 1.189e-04（<3%）；边界 q 相对 Linf 1.446e-04（<3%）；内部场相对最大 2.348e-05（<3%）；逐环通量相对 7.330e-05（<3%）。
- `angular_cos` @ 256/128：边界 q 相对 L2 3.924e-04（<3%）；边界 q 相对 Linf 5.455e-04（<3%）；内部场相对最大 3.882e-05（<3%）；逐环通量相对 5.060e-16（<3%）。
- 广告预算 64/32 面元同样全部 <3%（radial_log 1.840e-03，angular_cos 6.315e-03）。
- **结论：全部关键量均 <3%**。ref 页码未核（WITHHELD），但调和性已机器验证，故 ref 合法性不依赖未核的引用。

## ⑧ 生态摩擦（如实记）

- `GET /api/v1/tensor/knowledge/search` → `engine_offline`（`127.0.0.1:8021` 连接被拒）：开工三查的第一查（capability）无法执行。
- `GET /api/v1/knowledge/search`（sediment store）对 `TensorBEM` / `annulus` 均 0 条。
- `GET /api/v1/tensor/negative-knowledge/search?q=TensorBEM` → 0 命中；`q=annulus` → 0 命中。
- 平台 `/openapi.json` 无 oss_solutions 路由，第三查只能以仓内 grep 代替（结果：既有能力 `tensorbem.multiloop` + `benchmarks/annulus_laplace`，故复用不新建）。
- 预存测试欠账（与本单无关，未修）：`tests/test_capability_coverage.py` 报 `tensorbem.shape_gradient`、`tensorbem.steady_kelvin` 未登记；`tests/test_doc_claim_audit.py` 报根目录若干 `dev_report_*.md` 未进 baseline。
