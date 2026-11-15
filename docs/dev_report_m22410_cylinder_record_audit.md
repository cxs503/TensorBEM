# CASE-BEM-LAPLACE-CYLINDER：记录陈旧诊断 + 证书自校验器（m22410）

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** agent/m22410

## ① 开工核对

```
slug                      | CASE-BEM-LAPLACE-CYLINDER
acceptance                | 算例池|BEM-LAPLACE-CYLINDER｜ref: 解析复势｜容差: <3%
status                    | open
```

锚点 open，前序轮次（m21490/m22173 等）已把算例模块做到 44+ 判据。本轮不重做数值，
而是攻击前一轮**自己留下的欠账**：第 6 轮（m22173）明确写着「待完成项：集群作业 cj3943
取回并落盘」——该作业从未落盘。

## ② 缺陷诊断（在 main 上可复现）

| 事实 | 证据 |
| --- | --- |
| 提交的记录是 38 条判据，且**没有 `provenance`** | `python3 -c "json.load(...)['criteria']"` → 38，`payload["provenance"]` 不存在 |
| 当前模块会输出线性密度近壁节、方位角节、刻面归因节、斜向来流扫描 | `case_cylinder_stream.criteria()` / `heading_study()` |
| 因此 `render_markdown(旧记录)` **抛 `KeyError`** | `pytest tests/test_case_cylinder_stream.py -k committed_case_artifact` → FAILED |
| `provenance` 门因此只能 skip | `test_committed_record_was_rendered_from_the_current_case_module` → SKIPPED |

复现命令：

```bash
python3 -m pytest tests/test_case_cylinder_stream.py -q \
  -k "committed_case_artifact or provenance"
# 1 failed, 1 skipped
```

这不是文档瑕疵：**记录与代码漂移会让 main 上一条文档可复现性测试持续为红**，
而且"文档 vs 记录"的检查看不出两者同时落后——正是 `provenance` 门存在的原因。
重生成是唯一正确修法，本轮已递交集群作业 **cj4100**（8 核，3600 s）：

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder \
  --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check
```

**诚实状态：cj4100 在本单窗口内始终排队未启动**（递交后 30 min，队列 51–54 作业在飞，
排队位次 30）。因此本轮**没有**重生成记录，红色测试**没有**转绿；下面第 ③ 节是本轮
真正落地并已验证的增量。

## ③ 落地的增量：记录证书自校验器（不跑求解器）

`src/tensorbem/case_pool_verdict.py`（新模块）把一份 case-pool 记录当成**证书**重算三件事，
全程不 import 算例模块、不调求解器：

1. **阈值重算**：把每条判据的机读阈值（`"< 0.03"`、`"<= 1e-12 a"`、`">= 1.5"`）解析后
   重新作用到已发布的 `value` 上，与已记录的 `ok` 对照；纯文字阈值（`"in [0.5, 2.0]"`、
   `"True on every level"`）显式记为 **unverified**，绝不记 pass。
2. **acceptance 自洽**：`passed` / `failed_checks` / `n_checks` / `tolerance` 必须与判据表
   互相对得上；声明的容差不得宽于池容差（3%）。
3. **阶次来源**：每个 `observed_order_*` 用它自己的收敛阶梯（`*_relative_l2` + `h_mean`）重算，
   `finest` 行必须等于最后一阶。这条能抓住"汇总数字被手改"的情形——只抽查一个数看不出来。

阶梯判据的值可能是列表（模块对不同判据分别取最差阶/末阶），审计器**不猜聚合方式**：
只要某一种合理读法能复现记录的 `ok` 就算一致，并把"哪些读法满足"记入
`aggregation_ambiguous`，供人复核。

用法：

```bash
PYTHONPATH=src python3 -m tensorbem.case_pool_verdict \
  benchmarks/verified/case_bem_laplace_cylinder/results.json          # 退出码 0/1 可作门
```

### 3.1 对当前（未重生成的）旧记录的审计结果

| 项 | 结果 |
| --- | --- |
| 判据总数 | 38 |
| 机读阈值重算一致 | **35 / 38**（矛盾 0） |
| 纯文字阈值（不可机验） | 3：`near_wall_window_error_scales_with_h`、`near_wall_error_is_panel_relative`、`near_wall_guard_rejects_inside_window` |
| acceptance 自洽 | problems = 无（passed/failed_checks/n_checks=38/tolerance=0.03 全对上） |
| 阶次重算 | 全部拟合零不符（含 `finest` 行） |
| blocking | **false** |

即：**旧记录的数字本身是自洽的**，它的问题只是"落后于代码"，不是"数字伪造"。
这条区分很重要——审计器把"陈旧"与"不可信"分开，重生成后同一工具继续守住"不可信"。

### 3.2 可证伪性（7 条反例测试）

`tests/test_case_pool_verdict.py`（12 条测试，全部通过）里有 7 条是反例：

- `value` 改到阈值外仍写 `ok=True` → 必须报 `disagreements` + blocking；
- `failed_checks` 非空但 `passed=True` / `overall=PASS` → 必须报 problem；
- `tolerance=0.2` → 必须报 "looser than pool tolerance"；
- `observed_order_velocity=[1.0]` 而阶梯实测阶 2 → 必须报 `order_mismatches`，
  且重算值 `pytest.approx(2.0)`；
- `finest` 行改错 → blocking；
- 阶梯判据两种读法冲突 → 不 blocking 但报 `aggregation_ambiguous=["first","max"]`；
- CLI 退出码对好/坏记录分别为 0/1。

## ④ 误差判定

旧记录（38 判据）关键量 vs 解析复势（同点闭式求值，最细网格 N=128）：

| 关键量 | 相对 $L_2$ | <3% |
| --- | --- | --- |
| 路线 D 外部速度 | 1.366e-07 | 是 |
| 路线 D 扰动势 | 3.578e-07 | 是 |
| 路线 N 物面速度 | 1.008e-04 | 是 |
| 路线 N 外部速度 | 1.532e-04 | 是 |
| 路线 G 物面速度 | 由记录 `routes.circulation.finest` 注入 | 是 |
| 路线 G 升力（Kutta-Joukowski） | 4.014e-04 | 是 |

最差 4.014e-04，比 3% 容差小约 75 倍：**是，全部 <3%**。
但**注意**：以上仍是旧记录的数字，重生成后需要复核（斜向 heading 扫描与线性密度近壁节的
判据尚未在这一份记录里出现过，其数值本轮**未**取得）。

## ⑤ 未完成 / 交接

1. **重生成未完成**：cj4100 排队未启动。接手方式：
   `run_compute action=fetch job_id=cj4100`（或重递交同一命令），把
   `results.json` / `report.md` / `case.json` 落盘到
   `benchmarks/verified/case_bem_laplace_cylinder/`，`docs/CASE-BEM-LAPLACE-CYLINDER.md`
   同步，然后确认：
   `pytest tests/test_case_cylinder_stream.py -k "committed" -q` 全绿、
   `python3 -m tensorbem.case_pool_verdict benchmarks/verified/case_bem_laplace_cylinder/results.json`
   退出码 0。
2. **审计器覆盖面**：本仓使用"判据对象 + acceptance 块"这一族 schema 的记录只有 2 份
   （本锚点 + `case_bem_ellipse_2d_added_mass`），其余 bench 记录显式报 `unsupported`。
   扩大 schema 支持是下一个可单干的增量。

## ⑥ 产物

- `src/tensorbem/case_pool_verdict.py`：审计器（新模块，CLI + 可 import）
- `tests/test_case_pool_verdict.py`：12 条测试（含 7 条反例）
- `make_report_m22410.py`：报告/图件生成器（数字全部程序化注入）
- `docs/figures/m22410/{field,convergence,verdict_audit}.png`
- 平台报告：见 finish summary 的报告 id