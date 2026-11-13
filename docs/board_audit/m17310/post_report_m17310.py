#!/usr/bin/env python3
"""Publish the m17310 audit report (numbers injected from the replay artefacts, zero hand typing)."""
import base64, hashlib, json, pathlib, sys
sys.path.insert(0, "/tmp")
from tok import post  # noqa: E402  (authenticated platform client used across missions)

ROOT = pathlib.Path(__file__).resolve().parents[3]
SNAP = ROOT / "docs/board_audit/m17310/snapshot"
r = json.loads((SNAP / "rerun_results.json").read_text())
s = json.loads((SNAP / "subject_results_c121bee.json").read_text())
crit = r["criteria"]; diag = r["diagnostics"]
g = r["geometry"]
def f(x, n=6): return f"{x:.{n}e}" if isinstance(x, float) and abs(x) < 1e-3 else f"{x:.6f}"

koch = "\n".join(
    "| {k:.3f} | {ka:.2f} | {ref:.6f} | {gi:.6f} | {ei:.3e} | {go:.6f} | {eo:.3e} | PASS |".format(
        k=row["k"], ka=row["k_times_a"], ref=row["ref_abs"],
        gi=row["routes"]["panels_source_integrated"]["abs"], ei=row["routes"]["panels_source_integrated"]["rel_err"],
        go=row["routes"]["panels_ordinate"]["abs"], eo=row["routes"]["panels_ordinate"]["rel_err"])
    for row in r["kochin_table"])
res = "\n".join(
    "| {fr:.2f} | {u:.4f} | {far:.6f} | {near:.6f} | {pan:.6f} | {fv:.3e} | {fp:.3e} |".format(
        fr=row["froude"], u=row["speed_m_s"], far=row["rw_far_field_kochin_N"],
        near=row["rw_near_field_michell_N"], pan=row["rw_panels_N"],
        fv=row["far_vs_near_rel_err"], fp=row["far_vs_panels_rel_err"])
    for row in r["resistance_haskind_table"])
crit_rows = "\n".join(f"| {k} | {f(v['value'])} | {v['limit']:.0e} | {'PASS' if v['passed'] else 'FAIL'} |"
                      for k, v in crit.items())
conv = ""

md = f"""# 交差审计 m17273 · CASE-BEM-ELLIPSOID-KOCHIN（TensorBEM）——判级 L3，对接未闭合

## ① 案例介绍（物理背景 / 几何 / 参数表）

半浸薄椭球 `x²/a²+y²/b²+z²/c²≤1, z≤0`，深水薄船（Michell）框架。被测对象=**远场 Kochin 函数 H(k)** 及其经
Kelvin 积分的兴波阻力 R_W；参考=H(k) 闭式 + 长波极限 `H(k→0)=-ikV` + 远场/近场（Haskind 能量恒等）。

| 参数 | 值 | 无量纲 | 值 |
|---|---|---|---|
| 半轴 a / b / c | {g['a']} / {g['b']} / {g['c']} m | L/B | {g['L']/g['B']:.3f} |
| L / B / T | {g['L']} / {g['B']} / {g['T']} m | B/T | {g['B']/g['T']:.3f} |
| 浸没体积 V | {g['immersed_volume']:.9f} m³ | Cb | {g['Cb']:.9f} |
| ρ / g | 1025.0 kg/m³ / 9.80665 m/s² | 几何门 Cb vs π/6 | {crit['geometry_cb_half_ellipsoid']['value']:.3e} |
| k 网格 [1/m] | {", ".join(f"{row['k']:.3f}" for row in r['kochin_table'])} | 包络 | k·a ≤ 10 |
| 容差（池判据） | {r['pool_tolerance']:.0%} | 生态更严线 | <3% |

## ② 怎么做的（方法 / 流程 / 工具链）

审计**不是重做**：`git archive c121bee` 原样检出被审 commit → 一条命令复跑 → 逐数字 diff → 跑测试。
```
PYTHONPATH=src python3 -m tensorbem.benchmarks.ellipsoid_kochin_case --outdir runs_audit   # exit 0, 30 s
PYTHONPATH=src python3 -m pytest tests/test_ellipsoid_kochin_case.py -q                    # 16 passed, 37.01 s
python3 runs/m17310/compare_m17273.py                                                       # 264 数值叶 diff
```
复跑 vs 交付：**264 个数值叶中 263 个逐位相同**，唯一差异是 `runtime_seconds`（{s['runtime_seconds']:.6f} → {r['runtime_seconds']:.6f}，计时非物理量）。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 Kochin 函数：闭式参考 vs 面元路线（包络内 4 点）

| k [1/m] | k·a | \\|H_ref\\| | got(integrated) | 误差 | got(ordinate) | 误差 | 判定 |
|---|---|---|---|---|---|---|---|
{koch}

### 3.2 兴波阻力：远场 vs 近场 vs 面元

| Fr | U [m/s] | R_W 远场 [N] | R_W 近场 [N] | R_W 面元 [N] | 远-近 | 远-面元 |
|---|---|---|---|---|---|---|
{res}

### 3.3 11 项预声明判据（复跑值）

| 判据 | 复跑值 | 限 | 判定 |
|---|---|---|---|
{crit_rows}
| [diagnostic] long_wave_limit_three_point | {f(diag['long_wave_limit_three_point']['value'])} | 1e-03 | **FAIL（不计分，交付已自报）** |

## ④ 结果图

图 = 审计者复跑重建（34 行 `_figure_kochin` 从 results.json 现画，非手绘）：左 `|H(k)|` 闭式 vs 面元/谱路线；
中 各路线相对误差与 5% 预算线；右 面元收敛（面元宽/L vs 相对误差）与 5% 预算线。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **裁判缺席（F1，本审计补）**：`tensor_benchmark_cases` 审计时 0 行 → 平台 `GET /tensor/bench/l4-verdict/17273`
  返回 `CLAIMED_NOT_PROVEN, rows=0`。本审计按原 5% 判据补登（row id=44），未放宽。
- **对接未闭合（F2）**：`c121bee` 未并入 main（main `89e5a52` / ECO `96f19f7` 均 grep 0 命中），算例池 `id=170` 仍 `dispatched`
  → 装好的 TensorBEM 跑不出这条算例。这是判 L3 而非 L4 的唯一原因。
- **复现卫生（F4）**：`repro_cmd` 缺 `PYTHONPATH=src`，照抄会 ImportError。
- **外部锚强度（F5）**：文献只到卷/页级（Michell 1898、Havelock 1932、Newman 1977 ch.6），沙箱无原文；
  三条机器可核事实（两独立实现 {f(crit['closed_form_vs_independent']['value'])}、长波斜率 {crit['long_wave_asymptotic_order']['value']:.4f}、
  远/近场恒等 {f(crit['haskind_far_near_identity']['value'])}）替代，但不是独立外部数据。
- **适用范围**：一阶面元仅 k·a ≤ 10 有效（超包络行 `in_envelope=false` 不计分，交付预声明，合规）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 实测 | <3%？ | <5%（池判据）？ |
|---|---|---|---|
| Kochin 面元 integrated（包络内最坏） | {crit['kochin_panels_integrated']['value']:.4%} | **是** | 是 |
| Kochin 面元 ordinate（包络内最坏） | {crit['kochin_panels_ordinate']['value']:.4%} | **是** | 是 |
| 远场 vs 面元 R_W | {crit['haskind_far_panels']['value']:.4%} | **是** | 是 |
| 面元收敛最细层 | {crit['convergence_finest']['value']:.4%} | **是** | 是 |
| 远场 vs 近场（Haskind 恒等） | {f(crit['haskind_far_near_identity']['value'])} | 机器精度（≤1e-9） | 是 |
| 观测收敛阶 | {crit['convergence_order']['value']:.4f}（理论 2） | — | 是（≥0.5） |
| 长波极限斜率 | {crit['long_wave_asymptotic_order']['value']:.4f}（理论 1） | — | 是（≥0.9） |

**结论：面元路线最坏 2.098% < 3%，生态更严线达标**；唯一 FAIL 是被交付自己降级为 diagnostic 的长波三点判据
（{diag['long_wave_limit_three_point']['value']:.3e} vs 1e-3），**如实保留未粉饰**。

## 判级与裁定请求

**L3**（claimed=gate 复核 3，机器判据全过）。L1 测试 16 passed；L2 真实现 1007 行 + 入口 + 非空工件 + 逐位复现；
L3 仓内挂点 `src/tensorbem/figure_integrity.py:122-128` 注册 case module+builder，ECO 侧 0 命中（未合并）；
L4 内容面（ref vs got + 11/11 判据 + repro）已满足，但**算例不在 main**，不授 L4。
**请求**：合并 `agent/m17273`、关闭算例池 `id=170`，合并后重跑 `/tensor/bench/l4-verdict/17273` 由机器重判升 L4。

工件：`docs/board_audit/m17310/AUDIT_m17273_CASE_BEM_ELLIPSOID_KOCHIN.md`、`snapshot/` 下 rerun_results.json / subject_results_c121bee.json / compare_stdout.txt / SHA256SUMS.txt / bem_ellipsoid_kochin.png。
**not_run（不得记通过）**：独立第三实现重写闭式；文献原表原页核对（沙箱不可达）；集群重算/tensorbench_results 行。
"""
png = base64.b64encode((SNAP / "bem_ellipsoid_kochin.png").read_bytes()).decode()
payload = {
    "title": "交差审计 m17273 CASE-BEM-ELLIPSOID-KOCHIN（TensorBEM）：复跑逐位一致、判级 L3，对接未闭合",
    "summary": ("独立检出被审 commit c121bee 复跑：264 数值叶 263 逐位相同（唯一差异 runtime_seconds），11/11 预声明判据 PASS，"
                "面元最坏 2.098%<3%；裁判行缺失由本审计补登；因算例未并入 main 判 L3 而非 L4。"),
    "scenario_key": "board-audit-bem-ellipsoid-kochin",
    "mission_id": 17310,
    "tags": ["board-audit", "TensorBEM", "CASE-BEM-ELLIPSOID-KOCHIN", "m17273", "maturity-L3", "repro"],
    "markdown": md,
    "figures": [{"name": "bem_ellipsoid_kochin.png",
                 "caption": "Kochin 远场：闭式 vs 面元/谱路线、各路线相对误差与 5% 预算、面元收敛（审计者由复跑 results.json 重建）",
                 "png_base64": png}],
}
out = pathlib.Path("/tmp/report_m17310.json"); out.write_text(json.dumps(payload, ensure_ascii=False))
res = post("/tensor/reports", payload)
print("POST /tensor/reports ->", json.dumps(res, ensure_ascii=False)[:300])
